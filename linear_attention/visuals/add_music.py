"""Generate a quiet original ambient bed and mix it under clean narration.

Requires NumPy and FFmpeg. The input should be the narration-only video from
add_narration.py; using that clean stem prevents music layering on rebuilds.
Generated WAVs and intermediate videos stay in ignored .render/.
"""

import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
# Slowly crossfading Cmaj9, Am7, Fmaj9, and Gsus2 voicings, in MIDI notes.
CHORDS = [
    (48, 55, 59, 62, 64),
    (45, 52, 55, 59, 60),
    (41, 48, 52, 55, 57),
    (43, 50, 55, 57, 62),
]


def ambient_bed(duration: float, rms_db: float) -> np.ndarray:
    """Return deterministic stereo PCM, with slow attacks and no percussion."""
    samples = round(duration * SAMPLE_RATE)
    bed = np.zeros((samples, 2), dtype=np.float32)
    rng = np.random.default_rng(7)
    stride = 16
    chord_duration = 32
    t = np.arange(chord_duration * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    envelope = np.sin(np.pi * t / chord_duration) ** 2

    for index, start_seconds in enumerate(
        range(-stride, int(duration) + stride, stride)
    ):
        chord = np.zeros((len(t), 2), dtype=np.float32)
        for voice, midi in enumerate(CHORDS[index % len(CHORDS)]):
            frequency = 440 * 2 ** ((midi - 69) / 12)
            phase = rng.uniform(0, 2 * np.pi)
            # Gentle detuning and weak harmonics keep this softer than a pure
            # sustained test tone; slow envelopes avoid clicks at transitions.
            tone = (
                0.65 * np.sin(2 * np.pi * frequency * 0.999 * t + phase)
                + 0.35 * np.sin(2 * np.pi * frequency * 1.001 * t + phase)
                + 0.10 * np.sin(4 * np.pi * frequency * t + phase)
                + 0.025 * np.sin(6 * np.pi * frequency * t + phase)
            ) * envelope
            pan = np.linspace(-0.45, 0.45, 5)[voice]
            chord[:, 0] += (tone * np.sqrt((1 - pan) / 2)).astype(np.float32)
            chord[:, 1] += (tone * np.sqrt((1 + pan) / 2)).astype(np.float32)

        first = start_seconds * SAMPLE_RATE
        lo, hi = max(0, first), min(samples, first + len(chord))
        if lo < hi:
            bed[lo:hi] += chord[lo - first : hi - first]

    # Ease in over eight seconds; let the ending disappear over twelve.
    fade_in = min(samples, 8 * SAMPLE_RATE)
    fade_out = min(samples, 12 * SAMPLE_RATE)
    bed[:fade_in] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_in)))[:, None]
    bed[-fade_out:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade_out)))[:, None]
    rms = np.sqrt(np.mean(bed.astype(np.float64) ** 2))
    bed *= 10 ** (rms_db / 20) / rms
    if not np.isfinite(bed).all() or np.max(np.abs(bed)) >= 1:
        raise ValueError("Music level would clip; reduce --music-db.")
    return np.rint(bed * 32767).astype("<i2")


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path, default=directory / ".render/kda_chunking_narrated.mp4"
    )
    parser.add_argument(
        "--output", type=Path, default=directory / ".render/kda_chunking_with_music.mp4"
    )
    parser.add_argument(
        "--music-db",
        type=float,
        default=-36,
        help="Ambient RMS level before speech ducking, in dBFS (default: -36).",
    )
    args = parser.parse_args()
    if args.video.resolve() == args.output.resolve():
        parser.error(
            "Use a separate output path to preserve the clean narration source."
        )
    duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(args.video),
            ],
            text=True,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    music = ambient_bed(duration, args.music_db)
    music_path = args.output.with_suffix(".wav")
    with wave.open(str(music_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(music.tobytes())

    filters = (
        "[0:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
        "asplit=2[voice][speech];"
        "[1:a:0]aformat=sample_rates=48000:channel_layouts=stereo[bed];"
        "[bed][speech]sidechaincompress=threshold=0.04:ratio=3:"
        "attack=40:release=700[ducked];"
        "[voice][ducked]amix=inputs=2:duration=first:normalize=0,"
        "alimiter=limit=0.794:level=0:latency=1[mix]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.video),
            "-i",
            str(music_path),
            "-filter_complex",
            filters,
            "-map",
            "0:v:0",
            "-map",
            "[mix]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-t",
            str(duration),
            str(args.output),
        ],
        check=True,
    )
    print(
        f"Original ambient bed: {music_path}; RMS before ducking: {args.music_db} dBFS"
    )
    print(f"Narrated video with music: {args.output}")


if __name__ == "__main__":
    main()
