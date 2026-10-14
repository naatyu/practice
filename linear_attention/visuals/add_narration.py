"""Synthesize timed local narration and mux it with the Manim video.

Optional dependencies: piper-tts==1.8.0 (including NumPy) and FFmpeg.
Voice downloads, intermediate clips, and the output live in ignored .render/.
The checked-in narration text follows the existing WebVTT scene timings.
"""

import argparse
import io
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig


def seconds(timestamp: str) -> float:
    hours, minutes, remainder = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(remainder)


def synthesize(
    voice: PiperVoice, text: str, length_scale: float
) -> tuple[int, np.ndarray]:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        voice.synthesize_wav(
            text,
            output,
            syn_config=SynthesisConfig(length_scale=length_scale, volume=0.8),
        )
    buffer.seek(0)
    with wave.open(buffer, "rb") as audio:
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        sample_rate = audio.getframerate()
        samples = np.frombuffer(
            audio.readframes(audio.getnframes()), dtype="<i2"
        ).copy()

    # Keep a little breathing room around speech, but remove long model silence.
    active = np.flatnonzero(np.abs(samples.astype(np.int32)) > 32)
    if not active.size:
        raise ValueError(f"Voice produced no speech for: {text}")
    start = max(0, int(active[0]) - round(sample_rate * 0.06))
    end = min(len(samples), int(active[-1]) + round(sample_rate * 0.12))
    return sample_rate, samples[start:end]


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=directory / "kda_chunking.mp4")
    parser.add_argument(
        "--output", type=Path, default=directory / ".render/kda_chunking_narrated.mp4"
    )
    args = parser.parse_args()
    if args.video.resolve() == args.output.resolve():
        parser.error("Use a separate output path so the input video stays available.")

    specification = json.loads((directory / "narration.json").read_text())
    timings = [
        (seconds(start), seconds(end))
        for start, end in re.findall(
            r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})",
            (directory / "kda_chunking.vtt").read_text(),
        )
    ]
    texts = specification["segments"]
    if len(timings) != len(texts):
        raise ValueError("Narration must contain one passage per WebVTT cue.")
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
    voice = PiperVoice.load(str(args.model))
    sample_rate = voice.config.sample_rate
    timeline = np.zeros(round(duration * sample_rate), dtype="<i2")
    previous_end = 0.0
    for index, ((start, end), text) in enumerate(zip(timings, texts, strict=True), 1):
        if not previous_end <= start < end <= duration + 0.01:
            raise ValueError(f"Invalid or overlapping cue {index}: {start}, {end}")
        previous_end = end
        available = end - start - 0.25
        length_scale = 1.05
        for _attempt in range(4):
            rate, samples = synthesize(voice, text, length_scale)
            assert rate == sample_rate
            spoken_duration = len(samples) / rate
            if spoken_duration <= available:
                break
            length_scale *= available / spoken_duration * 0.97
            if length_scale < 0.72:
                raise ValueError(
                    f"Shorten passage {index}: speech would be too rushed."
                )
        else:
            raise ValueError(f"Passage {index} does not fit its scene.")
        offset = round((start + 0.08) * rate)
        timeline[offset : offset + len(samples)] = samples
        print(
            f"Passage {index:02}: {spoken_duration:.2f}s / {available:.2f}s available",
            flush=True,
        )

    wav_path = args.output.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(timeline.tobytes())
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
            str(wav_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-af",
            "loudnorm=I=-18:TP=-1.5:LRA=7",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
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
    print(f"Narrated video: {args.output}")


if __name__ == "__main__":
    main()
