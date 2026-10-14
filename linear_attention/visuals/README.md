# Watching KDA chunking happen

The educational linear-attention/KDA track is complete. This lesson remains
review material, alongside the explicitly AI-authored chunkwise worked example.

[![Watch the chunking animation](kda_chunking_poster.jpg)](kda_chunking.mp4)

[Watch or download the video](kda_chunking.mp4). It is a captioned
Manim animation with synthetic English narration and a quiet ambient music
bed; pause whenever you want to check a number. The separate
[WebVTT captions](kda_chunking.vtt) are available too. GitHub may download the
MP4 instead of playing it inline.

The video is 4 minutes 16 seconds, at 720p / 30 fps. Chapters:

- `0:09` — why the original loop is sequential
- `0:36` — the three-token numerical example
- `0:50` — decay of earlier writes versus incoming memory
- `1:22` — key alignment and the writer's beta
- `1:49` — the triangle of prediction-error dependencies
- `2:08` — the block error system and its numerical solution
- `2:53` — outputs and outgoing memory
- `3:19` — work within chunks and state between chunks
- `3:51` — what is feasible to implement for learning

The lesson follows one three-token example from individual memory updates to
one lower-triangular error system, then shows what moves between chunks.
It uses Manim Community for moving vectors, fading memory channels, and
animated matrix entries. This teaching asset was built with AI assistance;
the core KDA implementation remains the learner's code.

## The idea before the algebra

Each token reads what memory already predicts, computes an error, and writes
a correction. The next token's prediction therefore depends on that write.
Putting the same token loop into smaller blocks does not remove this
dependency.

But the earlier error affects the later prediction *linearly*. We can collect
those effects into a triangular system for a whole chunk, solve its errors,
then calculate the outputs and outgoing state with matrix operations.
The causal dependencies remain present. A triangular solve itself has
dependencies; a block formulation exposes compiled, batched work rather than
independent predictions for every token. Actual speed depends on the backend.

This is an educational derivation of the recurrent KDA equation. The
[Kimi Linear paper, section 3.1 and appendix B](https://arxiv.org/pdf/2510.26692)
uses WY transforms and specialized kernels. The animation does not reproduce
those kernels or claim a benchmark speedup.

## Follow a single write

Let `g_t = log(alpha_t)`, one log-retention value per key channel. Inside the
chunk, `G_t` is the cumulative sum through token `t`.

A correction written at token `i` is added **after** that token's decay.
When token `t >= i` reads it, the surviving part is scaled by
`r(t, i) = exp(G_t - G_i)`. This includes gates `i+1 ... t`. At `i=t`, the
factor is one: a new write has not yet been decayed.

The incoming state existed before the chunk. Its contribution at token `t`
is scaled by `exp(G_t)`, which includes gates `0 ... t`. This distinction
prevents the off-by-one error we discussed.

## Turn the dependencies into equations

For one head, with row-vector keys and an incoming state `S_in`:

```text
b_t = v_t - (k_t * exp(G_t)) @ S_in
a_ti = beta_i * dot(k_t, r(t, i) * k_i), for i < t
e_t = b_t - sum(a_ti * e_i, i < t)
```

Here `*` scales each key channel. The writer's `beta_i` belongs in the
coefficient because it controlled the strength of that earlier write.
Key alignment can be negative, so these coefficients are not probabilities.

Put `a_ti` in a strictly lower-triangular matrix `A` and stack the errors:

```text
(I + A) E = B
```

`I + A` has shape `[B, H, C, C]`, and the right-hand side and errors have
shape `[B, H, C, Dv]`. Upper-triangular entries are zero because future writes
cannot affect the current prediction. The diagonal of `A` is zero because
the error is computed before the current write. A unit-diagonal triangular
solve gives `E`; no explicit matrix inverse is needed.

For output readouts, use query vectors instead of the prediction keys:

```text
m_ti = beta_i * dot(q_t, r(t, i) * k_i), for i <= t
o_t = (q_t * exp(G_t)) @ S_in + sum(m_ti * e_i, i <= t)
```

This output matrix `M` **includes the diagonal**: queries read memory after
their own correction is written. Finally, decay the incoming state through
the chunk and add every correction with its surviving retention to obtain
`S_out`. Carry that one matrix to the next chunk.

## The numbers in the video

One head, two key channels, one value feature, with `S_in = [1, 1]^T` and
`q_t = k_t` for readability:

| Token | Key | Retention | Beta | Value |
| --- | --- | --- | --- | --- |
| 0 | `[1, 0]` | `[0.8, 0.9]` | 0.5 | 2 |
| 1 | `[0.6, 0.8]` | `[0.5, 0.8]` | 0.75 | 2 |
| 2 | `[0.8, 0.6]` | `[0.9, 0.5]` | 1 | 3 |

```text
        [1      0      0]       [1.2   ]
I + A = [.150   1      0]   B = [1.184 ]
        [.180  .504    1]       [2.496 ]

(I + A) E = B

E       = [1.2, 1.004, 1.773984]^T
O       = [1.4, 1.749, 3]^T
S_out   = [2.4558072, 1.7255904]^T
```

The video rounds some labels to three decimals; the source calculations keep
full precision. Reproduce the numerical check without Manim:

```bash
uv run python linear_attention/visuals/chunking_example.py
```

Both errors and readouts, plus the final state, are checked against the
original sequential recurrence. The video's outputs and state have also
been checked against this repo's `recurrent_kda`.

## From the video to code

[chunked_kda.py](../chunked_kda.py) now implements this regrouping with
log-prefix differences and `torch.linalg.solve_triangular`. Codex wrote it
at the learner's request as a worked example, not as learner-authored code.
Its tests compare against the recurrence, including gradients, nonzero
incoming state, chunks of size one, and a short final chunk. Follow the four
numbered sections with the [code walkthrough](../SOLUTIONS.md#chunkwise-code-walkthrough-ai-authored).

Building pairwise decay naively can allocate `[B, H, C, C, Dk]`, which is
larger than the `[B, H, C, C]` coefficient matrix. Mask noncausal log
differences **before exponentiating** so their positive exponents cannot
overflow. Optimized WY kernels handle this work more carefully; implementing
their tiling, backward pass, and GPU scheduling is substantially more work.

The layer remains recurrent by default. Set a positive `chunk_size` in
`KimiDeltaAttention` to use the worked chunkwise operator with the same cache.

## Render it again

The source is [kda_chunking_scene.py](kda_chunking_scene.py), using
[chunking_example.py](chunking_example.py) for the verified numbers.
Use an optional rendering environment with Manim Community 0.19.0.
For example, with micromamba or conda-forge:

```bash
micromamba create -n kda-video -c conda-forge python=3.12 manim=0.19.0 ffmpeg
micromamba activate kda-video
cd linear_attention/visuals
manim -qm --media_dir .render -o kda_chunking kda_chunking_scene.py KDAChunking
```

`-qm` renders 720p at 30 fps; `-ql` gives a faster preview. The scene uses
DejaVu Sans text and does not need LaTeX. It writes the MP4 under
`.render/videos/kda_chunking_scene/720p30/` and the WebVTT captions under
`.render/`. `.render/` is ignored; the linked MP4, poster, and captions here
are the checked-in teaching assets. Manim is not required for the repository's
attention exercises. For a pip/uv setup, see the
[Manim installation instructions](https://docs.manim.community/en/stable/installation/uv.html)
and the pinned [rendering requirements](requirements.txt).

## Rebuild the voice track

The spoken passages are editable in [narration.json](narration.json).
[add_narration.py](add_narration.py) synthesizes each passage locally, fits it
to its matching caption interval, leaves reading pauses, normalizes volume,
and adds an AAC track while copying the original video frames. It checks for
overlapping intervals and rejects text that would need excessive speeding up.

The voice is Piper's `en_US-ljspeech-high`, a synthetic US English voice.
See the [Piper project](https://github.com/OHF-Voice/piper1-gpl) and
[voice model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ljspeech/high/MODEL_CARD)
for attribution. Voice models and generated WAVs stay in ignored `.render/`.
After rendering the silent scene, run from `linear_attention/visuals`:

```bash
uv run --no-project --with piper-tts==1.8.0 python -m piper.download_voices \
  en_US-ljspeech-high --download-dir .render/voices
uv run --no-project --with piper-tts==1.8.0 python add_narration.py \
  --model .render/voices/en_US-ljspeech-high.onnx \
  --video .render/videos/kda_chunking_scene/720p30/kda_chunking.mp4
```

The result is `.render/kda_chunking_narrated.mp4`. The default input is the
published video if `--video` is omitted; only its image track is copied, so
rerunning synthesis does not layer two voices together. FFmpeg must be on
`PATH`. The checked-in video already includes this narration and the music
described below.

## Rebuild the background music

[add_music.py](add_music.py) creates an original ambient bed from slowly
crossfading chord tones, with an eight-second fade-in and twelve-second
fade-out. There are no external song recordings. The bed starts at -36 dBFS
RMS, well below the voice, and speech automatically ducks it further. A
limiter leaves headroom; the video frames and duration are copied unchanged.

After generating the clean narration-only video above, run:

```bash
uv run --no-project --with numpy==2.5.4 python add_music.py
```

This produces `.render/kda_chunking_with_music.mp4`. The script defaults to
the clean `.render/kda_chunking_narrated.mp4` input so rebuilding does not
layer music onto an existing music mix. `--music-db -40` makes the bed quieter;
`--music-db -32` makes it louder. Generated audio stays in ignored `.render/`.
