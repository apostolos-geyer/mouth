# localtranscription

Live local transcription with Qwen3-ASR + Qwen3-ForcedAligner, running on MPS.

Adapted from the offline pipeline at `~/Desktop/school/spring-2026/tools/qwen-transcriber/`
(built for the entrepreneur interview) to run against the microphone in real time.

## Install

```sh
uv tool install -e ".[mlx,diarize]"    # a system-wide `lt`, live against this clone
uv tool update-shell                   # once, if uv's bin dir isn't on your PATH yet
```

`-e` links the install to `src/` instead of copying it, so edits are live and there's no
reinstall step at all. Drop it for a snapshot — but then `--force` is *not* enough to
update one. uv caches the built wheel for a local path, so a rebuild needs
`--refresh-package localtranscription`, and without it the install silently stays behind.

That failure is worth recognising, because it doesn't look like a stale binary: an old
`lt` paired with a current config file rejects its own config.

```
$ lt tui
Invalid value for --config: unknown option --min-speech.
```

The extras are the two heavy optional paths — `mlx` is the MLX backend (`--backend mlx`,
and what `lt quantize` needs), `diarize` is `lt diarize`. Neither is a default dependency,
because the first pulls the whole mlx stack and the second coremltools. `uv tool install .`
gets the torch path alone.

Or run it out of the repo without installing anything:

```sh
uv sync
uv run lt tui          # or: uv run localtranscription tui
```

## Commands

```sh
lt tui                    # full-screen live view (q quit · p pause · c clear)
lt cli                    # streaming output to stdout
lt dictate                # speech to stdout, then exit (--hold for hold-to-talk)
lt transcribe FILE        # a file, as fast as the machine can (~17x realtime)
lt transcribe FILE --speakers   # ...and label who said what
lt devices                # list microphones
lt languages              # list supported ASR languages
lt backends               # which inference backends are installed
lt models                 # local checkpoints available to --model
lt quantize               # build a quantised checkpoint (the big perf win)
lt diarize FILE           # who spoke when, offline
lt paths                  # where config, transcripts and checkpoints live
lt config                 # defaults for the flags you always pass
lt tune                   # set it up for your machine and your voice
lt cadence 10             # what the partial schedule costs on a 10s utterance

lt tui -l Greek -m 2      # language + mic index
lt cli --wav clip.m4a     # replay a file *in real time*, as if it were the mic
lt transcribe clip.m4a    # the same file, at full speed, for the transcript
lt tui --context "Aristotle, peripatetic, Lyceum"   # words to expect
lt cli --no-record        # don't save audio

lt dictate -b mlx -M qwen3-asr-1.7b-q8g64 | pbcopy    # same --backend/--model as anywhere
```

First run downloads ~5GB of weights. After that the model loads in about 5s.

`--language` is a hint, not a hard constraint — speaking Greek with the English default
still produces Greek, but inconsistently, romanizing the same phrase on one pass and not
the next. Setting it properly is worth it.

## Setup

```sh
lt tune                   # say a few things; it measures and writes your config
lt tune --wav clip.flac   # measure against a recording instead
```

Asks you to say a few things, times how fast this machine transcribes them, and offers a
choice about how quickly text should appear while you talk. Everything it suggests is
measured on the spot: `--min-speech` comes from the shortest phrase you actually said,
and each option is marked with what it would cost *here*.

That last part matters more than it sounds. Whether the fastest option is affordable is a
property of your machine, not of the option — an early version wrote "needs the speed-up
to be affordable at all" into the description of a profile, which baked one laptop's
answer into every laptop's menu. The descriptions now say what the experience is; the
verdict beside them is measured.

## Transcribing a file

```sh
lt transcribe lecture.m4a
```

Same VAD, same model, same outputs as a live session — but the audio is already on disk,
so nothing waits on a clock. Measured on an M3 Max with the 8-bit checkpoint: **17x
realtime**, a 32s clip in 1.9s. Partials are off, because nobody is watching text land and
provisional passes are the expensive half of a live session.

`lt cli --wav` still exists and still paces to the clock — that is for *watching* a replay,
which is a different thing from wanting the transcript.

```sh
lt transcribe interview.m4a --speakers      # or -n 2 if you know the count
```

adds `.rttm`, `.speakers.json` (every word with a speaker) and `.speakers.md`:

```
**speaker 0**  [00:04]
Not, not super at at liberty. But I'll tell you about it.

**speaker 1**  [00:15]
Okay.
```

Checked against a 28-minute 3-speaker interview with a hand-corrected reference
transcript: **6520 words against the reference's 6400**, 3 speakers found unprompted, 16x
realtime for the transcription and 66x for the diarization.

Getting there found two bugs worth naming, because both failed silently:

- **The drain gave up.** A file arrives faster than the model consumes it, so the queue
  builds a backlog. `shutdown_timeout` is 2s — sized for quitting a live session, where one
  abandoned utterance sits among many — and it truncated the interview to **826 words**.
- **Calibration assumed the recording starts quiet.** The threshold is 3x the noise floor,
  and the floor was measured as the median of the first second. This interview had its
  pauses edited out, so it opens on speech: 0.146 against a real floor of 0.004, and 36%
  of the words never reached the model. A file can look at all of itself before deciding,
  so it now takes a low percentile of the whole recording. Mic audio is unaffected — on a
  recorded utterance both estimators produce exactly 0.005.

One decode and one pass over the file — running `lt diarize` afterwards would re-read and
re-analyse it, and leave you pasting a generated filename between two commands.

The prose is attributed per **utterance** and rendered from the utterance's own text, which
took two goes to get right. Per *word* (the obvious choice, since only words carry
timings), every short function word landing in a gap between turns came back unattributed
and broke the paragraph — one clause became eight blocks, three of them the single word
"it". And rebuilt from the aligner's word list, `"Not, not super at at liberty."` reads
`"Not not super at at liberty"`, because the word list has no punctuation. Word-level
labels are right for `.speakers.json`, where something computes with them, and wrong for
prose.

An utterance with the same voice either side of it is folded into that voice — the turns
don't tile the recording, so a short reply inside someone's paragraph comes back
unattributed. Between *different* speakers it stays unattributed, which is the case where
guessing would invent an attribution the audio doesn't support.

## Context

```sh
lt tui --context "Aristotle, peripatetic, the Lyceum"
```

Qwen3-ASR biases decoding toward words you tell it to expect: names, jargon, spellings.
It's a session setting rather than a per-utterance argument, and it's mutable, so the TUI
can edit it while running — `k` opens the field, and the next utterance uses it.

Off the `Backend` protocol (which stays one method wide) and onto a `Biasable` capability
protocol, alongside `Streaming` and `Drafting`.

## Config

The flags you'd otherwise type every time, in a file:

```sh
lt config --init          # write a commented starter
lt config                 # show what it sets, per command
lt config --edit          # open it in $EDITOR
```

```toml
# ~/.config/localtranscription/config.toml

backend = "mlx"
model = "qwen3-asr-1.7b-q8g64"
language = "Greek"

[dictate]
hold = true
record = false

[diarize]
threshold = 0.7
```

Nothing in it is a new setting: every key is an existing flag, and the file only changes
what that flag **defaults** to. A flag you type still wins — not by convention but by
construction, because this feeds Click's `default_map` and the layering happens inside the
parser. There's no per-flag plumbing to forget and no "was this passed?" sentinel to get
wrong, which is where hand-rolled versions of this leak.

A bare key reaches **every command that has that option**, which is nearly all of them —
"how this machine transcribes" is a setting, not a per-command opinion. Only three
commands opt out, each because a flag name genuinely means something else there:
`diarize` (`--threshold` is a cosine distance between voices, not an RMS gate), `quantize`
(`--model` is the checkpoint to convert), and `models` (`--dir` is where to look).

This was the other way round at first — a hand-written list of commands bare keys *did*
reach — and it was wrong three times, once for every command added after it: `tune`
benchmarked stock torch while the config pointed everything else at a quantised MLX
checkpoint, `cadence` printed the shipped schedule rather than the configured one, and
`transcribe` gated speech at 0.3s against a config asking for 0.15s. Every one a silent
wrong answer rather than a failure. Inverted, a new command inherits settings by default
and only a real collision needs writing down.

Everything else takes a table named after the command, because the same flag name doesn't
mean the same thing everywhere: `--threshold` is an RMS gate to a session and a cosine
distance to `diarize`, and a bare key that reached both would collapse every speaker into
one.

Keys are checked against the real CLI, not a list kept in parallel with it. So either
spelling resolves — `max-gap` as `--help` prints it, `max_gap` as the parameter is named,
and `--dir` or `model_dir` for the one place they differ — and an unknown key is an error
with a suggestion rather than a shrug:

```
$ lt config
[dictate] has no --holdd. Did you mean --hold?
```

That's the point of validating at all. A config file is written once and read never; a
typo that silently does nothing is a setting you believe is on for months.

`--config PATH` or `LT_CONFIG` point somewhere else, and a file named by hand must exist —
a typo'd path shouldn't silently fall back to your defaults. `lt config` does its own
loading, so it still reports on a file too broken for any other command to run.

One TOML rule worth knowing, since it bites here: bare keys must come **before** the first
`[table]`, or they land inside it.

## Layout

```
src/localtranscription/
  paths.py       XDG config/data/cache locations
  config.py      the config file, layered under the flags
  vad.py         VAD + Cadence (when partials fire)
  engine.py      model loading, inference worker, session driver
  backends.py    torch / mlx behind one transcribe() method
  quantize.py    build quantised MLX checkpoints (`lt quantize`)
  sources.py     mic and wav frame sources, and remembered VAD thresholds
  audio.py       decode wav/flac/m4a/mp3/mp4 to 16k mono
  recorder.py    per-utterance audio + manifest
  formats.py     txt / words.json / srt / timestamped.md / rttm
  diarize/       who spoke when
    coreml.py      model loading + per-model compute units
    offline.py     segment -> embed -> cluster, whole recording at once
  tui.py         Textual front end
  app.py         typer entrypoint
pocs/
  live.py        v1 — minimal, argparse, transcribes at pauses only
  live2.py       v2 — typer + Textual, fixed-cadence partials
tests/           CPU-only: VAD, cadence, recorder, formats
```

`pocs/` are self-contained PEP 723 scripts (`./pocs/live2.py` just runs). They're frozen
reference points; v3 is the package.

## How it works

`qwen_asr` exposes a `streaming_transcribe()` API, but it's **vLLM-only** and drops
timestamps, so it's no use on a Mac. Instead the mic stream is cut into utterances by an
energy VAD and each is pushed through the same `transcribe()` call the offline tool used —
so the forced aligner stays in play and word-level timestamps survive, re-based onto a
session-wide clock.

```
mic (16k mono) -> 30ms frames -> RMS VAD -> chunk -> queue -> transcribe() -> text + words
```

- Threshold auto-calibrates off 1s of room noise (`--threshold` overrides). Speech opens
  after 90ms above it, closes after 750ms below.
- 300ms pre-roll before each onset so word starts aren't clipped; closing silence trimmed
  to 210ms.
- An utterance needs 300ms of *voiced* audio to count, so coughs and key clicks never
  reach the model to be hallucinated over. `--min-speech` moves that gate, and single
  words want it lower — see below.
- Inference runs on a worker thread; capture never blocks.
- Utterances cap at 30s (the forced aligner's own limit is 180s).

### `--min-speech`: the gate cuts real words

The 300ms is voiced *frames*, not clip length — frames whose RMS clears the threshold. So
it counts the vowel and little else: measured across 237 recorded utterances, a spoken
"Claude" is **exactly 10 voiced frames against a gate of 10**, because the `/kl/` burst
and the final `/d/` read as silence.

```
gate: 10 voiced frames
  0.72s   10 voiced  +0 over   'Claude.'
  0.66s   11 voiced  +1 over   'Are illegal.'
  0.75s   18 voiced  +8 over   'Because.'
```

One frame quicker and the word is discarded before the model sees it — no transcript, no
recording, nothing to debug. That is the failure mode people hit as "short words don't
register", and the answer is `--min-speech 0.15` (or in the config file, where it belongs
if you dictate single words at all).

It stays at 0.3 by default because the gate is a duration heuristic standing in for a
detector that can't tell speech from a keyboard clack — an RMS threshold measures
loudness, not voice, so the only cheap signal for "that was junk" is that junk is short.
0.15 is still five frames, comfortably above a click and above the 120ms cough the gate
was written for. A speech-probability VAD (Silero) is the real fix and would let this go
to near zero; it isn't here yet.

### Partials and adaptive cadence

Text appears while you're still talking. Partial passes transcribe the utterance so far
and land **in the transcript itself** — the in-progress line is a live widget (dim italic,
trailing `▌`) that keeps rewriting until the final pass hardens it in place, so healing is
visible exactly where the text will end up.
Partials **heal** — because each re-transcribes the whole prefix rather than appending, a
word already on screen can be revised:

```
… The quick brown fox.
… The quick brown fox jumps over the lazy.
  The quick brown fox jumps over the lazy dog.
```

The schedule is where the cost lives. Every partial re-encodes its whole prefix, so fixed
spacing `c` fires at `c, 2c, 3c…` and sums to ~`n²/2c` — quadratic in utterance length.
Geometric spacing makes the sum dominated by the final pass, so it's cheaper *and* the
first partial lands sooner:

| 10s utterance | first text | audio processed |
|---|---|---|
| fixed 1.2s (v2) | 1.2s | 53.2s (5.3x realtime) |
| adaptive (v3) | 0.4s | 31.3s (3.1x realtime) |
| `--partials stream` | 0.4s | 10.0s (1.0x realtime) |

`lt cadence <seconds>` prints this for any setting. `--growth 1.0` reverts to fixed
spacing, which is the honest baseline for benchmarking.

Two things keep partials affordable: they skip the forced aligner
(`return_time_stamps=False` — their timestamps get discarded anyway), and they're
droppable, so a stale partial never starves the finals that get saved.

**Known limit:** `--max-gap` caps how long a partial may lag, but once it binds, spacing
is constant again and cost returns to quadratic — so long utterances cost more than the
geometric schedule implies.

### `--partials stream`: the quadratic term, removed

`mlx-qwen3-asr` has an incremental decoder: feed it only the audio it hasn't seen and it
keeps its KV cache across turns. Partials then cost time proportional to the utterance
rather than to the square of it. Opt in with `--partials stream` (mlx only — qwen's own
streaming path is vLLM-only, and vLLM has no Metal support).

Measured on a real 11.4s utterance, q8 on an M3 Max:

| | partials | partial compute | first text | final |
|---|---|---|---|---|
| `--partials reencode` (default) | 7 | 1.94s | 0.84s | 0.61s |
| `--partials stream` | 5 | **0.78s** | 0.81s | 0.65s |

**2.5x less compute for provisional text, at the same latency.** The VAD switches to
emitting deltas, so partial audio sums to the utterance's own length instead of ~n²/2c —
on a 12s utterance, 35.3s of re-processed audio becomes 11.9s.

It is not the default, because the two modes differ in more than cost:

- `reencode` **heals**. Each pass re-transcribes the whole prefix, so a word already on
  screen can be revised — which is the behaviour the live view is built around.
- `stream` **appends**. The decoder commits to a prefix (`stable_text`, monotonic by
  design) and only the tail after it moves. Provisional segments carry that prefix as
  `Segment.stable` so a front end can render it settled and the rest as still-moving.

The saved transcript is identical either way: the final pass is a full `transcribe()` over
the whole utterance with the forced aligner, and nothing about partials touches it.
Verified — both modes produced the same text and the same 55 timed words.

Earlier notes here called the 30s case "marginal" on the assumption torch ran ~6x realtime.
Measurement says otherwise: torch does **~10-15x realtime** for clips of 2s and up (0.19s
for 2s, 0.57s for 8s, 2.07s for 32s), so a 30s utterance's ~193s of scheduled audio is
about 16s of compute — roughly half realtime, comfortable rather than marginal. The ~6x
figure came from a single 3.2s clip where fixed per-call overhead dominates.

## Dictation

`lt tui` and `lt cli` are sessions. `lt dictate` is one dictation:

```sh
lt dictate | pbcopy                 # talk, stop talking, it's on the clipboard
lt dictate | tee -a ~/notes.md
lt dictate || say "nothing heard"
```

Talk; stop talking; the text is on **stdout**, and nothing else ever is. Status, errors and
telemetry go to stderr, so it pipes with no flags and no `2>/dev/null`. There's no trailing
newline when stdout isn't a terminal — the caller is usually pasting into a text field, and
a stray newline sends the message. Nothing was heard means exit 1 with an empty stdout, so
`||` works and `| pbcopy` never clobbers the clipboard with nothing.

No daemon, no socket, no UI in here. It's a surface for other programs to compose on.

### SIGINT means "I stopped talking"

Not "abort". The utterance in progress is still transcribed and printed:

```sh
lt dictate --hold > /tmp/said &   # key down
kill -INT %1                      # key up — and the text still lands
```

That is what makes hold-to-talk work from any hotkey manager without a daemon or a
protocol to invent.

### Two ways to decide you're done

The termination policy is the thing that varies, and a key-driven flow wants the second:

| | ends when | for |
|---|---|---|
| default | the VAD sees 750ms of silence | a bare `lt dictate \| pbcopy`, nothing driving it |
| `--hold` | you signal | hold-to-talk |

Without `--hold`, the pause that closes an utterance also ends the command — so thinking
for a second mid-sentence truncates the dictation there, while the key is still down and
you're still talking:

```
$ lt dictate --wav paused.wav            # speech, 1.5s pause, more speech
But it was like private capital that funded the acquisition.

$ lt dictate --wav paused.wav --hold
But it was like private capital that funded the acquisition. Super appreciate. Thank
you, Charlie. Absolutely, happy to. And so, if you have more questions or you need more.
```

`--hold` costs nothing in release latency. Utterances still close on their own pauses and
transcribe as they go, so when the signal lands only the last one is outstanding — the
join is over work already done. They're joined with a space, because these are pauses
inside one dictation rather than separate lines, and the destination is a text field.

### Events, for anything that wants to draw

`--events` puts JSON lines on stderr. stdout stays the transcript, so a front end
subscribes to one without disturbing the other.

| event | carries | |
|---|---|---|
| `loading` | `model`, `backend`, `detail` | |
| `ready` | `threshold`, `calibrated`, `load` | **the mic is live — talk now** |
| `speech` | — | onset |
| `level` | `rms`, `speech` | ~33/s, for a meter |
| `partial` | `text` | only with `--interim 0.4` |
| `final` | `text`, `took`, `audio` | |
| `empty`, `error` | `message` | |

`ready` is the one that matters. Without it a UI is guessing when the mic opened, and you
clip the front of every dictation — the 300ms pre-roll only covers a mic that is *already*
open, which a process launched on a keypress isn't yet.

### What makes it start fast

`--backend` and `--model` are the same flags as everywhere else, and torch on upstream
weights is still the default — a fresh checkout has no quantised checkpoint, and
`lt quantize` is a deliberate step. What changes for dictation is that **launch cost is
now part of the interaction**, so it's worth knowing what each checkpoint costs. Every
one of these on an M3 Max, `-t` given so no calibration, best of 2:

| `-M` | on disk | launch → ready | 5s utterance |
|---|---|---|---|
| torch bf16 (upstream) | 4.40 GB | 6.35s | 0.59s |
| `qwen3-asr-1.7b-mxfp4` | 1.26 GB | 0.44s | **0.19s** |
| `qwen3-asr-1.7b-nvfp4` | 1.33 GB | 0.41s | 0.21s |
| `qwen3-asr-1.7b-q4g64` | 1.33 GB | **0.39s** | 0.23s |
| `qwen3-asr-1.7b-q5g32` | 1.77 GB | 0.40s | 0.26s |
| `qwen3-asr-1.7b-q6g64` | 1.92 GB | 0.40s | 0.26s |
| `qwen3-asr-1.7b-q8g32` | 2.65 GB | 0.43s | 0.28s |
| `qwen3-asr-1.7b-q8g64` | 2.50 GB | 0.42s | 0.25s |

Two things fall out of that, neither of them obvious:

- **Launch is flat across every quantisation** — 0.39s to 0.44s from 1.26 GB to 2.65 GB.
  Once you're on MLX, time-to-ready is process startup and imports, not weight size, so
  there is nothing to buy by dropping bits. The 15x gap is torch vs MLX, not 4-bit vs
  8-bit.
- **Inference spread is real but small**: 0.19s to 0.28s. On this clip every checkpoint
  produced text identical to torch's *except* `mxfp4`, which ran fastest and got the tail
  of the utterance wrong. That's one utterance, not a WER measurement — treat it as a
  reason to check your own recordings before trusting the fast end, not as a verdict.

So the whole menu costs about the same to start, and the trade left is accuracy against
~0.1s of inference. Which end of that you want isn't something this tool should decide;
`lt models` lists what you have.

Two things get startup there in the first place, and the second is the bigger one:

- **No forced aligner.** Nothing here consumes word timings, so `load_backend(align=False)`
  skips those weights entirely rather than just passing `timestamps=False`. On torch that
  halves the load (4.30s → 2.08s); on mlx q8 it's marginal (0.29s → 0.26s, plus ~0.04s off
  each utterance) because there wasn't much load left to save.
- **A remembered threshold.** Calibrating the room costs a full second — *more than loading
  a quantised model does*, which would make it the reason dictation felt slow. It's cached
  per input device in `$XDG_CACHE_HOME/localtranscription/calibration.json` and reused;
  `--recalibrate` retakes it, `--threshold` skips it. Keyed per device because a laptop mic
  and a desk condenser don't share one, and never written from `--wav`, whose room is a file.

So it starts on a keypress with any of them, and there is nothing to keep resident. That was
the open question a daemon would have existed to answer, and at ~0.4s it does not need one.

## Where things go

Nothing is written into the working directory: a session run inside a repo would
otherwise drop transcripts and audio into it, and `.gitignore` becomes load-bearing.
Defaults follow the XDG Base Directory spec.

```sh
lt paths                  # show them, and which flag overrides each
```

| | default | override |
|---|---|---|
| config | `$XDG_CONFIG_HOME/localtranscription/config.toml` | `--config` |
| transcripts | `$XDG_DATA_HOME/localtranscription/out` | `--out` |
| recordings | `$XDG_DATA_HOME/localtranscription/recordings` | `--record-dir` |
| checkpoints | `$XDG_CACHE_HOME/localtranscription/models` | `--model` |

Falling back to `~/.config`, `~/.local/share` and `~/.cache`. Checkpoints live under the **cache**
because `lt quantize` rebuilds any of them from upstream weights — losing that directory
costs time, not work. Transcripts and recordings are data and don't.

macOS's own convention is `~/Library/Application Support`; the env vars are honoured, so
`XDG_DATA_HOME=~/Library/Application\ Support` gets that without a code change. A relative
or empty `XDG_*` value is ignored, per the spec — honouring one would put user data
wherever the process happened to start, which is the failure this avoids.

## Recordings

On by default. Every utterance is written as FLAC alongside a `manifest.jsonl` entry:

```json
{"id": "utt-0001", "audio": "utt-0001.flac", "start": 0.99, "duration": 3.18,
 "hypothesis": "...", "words": [...],
 "interims": [{"at": 0.4, "text": "…", "took": 0.31}, ...]}
```

`interims` is the schedule the partials actually fired on. That's deliberate: replaying a
real session's exact schedule is the only like-for-like way to measure whether caching
helped. It's also the substrate for the correction loop — audio paired with what the model
thought it heard.

## Backends

Optional capabilities — incremental streaming, drafted partials — are `runtime_checkable`
Protocols (`Streaming`, `Drafting`) rather than a `streaming = True` flag beside the
method, and `Backend` stays one method wide. `isinstance` matches on the method itself, so
a backend cannot claim a capability it doesn't have or grow one and forget to announce it;
it also narrows the type, so the call type-checks without a suppression while reaching for
`open_draft()` *without* probing is still an error. The `getattr` version needed a
suppression broad enough to hide both.


Inference sits behind a `Backend` Protocol in `backends.py`, selected with `--backend`:

```sh
lt backends                # which are installed
lt models                  # checkpoints available to --model
lt tui --backend torch     # default: PyTorch + transformers on MPS
lt tui --backend mlx       # MLX port (needs: uv sync --extra mlx)

lt tui -M qwen3-asr-1.7b-q8g64 -b mlx          # a quantised checkpoint, by name
lt tui --aligner Qwen/Qwen3-ForcedAligner-0.6B --dtype bf16
```

Weights are configuration, not constants. `--model` and `--aligner` each take a checkpoint
name (resolved against the checkpoint directory — `lt models` lists them), a path, or an HF
repo id, and `--dtype` picks compute precision (`auto` = bf16 on torch,
fp16 on mlx). Quantisation is read off the checkpoint rather than passed as a flag,
because it is a property of the weights on disk.

The engine's whole demand on a model is one method — hand it audio, get back text and
optionally word timings:

```python
def transcribe(self, audio, sample_rate, *, language, timestamps) -> Transcription
```

Results normalise to our own `Word`/`Transcription` types rather than passing a library's
objects through, which is what keeps the boundary real: torch calls them `time_stamps`
with `.start_time` attributes, MLX calls them `segments` with `["start"]` keys, and the
engine knows about neither. `--device` is torch-only — MLX uses unified memory and has no
device argument.

### Measured: quantisation is the whole game

An earlier revision of this file concluded "torch is faster here" — MLX measured ~2x
*slower* at every clip length, against the port's advertised 3-4x. That was true, and it
was the wrong comparison: it pitted torch bf16 against MLX **fp16**, and unquantised is
not how you run MLX. Re-measured on this machine (M3 Max, 40-core GPU), same 1.7B weights,
per `transcribe()` call:

| clip | torch bf16/MPS | mlx fp16 | mlx q8/g64 | mlx q4/g64 |
|---|---|---|---|---|
| 0.5s | 0.099s | 0.223s | **0.048s** | 0.044s |
| 2s | 0.284s | 0.613s | **0.129s** | 0.110s |
| 8s | 1.080s | 2.362s | **0.470s** | 0.379s |
| 16s | 2.110s | 4.401s | **0.919s** | 0.719s |
| 30s | 2.658s | 7.216s | **1.961s** | 1.135s |
| load | 10.0s | 0.69s | **0.29s** | 0.24s |
| on disk | 4.4 GB | 4.4 GB | 2.50 GB | 1.33 GB |

So the port is 2.1-2.7x slower than torch unquantised and **2.2x faster at 8-bit**. The
model is memory-bandwidth-bound: shrinking the weights *is* the optimisation, and fp16 vs
bf16 is noise beside it. Nothing about the model tier changes — these are the same 1.7B
weights, quantised, not a smaller model.

8-bit is the recommended setting. Upstream measures it at +0.04pp WER; end-to-end on this
project's own recordings (34s, 5 utterances, 18 timed words), torch bf16 and mlx q8
produced **identical text and identical word timestamps** — all 18 within 20ms.

Quantise the aligner too. With the *fp16* MLX aligner one word ("Oh", an isolated
interjection) landed at 1.28s against torch's 0.24s; with an 8-bit aligner every word
matches torch exactly. Same command, different flag:

```sh
lt quantize Qwen/Qwen3-ForcedAligner-0.6B    # -> models/qwen3-forcedaligner-0.6b-q8g64
lt tui -b mlx -M models/qwen3-asr-1.7b-q8g64 --aligner models/qwen3-forcedaligner-0.6b-q8g64
```

4-bit costs +0.43pp WER upstream for another ~1.7x on long clips.

```sh
lt quantize                                  # 8-bit by default -> models/qwen3-asr-1.7b-q8g64
lt quantize --bits 4                         # speed-first
lt quantize --mode mxfp4                     # MLX float modes: mxfp4, mxfp8, nvfp4
lt models                                    # what's on disk
lt tui --backend mlx -M models/qwen3-asr-1.7b-q8g64
```

`torch` stays the **default** because a fresh checkout has no quantised checkpoint and
`lt quantize` is a deliberate step. Once you've run it, mlx is the fast path.

One upstream limitation worth knowing: `mlx_qwen3_asr`'s loader reads `bits` and
`group_size` out of `quantization_config.json` but always re-quantises with
`mode="affine"`. The `mxfp4`/`mxfp8`/`nvfp4` modes save `.scales` and no `.biases`, so an
affine parameter tree can't be filled and `load_weights` raises. `backends._load_mlx_model`
mirrors their load path and honours `mode`, which is what makes those formats selectable
at all — it's a ~3-line change upstream.

Two bugs in the adapter were only found by running it, both because their README disagrees
with their code: `dtype` is an `mx.Dtype`, not the documented string (a string reaches
`x.astype("float16")` and raises), and `forced_aligner` is `Optional[str | ForcedAligner]`,
not the documented bool — `True` passes straight through `_resolve_aligner` and is returned
*as* the aligner, so every timestamped call would have died. The aligner is also built once
and passed as an instance: given `None` or a string, `_resolve_aligner` constructs a fresh
0.6B aligner on **every call**.

## Making reencode partials cheaper

Partials re-encode *and* re-decode the whole prefix every pass. The obvious target is the
re-encoding — and it's the wrong one. Measured per partial on the q8 MLX path:

| prefix | mel | encoder | generate | encoder's share |
|---|---|---|---|---|
| 2s | 0.000s | 0.007s | 0.125s | 5% |
| 8s | 0.001s | 0.017s | 0.336s | 5% |
| 16s | 0.001s | 0.029s | 0.693s | 4% |
| 30s | 0.001s | 0.052s | 1.508s | 3% |

**The encoder is 3–5% of a partial.** Caching its output — the plan in the original
analysis below — is sound, exactly reusable, and worth at most 5%, and only past 8s where
an attention window completes. The cost is `generate`, which redecodes the entire
transcript one token at a time at ~9ms per token regardless of what the token is.

### `--partials x-draft`: the previous partial is a free draft

Experimental, mlx only. A third value of `--partials`, not a flag beside it — it changes
how a partial is decoded, which is the one thing `--partials` selects.

A partial's answer is almost exactly the previous partial's answer plus a few words. That
makes the previous answer a draft, and `step_many` verifies a whole draft in one pass over
the weights instead of one pass per token:

| draft length | `step_many` | k sequential steps | |
|---|---|---|---|
| 8 | 19.8ms | 71.6ms | 3.6x |
| 16 | 21.7ms | 143.1ms | 6.6x |
| 32 | 19.9ms | 286.2ms | 14.4x |
| 64 | 30.9ms | 572.5ms | **18.5x** |

Replaying the real partial cadence over seven clips — the test fixture plus six recorded
utterances, 67 partials — against full decode:

```
  interview-excerpt   32.0s  14 partials   base 7.99s   draft 4.15s   1.93x
  utt-0001             3.6s   5 partials   base 0.55s   draft 0.42s   1.32x
  utt-0001            12.1s   8 partials   base 2.09s   draft 1.35s   1.54x
  utt-0001             4.3s   5 partials   base 0.45s   draft 0.35s   1.27x
  utt-0001            30.0s  14 partials   base 3.95s   draft 2.40s   1.65x
  utt-0002            30.0s  14 partials   base 6.18s   draft 4.50s   1.37x
  utt-0003            10.0s   7 partials   base 1.10s   draft 0.83s   1.31x
  TOTAL                                    base 22.3s   draft 14.0s   1.59x
```

59% of tokens came from the draft. Two things make that number what it is:

- **The draft is located by text, not by index.** One word inserted near the start shifts
  every later token by one, and index alignment never recovers — which is exactly what a
  partial does when a revision lands. Matching the tail of the generated tokens against an
  n-gram index of the previous answer re-finds the place. This alone took acceptance from
  31% to 59%, and turned two clips that were *slower* into 1.37x and 1.65x.
- **A break-even guard, judged per pass.** A verification costs ~3.5 sequential steps and
  replaces accepted+1 of them, so it needs ~3 accepted tokens to pay. Below that, drafting
  stops for the rest of the pass. Judged per *pass*, not per utterance: the first partials
  carry almost no text to draft from, and latching on their low acceptance switched
  drafting off for precisely the long later passes it pays best on.

**Lossless by construction, not bit-identical.** A draft token is accepted only where it
equals the model's own argmax there, and decoding is greedy, so the accepted path is the
path plain decoding would have taken — and healing survives, since a word the model wants
to revise fails to match and decode resumes there. The caveat is float: `step_many`
batches k positions into one matmul, `step` does them one at a time, and they disagree in
the last bits. On a near tie the argmax can flip. Across those 67 partials, output matched
the library on six clips of seven; the seventh — music bleeding into speech — dropped a
duplicated word (`"real realistic"` → `"realistic"`). Running the same loop with
`WINDOW = 0` reproduces the library exactly on that clip, which is what isolates it to the
batched kernel rather than the accept logic.

That is why this is partials-only. **Finals never take this path**: they run the aligner,
they are what gets saved, and they stay on the library's own `transcribe()`.

### Batching, and why finals aren't

Batching is the obvious lever for `lt transcribe`, which produces a queue of finished
utterances with nothing waiting on them. It does not pay, and the reason is upstream:
`mlx_qwen3_asr.transcribe_batch` is `for index, audio in enumerate(audios)` — a
convenience wrapper, not a batched forward pass. Measured over 12 real utterances
(0.7–3.6s), with and without the aligner, at batch 4 and 12, sorted by length and not:

```
text only            sequential 1.39s   batched 1.39s   1.00x
with timestamps      sequential 1.79s   batched 1.64s   1.09x
```

Noise, and sometimes worse — a mixed-length batch pads to its longest member. Real
batching would mean driving prefill and decode across sequences directly, below
`transcribe()`, with per-sequence EOS tracking and one KV cache each. That is a bigger
job than the drafted decoder and it buys speed on a path already running at 17x realtime.

Where batching *is* used, it was worth it and it's measured: the diarizer's segmentation
runs batch-32 (2.1x over single dispatches, purely from amortising CoreML's per-call
overhead), its embeddings go one dispatch per window across all local speakers, its
pairwise distances are a matmul rather than a scalar loop (143x at two hours), and
`--partials x-draft` verifies a 64-token draft in one `step_many` instead of 64 sequential
steps (18.5x). The pattern holds: batching pays where it amortises a fixed per-call cost,
and does nothing where the work is one autoregressive decode after another.

## Encoder + KV caching, and what's left

`--partials stream` uses the decoder-side half of this. The notes below are why it
works, and what the encoder side would still add.

### Original analysis

Partials re-encode audio already processed. The architecture is friendlier to fixing this
than expected:

- **The audio encoder is chunk-local.** `Qwen3ASRAudioAttention.is_causal = False`, but
  `_prepare_attention_mask` builds a *block-diagonal* mask over fixed `n_window * 2`
  chunks — bidirectional within a chunk, zero across chunks. Completed chunks never see
  later audio, so their encoder output is identical no matter what follows. Only the
  ragged tail chunk needs recomputing.
- **The KV cache follows from that.** The text side is causal with standard
  `past_key_values` / `DynamicCache`. The decoder prefix is `[prompt][audio embeddings]`,
  and if those embeddings are stable, the prefix KV is reusable.

This is why qwen gated streaming to vLLM: their streaming re-feeds all audio too and
leans on vLLM's automatic prefix caching. vLLM has no Metal support, so on a Mac it has
to be hand-rolled — drop below `transcribe()` and drive
`Qwen3ASRForConditionalGeneration` directly.

## Diarization

Who spoke when, as a separate stage from what was said.

```sh
uv sync --extra diarize
lt diarize meeting.m4a                       # wav, flac, m4a, mp3, mp4
lt diarize meeting.m4a -o meeting.rttm       # RTTM for dscore / pyannote.metrics
lt diarize meeting.m4a --words out/x.words.json   # label an existing transcript
lt diarize meeting.m4a -n 3                  # exact speaker count, if known
```

It runs the pyannote community-1 family through **CoreML**, not torch: the segmentation
and embedding networks go to the ANE/GPU, which leaves the Metal GPU free for ASR. On an
M3 Max, a 28-minute 3-speaker interview diarizes in **25s — 65x realtime**. pyannote's own
torch pipeline on MPS is ~24x, and its weights are gated; these conversions are not.

Three stages, which is what the model set dictates:

1. **Segment.** 10s in, `(589, 7)` out. The 7 is a *powerset* — one class per subset of up
   to 3 simultaneous speakers, so a single argmax decides overlap as well as identity.
2. **Embed.** Each locally-detected speaker gets a 256-d embedding of only its own frames.
   Speaker ids inside a window are arbitrary; embeddings are what link a voice across them.
3. **Cluster.** Agglomerative on cosine distance over the whole recording, which is what
   makes "speaker 1" the same person at 00:05 and at 27:00.

Windows are 10s wide and 1s apart, so every frame is decided by ten independent looks.

### What the tuning cost

Two settings had to be found against real audio, because the obvious ones both failed:

- **Cluster only on clean embeddings.** An embedding from frames where two people overlap
  describes neither; a short one describes the mask. Filtering to solo speech of at least
  2s took anchor purity from 59% to 100% on the interview.
- **...but relax that when there isn't enough.** The same filter on a 32s excerpt of rapid,
  overlapping turn-taking left **2 usable embeddings out of 46**, and the clip collapsed to
  one speaker. `_reliable` now walks a ladder and stops at the first rung with enough to
  work on. Filters that assume a long recording fail exactly where the recording is short.

Average linkage, not complete. Complete also resists the chaining that made an early
version report 98.6% of a recording as one speaker — but its threshold is set by a
cluster's *worst* pair, so the value tuned on 2201 embeddings merged everything on 46.

Accuracy: community-1 reports 10.6% DER on AMI SDM using PLDA-scored VBx clustering. This
uses cosine agglomerative clustering, which is simpler and faster but weaker at separating
similar voices — expect worse than 10.6%. `PLDA.mlmodelc` and `plda-parameters.json` ship
in the same repo, so that is the upgrade path, and `cluster_embeddings` is the only
function that would change.

### Apple Silicon specifics

Two things here are not generic numpy:

- **Batched segmentation.** The repo ships a batch-32 export of the same network. One
  dispatch covering 32 windows is 4.08 ms/window against 8.42 ms/window for 32 single
  dispatches — 2.1x, purely from amortising CoreML's per-call overhead. Windows are cut
  per batch, not stacked up front, because stacking 28 minutes of them is 1.07 GB.
- **Pairwise distances via matmul.** `scipy.pdist(metric="cosine")` is a scalar C loop, and
  this is the one axis that grows with recording length. For unit-norm rows the matrix is
  `1 - X @ X.T`, which goes to MLX/Metal when available and to Accelerate (AMX) otherwise:

  | embeddings | scipy pdist | Accelerate | MLX/Metal |
  |---|---|---|---|
  | 1471 (28 min) | 201 ms | 2.1 ms | 2.1 ms |
  | 3000 | 836 ms | 9.1 ms | 7.2 ms |
  | 8000 (~2 hr) | 5964 ms | 105 ms | **41.6 ms** |

  Identical results to 3e-7, and 143x at the size that matters.

Compute units are chosen per model, not per process: segmentation runs the same on ANE or
GPU, while the embedding model is 2.7x faster letting CoreML choose (11.6 ms) than pinned
to the ANE (31.5 ms). `--compute-units` overrides.

## Shutdown and resource bounds

Quitting is immediate — measured at ~0.12s even with an inference deliberately wedged for
60s — and completed utterances are still saved.

- Quit drops queued **partials** (disposable by definition) but keeps queued **finals**
  (the product). A wav ending naturally still finishes its backlog.
- The stop event is owned by the front end, not created inside `run_session`, so quitting
  during model load or calibration actually stops something.
- Results are read off the worker, not from `run_session`'s return value — the UI closes
  before that returns, and reading the return value silently discarded finished work.
- An inference that won't return is abandoned after `SHUTDOWN_TIMEOUT`; the worker is a
  daemon thread.
- **`main()` always hard-exits.** Loading spins up ThreadPoolExecutors inside
  huggingface/transformers, and `concurrent.futures` registers an atexit hook that joins
  those workers *non-daemonically* — so Ctrl-C during load hangs in `_python_exit ->
  t.join()` with all our work already done. Measured: 10s+ and still hanging on a plain
  return, 0.35s with the hard exit (rc 130). Real exceptions still print their traceback
  first.

Nothing is leaked across exit — process death releases memory, fds, and the Metal context,
and no child processes are spawned. Three unbounded-growth paths inside a long session are
capped explicitly:

| path | bound |
|---|---|
| `SessionRecorder._pending` interims | freed when a final is empty or raises; hard cap `MAX_PENDING` |
| TUI transcript widgets | `MAX_LINES` (view only; saved transcript is unaffected) |
| truncated manifest line from a hard exit | skipped on load rather than failing the session |

The inference queue is deliberately *not* bounded: capping it would mean dropping finals,
i.e. losing transcript. Depth is surfaced as `queue N` in the HUD instead.

## Gotchas

- **Don't subclass `threading.Thread`.** CPython keeps private attributes on it and they
  move between versions: `_stop` was a method in 3.12, 3.13 added `_handle`. Both collided
  with names used here and broke at runtime — the second only after uv silently picked a
  different interpreter, since `requires-python` doesn't pin one. Everything holds a
  thread instead of being one.
- **Load the model before starting the TUI.** Loading spawns a subprocess and Textual's
  replacement stdout has no real fileno for it to inherit — `bad value(s) in fds_to_keep`.
- `max_new_tokens=2048` is carried from the offline tool; the 512 default silently
  truncated long chunks there.

## Checks

```sh
uv run ruff check src tests    # lint
uv run ruff format src tests   # format (line length 92)
uv run ty check                # types
uv run pytest tests/ -q        # ~25s, offline
```

All four are clean. Two of ruff's defaults are disabled because they invert this
codebase's design rather than critique it: **PLC0415** (imports inside functions) fires
146 times on the lazy imports that let `lt devices` run without importing torch and let
dictation start on a keypress, and **B008** (calls in argument defaults) is typer's API.
Complexity metrics are off for the same reason — `segment_utterances` and the decode loop
are long because they're state machines, and splitting them would spread the state.

`ty` rather than pyrefly for the type gate: on this codebase pyrefly's `basic` preset
finds only the mlx imports, and `strict` finds 183, of which 107 are "annotate this
parameter" — a migration, not a review. ty found the real annotation bugs at zero
configuration. pyrefly stays installed for `pyrefly infer` and `stubgen`, which ty has no
equivalent of. Both are pre-1.0-ish in different ways: ty is 0.0.73, pyrefly is 1.2.

Three suppressions exist, each with its reason at the site: `mlx.core` is a compiled
extension shipped with no stubs and no `py.typed`, so no checker can resolve it; the
partial-decoder capabilities are probed with `getattr` on purpose and so aren't on the
`Backend` protocol; and `takes_device` is a runtime discriminator over two constructors
that genuinely differ.

There is **no static shape checking for MLX arrays** — pyrefly's tensor support is
torch-only, and the runtime option (jaxtyping + beartype, which does work on `mx.array`)
costs a per-call check in a decode loop, which is the wrong trade here.

## Tests

```sh
uv run pytest tests/ -q      # ~25s, offline
```

CPU-only for VAD, cadence, recorder, formats and the diarization logic. The end-to-end
diarization tests run against a vendored 32-second excerpt in `tests/fixtures/`, whose
speaker turns were labelled from the transcript's *words* rather than from any diarizer
output — so they check correctness, not just that nothing changed. They skip rather than
download when the CoreML weights aren't already cached, which keeps the suite offline.

`tests/test_core.py` also pins the `mlx_qwen3_asr` private names that
`backends._load_mlx_model` reimplements, so an upstream rename fails here with the reason
instead of surfacing as an ImportError halfway through `lt tui`.
