"""Energy VAD: cut a frame stream into utterances, with provisional passes along the way."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples

SPEECH_FRAMES_TO_START = 3  # 90ms over threshold opens an utterance
SILENCE_FRAMES_TO_END = 25  # 750ms under threshold closes it
PREROLL_FRAMES = 10  # 300ms kept before onset so word starts aren't clipped
TAIL_FRAMES = 7  # 210ms of the closing silence kept; the rest is trimmed
# Gate on *voiced* frames, not clip length: every clip carries pre-roll plus trailing
# silence, so a length check would pass a 120ms cough as a ~1.1s utterance.
#
# It is a duration heuristic standing in for a detector that can't tell speech from a
# keyboard clack, and it cuts real words: "Claude" said at speed measures 10 voiced
# frames -- exactly the gate, no margin -- because only the vowel clears an RMS
# threshold, while the /kl/ burst and final /d/ read as silence. Hence --min-speech.
MIN_SPEECH_SEC = 0.3
MAX_UTTERANCE_SEC = 30.0  # forced aligner tops out at 180s; flush well before


@dataclass
class Chunk:
    """A slice of audio to transcribe.

    final=False is a provisional look at an utterance still in progress; it gets
    superseded by the final pass over the same (longer) audio.

    incremental=True means `audio` is only what's new since the previous provisional
    chunk, not the whole prefix -- for a backend that keeps its own decoder state. The
    final chunk always carries the complete utterance either way, because that is the
    pass that runs the aligner and gets saved.
    """

    audio: np.ndarray
    start: float
    final: bool
    incremental: bool = False


@dataclass
class Cadence:
    """When the next provisional pass fires, measured in seconds of utterance audio.

    Every partial re-encodes its whole prefix, so the schedule decides the total cost.
    Fixed spacing c fires at c, 2c, 3c... and sums to ~n^2/2c -- quadratic in utterance
    length. Spacing geometrically makes the sum dominated by the final pass, so total
    work is linear in n, *and* the first partial lands sooner because the floor is small.

    growth=1.0 degenerates to fixed spacing of `first`, which is what the v4 caching
    benchmark wants as a baseline.

    Caveat on max_gap: once it binds, spacing is constant again and cost goes back to
    quadratic -- it buys freshness on long utterances by giving up the linearity. It's
    still far cheaper than a small fixed cadence, but at 30s the schedule approaches the
    model's own throughput. Per-chunk encoder caching is the real fix; until then this is
    a latency/compute dial, and `m cadence <seconds>` prints the cost of any setting.
    """

    first: float = 0.4  # also the floor on spacing between partials
    growth: float = 1.6
    max_gap: float = 3.0  # so a long utterance never goes quiet

    def next_at(self, last: float) -> float:
        if last <= 0:
            return self.first
        return min(max(last * self.growth, last + self.first), last + self.max_gap)

    def schedule(self, upto: float) -> list[float]:
        """The firing points across an utterance of `upto` seconds. For tests/benchmarks."""
        out, t = [], self.next_at(0.0)
        while t <= upto:
            out.append(round(t, 3))
            t = self.next_at(t)
        return out


def segment_utterances(
    frame_iter,
    threshold: float,
    on_level=None,
    cadence: Cadence | None = None,
    incremental: bool = False,
    min_speech: float = MIN_SPEECH_SEC,
    cuts=(),
):
    """Cut a stream of fixed-size frames into utterances.

    Yields Chunks. on_level(rms, in_speech) is called per frame so a UI can draw a meter
    without re-reading the audio.

    With a cadence, an in-progress utterance also emits provisional Chunks carrying
    everything heard so far -- so text appears while you're still talking. Re-sending the
    whole prefix is what qwen's own streaming mode does; cutting at a fixed boundary
    instead would slice mid-word and break the way partials heal.

    With incremental=True the provisional chunks instead carry only the audio since the
    last one. That is only correct for a backend holding its own decoder state across the
    utterance -- otherwise each chunk is a fragment with no context.

    min_speech is the voiced audio an utterance needs to count at all. Lower it for
    single words, at the cost of letting shorter noise through to the model.

    cuts are extra times, in seconds, where an open utterance must end -- speaker changes,
    in practice. Silence is the only boundary this can find on its own, which is enough
    for a microphone and not enough for a recording someone has edited the pauses out of:
    with no silence to close on, utterances run to MAX_UTTERANCE_SEC and each one holds
    several people talking. A cut is a boundary the audio does not contain.
    """
    preroll = deque(maxlen=PREROLL_FRAMES)
    utterance: list[np.ndarray] = []
    speech_run = 0
    silence_run = 0
    voiced = 0
    sent = 0  # frames of the current utterance already emitted, for incremental mode
    in_speech = False
    utt_start = 0.0
    consumed = 0
    next_interim = None

    min_voiced = min_speech * SAMPLE_RATE / FRAME_LEN
    pending = sorted(float(c) for c in cuts)
    cut_i = 0

    def build(utterance, silence_run, voiced):
        if voiced < min_voiced:
            return None  # a blip, not speech
        trim = max(0, silence_run - TAIL_FRAMES)
        kept = utterance[: len(utterance) - trim] if trim else utterance
        return np.concatenate(kept)

    for frame in frame_iter:
        consumed += 1
        now = consumed * FRAME_LEN / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(frame**2)))
        loud = rms > threshold
        # Consume every cut this frame has reached. One landing in silence is simply
        # gone -- the boundary it marks already exists.
        at_cut = False
        while cut_i < len(pending) and pending[cut_i] <= now:
            cut_i += 1
            at_cut = True
        if on_level is not None:
            on_level(rms, in_speech)

        if not in_speech:
            preroll.append(frame)
            speech_run = speech_run + 1 if loud else 0
            if speech_run >= SPEECH_FRAMES_TO_START:
                in_speech = True
                silence_run = 0
                voiced = speech_run
                utterance = list(preroll)
                utt_start = max(0.0, now - len(utterance) * FRAME_LEN / SAMPLE_RATE)
                next_interim = cadence.next_at(0.0) if cadence else None
                sent = 0
                preroll.clear()
            continue

        utterance.append(frame)
        if loud:
            silence_run = 0
            voiced += 1
        else:
            silence_run += 1
        utt_sec = len(utterance) * FRAME_LEN / SAMPLE_RATE

        if silence_run >= SILENCE_FRAMES_TO_END or utt_sec >= MAX_UTTERANCE_SEC or at_cut:
            audio = build(utterance, silence_run, voiced)
            if audio is not None:
                yield Chunk(audio, utt_start, final=True)
            in_speech = False
            speech_run = 0
            voiced = 0
            sent = 0
            utterance = []
            next_interim = None
            preroll.clear()
        elif next_interim is not None and voiced >= min_voiced and utt_sec >= next_interim:
            assert cadence is not None  # next_interim is only ever set from one
            next_interim = cadence.next_at(utt_sec)
            if incremental:
                fresh = utterance[sent:]
                sent = len(utterance)
                if fresh:
                    yield Chunk(
                        np.concatenate(fresh), utt_start, final=False, incremental=True
                    )
            else:
                yield Chunk(np.concatenate(utterance), utt_start, final=False)

    if in_speech and utterance:  # stream ended mid-utterance
        audio = build(utterance, silence_run, voiced)
        if audio is not None:
            yield Chunk(audio, utt_start, final=True)
