"""`lt tune`: measure this machine and this voice, then pick settings from the numbers.

Every default in this tool is a measurement taken on one M3 Max against one speaker, and
three of them are the difference between "near realtime" and "why is it dropping words":

    --min-speech   the gate a short word has to clear. Set from a table, it cuts real
                   words -- a quick "Claude" measured exactly the 0.3s default here, with
                   no margin. Set from *your* shortest word, it doesn't.
    the cadence    when partials fire. Tighter means the text on screen is fresher, and
                   costs compute this machine may or may not have.
    --x-partial-draft   whether a tighter cadence is affordable at all.

None of that is guessable from the hardware alone: it depends on the checkpoint, the
room, how fast you talk, and how stable the audio is. So measure it. This module does the
arithmetic and the front end in app.py does the talking, which is also what makes the
analysis testable without a microphone in the room.
"""

from __future__ import annotations

import itertools
import platform
import subprocess
import time
from dataclasses import dataclass

import numpy as np

from .vad import FRAME_LEN, MAX_UTTERANCE_SEC, SAMPLE_RATE, Cadence

# ---------------------------------------------------------------- the machine


@dataclass
class Machine:
    chip: str
    cores: int
    memory_gb: float
    platform: str

    @property
    def apple_silicon(self) -> bool:
        return self.platform == "Darwin" and self.chip.startswith("Apple")


def _sysctl(key: str) -> str:
    """One sysctl value, or "" if this isn't a Mac or the key is gone."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def machine() -> Machine:
    """What we are running on. Best-effort: nothing here is load-bearing enough to fail."""
    import os

    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    mem = _sysctl("hw.memsize")
    return Machine(
        chip=chip or "unknown",
        cores=os.cpu_count() or 0,
        # GiB, not GB: a 128 GB machine reports 137438953472 bytes, and printing
        # "137.4 GB" back at its owner reads as a bug in the tool.
        memory_gb=round(int(mem) / (1024**3), 1) if mem.isdigit() else 0.0,
        platform=platform.system(),
    )


# ---------------------------------------------------------------- what you said


@dataclass
class Sample:
    """One recorded utterance, with the measurement the VAD would have made of it."""

    label: str
    audio: np.ndarray
    voiced: int  # frames over the threshold -- the number --min-speech is compared against

    @property
    def seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def voiced_sec(self) -> float:
        return self.voiced * FRAME_LEN / SAMPLE_RATE


def count_voiced(audio: np.ndarray, threshold: float) -> int:
    """Frames whose RMS clears the threshold. The same quantity segment_utterances gates on.

    Not the same as the clip's length, and that gap is the whole point: a clip carries
    pre-roll and a trailing tail, and only the loud middle counts. For "Claude" that is
    the vowel -- the /kl/ burst and final /d/ read as silence.
    """
    n = len(audio) // FRAME_LEN
    if n == 0:
        return 0
    frames = audio[: n * FRAME_LEN].reshape(n, FRAME_LEN)
    return int((np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) > threshold).sum())


def suggest_min_speech(
    samples: list[Sample],
    *,
    margin: float = 0.75,
    floor: float = 0.05,
    ceiling: float = 0.3,
) -> float:
    """A gate the shortest thing you actually said clears, with room to spare.

    Below the shortest sample rather than at it: you will say a shorter one tomorrow. The
    floor is there because the gate is still the only thing standing between a key click
    and the model, and the ceiling is the shipped default -- tuning should never make a
    machine *less* able to hear a short word than an untuned one.
    """
    voiced = [s.voiced_sec for s in samples if s.voiced_sec > 0]
    if not voiced:
        return ceiling
    return round(min(max(min(voiced) * margin, floor), ceiling), 3)


# ---------------------------------------------------------------- what it costs


@dataclass
class CostModel:
    """Partial cost as fixed overhead plus a term proportional to the prefix.

    Both halves are real and they trade off differently: the fixed part is dispatch and a
    short decode, and dominates the sub-2s partials that make up most of a session; the
    proportional part is redecoding a growing transcript, and dominates long utterances.
    A single "x realtime" number hides which one you are up against.
    """

    fixed: float
    per_sec: float
    samples: int = 0

    def at(self, prefix: float) -> float:
        return max(0.0, self.fixed + self.per_sec * prefix)


def fit(points: list[tuple[float, float]]) -> CostModel:
    """Least squares over (prefix seconds, seconds taken).

    Degenerate inputs fall back to a flat mean rather than raising: a tuning run that
    happened to measure one prefix length should produce a worse model, not a crash.
    """
    if not points:
        return CostModel(0.0, 0.0, 0)
    xs = np.array([p for p, _ in points], dtype=float)
    ys = np.array([t for _, t in points], dtype=float)
    if len(points) < 2 or float(xs.std()) < 1e-6:
        return CostModel(float(ys.mean()), 0.0, len(points))
    slope, intercept = np.polyfit(xs, ys, 1)
    return CostModel(float(intercept), float(slope), len(points))


# ---------------------------------------------------------------- the profiles


@dataclass
class Profile:
    """A choice about how often text should appear, in words a user can weigh.

    `blurb` describes the experience and nothing else. It deliberately says nothing
    about whether the machine can afford it -- that answer is measured, differs between
    machines, and an earlier version of this hard-coded one machine's answer into the
    description of the option.
    """

    name: str
    cadence: Cadence
    blurb: str


PROFILES = [
    Profile("calm", Cadence(0.4, 1.6, 3.0), "text catches up in comfortable chunks"),
    Profile(
        "balanced", Cadence(0.15, 1.25, 1.2), "text follows along a sentence at a time"
    ),
    Profile(
        "instant", Cadence(0.15, 1.15, 0.6), "text chases your voice as closely as it can"
    ),
]


@dataclass
class Verdict:
    """What a profile costs on this machine, for an utterance of a given length."""

    profile: Profile
    partials: int
    compute: float  # seconds of inference, drafted
    compute_full: float  # the same without drafting
    stale_avg: float
    stale_max: float
    length: float
    speedup: float = 1.0

    @property
    def load(self) -> float:
        """Inference seconds per second of speech. Over 1.0 and the worker falls behind."""
        return self.compute / self.length if self.length else 0.0

    @property
    def load_full(self) -> float:
        return self.compute_full / self.length if self.length else 0.0

    @property
    def headroom(self) -> str:
        """How hard this works the machine, for someone who will never read a ratio."""
        if self.load > 1.0:
            return "too much"
        if self.load > SUSTAINABLE:
            return "strained"
        if self.load > 0.4:
            return "works for it"
        return "easy"

    @property
    def safe_without_drafting(self) -> bool:
        """Whether losing drafting costs freshness rather than correctness.

        Drafting stops paying on unstable audio -- music, crosstalk -- and the guard
        switches it off mid-utterance. A profile that only fits *with* it doesn't degrade,
        it falls off: the worker drops partials and the text stops moving.
        """
        return self.load_full <= SUSTAINABLE


#: Inference-to-speech ratio we are willing to plan for. Not 1.0: the queue also carries
#: finals, which are what actually gets saved, and they must never wait behind partials.
SUSTAINABLE = 0.75


def evaluate(
    profile: Profile, drafted: CostModel, full: CostModel, length: float
) -> Verdict:
    """Play a profile's schedule out over one utterance and add up what it would cost.

    Staleness is the number that describes the experience: at each partial, how far behind
    the text on screen is -- the wait since the previous one, plus the time this one takes
    to compute. The maximum is what you notice, because it is the longest the text sits
    still while you are still talking.
    """
    points = profile.cadence.schedule(length) or [length]
    gaps = [points[0]] + [b - a for a, b in itertools.pairwise(points)]
    costs = [drafted.at(p) for p in points]
    stale = [g + c for g, c in zip(gaps, costs, strict=True)]
    total, total_full = sum(costs), sum(full.at(p) for p in points)
    return Verdict(
        profile=profile,
        partials=len(points),
        compute=total,
        compute_full=total_full,
        stale_avg=float(np.mean(stale)),
        stale_max=float(max(stale)),
        length=length,
        speedup=(total_full / total) if total > 0 else 1.0,
    )


def recommend(verdicts: list[Verdict]) -> Verdict:
    """The freshest profile that still keeps up, preferring one that degrades gracefully.

    Two passes rather than one score: a profile that only fits while drafting is accepting
    well is a worse recommendation than a slightly staler one that fits either way, and no
    weighted sum expresses that as clearly as looking for it first.
    """
    ordered = sorted(verdicts, key=lambda v: v.stale_max)
    for v in ordered:
        if v.load <= SUSTAINABLE and v.safe_without_drafting:
            return v
    for v in ordered:
        if v.load <= SUSTAINABLE:
            return v
    return max(verdicts, key=lambda v: -v.load)  # nothing fits; the cheapest is least bad


# ---------------------------------------------------------------- the config


def render(
    *,
    backend: str,
    model: str | None,
    aligner: str | None,
    min_speech: float,
    profile: Profile,
    drafting: bool,
    note: str = "",
) -> str:
    """The config file this tuning run implies, as text to show before it is written."""
    lines = [
        "# localtranscription -- written by `lt tune`.",
    ]
    if note:
        lines += [f"# {line}" for line in note.splitlines()]
    lines += ["", f'backend = "{backend}"']
    if model:
        lines.append(f'model = "{model}"')
    if aligner:
        lines.append(f'aligner = "{aligner}"')
    lines += [
        "",
        "# The gate a word has to clear to count as speech at all, measured against the",
        "# shortest phrase recorded during tuning.",
        f"min-speech = {min_speech:g}",
    ]
    if drafting:
        lines += [
            "",
            "# Experimental: decode each partial against the previous one as a draft.",
            "x-partial-draft = true",
        ]
    c = profile.cadence
    lines += [
        "",
        f"# Cadence profile: {profile.name} -- {profile.blurb}.",
        "[tui]",
        f"interim = {c.first:g}",
        f"growth = {c.growth:g}",
        f"max-gap = {c.max_gap:g}",
        "",
    ]
    return "\n".join(lines)


def capture(source, threshold: float, *, limit: float = 12.0) -> Sample | None:
    """Wait for one utterance from a frame source and return it, or None if nothing came.

    Runs the real VAD with the gate wide open (min_speech=0), because the whole point is
    to measure how short the shortest thing you say actually is -- gating that measurement
    by the value being chosen would only ever confirm the current setting.
    """
    import threading

    from .vad import segment_utterances

    stop = threading.Event()
    deadline = time.monotonic() + limit
    watch = threading.Timer(limit, stop.set)
    watch.daemon = True
    watch.start()
    try:
        for chunk in segment_utterances(source.frames(stop), threshold, min_speech=0.0):
            if chunk.final:
                return Sample("", chunk.audio, count_voiced(chunk.audio, threshold))
            if time.monotonic() > deadline:
                break
    finally:
        watch.cancel()
        stop.set()
    return None


def bench_lengths(seconds: float) -> list[float]:
    """Prefix lengths to time a partial at, spanning what a real cadence would ask for.

    Geometric, because that is the shape of the schedule being modelled -- sampling
    linearly would fit the model mostly on long prefixes a session rarely reaches.
    """
    top = min(seconds, MAX_UTTERANCE_SEC)
    out, t = [], 0.5
    while t < top:
        out.append(round(t, 2))
        t *= 1.8
    out.append(round(top, 2))
    return out
