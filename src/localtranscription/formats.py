"""Transcript writers -- same four artifacts the original offline tool produced."""

from __future__ import annotations

import json
import time
from pathlib import Path


def fmt_srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_clock(t: float) -> str:
    mm, ss = divmod(int(t), 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"


def words_to_srt(items, max_words_per_cue=12, max_gap=0.8):
    cues, cur = [], []
    for it in items:
        if cur and (it["start"] - cur[-1]["end"] > max_gap or len(cur) >= max_words_per_cue):
            cues.append(cur)
            cur = []
        cur.append(it)
    if cur:
        cues.append(cur)

    lines = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_time(cue[0]['start'])} --> {fmt_srt_time(cue[-1]['end'])}")
        lines.append(" ".join(w["text"] for w in cue))
        lines.append("")
    return "\n".join(lines)


def words_to_timestamped_md(items, group_seconds=20):
    lines = []
    bucket_start = None
    bucket_words = []

    def flush():
        if bucket_words:
            # A non-empty bucket has a start: both are set on the same word.
            assert bucket_start is not None
            lines.append(f"**[{fmt_clock(bucket_start)}]** {' '.join(bucket_words)}")
            lines.append("")

    for it in items:
        if bucket_start is None:
            bucket_start = it["start"]
        if it["start"] - bucket_start > group_seconds:
            flush()
            bucket_start = it["start"]
            bucket_words = []
        bucket_words.append(it["text"])
    flush()
    return "\n".join(lines)


def write_outputs(out_dir: Path, segments, words, stem=None) -> Path | None:
    """Write the same four artifacts the offline tool produced."""
    segments = sorted(segments, key=lambda s: s[0])
    words = sorted(words, key=lambda w: w["start"])
    if not segments:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"session-{time.strftime('%Y%m%d-%H%M%S')}"

    (out_dir / f"{stem}.txt").write_text(
        "\n".join(text for _, text in segments) + "\n", encoding="utf-8"
    )
    (out_dir / f"{stem}.words.json").write_text(json.dumps(words, indent=2), encoding="utf-8")
    if words:
        (out_dir / f"{stem}.srt").write_text(words_to_srt(words), encoding="utf-8")
        (out_dir / f"{stem}.timestamped.md").write_text(
            words_to_timestamped_md(words), encoding="utf-8"
        )
    return out_dir / stem


def rttm(turns, name: str) -> str:
    """NIST RTTM: what every diarization scorer (dscore, pyannote.metrics) reads.

    Ten space-separated fields; the ones that matter here are onset and duration (not
    onset and offset -- a common way to produce a file that scores as garbage).
    """
    return "".join(
        f"SPEAKER {name} 1 {t.start:.3f} {t.duration:.3f} "
        f"<NA> <NA> spk{t.speaker} <NA> <NA>\n"
        for t in turns
    )
