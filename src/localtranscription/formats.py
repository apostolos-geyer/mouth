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
        if cur and (
            it["start"] - cur[-1]["end"] > max_gap or len(cur) >= max_words_per_cue
        ):
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


def speaker_md(segments, words, turns) -> str:
    """A transcript grouped by who was talking.

    Attributed per *utterance*, not per word, and rendered from the utterance's own text.
    Both halves of that matter, and the first version got both wrong:

    Per word, every short function word that happened to land in a gap between turns came
    back unattributed and broke the paragraph -- one clause became eight blocks, three of
    them the single word "it". Word-level labels are the right answer for `.speakers.json`,
    where something is going to compute with them, and the wrong one for prose.

    From the utterance's text, because the aligner's word list has no punctuation: rebuilt
    from words, "Not, not super at at liberty." reads "Not not super at at liberty". The
    utterance already holds the sentence as the model wrote it.

    An utterance whose words are split across speakers goes to whoever holds most of it.
    The VAD cuts on silence and not on turns, so a genuine interruption mid-utterance
    exists; it is rare in practice and `.speakers.json` still has the detail.
    """
    from .diarize import label_words

    labelled = label_words(words, turns) if words else []
    bounds = [start for start, _ in segments] + [float("inf")]

    owned = []
    for i, (start, text) in enumerate(segments):
        tally: dict = {}
        for w in labelled:
            if start <= w["start"] < bounds[i + 1] and w.get("speaker") is not None:
                tally[w["speaker"]] = tally.get(w["speaker"], 0) + 1
        owned.append([max(tally, key=lambda k: tally[k]) if tally else None, start, text])

    # An unclaimed utterance with the same voice either side of it belongs to that voice.
    # The diarizer's turns do not tile the recording -- it declines to place low-energy
    # frames -- so a short reply inside one person's paragraph comes back unattributed and
    # would otherwise cut the paragraph in three. Between *different* speakers it stays
    # unattributed, which is the case where guessing would invent an attribution.
    for i, row in enumerate(owned):
        if row[0] is not None:
            continue
        before = next((r[0] for r in reversed(owned[:i]) if r[0] is not None), None)
        after = next((r[0] for r in owned[i + 1 :] if r[0] is not None), None)
        if before is not None and before == after:
            row[0] = before

    lines, run, who = [], [], object()
    for spk, start, text in owned:
        if spk != who:
            if run:
                lines += [
                    f"**{_who(who)}**  [{fmt_clock(run[0][0])}]",
                    " ".join(t for _, t in run),
                    "",
                ]
            run, who = [], spk
        run.append((start, text))
    if run:
        lines += [
            f"**{_who(who)}**  [{fmt_clock(run[0][0])}]",
            " ".join(t for _, t in run),
            "",
        ]
    return "\n".join(lines)


def _who(speaker) -> str:
    return "unattributed" if speaker is None else f"speaker {speaker}"


def write_outputs(out_dir: Path, segments, words, stem=None, turns=None) -> Path | None:
    """Write the same four artifacts the offline tool produced.

    With `turns`, three more: the RTTM every diarization scorer reads, the words with a
    speaker attached, and a transcript grouped by speaker.
    """
    segments = sorted(segments, key=lambda s: s[0])
    words = sorted(words, key=lambda w: w["start"])
    if not segments:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"session-{time.strftime('%Y%m%d-%H%M%S')}"

    (out_dir / f"{stem}.txt").write_text(
        "\n".join(text for _, text in segments) + "\n", encoding="utf-8"
    )
    (out_dir / f"{stem}.words.json").write_text(
        json.dumps(words, indent=2), encoding="utf-8"
    )
    if words:
        (out_dir / f"{stem}.srt").write_text(words_to_srt(words), encoding="utf-8")
        (out_dir / f"{stem}.timestamped.md").write_text(
            words_to_timestamped_md(words), encoding="utf-8"
        )
    if turns:
        from .diarize import label_words

        (out_dir / f"{stem}.rttm").write_text(rttm(turns, stem), encoding="utf-8")
        if words:
            # Words aligned per speaker block already carry one; re-deriving it from
            # timings would be strictly worse than what construction gave us.
            labelled = words if all("speaker" in w for w in words) else label_words(words, turns)
            (out_dir / f"{stem}.speakers.json").write_text(
                json.dumps(labelled, indent=2), encoding="utf-8"
            )
            (out_dir / f"{stem}.speakers.md").write_text(
                speaker_md(segments, words, turns), encoding="utf-8"
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
