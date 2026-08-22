"""Diarization tests.

Everything here runs off the repo: the CPU-side logic on arrays built in-test, and the
end-to-end path on a vendored 32-second excerpt with hand-labelled turns. The end-to-end
tests skip when the CoreML weights aren't present rather than reaching for the network
mid-suite, so `pytest tests/` stays offline and fast by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mouth.diarize import Turn, label_words, merge_turns
from mouth.diarize.offline import (
    MIN_CORE_EMBEDDINGS,
    OfflineConfig,
    _reliable,
    cluster_embeddings,
    cosine_condensed,
    powerset,
)
from mouth.formats import rttm

FIXTURES = Path(__file__).parent / "fixtures"
EXCERPT = FIXTURES / "interview-excerpt.flac"
TRUTH = FIXTURES / "interview-excerpt.json"


# ---------------------------------------------------------------- powerset


def test_diarizer_protocol_is_duck_typed():
    """The Protocol is the module's declared seam, so something has to meet it.

    isinstance, not issubclass: Diarizer carries data members (name, detail), and
    runtime_checkable only supports issubclass for method-only protocols.
    """
    from mouth.diarize import Diarizer

    class Fake:
        name, detail = "fake", "fake"

        def diarize(self, audio, sample_rate):
            return [Turn(0.0, 1.0, 0)]

    assert isinstance(Fake(), Diarizer)


def test_offline_diarizer_exposes_the_protocol_surface():
    """Checked without loading any weights, so it runs everywhere."""
    from mouth.diarize.offline import OfflineDiarizer

    assert OfflineDiarizer.name == "offline"
    assert callable(OfflineDiarizer.diarize)


def test_class_table_matches_the_powerset():
    """The decode gather table must agree with the class list it replaced."""
    import numpy as np

    from mouth.diarize.offline import _CLASS_TABLE, powerset

    classes = powerset()
    assert _CLASS_TABLE.shape == (len(classes), 3)
    for i, cls in enumerate(classes):
        assert set(np.flatnonzero(_CLASS_TABLE[i])) == set(cls)


def test_powerset_is_silence_then_singles_then_pairs():
    classes = powerset(3, 2)
    assert len(classes) == 7, "the segmentation model has exactly 7 output classes"
    assert classes[0] == ()
    assert classes[1:4] == [(0,), (1,), (2,)]
    assert classes[4:] == [(0, 1), (0, 2), (1, 2)]


def test_powerset_covers_every_subset_once():
    classes = powerset(3, 2)
    assert len(set(classes)) == len(classes)


# ---------------------------------------------------------------- turns


def test_overlap_is_zero_for_disjoint_turns():
    assert Turn(0, 1, 0).overlap(2, 3) == 0.0
    assert Turn(0, 2, 0).overlap(1, 3) == pytest.approx(1.0)


def test_merge_joins_same_speaker_across_a_short_gap():
    merged = merge_turns([Turn(0, 1, 0), Turn(1.2, 2, 0)], max_gap=0.5)
    assert merged == [Turn(0, 2, 0)]


def test_merge_keeps_different_speakers_apart():
    merged = merge_turns([Turn(0, 1, 0), Turn(1.1, 2, 1)], max_gap=0.5)
    assert [t.speaker for t in merged] == [0, 1]


def test_merge_drops_slivers_but_only_after_merging():
    # Two 0.2s fragments of one speaker are a 0.45s turn, not two things to discard.
    merged = merge_turns(
        [Turn(0, 0.2, 0), Turn(0.25, 0.45, 0)], max_gap=0.5, min_duration=0.25
    )
    assert merged == [Turn(0, 0.45, 0)]
    assert merge_turns([Turn(0, 0.1, 0)], min_duration=0.25) == []


def test_label_words_picks_the_turn_with_most_overlap():
    turns = [Turn(0, 1, 0), Turn(1, 2, 1)]
    words = [
        {"text": "a", "start": 0.0, "end": 0.9},
        {"text": "b", "start": 1.1, "end": 1.9},
    ]
    assert [w["speaker"] for w in label_words(words, turns)] == [0, 1]


def test_label_words_leaves_silence_unlabelled():
    # Not being anyone's word is a real answer; forcing it to the nearest turn invents data.
    out = label_words([{"text": "x", "start": 9.0, "end": 9.5}], [Turn(0, 1, 0)])
    assert out[0]["speaker"] is None


# ---------------------------------------------------------------- distances


def test_cosine_condensed_matches_scipy():
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(0)
    for n in (2, 5, 64, 300):
        x = rng.standard_normal((n, 256)).astype(np.float32)
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        assert np.allclose(cosine_condensed(unit), pdist(unit, metric="cosine"), atol=1e-5)


def test_cosine_condensed_is_non_negative():
    """Round-off must not produce a negative distance; linkage rejects those."""
    unit = np.repeat(np.eye(1, 8), 4, axis=0).astype(np.float32)  # four identical rows
    assert cosine_condensed(unit).min() >= 0.0


# ---------------------------------------------------------------- reliability ladder


def test_reliability_prefers_solo_speech_when_there_is_enough():
    n = MIN_CORE_EMBEDDINGS * 3
    voiced = np.full(n, 5.0)
    shared = np.zeros(n)
    shared[: n // 2] = 1.0  # half are contaminated by overlap
    assert _reliable(voiced, shared).sum() == n - n // 2


def test_reliability_falls_back_rather_than_starving_the_clusterer():
    """A short, overlapping recording must not be filtered down to nothing.

    This is the 32s-excerpt case: the strict rung left 2 usable embeddings of 46 and the
    whole clip collapsed to one speaker.
    """
    voiced = np.full(20, 3.0)
    shared = np.full(20, 1.0)  # nothing is solo
    mask = _reliable(voiced, shared)
    assert mask.sum() >= MIN_CORE_EMBEDDINGS


def test_reliability_always_returns_something():
    assert _reliable(np.array([0.1]), np.array([9.0])).all()


# ---------------------------------------------------------------- clustering


def _two_blobs(rng, n=30, dim=32, spread=0.15):
    """Two tight clusters on near-orthogonal axes.

    Clustering is on cosine distance, so the blobs have to differ in *direction*: two
    Gaussians offset along the same ray normalise onto each other.
    """
    a = np.eye(1, dim)[0] + rng.standard_normal((n, dim)) * spread
    b = np.eye(1, dim, 1)[0] + rng.standard_normal((n, dim)) * spread
    return np.vstack([a, b])


def test_cluster_separates_two_distinct_voices():
    rng = np.random.default_rng(1)
    x = _two_blobs(rng)
    labels = cluster_embeddings(x, np.ones(len(x), bool), OfflineConfig())
    assert len(set(labels)) == 2
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[30]


def test_cluster_respects_an_exact_speaker_count():
    rng = np.random.default_rng(2)
    x = _two_blobs(rng)
    labels = cluster_embeddings(x, np.ones(len(x), bool), OfflineConfig(num_speakers=1))
    assert len(set(labels)) == 1


def test_cluster_clamps_to_max_speakers():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((40, 32)) * 10  # no structure: threshold alone would over-split
    labels = cluster_embeddings(x, np.ones(len(x), bool), OfflineConfig(max_speakers=2))
    assert len(set(labels)) <= 2


def test_unreliable_embeddings_are_assigned_but_do_not_define_clusters():
    """The whole point of the split: junk gets a label, it doesn't get a vote."""
    rng = np.random.default_rng(4)
    x = _two_blobs(rng)
    reliable = np.zeros(len(x), bool)
    reliable[:10] = True  # only speaker A is trusted
    reliable[30:40] = True  # and only speaker B
    labels = cluster_embeddings(x, reliable, OfflineConfig())
    assert len(labels) == len(x)
    assert len(set(labels)) == 2


def test_cluster_handles_a_single_embedding():
    labels = cluster_embeddings(np.ones((1, 16)), np.ones(1, bool), OfflineConfig())
    assert labels.tolist() == [0]


# ---------------------------------------------------------------- rttm


def test_rttm_writes_duration_not_offset():
    """The classic way to produce a file that scores as garbage."""
    line = rttm([Turn(1.5, 3.25, 0)], "clip").strip().split()
    assert line[3] == "1.500" and line[4] == "1.750"
    assert line[7] == "spk0"
    assert len(line) == 10


# ---------------------------------------------------------------- end to end


def _diarizer():
    pytest.importorskip("coremltools")
    from mouth.diarize.coreml import DIARIZATION_REPO, fetch
    from mouth.diarize.offline import OfflineDiarizer

    try:  # cached weights only -- the suite shouldn't depend on the network
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        fetch(DIARIZATION_REPO, ["Segmentation.mlmodelc/**", "wespeaker.mlmodelc/**"])
    except Exception as e:  # noqa: BLE001 - any hub failure means "not available here"
        pytest.skip(f"diarization weights not cached: {type(e).__name__}")
    return OfflineDiarizer(OfflineConfig())


@pytest.fixture(scope="module")
def excerpt_turns():
    # audio.load, not soundfile directly: this is the decode path `lt diarize` uses, so
    # the end-to-end test exercises it rather than a parallel one that could rot.
    from mouth.audio import SAMPLE_RATE, load

    return _diarizer().diarize(load(EXCERPT), SAMPLE_RATE), json.loads(TRUTH.read_text())


def test_excerpt_finds_exactly_two_speakers(excerpt_turns):
    turns, truth = excerpt_turns
    assert len({t.speaker for t in turns}) == len(truth["speakers"]) == 2


def test_excerpt_attributes_every_labelled_turn_correctly(excerpt_turns):
    """Each hand-labelled span must land mostly on one predicted speaker, and the two
    ground-truth speakers must map to different predicted ids."""
    turns, truth = excerpt_turns
    dominant = {}
    for span in truth["turns"]:
        share: dict[int, float] = {}
        for t in turns:
            ov = t.overlap(span["start"], span["end"])
            if ov:
                share[t.speaker] = share.get(t.speaker, 0.0) + ov
        assert share, f"nothing predicted over {span['text'][:40]!r}"
        top = max(share, key=lambda spk: share[spk])
        assert share[top] / sum(share.values()) > 0.6, f"{span['speaker']} span is muddled"
        dominant.setdefault(span["speaker"], set()).add(top)

    assert all(len(v) == 1 for v in dominant.values()), "a speaker changed id mid-recording"
    assert dominant["C"] != dominant["S"], "the two speakers collapsed into one"


# ---------------------------------------------------------------- speaker transcript


def _turn(start, end, speaker):
    from mouth.diarize.offline import Turn

    return Turn(start, end, speaker)


def _words(spec):
    """spec: [(text, start)] -> aligned words, half a second each."""
    return [{"text": t, "start": s, "end": s + 0.5} for t, s in spec]


def test_speaker_transcript_keeps_the_sentence_intact():
    """Rendered from the utterance's own text, not rebuilt from aligner words.

    The word list carries no punctuation, so a transcript rebuilt from it reads "Not not
    super at at liberty" where the model wrote "Not, not super at at liberty."
    """
    from mouth.formats import speaker_md

    md = speaker_md(
        [(0.0, "Not, not super at at liberty.")],
        _words([("Not", 0.0), ("not", 0.6), ("super", 1.2)]),
        [_turn(0.0, 3.0, 0)],
    )
    assert "Not, not super at at liberty." in md
    assert "**speaker 0**" in md


def test_speaker_transcript_groups_a_run_rather_than_fragmenting_it():
    """Attribution is per utterance, so a function word landing in a gap between turns
    cannot break a paragraph into one block per word -- which is what per-word grouping
    did: one clause became eight blocks, three of them the single word "it"."""
    from mouth.formats import speaker_md

    md = speaker_md(
        [(0.0, "One."), (1.0, "Two."), (2.0, "Three.")],
        # The middle utterance's word falls in a gap and is unattributed.
        _words([("One", 0.0), ("Two", 1.0), ("Three", 2.0)]),
        [_turn(0.0, 0.9, 0), _turn(1.9, 3.0, 0)],
    )
    assert md.count("**speaker 0**") == 1, md
    assert "One. Two. Three." in md


def test_a_mixed_utterance_goes_to_whoever_holds_most_of_it():
    from mouth.formats import speaker_md

    md = speaker_md(
        [(0.0, "Mostly mine but you got a word in.")],
        _words([("Mostly", 0.0), ("mine", 0.6), ("but", 1.2), ("you", 5.0)]),
        [_turn(0.0, 2.0, 0), _turn(4.9, 6.0, 1)],
    )
    assert "**speaker 0**" in md and "**speaker 1**" not in md


def test_an_utterance_nobody_claims_says_so():
    """Silence between speakers is a real answer; guessing invents an attribution."""
    from mouth.formats import speaker_md

    md = speaker_md(
        [(0.0, "Who said this?")], _words([("Who", 0.0)]), [_turn(9.0, 10.0, 0)]
    )
    assert "**unattributed**" in md


# ---------------------------------------------------------------- speaker blocks


def test_blocks_join_one_voice_and_split_on_the_other():
    from mouth.diarize import speaker_blocks

    blocks = speaker_blocks(
        [(0.0, "Hi."), (2.0, "Still me."), (4.0, "Now me.")],
        [_turn(0, 1.9, 0), _turn(2, 3.9, 0), _turn(4, 6, 1)],
        6.0,
    )
    assert [(b[0], b[3]) for b in blocks] == [(0, "Hi. Still me."), (1, "Now me.")]
    # A block spans from its first utterance to the end of its last -- that span is what
    # gets handed to the aligner, so it has to be contiguous and in recording time.
    assert blocks[0][1] == 0.0 and blocks[0][2] == 4.0
    assert blocks[1][1] == 4.0 and blocks[1][2] == 6.0


def test_a_block_nobody_claims_stays_its_own():
    """Silence between speakers is a real answer here too."""
    from mouth.diarize import speaker_blocks

    blocks = speaker_blocks(
        [(0.0, "Mine."), (2.0, "Nobody's."), (4.0, "Mine again.")],
        [_turn(0, 1.9, 0), _turn(4, 6, 0)],
        6.0,
    )
    assert [b[0] for b in blocks] == [0, None, 0]


def test_blocks_of_nothing():
    from mouth.diarize import speaker_blocks

    assert speaker_blocks([], [_turn(0, 1, 0)], 1.0) == []
    assert [b[0] for b in speaker_blocks([(0.0, "Hi.")], [], 1.0)] == [None]


def test_standalone_alignment_is_a_capability_not_a_requirement():
    """Timing known text over a chosen span is what lets a whole block be timed at once.

    Kept off Backend, like the other three: a backend that only transcribes is still a
    backend, and every fake in these tests is one.
    """
    from mouth.backends import Aligning, Backend

    class Plain:
        name = detail = "plain"

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            return None

    assert isinstance(Plain(), Backend)
    assert not isinstance(Plain(), Aligning)
