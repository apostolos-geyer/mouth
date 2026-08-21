"""Offline diarization: the pyannote community-1 family, on CoreML.

Three stages, which is what the model set dictates:

1. **Segment.** A 10s window in, a (589, 7) map out. The 7 is a *powerset*: not "three
   speaker probabilities" but one class per subset of up to 3 speakers that could be
   talking at once -- silence, each speaker alone, each pair. One argmax therefore
   decides overlap too, which is why this beats three independent sigmoids.
2. **Embed.** Each locally-detected speaker gets a 256-d embedding of just its own
   frames, via a mask. Speaker ids inside a window are arbitrary, so embeddings are the
   only thing that can link a voice across windows.
3. **Cluster.** Agglomerative on cosine distance over every window's embeddings, which
   turns per-window local ids into session-wide ones.

Windows overlap heavily (10s wide, 1s apart by default) so every frame is decided by ten
independent looks, and the vote is what makes boundaries stable.

Two choices in `_cluster` were made against a real 28-minute 3-speaker interview, because
the obvious settings both failed on it:

- **Cluster only on clean embeddings.** An embedding taken from frames where two people
  overlap describes neither of them, and a short one describes the mask. On a 28-minute
  interview, clustering on solo speakers holding at least `MIN_CLUSTER_SEC` took anchor
  purity from 59% to 100%. Everything else is then assigned to the nearest resulting
  centroid rather than being allowed to influence where the centroids are.
- **...but relax that when there isn't enough of it.** The same filter applied to a 32s
  excerpt of rapid, overlapping turn-taking left 2 usable embeddings out of 46, and
  everything collapsed to one speaker. Filters that assume a long recording fail exactly
  where the recording is short, so `_reliable` walks a ladder and stops at the first rung
  with enough to cluster on. Unfiltered, that excerpt separates correctly.

Average linkage is used, not complete. Complete resists the chaining that made an early
version merge a whole recording into one speaker, but its threshold is set by the *worst*
pair in a cluster, so the value that worked on 2201 embeddings merged everything on 46 --
scale-dependent in the one dimension that varies most. Once the embedding set is filtered,
average linkage separates correctly at the same threshold on both.

Accuracy note: community-1 reports 10.6% DER on AMI SDM using PLDA-scored VBx clustering.
This uses cosine agglomerative clustering instead, which is simpler and materially
faster but not as good at separating similar voices -- expect worse than 10.6%. The repo
ships PLDA.mlmodelc and plda-parameters.json, so that is the upgrade path, and
`_cluster` is the only function that would change.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Turn, merge_turns
from .coreml import DIARIZATION_REPO, DiarizationUnavailable, resolve


def cosine_condensed(unit: np.ndarray) -> np.ndarray:
    """Condensed pairwise cosine distances for unit-norm rows.

    scipy's ``pdist(metric="cosine")`` is a scalar C loop over every pair, and this is an
    O(n^2) problem on the one axis that grows: a longer recording is exactly when it
    hurts. For unit-norm rows the whole matrix is ``1 - X @ X.T``, one matmul.

    Measured on an M3 Max (n = embeddings, 256-d), producing the identical result to 3e-7:

        n      pdist     Accelerate    MLX/Metal
        1471   201 ms      2.1 ms        2.1 ms
        3000   836 ms      9.1 ms        7.2 ms
        8000  5964 ms    105.0 ms       41.6 ms

    MLX wins once the matrix is big enough to cover the dispatch, and by 143x over scipy
    at 8000 -- about two hours of audio. numpy is not a slow fallback here either: it is
    linked against Accelerate, so the matmul lands on the AMX coprocessor.

    The triangle is taken with scipy's ``squareform`` rather than ``triu_indices`` fancy
    indexing, which was itself two thirds of the cost at n=1471.
    """
    from scipy.spatial.distance import squareform

    gram = _gram(unit)
    dist = 1.0 - gram
    # Round-off can push a self-similar pair a hair below zero, which linkage reads as a
    # negative distance and refuses.
    np.clip(dist, 0.0, 2.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return squareform(dist, checks=False).astype(np.float64)


def _gram(unit: np.ndarray) -> np.ndarray:
    """X @ X.T on the GPU when MLX is installed, on AMX via Accelerate otherwise."""
    try:
        import mlx.core as mx
    except ImportError:
        return unit @ unit.T
    a = mx.array(np.ascontiguousarray(unit, dtype=np.float32))
    out = a @ a.T
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


WINDOW_SEC = 10.0          # what the segmentation model was exported for
LOCAL_SPEAKERS = 3         # powerset width: at most 3 voices inside one window
EMBED_BATCH = 3            # the embedding model takes exactly one waveform per local speaker

# A local speaker with less than this much voice in a window gets no embedding: too few
# frames and the vector describes the mask, not the person.
MIN_ACTIVITY_SEC = 0.5

# Stricter bar to be allowed to *define* a cluster, as opposed to merely being assigned to
# one. See the module docstring: this is what stopped a 28-minute interview collapsing to
# a single speaker.
MIN_CLUSTER_SEC = 2.0
# Frames a speaker shares with another before its embedding is considered contaminated.
MAX_OVERLAP_SEC = 0.10
# Below this many clustering embeddings the filter is doing more harm than good.
MIN_CORE_EMBEDDINGS = 10


def powerset(num_speakers: int = LOCAL_SPEAKERS, max_simultaneous: int = 2) -> list[tuple]:
    """Class index -> which speakers are talking. Silence, singles, then pairs.

    Matches pyannote's ordering, and deriving it beats a hard-coded table of 7 tuples
    that silently means something else if a model ever ships with different widths.
    """
    out: list[tuple] = []
    for size in range(max_simultaneous + 1):
        out.extend(itertools.combinations(range(num_speakers), size))
    return out


@dataclass
class OfflineConfig:
    hop_sec: float = 1.0            # distance between window starts; 10 looks per frame
    # Cosine distance above which complete linkage refuses to merge. On the interview this
    # was measured against, 0.90-1.05 all give the correct speaker count and the nearest
    # merge heights are 0.87 and 1.07, so 0.95 sits in the middle of a real plateau rather
    # than on a cliff.
    threshold: float = 0.65
    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 8
    compute_units: str = "ALL"
    # The batch-32 export of the same segmentation network. One dispatch covering 32
    # windows measures 4.08 ms/window against 8.42 ms/window for 32 single dispatches on
    # an M3 Max -- 2.1x, purely from amortising CoreML's per-call overhead. Set
    # segmentation_batch=1 with "pyannote_segmentation" to use the unbatched export.
    segmentation_model: str = "Segmentation"
    segmentation_batch: int = 32
    embedding_model: str = "wespeaker"
    max_gap: float = 0.5
    min_duration: float = 0.25


def _reliable(voiced: np.ndarray, shared: np.ndarray) -> np.ndarray:
    """Which embeddings are clean enough to decide where the clusters are.

    A ladder, not a fixed rule. The strict rung -- solo speech, at least MIN_CLUSTER_SEC
    of it -- is what makes a long recording cluster correctly, and it is also what leaves
    a short overlapping one with 2 usable embeddings out of 46. So each rung is tried in
    turn and the first with enough to work on wins; the last rung is "everything", which
    is always available.
    """
    solo = shared <= MAX_OVERLAP_SEC
    for mask in (solo & (voiced >= MIN_CLUSTER_SEC),
                 voiced >= MIN_CLUSTER_SEC,
                 solo,
                 np.ones(len(voiced), dtype=bool)):
        if mask.sum() >= MIN_CORE_EMBEDDINGS:
            return mask
    return np.ones(len(voiced), dtype=bool)


def cluster_embeddings(embeddings: np.ndarray, reliable: np.ndarray,
                       cfg: "OfflineConfig") -> np.ndarray:
    """Embeddings -> a global speaker id each.

    `reliable` marks the ones clean enough to decide where the clusters *are*; the
    rest are assigned to whichever resulting centroid they're nearest. See the module
    docstring for why that split matters.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    # Cosine only compares direction, so a loud stretch and a quiet stretch of the same
    # voice stay close.
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    if len(norm) == 1:
        return np.zeros(1, dtype=int)

    core = norm[reliable] if reliable.any() else norm
    if len(core) == 1:
        return np.zeros(len(norm), dtype=int)

    # Average linkage over the *filtered* set. Complete linkage also resists the chaining
    # that made an early version merge a whole recording into one speaker, but it sets its
    # threshold by a cluster's worst pair, so the value tuned on 2201 embeddings merged
    # everything on 46 -- scale-dependent exactly where recordings vary most.
    tree = linkage(cosine_condensed(core), method="average")

    if cfg.num_speakers:
        k = max(1, min(cfg.num_speakers, len(core)))
        core_labels = fcluster(tree, t=k, criterion="maxclust") - 1
    else:
        core_labels = fcluster(tree, t=cfg.threshold, criterion="distance") - 1
        found = len(np.unique(core_labels))
        # The threshold is a similarity judgement, not a count; clamp it to what the
        # caller says is possible rather than letting outliers invent a speaker.
        bounded = min(max(found, cfg.min_speakers), cfg.max_speakers, len(core))
        if bounded != found:
            core_labels = fcluster(tree, t=bounded, criterion="maxclust") - 1

    n = int(core_labels.max()) + 1
    centroids = np.stack([core[core_labels == c].mean(axis=0) for c in range(n)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8

    if not reliable.any():
        return core_labels
    labels = np.empty(len(norm), dtype=int)
    labels[reliable] = core_labels
    rest = ~reliable
    if rest.any():  # nearest centroid by cosine similarity
        labels[rest] = (norm[rest] @ centroids.T).argmax(axis=1)
    return labels


class OfflineDiarizer:
    """Diarize a whole recording at once. See module docstring for the pipeline."""

    name = "offline"

    def __init__(self, cfg: Optional[OfflineConfig] = None, on_status=None):
        self.cfg = cfg or OfflineConfig()
        say = on_status or (lambda m: None)
        say("loading segmentation")
        self._seg = resolve(DIARIZATION_REPO, self.cfg.segmentation_model,
                            self.cfg.compute_units)
        say("loading speaker embedding")
        self._emb = resolve(DIARIZATION_REPO, self.cfg.embedding_model,
                            self.cfg.compute_units)
        self.detail = f"coreml · {self.cfg.compute_units.lower()}"
        self._classes = powerset()

    # ---------------------------------------------------------------- stages

    def _decode(self, scores: np.ndarray) -> np.ndarray:
        """(n_frames, 7) powerset scores -> (n_frames, LOCAL_SPEAKERS) binary activity.

        argmax over the powerset, not a per-speaker threshold: the classes are mutually
        exclusive by construction, so one decision covers overlap as well as identity.
        """
        chosen = scores.argmax(axis=1)
        active = np.zeros((scores.shape[0], LOCAL_SPEAKERS), dtype=bool)
        for cls in np.unique(chosen):
            for spk in self._classes[int(cls)]:
                active[chosen == cls, spk] = True
        return active

    def _segment_all(self, audio: np.ndarray, starts: list[int], win: int) -> list[np.ndarray]:
        """Segment every window, `segmentation_batch` at a time.

        Windows are cut per batch rather than stacked up front: at a 1s hop, materialising
        all of them for a 28-minute recording is 1.07 GB of float32, and it scales with
        length. One batch is 20 MB.
        """
        batch = max(1, self.cfg.segmentation_batch)
        n_classes = len(self._classes)
        buf = np.zeros((batch, 1, win), dtype=np.float16)
        out: list[np.ndarray] = []
        for i in range(0, len(starts), batch):
            group = starts[i:i + batch]
            buf[:] = 0.0  # the export's batch dimension is fixed; a short tail stays padded
            for j, s0 in enumerate(group):
                buf[j, 0] = audio[s0:s0 + win]
            result = self._seg.predict({"audio": buf})
            scores = np.asarray(next(iter(result.values())), dtype=np.float32)
            scores = scores.reshape(batch, -1, n_classes)
            out.extend(self._decode(row) for row in scores[:len(group)])
        return out

    def _embed(self, window: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Masked embeddings for the window's local speakers -> (LOCAL_SPEAKERS, 256)."""
        n_frames = active.shape[0]
        waveform = np.repeat(window.reshape(1, -1), EMBED_BATCH, axis=0).astype(np.float16)
        mask = active.T.astype(np.float16)[:EMBED_BATCH]
        if mask.shape[0] < EMBED_BATCH:  # defensive: model batch is fixed
            mask = np.pad(mask, ((0, EMBED_BATCH - mask.shape[0]), (0, 0)))
        out = self._emb.predict({"waveform": waveform, "mask": mask.reshape(EMBED_BATCH, n_frames)})
        emb = out.get("embedding")
        if emb is None:  # the converted models don't all agree on the output name
            emb = next(v for v in out.values() if np.ndim(v) == 2 and np.shape(v)[0] == EMBED_BATCH)
        return np.asarray(emb, dtype=np.float32)

    # clustering lives at module scope; see cluster_embeddings

    # ---------------------------------------------------------------- driver

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[Turn]:
        cfg = self.cfg
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            raise DiarizationUnavailable(
                f"diarization expects 16kHz audio, got {sample_rate}Hz"
            )
        win = int(WINDOW_SEC * sample_rate)
        hop = max(1, int(cfg.hop_sec * sample_rate))
        total = len(audio)
        if total == 0:
            return []
        # The model's input length is fixed, so a short recording is padded rather than
        # refused -- a 3s clip is a normal thing to diarize.
        padded = np.pad(audio, (0, max(0, win - total)))

        starts = list(range(0, max(1, len(padded) - win + 1), hop))
        if starts[-1] + win < len(padded):
            starts.append(len(padded) - win)

        # Pass 1: segment every window, and embed the speakers it found. `reliable` records
        # which of those embeddings are clean enough to define a cluster.
        activities = self._segment_all(padded, starts, win)

        per_window, vectors, owners, voiced_of, shared_of = [], [], [], [], []
        for w_i, s0 in enumerate(starts):
            active = activities[w_i]
            per_window.append((s0, active))
            frame_sec = WINDOW_SEC / active.shape[0]
            voiced = active.sum(axis=0) * frame_sec
            enough = voiced >= MIN_ACTIVITY_SEC
            if not enough.any():
                continue
            emb = self._embed(padded[s0:s0 + win], active)
            for k in np.flatnonzero(enough):
                others = [j for j in range(LOCAL_SPEAKERS) if j != k]
                shared = (active[:, k] & active[:, others].any(axis=1)).sum() * frame_sec
                vectors.append(emb[k])
                owners.append((w_i, int(k)))
                voiced_of.append(float(voiced[k]))
                shared_of.append(float(shared))

        if not vectors:
            return []

        core_mask = _reliable(np.array(voiced_of), np.array(shared_of))

        # Pass 2: one clustering over the whole recording. Doing it globally is the point --
        # per-window ids are arbitrary, so only a session-wide view makes "speaker 1" mean
        # the same person at 00:05 and at 40:00.
        labels = cluster_embeddings(np.stack(vectors), core_mask, cfg)
        n_speakers = int(labels.max()) + 1
        assignment = {owner: int(lab) for owner, lab in zip(owners, labels)}

        # Pass 3: vote. Every frame sits under ~WINDOW_SEC/hop windows; a speaker holds the
        # frame if more than half of the windows that saw it agree.
        n_frames_win = per_window[0][1].shape[0]
        frame_sec = WINDOW_SEC / n_frames_win
        n_global = int(np.ceil(len(padded) / sample_rate / frame_sec)) + 1
        votes = np.zeros((n_global, n_speakers), dtype=np.float32)
        seen = np.zeros(n_global, dtype=np.float32)
        for w_i, (s0, active) in enumerate(per_window):
            off = int(round(s0 / sample_rate / frame_sec))
            end = min(off + n_frames_win, n_global)
            span = end - off
            seen[off:end] += 1
            for k in range(LOCAL_SPEAKERS):
                lab = assignment.get((w_i, k))
                if lab is not None:
                    votes[off:end, lab] += active[:span, k]

        held = votes > (seen[:, None] * 0.5)
        held[seen == 0] = False

        # Frames -> turns, then trim anything past the real end of the audio.
        turns: list[Turn] = []
        for spk in range(n_speakers):
            on = held[:, spk]
            if not on.any():
                continue
            edges = np.diff(on.astype(np.int8), prepend=0, append=0)
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                turns.append(Turn(a * frame_sec, b * frame_sec, spk))

        limit = total / sample_rate
        clipped = [Turn(t.start, min(t.end, limit), t.speaker) for t in turns if t.start < limit]
        return merge_turns(clipped, max_gap=cfg.max_gap, min_duration=cfg.min_duration)
