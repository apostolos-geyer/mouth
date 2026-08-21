"""CPU-only tests: VAD, cadence, recorder. No model, no GPU."""

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from localtranscription import config as cfgfile
from localtranscription.backends import Backend, Transcription, Word
from localtranscription.formats import words_to_srt, write_outputs
from localtranscription.recorder import SessionRecorder, load_manifest
from localtranscription.vad import FRAME_LEN, SAMPLE_RATE, Cadence, segment_utterances

rng = np.random.default_rng(0)


def frames(spec):
    """spec: [(loud, seconds)] -> list of frames."""
    out = []
    for loud, secs in spec:
        n = int(secs * SAMPLE_RATE / FRAME_LEN)
        amp = 0.3 if loud else 0.0005
        out += [rng.normal(0, amp, FRAME_LEN).astype(np.float32) for _ in range(n)]
    return out


def run(spec, cadence=None, threshold=0.01, **kw):
    return list(segment_utterances(iter(frames(spec)), threshold, cadence=cadence, **kw))


# ---------------------------------------------------------------- VAD


def test_silence_yields_nothing():
    assert run([(False, 3)]) == []


def test_single_utterance():
    chunks = run([(False, 1), (True, 2), (False, 2)])
    assert len(chunks) == 1 and chunks[0].final
    # pre-roll pulls the start back before onset; tail silence is trimmed
    assert 0.6 <= chunks[0].start <= 1.05
    assert 2.0 <= len(chunks[0].audio) / SAMPLE_RATE <= 2.6


def test_two_utterances_split():
    chunks = run([(False, 1), (True, 1.5), (False, 1.5), (True, 2), (False, 1.5)])
    assert len(chunks) == 2


def test_brief_pause_does_not_split():
    # 0.4s < the 750ms close threshold
    assert len(run([(False, 1), (True, 1), (False, 0.4), (True, 1), (False, 1.5)])) == 1


def test_blip_dropped():
    """A 120ms cough must not reach the model, despite pre-roll padding it past 1s."""
    assert run([(False, 1), (True, 0.12), (False, 2)]) == []


def test_min_speech_moves_the_gate():
    """The gate is a duration heuristic, and it cuts real words.

    A quick "Claude" measures ten voiced frames -- exactly the 0.3s default, no margin --
    because only the vowel clears an RMS threshold while the /kl/ burst and final /d/
    read as silence. So the gate has to be movable, in both directions.
    """
    short = [(False, 1), (True, 0.18), (False, 2)]
    assert run(short) == []
    assert len(run(short, min_speech=0.15)) == 1
    # And up: raising it past an utterance discards one that the default keeps.
    assert len(run([(False, 1), (True, 0.5), (False, 2)])) == 1
    assert run([(False, 1), (True, 0.5), (False, 2)], min_speech=1.0) == []


def test_min_speech_zero_keeps_anything_that_opened():
    """0 is not "off": the 90ms that opens an utterance is still required."""
    assert len(run([(False, 1), (True, 0.12), (False, 2)], min_speech=0.0)) == 1
    assert run([(False, 3)], min_speech=0.0) == []


def test_flushes_when_stream_ends_mid_utterance():
    chunks = run([(False, 1), (True, 2)])
    assert len(chunks) == 1 and chunks[0].final


def test_long_speech_splits_at_cap():
    chunks = run([(False, 1), (True, 65), (False, 1.5)])
    assert len(chunks) == 3
    assert all(len(c.audio) / SAMPLE_RATE <= 30.01 for c in chunks)


# ---------------------------------------------------------------- cadence


def test_cadence_geometric_beats_fixed():
    """The whole point: geometric spacing costs less AND shows text sooner."""
    length = 20.0
    adaptive = Cadence()
    fixed = Cadence(first=1.2, growth=1.0)

    a, f = adaptive.schedule(length), fixed.schedule(length)
    assert a[0] < f[0], "adaptive must show first text sooner"
    assert sum(a) < sum(f), "adaptive must re-encode less audio"


def test_cadence_growth_one_is_fixed_spacing():
    pts = Cadence(first=1.0, growth=1.0).schedule(5.0)
    assert pts == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_cadence_max_gap_caps_long_utterances():
    pts = Cadence(first=0.4, growth=1.6, max_gap=3.0).schedule(40)
    gaps = [b - a for a, b in zip(pts, pts[1:])]
    assert max(gaps) <= 3.0 + 1e-9, f"gap exceeded max_gap: {max(gaps)}"


def cost(cadence, n):
    """Audio re-encoded across every pass, partials plus the final."""
    return sum(cadence.schedule(n)) + n


def test_uncapped_geometric_cadence_is_near_linear():
    """Pure geometric spacing: doubling the utterance roughly doubles the work."""
    c = Cadence(max_gap=float("inf"))
    ratio = cost(c, 40) / cost(c, 20)
    assert ratio < 2.5, f"expected near-linear, got {ratio}"


def test_max_gap_trades_cost_for_freshness():
    """max_gap caps staleness but reintroduces fixed spacing -- and so quadratic growth.

    Documented deliberately: once the cap binds, partials fire at a constant interval,
    which is the very thing geometric spacing exists to avoid. It stays far cheaper than
    a small fixed cadence, but it is not linear, and on long utterances it approaches the
    model's own throughput. Chunk-level encoder caching is what actually dissolves this.
    """
    capped = Cadence(max_gap=3.0)
    uncapped = Cadence(max_gap=float("inf"))
    fixed = Cadence(first=1.2, growth=1.0)

    r_capped = cost(capped, 40) / cost(capped, 20)
    r_uncapped = cost(uncapped, 40) / cost(uncapped, 20)
    r_fixed = cost(fixed, 40) / cost(fixed, 20)

    assert r_uncapped < r_capped < r_fixed
    # still a large absolute saving where it matters
    assert cost(capped, 30) < 0.5 * cost(fixed, 30)


def test_interims_emitted_and_grow_monotonically():
    chunks = run([(False, 1), (True, 8), (False, 1.5)], cadence=Cadence())
    interims = [c for c in chunks if not c.final]
    finals = [c for c in chunks if c.final]
    assert len(finals) == 1
    assert len(interims) >= 4
    lens = [len(c.audio) for c in interims]
    assert lens == sorted(lens) and len(set(lens)) == len(lens)
    assert all(c.start == finals[0].start for c in interims)


def test_no_cadence_means_no_interims():
    chunks = run([(False, 1), (True, 8), (False, 1.5)], cadence=None)
    assert all(c.final for c in chunks)


def test_blip_emits_no_interims_either():
    assert run([(False, 1), (True, 0.12), (False, 2)], cadence=Cadence()) == []


# ---------------------------------------------------------------- recorder


def test_recorder_writes_audio_and_manifest(tmp_path):
    rec = SessionRecorder(tmp_path, "sess")
    audio = rng.normal(0, 0.1, SAMPLE_RATE * 2).astype(np.float32)
    rec.note_interim(1.0, 0.4, "hello", 0.2)
    rec.note_interim(1.0, 0.8, "hello world", 0.3)
    rec.add(audio, 1.0, "hello world", [{"text": "hello", "start": 1.0, "end": 1.4}],
            "English", 0.5)

    entries = load_manifest(tmp_path / "sess")
    assert len(entries) == 1
    e = entries[0]
    assert Path(e["audio_path"]).exists()
    assert e["hypothesis"] == "hello world"
    assert e["duration"] == 2.0
    # the partial trajectory is what the caching benchmark replays
    assert [i["at"] for i in e["interims"]] == [0.4, 0.8]
    assert e["words"][0]["text"] == "hello"


def test_recorded_audio_round_trips(tmp_path):
    import soundfile as sf

    rec = SessionRecorder(tmp_path, "sess")
    audio = np.sin(np.arange(SAMPLE_RATE) * 0.05).astype(np.float32) * 0.5
    path = rec.add(audio, 0.0, "x", [], "English", 0.1)
    back, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE
    assert len(back) == len(audio)
    assert np.abs(back - audio).max() < 1e-3  # PCM_16 quantisation only


def test_interims_isolated_per_utterance(tmp_path):
    rec = SessionRecorder(tmp_path, "sess")
    a = rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32)
    rec.note_interim(1.0, 0.4, "first", 0.1)
    rec.note_interim(5.0, 0.4, "second", 0.1)
    rec.add(a, 1.0, "first done", [], "English", 0.2)
    rec.add(a, 5.0, "second done", [], "English", 0.2)
    e1, e2 = load_manifest(tmp_path / "sess")
    assert [i["text"] for i in e1["interims"]] == ["first"]
    assert [i["text"] for i in e2["interims"]] == ["second"]


# ---------------------------------------------------------------- formats


def test_write_outputs_produces_four_files(tmp_path):
    words = [{"text": "hello", "start": 0.0, "end": 0.5},
             {"text": "world", "start": 0.5, "end": 1.0}]
    base = write_outputs(tmp_path, [(0.0, "hello world")], words, stem="t")
    for ext in ("txt", "words.json", "srt", "timestamped.md"):
        p = tmp_path / f"t.{ext}"
        assert p.exists() and p.stat().st_size > 0
    assert "00:00:00,000 --> 00:00:01,000" in (tmp_path / "t.srt").read_text()


def test_write_outputs_empty_is_noop(tmp_path):
    assert write_outputs(tmp_path, [], [], stem="t") is None


# ---------------------------------------------------------------- shutdown


class SlowBackend:
    """Stands in for an inference that won't be interrupted."""

    name = "slow"
    detail = "fake"

    def __init__(self, delay=5.0):
        self.delay = delay
        self.calls = 0

    def transcribe(self, audio, sample_rate, *, language, timestamps):
        import time as _t
        self.calls += 1
        _t.sleep(self.delay)
        return Transcription(text="x")


def test_close_without_drain_drops_partials_but_keeps_finals():
    """Quitting must not sit through a backlog -- but must not lose utterances either."""
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    w = Transcriber(SlowBackend(delay=0.0), "English")
    a = np.zeros(1600, dtype=np.float32)
    for _ in range(5):
        w.submit(Chunk(a, 0.0, final=False))
    w.submit(Chunk(a, 1.0, final=True))
    # never started: inspect what close() leaves for the worker to do
    w.close(drain=False, timeout=0.1)
    remaining = []
    while not w.work.empty():
        remaining.append(w.work.get_nowait())
    finals = [c for c in remaining if getattr(c, "final", False)]
    assert len(finals) == 1, "a queued final must survive the quit path"
    assert w.dropped >= 5, "queued partials should have been discarded"


def test_close_gives_up_on_wedged_inference():
    """A stuck inference must not hold the process open."""
    import time as _t
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    w = Transcriber(SlowBackend(delay=30.0), "English")
    w.start()
    w.submit(Chunk(np.zeros(1600, dtype=np.float32), 0.0, final=True))
    _t.sleep(0.2)  # let it get into transcribe()

    t0 = _t.monotonic()
    finished = w.close(drain=False, timeout=0.5)
    elapsed = _t.monotonic() - t0

    assert not finished, "should report that it abandoned the worker"
    assert elapsed < 1.5, f"close() blocked for {elapsed:.1f}s"
    assert w.thread.daemon, "abandoned thread must be daemon or exit still hangs"


# ---------------------------------------------------------------- leak paths


def test_pending_interims_freed_when_final_has_no_text(tmp_path):
    """An utterance that transcribes to nothing must not strand its partials."""
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    class EmptyFinal:
        name, detail = "fake", "fake"

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            return Transcription(text="" if timestamps else "partial text")

    rec = SessionRecorder(tmp_path, "sess")
    w = Transcriber(EmptyFinal(), "English", recorder=rec)
    a = np.zeros(1600, dtype=np.float32)
    w._transcribe_one(Chunk(a, 3.0, final=False))
    assert rec._pending, "interim should be buffered"
    w._transcribe_one(Chunk(a, 3.0, final=True))
    assert not rec._pending, "empty final leaked its buffered interims"


def test_pending_interims_freed_when_final_raises(tmp_path):
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    class Boom:
        name, detail = "fake", "fake"

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            if timestamps:
                raise RuntimeError("boom")
            return Transcription(text="partial")

    rec = SessionRecorder(tmp_path, "sess")
    errors = []
    w = Transcriber(Boom(), "English", recorder=rec, on_error=lambda o, m: errors.append(m))
    a = np.zeros(1600, dtype=np.float32)
    w._transcribe_one(Chunk(a, 3.0, final=False))
    w.submit(Chunk(a, 3.0, final=True))
    w.start()
    w.close(timeout=2.0)
    assert errors, "the failure should have been reported"
    assert not rec._pending, "failed final leaked its buffered interims"


def test_pending_is_hard_capped(tmp_path):
    """Even an unforeseen stranding path must not grow without bound."""
    rec = SessionRecorder(tmp_path, "sess")
    for i in range(rec.MAX_PENDING * 4):
        rec.note_interim(float(i), 0.4, "x", 0.1)
    assert len(rec._pending) <= rec.MAX_PENDING


def test_manifest_survives_a_truncated_line(tmp_path):
    """A hard exit can cut the last append in half; the session should still load."""
    rec = SessionRecorder(tmp_path, "sess")
    a = np.zeros(SAMPLE_RATE, dtype=np.float32)
    rec.add(a, 0.0, "good one", [], "English", 0.1)
    with (tmp_path / "sess" / "manifest.jsonl").open("a") as fh:
        fh.write('{"id": "utt-0002", "audio": "utt-0')  # truncated mid-write
    entries = load_manifest(tmp_path / "sess")
    assert len(entries) == 1 and entries[0]["hypothesis"] == "good one"


def test_tui_transcript_widget_tree_is_bounded():
    """One widget per utterance would grow forever on an all-day session.

    Only the view is capped -- the saved transcript comes from the worker, not the UI.
    """
    import asyncio

    from localtranscription.engine import Config, Segment
    from localtranscription.tui import build_tui

    app = build_tui(Config(record=False), object())
    app.pipeline = lambda: None  # no capture: this is about the widget tree alone

    async def drive():
        async with app.run_test() as pilot:
            cap = app.transcript.MAX_LINES
            total = cap * 2
            for i in range(total):
                app.transcript.partial(float(i), f"p{i}")
                app.transcript.final(Segment(float(i), f"utterance {i}", 1.0, 0.1))
                if i % 50 == 0:
                    await pilot.pause(0.01)
            await pilot.pause(0.4)
            assert len(app.transcript.children) <= cap + 60
            assert f"utterance {total - 1}" in str(app.transcript.children[-1].content)

    asyncio.run(asyncio.wait_for(drive(), timeout=120))


def test_ctrl_c_during_model_load_exits_immediately(tmp_path):
    """Ctrl-C while loading must not hang on atexit.

    huggingface/transformers leave ThreadPoolExecutor workers behind during load, and
    concurrent.futures joins those non-daemonically at interpreter shutdown. Without a
    hard exit the process sits in `_python_exit -> t.join()` with all our work done.
    """
    import subprocess
    import textwrap
    import time as _t

    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import concurrent.futures, sys, time
        sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
        import localtranscription.app as m

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        ex.submit(time.sleep, 120)      # the wedged non-daemon worker
        time.sleep(0.2)

        def boom(*a, **k):
            raise KeyboardInterrupt()

        m._load = boom          # stub the backend load seam: never touch a real model
        sys.argv = ["lt", "tui", "--no-record"]
        m.main()
    """))

    t0 = _t.monotonic()
    proc = subprocess.run([sys.executable, str(child)], capture_output=True, timeout=30)
    elapsed = _t.monotonic() - t0

    assert elapsed < 10, f"exit took {elapsed:.1f}s; atexit is blocking"
    assert proc.returncode == 130, f"expected SIGINT convention, got {proc.returncode}"


# ---------------------------------------------------------------- backends


def test_registered_backends_satisfy_the_protocol():
    """Every backend must structurally match Backend without being imported/loaded."""
    from localtranscription.backends import BACKENDS

    for name, cls in BACKENDS.items():
        assert hasattr(cls, "transcribe"), name
        sig = inspect.signature(cls.transcribe)
        assert list(sig.parameters)[:3] == ["self", "audio", "sample_rate"], name
        assert sig.parameters["language"].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert sig.parameters["timestamps"].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_fake_backend_is_a_backend():
    """runtime_checkable Protocol: duck typing is the whole contract."""
    assert isinstance(_FakeBackend(), Backend)


def test_engine_offsets_words_from_backend_onto_session_clock():
    """Word times come back relative to the chunk; the engine re-bases them."""
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    class Two:
        name, detail = "fake", "fake"

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            return Transcription(text="a b", words=[Word("a", 0.0, 0.4), Word("b", 0.5, 0.9)])

    w = Transcriber(Two(), "English")
    w._transcribe_one(Chunk(np.zeros(1600, dtype=np.float32), 10.0, final=True))
    assert [x["start"] for x in w.words] == [10.0, 10.5]
    assert [x["end"] for x in w.words] == [10.4, 10.9]


def test_unknown_backend_is_rejected_with_a_useful_message():
    from localtranscription.backends import BackendUnavailable, load_backend

    try:
        load_backend("tensorflow")
    except BackendUnavailable as e:
        assert "unknown backend" in str(e) and "torch" in str(e)
    else:
        raise AssertionError("should have refused an unknown backend")


def test_missing_mlx_dependency_explains_the_fix():
    """The mlx package isn't installed here; the failure must be actionable."""
    from localtranscription.backends import MlxBackend, BackendUnavailable, available

    if available("mlx"):
        pytest.skip("mlx-qwen3-asr is installed")
    with pytest.raises(BackendUnavailable) as ei:
        MlxBackend()
    assert "--extra mlx" in str(ei.value) and "--backend torch" in str(ei.value)


def test_mlx_adapter_passes_dtype_and_aligner_in_the_shapes_the_library_wants(monkeypatch):
    """Regression for two bugs their README caused, neither visible without running it.

    - dtype must be an mx.Dtype; the documented string reaches x.astype("float16") and
      raises TypeError during load.
    - forced_aligner must be a ForcedAligner *instance*; the documented bool passes through
      _resolve_aligner and is returned as the aligner, killing every timestamped call.
      Passing None or a str would rebuild a 0.6B aligner on every single call.
    """
    pytest.importorskip("mlx_qwen3_asr")
    import mlx.core as mx
    import mlx_qwen3_asr as m

    from localtranscription.backends import MlxBackend

    seen = {}

    class FakeAligner:
        def __init__(self, model_path=None, dtype=None, **kw):
            seen["aligner_dtype"] = dtype
            seen["aligner_model"] = model_path

    class FakeSession:
        def __init__(self, model=None, *, dtype=None, **kw):
            seen["session_dtype"] = dtype

        def transcribe(self, audio, **kw):
            seen.setdefault("calls", []).append(kw)
            return type("R", (), {"text": "hi", "segments": None, "language": "English"})()

    monkeypatch.setattr(m, "Session", FakeSession)
    monkeypatch.setattr(m, "ForcedAligner", FakeAligner)

    b = MlxBackend()
    assert isinstance(seen["session_dtype"], mx.Dtype), "dtype must not be a string"
    assert isinstance(seen["aligner_dtype"], mx.Dtype)
    assert "ForcedAligner" in seen["aligner_model"]

    b.transcribe(np.zeros(1600, dtype=np.float32), 16000, language="English", timestamps=True)
    b.transcribe(np.zeros(1600, dtype=np.float32), 16000, language="English", timestamps=False)
    final, partial = seen["calls"]
    assert isinstance(final["forced_aligner"], FakeAligner), "finals need the instance"
    assert partial["forced_aligner"] is None, "partials must skip the aligner entirely"
    assert final["max_new_tokens"] == 2048, "parity with torch's truncation fix"


# --------------------------------------------------------------- upstream contract

def test_mlx_loader_upstream_symbols_still_exist():
    """Pin the mlx_qwen3_asr internals backends._load_mlx_model reimplements.

    We mirror their load path rather than patching site-packages, because their loader
    hard-codes nn.quantize(mode="affine") and so cannot open an mxfp4/mxfp8/nvfp4
    checkpoint. The cost of that choice is a dependency on private names: if a release
    renames one, the failure would otherwise surface as an ImportError halfway through
    `lt tui`, after the spinner. Fail here instead, with the reason.
    """
    pytest.importorskip("mlx_qwen3_asr")
    from mlx_qwen3_asr import load_models

    required = [
        "_cast_tree_dtype",
        "_load_safetensors",
        "_materialize_tied_lm_head_weights",
        "_quantized_module_paths",
        "_resolve_path",
    ]
    missing = [n for n in required if not hasattr(load_models, n)]
    assert not missing, (
        f"mlx_qwen3_asr.load_models no longer exports {missing}. "
        "backends._load_mlx_model mirrors their load path and needs these; "
        "re-derive it from the installed version."
    )


def test_mlx_quantize_supports_the_modes_we_offer():
    """`lt quantize --mode` only lists modes MLX can actually produce."""
    mx = pytest.importorskip("mlx.core")
    import inspect

    import mlx.nn as nn

    from localtranscription.quantize import MODES

    assert "mode" in inspect.signature(nn.quantize).parameters, (
        "mlx.nn.quantize lost its `mode` parameter; `lt quantize --mode` and "
        "backends._load_mlx_model both depend on it."
    )
    doc = mx.quantize.__doc__ or ""
    for mode in MODES:
        assert mode in doc, f"MLX no longer documents quantization mode {mode!r}"


# --------------------------------------------------------------- xdg paths

def test_paths_default_under_xdg(monkeypatch, tmp_path):
    from localtranscription import paths

    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert paths.out_dir() == tmp_path / ".local/share/localtranscription/out"
    assert paths.record_dir() == tmp_path / ".local/share/localtranscription/recordings"
    assert paths.models_dir() == tmp_path / ".cache/localtranscription/models"


def test_paths_honour_xdg_env(monkeypatch, tmp_path):
    from localtranscription import paths

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "d"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    assert paths.out_dir() == tmp_path / "d/localtranscription/out"
    assert paths.models_dir() == tmp_path / "c/localtranscription/models"


def test_paths_ignore_relative_xdg_values(monkeypatch, tmp_path):
    """The spec says a relative XDG_* value is invalid and must be ignored.

    Honouring one would put user data wherever the process happened to be started, which
    is the failure this module exists to prevent.
    """
    from localtranscription import paths

    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert paths.out_dir() == tmp_path / ".local/share/localtranscription/out"


def test_paths_ignore_empty_xdg_values(monkeypatch, tmp_path):
    from localtranscription import paths

    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert paths.models_dir() == tmp_path / ".cache/localtranscription/models"


def test_config_defaults_are_not_relative(monkeypatch, tmp_path):
    """A default Config must never write into the working directory."""
    from localtranscription.engine import Config

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    cfg = Config()
    assert cfg.out_dir.is_absolute() and cfg.record_dir.is_absolute()
    assert tmp_path in cfg.out_dir.parents


def test_config_defaults_are_independent_instances(monkeypatch, tmp_path):
    """default_factory, not a shared mutable default."""
    from localtranscription.engine import Config

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert Config().out_dir == Config().out_dir


# ------------------------------------------------- checkpoint reference resolution

def _make_checkpoint(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


def test_bare_name_resolves_against_the_checkpoint_dir(monkeypatch, tmp_path):
    from localtranscription.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    models = tmp_path / "localtranscription" / "models"
    _make_checkpoint(models, "qwen3-asr-1.7b-q8g64")
    assert resolve_checkpoint("qwen3-asr-1.7b-q8g64") == str(models / "qwen3-asr-1.7b-q8g64")


def test_stale_models_prefix_still_resolves(monkeypatch, tmp_path):
    """`-M models/<name>` predates the move to XDG and must keep working.

    Regression: it used to fall through to the Hub and fail as `401 Unauthorized` for a
    repo that never existed.
    """
    from localtranscription.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    models = tmp_path / "localtranscription" / "models"
    _make_checkpoint(models, "qwen3-asr-1.7b-q8g64")
    assert resolve_checkpoint("models/qwen3-asr-1.7b-q8g64") == str(models / "qwen3-asr-1.7b-q8g64")


def test_existing_path_wins(monkeypatch, tmp_path):
    from localtranscription.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    here = _make_checkpoint(tmp_path / "elsewhere", "mine")
    assert resolve_checkpoint(str(here)) == str(here)


def test_hf_repo_ids_pass_through(monkeypatch, tmp_path):
    from localtranscription.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for repo in ("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B"):
        assert resolve_checkpoint(repo) == repo


def test_missing_local_ref_fails_here_not_at_the_hub(monkeypatch, tmp_path):
    """A name meant as a path must not be reported as a missing repository."""
    from localtranscription.backends import BackendUnavailable, resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    models = tmp_path / "localtranscription" / "models"
    _make_checkpoint(models, "real-one")
    for ref in ("models/nope", "./nope", "some/deep/path"):
        with pytest.raises(BackendUnavailable) as exc:
            resolve_checkpoint(ref)
        assert "real-one" in str(exc.value), "the error should list what is available"


def test_missing_checkpoint_dir_is_not_a_crash(monkeypatch, tmp_path):
    from localtranscription.backends import BackendUnavailable, resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "absent"))
    with pytest.raises(BackendUnavailable) as exc:
        resolve_checkpoint("models/nope")
    assert "lt quantize" in str(exc.value)


# ------------------------------------------------------------ incremental partials

def _speech_frames(voiced_frames: int, tail: int = 40):
    from localtranscription.vad import FRAME_LEN

    loud = np.full(FRAME_LEN, 0.5, dtype=np.float32)
    quiet = np.zeros(FRAME_LEN, dtype=np.float32)
    return [quiet] * 5 + [loud] * voiced_frames + [quiet] * tail


def test_incremental_partials_do_not_resend_the_prefix():
    """The whole point: partial audio should sum to the utterance, not to n^2/2c."""
    from localtranscription.vad import Cadence, SAMPLE_RATE, segment_utterances

    frames = _speech_frames(400)
    whole = [c for c in segment_utterances(iter(frames), 0.1, cadence=Cadence())]
    delta = [c for c in segment_utterances(iter(frames), 0.1, cadence=Cadence(),
                                           incremental=True)]

    partial_audio = lambda cs: sum(len(c.audio) for c in cs if not c.final) / SAMPLE_RATE
    final = next(c for c in delta if c.final)
    assert partial_audio(delta) <= len(final.audio) / SAMPLE_RATE + 0.1, \
        "incremental partials should never exceed the utterance's own length"
    assert partial_audio(whole) > 2 * partial_audio(delta)


def test_incremental_partials_are_flagged():
    from localtranscription.vad import Cadence, segment_utterances

    chunks = list(segment_utterances(iter(_speech_frames(400)), 0.1,
                                     cadence=Cadence(), incremental=True))
    assert all(c.incremental for c in chunks if not c.final)
    assert not any(c.incremental for c in chunks if c.final)


def test_final_chunk_still_carries_the_whole_utterance():
    """Finals run the aligner and get saved, so they must never be a delta."""
    from localtranscription.vad import Cadence, SAMPLE_RATE, segment_utterances

    whole = [c for c in segment_utterances(iter(_speech_frames(400)), 0.1, cadence=Cadence())
             if c.final]
    delta = [c for c in segment_utterances(iter(_speech_frames(400)), 0.1,
                                           cadence=Cadence(), incremental=True) if c.final]
    assert len(whole) == len(delta) == 1
    assert len(whole[0].audio) == len(delta[0].audio)


class _FakeBackend:
    """Returns text for anything."""

    name, detail = "fake", "fake"

    def transcribe(self, audio, sample_rate, *, language, timestamps):
        return Transcription(text="hi", words=[Word("hi", 0.0, 0.2)])


class _SilentBackend:
    """Returns nothing -- the branch where a final produces no text at all."""

    name, detail = "silent", "fake"

    def transcribe(self, audio, sample_rate, *, language, timestamps):
        return Transcription(text="")


class _FakeStream:
    """Records what it was fed, so the engine's contract with a stream is observable."""

    def __init__(self):
        self.fed: list[int] = []
        self.closes = 0
        self.stable = ""

    def feed(self, pcm):
        self.fed.append(len(pcm))
        return "hello " * len(self.fed)

    def close(self):
        self.closes += 1
        self.fed = []


def test_stream_is_reset_between_utterances():
    """A new utterance must not inherit the previous one's decoder state."""
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    stream = _FakeStream()
    worker = Transcriber(_FakeBackend(), "English", stream=stream)
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 5.0, False, incremental=True))
    assert stream.closes == 1, "changing utterance should close the previous stream"


def test_final_closes_the_stream_even_when_it_yields_nothing():
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    stream = _FakeStream()
    worker = Transcriber(_SilentBackend(), "English", stream=stream)
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 0.0, True))
    assert stream.closes >= 1


def test_incremental_partials_are_never_dropped_as_stale():
    """A delta the decoder hasn't seen can't be skipped -- it would hole the stream."""
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    worker = Transcriber(_FakeBackend(), "English", stream=_FakeStream())
    worker.submit(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker.submit(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker.submit(Chunk(np.zeros(1600, np.float32), 0.0, True))
    worker.start()
    worker.close(drain=True, timeout=5)
    assert worker.dropped == 0


# ---------------------------------------------------------------- dictate


def test_threshold_cache_round_trips_per_device(tmp_path):
    """A threshold describes a mic in a room; a laptop mic and a condenser don't share one."""
    from localtranscription.sources import cached_threshold, remember_threshold

    cal = tmp_path / "calibration.json"
    assert cached_threshold(None, cal) is None

    remember_threshold(None, 0.0123, cal)
    remember_threshold(3, 0.0456, cal)

    assert cached_threshold(None, cal) == pytest.approx(0.0123)
    assert cached_threshold(3, cal) == pytest.approx(0.0456)
    assert cached_threshold(9, cal) is None


def test_threshold_cache_treats_damage_as_a_miss(tmp_path):
    """A stale cache must cost a calibration, never a failed session."""
    from localtranscription.sources import cached_threshold, remember_threshold

    cal = tmp_path / "calibration.json"
    for junk in ("{ not json", "[]", '{"default": {}}', '{"default": {"threshold": 0}}'):
        cal.write_text(junk)
        assert cached_threshold(None, cal) is None

    # ...and writing over the damage still works.
    remember_threshold(None, 0.02, cal)
    assert cached_threshold(None, cal) == pytest.approx(0.02)


def test_transcriber_can_skip_the_aligner():
    """timestamps=False must reach the backend on finals, not just on partials."""
    from localtranscription.engine import Transcriber
    from localtranscription.vad import Chunk

    seen = []

    class Spy:
        name = detail = "spy"

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            seen.append(timestamps)
            return Transcription(text="x")

    audio = np.zeros(FRAME_LEN, dtype=np.float32)
    for timestamps, expected in ((True, [True]), (False, [False])):
        seen.clear()
        w = Transcriber(Spy(), "English", timestamps=timestamps)
        w.start()
        w.submit(Chunk(audio, 0.0, final=True))
        assert w.close(timeout=5)
        assert seen == expected


def _dictate(tmp_path, spec, *args):
    """Run `lt dictate --wav` in a child with the model seam stubbed out."""
    import subprocess
    import textwrap

    import soundfile as sf

    wav = tmp_path / "in.wav"
    sf.write(wav, np.concatenate(frames(spec)), SAMPLE_RATE)

    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
        import localtranscription.app as m
        from localtranscription.backends import Transcription

        class Fake:
            name = detail = "fake"
            def transcribe(self, audio, sample_rate, *, language, timestamps):
                assert timestamps is False, "dictate must never run the aligner"
                return Transcription(text="hello there")

        m._load = lambda *a, **k: Fake()
        sys.argv = ["lt", "dictate", "--wav", {str(wav)!r}, *{list(args)!r}]
        m.main()
    """))
    # XDG_CONFIG_HOME at a scratch dir: the child would otherwise read the developer's
    # own config.toml, whose whole purpose is to change what `lt dictate` does.
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "config")}
    return subprocess.run([sys.executable, str(child)], capture_output=True, timeout=180,
                          env=env)


def test_dictate_puts_only_the_text_on_stdout(tmp_path):
    """The contract other programs compose on: stdout is the transcript and nothing else.

    No trailing newline when piped either -- the caller is pasting into a text field, and
    a stray newline sends the message.
    """
    proc = _dictate(tmp_path, [(False, 1.2), (True, 1.0), (False, 1.0)])

    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == b"hello there"


def test_dictate_stops_after_one_utterance(tmp_path):
    """Two utterances in, one out: the command is a single dictation, not a session."""
    proc = _dictate(
        tmp_path,
        [(False, 1.2), (True, 0.8), (False, 1.0), (True, 0.8), (False, 1.0)],
        "--events",
    )

    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == b"hello there"
    events = [json.loads(line) for line in proc.stderr.splitlines() if line.strip()]
    assert sum(e["event"] == "final" for e in events) == 1


def test_dictate_exits_nonzero_on_silence(tmp_path):
    """`lt dictate | pbcopy` must not clobber the clipboard with nothing."""
    proc = _dictate(tmp_path, [(False, 3.0)], "--wait", "1")

    assert proc.returncode == 1
    assert proc.stdout == b""


def test_dictate_events_carry_what_a_meter_needs(tmp_path):
    """The stderr stream is the UI's whole input: ready to talk, levels, the text."""
    proc = _dictate(tmp_path, [(False, 1.2), (True, 1.0), (False, 1.0)], "--events")

    assert proc.returncode == 0, proc.stderr.decode()
    events = [json.loads(line) for line in proc.stderr.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]

    # "ready" is what tells a UI the mic is live -- without it you guess, and clip.
    assert "ready" in kinds and "speech" in kinds and "final" in kinds
    assert kinds.index("ready") < kinds.index("speech") < kinds.index("final")

    levels = [e for e in events if e["event"] == "level"]
    assert len(levels) > 30 and all("rms" in e and "speech" in e for e in levels)
    assert any(e["speech"] for e in levels)
    assert [e for e in events if e["event"] == "final"][0]["text"] == "hello there"


def test_dictate_hold_keeps_going_through_pauses(tmp_path):
    """Hold-to-talk: a pause mid-thought is not the end of the dictation.

    Without --hold the 750ms that closes an utterance also ends the command, so
    everything said after thinking for a second is lost while the key is still down.
    """
    spec = [(False, 1.2), (True, 0.8), (False, 1.2), (True, 0.8), (False, 1.0)]

    stops = _dictate(tmp_path, spec, "--events")
    holds = _dictate(tmp_path, spec, "--events", "--hold")

    assert stops.stdout == b"hello there"
    assert holds.stdout == b"hello there hello there"

    finals = lambda p: sum(  # noqa: E731
        json.loads(line)["event"] == "final"
        for line in p.stderr.splitlines() if line.strip()
    )
    assert finals(stops) == 1
    assert finals(holds) == 2


# ------------------------------------------------------------------ config file


def _cli_params():
    """The real CLI's option table, so these tests can't drift from the real flags."""
    import typer.main

    from localtranscription.app import _params, app

    return _params(typer.main.get_command(app))


def _map(text):
    import tomllib

    return cfgfile.default_map(tomllib.loads(text), _cli_params())


def test_bare_keys_reach_every_command_that_listens():
    m = _map('backend = "mlx"\nlanguage = "Greek"\n')
    assert set(m) == {"tui", "cli", "dictate"}
    assert all(v == {"backend": "mlx", "language": "Greek"} for v in m.values())


def test_a_table_overrides_the_bare_key():
    m = _map("record = true\n\n[dictate]\nrecord = false\n")
    assert m["tui"]["record"] is True
    assert m["dictate"]["record"] is False


def test_a_bare_key_never_reaches_diarize():
    """The reason bare keys are scoped at all.

    --threshold is an RMS gate to a session and a cosine distance to `diarize`. One
    number cannot be both, and a config that quietly sent 0.02 to the clusterer would
    collapse every speaker into one.
    """
    assert "diarize" not in _map("threshold = 0.02\n")
    assert _map("[diarize]\nthreshold = 0.7\n") == {"diarize": {"threshold": 0.7}}


def test_an_option_belonging_elsewhere_says_where_it_goes():
    with pytest.raises(cfgfile.ConfigError, match=r"\[quantize\]"):
        _map("bits = 4\n")


def test_a_typo_is_an_error_with_a_suggestion():
    """Silence is the wrong answer here: a dead key is a setting you believe is on."""
    with pytest.raises(cfgfile.ConfigError, match="--backend"):
        _map('bakend = "mlx"\n')
    with pytest.raises(cfgfile.ConfigError, match="--hold"):
        _map("[dictate]\nholdd = true\n")
    with pytest.raises(cfgfile.ConfigError, match="dictate"):
        _map("[dictat]\nhold = true\n")


def test_either_spelling_of_a_flag_resolves():
    """--max-gap is what --help prints; max_gap is what the parameter is called."""
    assert _map("max-gap = 5.0\n")["tui"] == _map("max_gap = 5.0\n")["tui"]
    # And where the two differ -- `lt models --dir` sets model_dir -- both still land.
    assert _map('[models]\ndir = "/ckpts"\n') == {"models": {"model_dir": "/ckpts"}}


def test_home_relative_paths_expand():
    """Click casts strings to Paths for us, but it does not expand `~`, and a config
    file is exactly where someone writes one."""
    out = _map('out = "~/notes"\n')["cli"]["out"]
    assert out == str(Path.home() / "notes")


def test_a_missing_file_is_not_a_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgfile.paths, "config_file", lambda: tmp_path / "nope.toml")
    assert cfgfile.locate() is None
    # But one named by hand was named for a reason.
    with pytest.raises(cfgfile.ConfigError):
        cfgfile.locate(tmp_path / "nope.toml")


def _invoke(tmp_path, text, *args):
    from typer.testing import CliRunner

    from localtranscription.app import app

    cfg = tmp_path / "config.toml"
    cfg.write_text(text)
    return CliRunner().invoke(app, ["--config", str(cfg), *args])


def test_a_flag_still_beats_the_file(tmp_path):
    """The whole claim of the feature: the file moves defaults, it doesn't pin values."""
    from_file = _invoke(tmp_path, "[cadence]\ninterim = 2.0\n", "cadence", "10")
    assert from_file.exit_code == 0
    assert "first text after 2s" in from_file.stdout

    typed = _invoke(tmp_path, "[cadence]\ninterim = 2.0\n", "cadence", "10",
                    "--interim", "0.5")
    assert typed.exit_code == 0
    assert "first text after 0.5s" in typed.stdout


def test_a_broken_config_stops_every_command_but_reports_itself(tmp_path):
    """`lt config` is how you find out what's wrong with the file, so it survives one."""
    broken = 'backend = "mlx"\nlanguage =\n'
    assert _invoke(tmp_path, broken, "cadence", "10").exit_code == 2

    shown = _invoke(tmp_path, broken, "config")
    assert shown.exit_code == 1
    assert "line 2" in shown.stdout
