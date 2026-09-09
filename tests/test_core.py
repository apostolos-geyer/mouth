"""CPU-only tests: VAD, cadence, recorder. No model, no GPU."""

import inspect
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mouth import config as cfgfile
from mouth.backends import Backend, Transcription, Word
from mouth.formats import write_outputs
from mouth.recorder import SessionRecorder, load_manifest
from mouth.vad import FRAME_LEN, SAMPLE_RATE, Cadence, segment_utterances

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
    gaps = [b - a for a, b in itertools.pairwise(pts)]
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
    rec.add(
        audio,
        1.0,
        "hello world",
        [{"text": "hello", "start": 1.0, "end": 1.4}],
        "English",
        0.5,
    )

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
    words = [
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.5, "end": 1.0},
    ]
    write_outputs(tmp_path, [(0.0, "hello world")], words, stem="t")
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
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

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

    from mouth.engine import Transcriber
    from mouth.vad import Chunk

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
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

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
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

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

    from mouth.engine import Config, Segment
    from mouth.tui import build_tui

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
    child.write_text(
        textwrap.dedent(f"""
        import concurrent.futures, sys, time
        sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
        import mouth.app as m

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        ex.submit(time.sleep, 120)      # the wedged non-daemon worker
        time.sleep(0.2)

        def boom(*a, **k):
            raise KeyboardInterrupt()

        m._load = boom          # stub the backend load seam: never touch a real model
        sys.argv = ["m", "tui", "--no-record"]
        m.main()
    """)
    )

    t0 = _t.monotonic()
    proc = subprocess.run(
        [sys.executable, str(child)],
        capture_output=True,
        timeout=30,
        check=False,
        # Scratch XDG_CONFIG_HOME: the child would otherwise read the developer's own
        # config, and a setting it doesn't like exits 2 before the signal is ever sent --
        # which reads as "the hard exit regressed" rather than "the config was rejected".
        env={**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "config")},
    )
    elapsed = _t.monotonic() - t0

    assert elapsed < 10, f"exit took {elapsed:.1f}s; atexit is blocking"
    assert proc.returncode == 130, f"expected SIGINT convention, got {proc.returncode}"


# ---------------------------------------------------------------- backends


def test_registered_backends_satisfy_the_protocol():
    """Every backend must structurally match Backend without being imported/loaded."""
    from mouth.backends import BACKENDS

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
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

    class Two:
        name, detail = "fake", "fake"

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            return Transcription(
                text="a b", words=[Word("a", 0.0, 0.4), Word("b", 0.5, 0.9)]
            )

    w = Transcriber(Two(), "English")
    w._transcribe_one(Chunk(np.zeros(1600, dtype=np.float32), 10.0, final=True))
    assert [x["start"] for x in w.words] == [10.0, 10.5]
    assert [x["end"] for x in w.words] == [10.4, 10.9]


def test_unknown_backend_is_rejected_with_a_useful_message():
    from mouth.backends import BackendUnavailable, load_backend

    try:
        load_backend("tensorflow")
    except BackendUnavailable as e:
        assert "unknown backend" in str(e) and "torch" in str(e)
    else:
        raise AssertionError("should have refused an unknown backend")


def test_missing_mlx_dependency_explains_the_fix():
    """The mlx package isn't installed here; the failure must be actionable."""
    from mouth.backends import BackendUnavailable, MlxBackend, available

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
    import mlx.core as mx  # ty: ignore[unresolved-import]
    import mlx_qwen3_asr as m

    from mouth.backends import MlxBackend

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

    b.transcribe(
        np.zeros(1600, dtype=np.float32), 16000, language="English", timestamps=True
    )
    b.transcribe(
        np.zeros(1600, dtype=np.float32), 16000, language="English", timestamps=False
    )
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
    `m tui`, after the spinner. Fail here instead, with the reason.
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
    """`m quantize --mode` only lists modes MLX can actually produce."""
    mx = pytest.importorskip("mlx.core")
    import inspect

    from mlx import nn

    from mouth.quantize import MODES

    assert "mode" in inspect.signature(nn.quantize).parameters, (
        "mlx.nn.quantize lost its `mode` parameter; `m quantize --mode` and "
        "backends._load_mlx_model both depend on it."
    )
    doc = mx.quantize.__doc__ or ""
    for mode in MODES:
        assert mode in doc, f"MLX no longer documents quantization mode {mode!r}"


# --------------------------------------------------------------- xdg paths


def test_paths_default_under_xdg(monkeypatch, tmp_path):
    from mouth import paths

    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert paths.out_dir() == tmp_path / ".local/share/mouth/out"
    assert paths.record_dir() == tmp_path / ".local/share/mouth/recordings"
    assert paths.models_dir() == tmp_path / ".cache/mouth/models"


def test_paths_honour_xdg_env(monkeypatch, tmp_path):
    from mouth import paths

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "d"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    assert paths.out_dir() == tmp_path / "d/mouth/out"
    assert paths.models_dir() == tmp_path / "c/mouth/models"


def test_paths_ignore_relative_xdg_values(monkeypatch, tmp_path):
    """The spec says a relative XDG_* value is invalid and must be ignored.

    Honouring one would put user data wherever the process happened to be started, which
    is the failure this module exists to prevent.
    """
    from mouth import paths

    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert paths.out_dir() == tmp_path / ".local/share/mouth/out"


def test_paths_ignore_empty_xdg_values(monkeypatch, tmp_path):
    from mouth import paths

    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert paths.models_dir() == tmp_path / ".cache/mouth/models"


def test_config_defaults_are_not_relative(monkeypatch, tmp_path):
    """A default Config must never write into the working directory."""
    from mouth.engine import Config

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    cfg = Config()
    assert cfg.out_dir.is_absolute() and cfg.record_dir.is_absolute()
    assert tmp_path in cfg.out_dir.parents


def test_config_defaults_are_independent_instances(monkeypatch, tmp_path):
    """default_factory, not a shared mutable default."""
    from mouth.engine import Config

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert Config().out_dir == Config().out_dir


# ------------------------------------------------- checkpoint reference resolution


def _make_checkpoint(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


def test_bare_name_resolves_against_the_checkpoint_dir(monkeypatch, tmp_path):
    from mouth.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    models = tmp_path / "mouth" / "models"
    _make_checkpoint(models, "qwen3-asr-1.7b-q8g64")
    assert resolve_checkpoint("qwen3-asr-1.7b-q8g64") == str(
        models / "qwen3-asr-1.7b-q8g64"
    )


def test_stale_models_prefix_still_resolves(monkeypatch, tmp_path):
    """`-M models/<name>` predates the move to XDG and must keep working.

    Regression: it used to fall through to the Hub and fail as `401 Unauthorized` for a
    repo that never existed.
    """
    from mouth.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    models = tmp_path / "mouth" / "models"
    _make_checkpoint(models, "qwen3-asr-1.7b-q8g64")
    assert resolve_checkpoint("models/qwen3-asr-1.7b-q8g64") == str(
        models / "qwen3-asr-1.7b-q8g64"
    )


def test_existing_path_wins(monkeypatch, tmp_path):
    from mouth.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    here = _make_checkpoint(tmp_path / "elsewhere", "mine")
    assert resolve_checkpoint(str(here)) == str(here)


def test_hf_repo_ids_pass_through(monkeypatch, tmp_path):
    from mouth.backends import resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for repo in ("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B"):
        assert resolve_checkpoint(repo) == repo


def test_missing_local_ref_fails_here_not_at_the_hub(monkeypatch, tmp_path):
    """A name meant as a path must not be reported as a missing repository."""
    from mouth.backends import BackendUnavailable, resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    models = tmp_path / "mouth" / "models"
    _make_checkpoint(models, "real-one")
    for ref in ("models/nope", "./nope", "some/deep/path"):
        with pytest.raises(BackendUnavailable) as exc:
            resolve_checkpoint(ref)
        assert "real-one" in str(exc.value), "the error should list what is available"


def test_missing_checkpoint_dir_is_not_a_crash(monkeypatch, tmp_path):
    from mouth.backends import BackendUnavailable, resolve_checkpoint

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "absent"))
    with pytest.raises(BackendUnavailable) as exc:
        resolve_checkpoint("models/nope")
    assert "m quantize" in str(exc.value)


# ------------------------------------------------------------ incremental partials


def _speech_frames(voiced_frames: int, tail: int = 40):
    from mouth.vad import FRAME_LEN

    loud = np.full(FRAME_LEN, 0.5, dtype=np.float32)
    quiet = np.zeros(FRAME_LEN, dtype=np.float32)
    return [quiet] * 5 + [loud] * voiced_frames + [quiet] * tail


def test_incremental_partials_do_not_resend_the_prefix():
    """The whole point: partial audio should sum to the utterance, not to n^2/2c."""
    from mouth.vad import SAMPLE_RATE, Cadence, segment_utterances

    frames = _speech_frames(400)
    whole = list(segment_utterances(iter(frames), 0.1, cadence=Cadence()))
    delta = list(segment_utterances(iter(frames), 0.1, cadence=Cadence(), incremental=True))

    def partial_audio(cs):
        return sum(len(c.audio) for c in cs if not c.final) / SAMPLE_RATE

    final = next(c for c in delta if c.final)
    assert partial_audio(delta) <= len(final.audio) / SAMPLE_RATE + 0.1, (
        "incremental partials should never exceed the utterance's own length"
    )
    assert partial_audio(whole) > 2 * partial_audio(delta)


def test_incremental_partials_are_flagged():
    from mouth.vad import Cadence, segment_utterances

    chunks = list(
        segment_utterances(
            iter(_speech_frames(400)), 0.1, cadence=Cadence(), incremental=True
        )
    )
    assert all(c.incremental for c in chunks if not c.final)
    assert not any(c.incremental for c in chunks if c.final)


def test_final_chunk_still_carries_the_whole_utterance():
    """Finals run the aligner and get saved, so they must never be a delta."""
    from mouth.vad import Cadence, segment_utterances

    whole = [
        c
        for c in segment_utterances(iter(_speech_frames(400)), 0.1, cadence=Cadence())
        if c.final
    ]
    delta = [
        c
        for c in segment_utterances(
            iter(_speech_frames(400)), 0.1, cadence=Cadence(), incremental=True
        )
        if c.final
    ]
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

    mode = "stream"
    incremental = True

    def __init__(self):
        self.fed: list[int] = []
        self.closes = 0
        self.stable = ""

    def text(self, pcm):
        self.fed.append(len(pcm))
        return "hello " * len(self.fed)

    def reset(self):
        self.closes += 1
        self.fed = []


def test_stream_is_reset_between_utterances():
    """A new utterance must not inherit the previous one's decoder state."""
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

    stream = _FakeStream()
    worker = Transcriber(_FakeBackend(), "English", partials=stream)
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 5.0, False, incremental=True))
    assert stream.closes == 1, "changing utterance should close the previous stream"


def test_final_closes_the_stream_even_when_it_yields_nothing():
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

    stream = _FakeStream()
    worker = Transcriber(_SilentBackend(), "English", partials=stream)
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker._transcribe_one(Chunk(np.zeros(1600, np.float32), 0.0, True))
    assert stream.closes >= 1


def test_incremental_partials_are_never_dropped_as_stale():
    """A delta the decoder hasn't seen can't be skipped -- it would hole the stream."""
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

    worker = Transcriber(_FakeBackend(), "English", partials=_FakeStream())
    worker.submit(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker.submit(Chunk(np.zeros(1600, np.float32), 0.0, False, incremental=True))
    worker.submit(Chunk(np.zeros(1600, np.float32), 0.0, True))
    worker.start()
    worker.close(drain=True, timeout=5)
    assert worker.dropped == 0


# ---------------------------------------------------------------- dictate


def test_threshold_cache_round_trips_per_device(tmp_path):
    """A threshold describes a mic in a room; a laptop mic and a condenser don't share one."""
    from mouth.sources import cached_threshold, remember_threshold

    cal = tmp_path / "calibration.json"
    assert cached_threshold(None, cal) is None

    remember_threshold(None, 0.0123, cal)
    remember_threshold(3, 0.0456, cal)

    assert cached_threshold(None, cal) == pytest.approx(0.0123)
    assert cached_threshold(3, cal) == pytest.approx(0.0456)
    assert cached_threshold(9, cal) is None


def test_threshold_cache_treats_damage_as_a_miss(tmp_path):
    """A stale cache must cost a calibration, never a failed session."""
    from mouth.sources import cached_threshold, remember_threshold

    cal = tmp_path / "calibration.json"
    for junk in ("{ not json", "[]", '{"default": {}}', '{"default": {"threshold": 0}}'):
        cal.write_text(junk)
        assert cached_threshold(None, cal) is None

    # ...and writing over the damage still works.
    remember_threshold(None, 0.02, cal)
    assert cached_threshold(None, cal) == pytest.approx(0.02)


def test_transcriber_can_skip_the_aligner():
    """timestamps=False must reach the backend on finals, not just on partials."""
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

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
    """Run `m dictate --wav` in a child with the model seam stubbed out."""
    import subprocess
    import textwrap

    import soundfile as sf

    wav = tmp_path / "in.wav"
    sf.write(wav, np.concatenate(frames(spec)), SAMPLE_RATE)

    child = tmp_path / "child.py"
    child.write_text(
        textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
        import mouth.app as m
        from mouth.backends import Transcription

        class Fake:
            name = detail = "fake"
            def transcribe(self, audio, sample_rate, *, language, timestamps):
                assert timestamps is False, "dictate must never run the aligner"
                return Transcription(text="hello there")

        m._load = lambda *a, **k: Fake()
        sys.argv = ["m", "dictate", "--wav", {str(wav)!r}, *{list(args)!r}]
        m.main()
    """)
    )
    # XDG_CONFIG_HOME at a scratch dir: the child would otherwise read the developer's
    # own config.toml, whose whole purpose is to change what `m dictate` does.
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "config")}
    return subprocess.run(
        [sys.executable, str(child)], capture_output=True, timeout=180, env=env, check=False
    )


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
    """`m dictate | pbcopy` must not clobber the clipboard with nothing."""
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
    assert next(e for e in events if e["event"] == "final")["text"] == "hello there"


def _live(tmp_path, spec, *args):
    """Run `m live --wav` in a child with the model seam stubbed out.

    A child process rather than CliRunner because the thing under test *is* the split
    between two real file descriptors, and a runner that captures them into one buffer
    cannot see it.
    """
    import subprocess
    import textwrap

    import soundfile as sf

    wav = tmp_path / "in.wav"
    sf.write(wav, np.concatenate(frames(spec)), SAMPLE_RATE)

    child = tmp_path / "child.py"
    child.write_text(
        textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
        import mouth.app as m
        from mouth.backends import Transcription

        class Fake:
            name = detail = "fake"
            def transcribe(self, audio, sample_rate, *, language, timestamps):
                return Transcription(text="hello there")

        m._load = lambda *a, **k: Fake()
        sys.argv = ["m", "live", "--wav", {str(wav)!r}, "--no-record",
                    "-o", {str(tmp_path / "out")!r}, *{list(args)!r}]
        m.main()
    """)
    )
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "config")}
    return subprocess.run(
        [sys.executable, str(child)], capture_output=True, timeout=180, env=env, check=False
    )


def test_live_plain_puts_only_the_text_on_stdout(tmp_path):
    """The point of the flag. Everything that is not an utterance -- the load status, the
    measured threshold, the closing summary -- is commentary, and commentary in the middle
    of `m live --plain | tee -a notes.md` lands in the notes."""
    proc = _live(tmp_path, [(False, 1.2), (True, 1.0), (False, 1.0)], "--plain")

    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == b"hello there\n"
    # And the commentary did not simply vanish -- it moved.
    assert b"VAD threshold" in proc.stderr and b"utterances" in proc.stderr


def test_live_plain_leaves_no_markup_or_wrapping_in_the_text(tmp_path):
    """Written straight to the fd, not through rich: a console would hard-wrap the line
    to a terminal width the consumer does not have, and read `[00:04]` in a transcript as
    markup rather than as something somebody said."""
    proc = _live(tmp_path, [(False, 1.2), (True, 1.0), (False, 1.0)], "--plain")

    assert proc.returncode == 0, proc.stderr.decode()
    assert b"\x1b[" not in proc.stdout, "no ANSI"
    assert proc.stdout.count(b"\n") == 1, "one line per utterance, unwrapped"


def test_live_rich_is_the_default_and_keeps_its_decoration(tmp_path):
    """The flag adds a mode; it does not quietly take the old one away. --rich prints the
    clock and what the utterance cost, on stdout, the way it always did."""
    proc = _live(tmp_path, [(False, 1.2), (True, 1.0), (False, 1.0)])

    assert proc.returncode == 0, proc.stderr.decode()
    assert b"hello there" in proc.stdout
    assert b"00:0" in proc.stdout, "the clock"
    assert b"VAD threshold" in proc.stdout, "commentary stays on stdout under --rich"
    assert proc.stderr == b""


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

    def finals(p):
        return sum(
            json.loads(line)["event"] == "final"
            for line in p.stderr.splitlines()
            if line.strip()
        )

    assert finals(stops) == 1
    assert finals(holds) == 2


# ------------------------------------------------------------------ config file


def _cli_params():
    """The real CLI's option table, so these tests can't drift from the real flags."""
    import typer.main

    from mouth.app import _params, app

    return _params(typer.main.get_command(app))


def _map(text):
    import tomllib

    return cfgfile.default_map(tomllib.loads(text), _cli_params())


def test_bare_keys_reach_every_command_with_the_option():
    """Reaching is the default; only a name collision opts out.

    The inclusion list this replaced was wrong three times in a row -- once for every
    command added after it -- and each time it was a silent wrong answer rather than a
    failure.
    """
    m = _map('backend = "mlx"\nlanguage = "Greek"\n')
    assert {"tui", "live", "dictate", "tune", "transcribe"} <= set(m)
    assert all(v == {"backend": "mlx", "language": "Greek"} for v in m.values())


def _cli_options():
    """Every command's parameters, off the built CLI: {long option: [(command, ...)]}."""
    import click
    import typer

    from mouth.app import app

    group = typer.main.get_command(app)
    ctx = click.Context(group)
    longs, shorts = {}, {}
    for name in group.list_commands(ctx):
        for prm in group.get_command(ctx, name).params:
            if prm.name == "help":
                continue
            spelled = [o for o in prm.opts if o.startswith("--")]
            key = spelled[0] if spelled else f"<{prm.name}>"
            kind = (
                "flag" if getattr(prm, "is_flag", False) else getattr(prm.type, "name", "?")
            )
            longs.setdefault(key, []).append((name, kind, prm.name))
            for short in (o for o in prm.opts if not o.startswith("--")):
                shorts.setdefault(short, []).append((name, key))
    return longs, shorts


def test_no_flag_name_means_two_things():
    """The invariant behind three renames, kept as a property rather than a memory.

    `--speakers` was a count to `m diarize` and a bare on/off switch to `m transcribe`,
    so `m transcribe interview.m4a --speakers 3` died on "unexpected extra argument".
    `--threshold` was an RMS gate and a cosine distance. `--out` was a directory and a
    single RTTM file. Each was a flag someone could reasonably think they understood.

    A name shared by two commands must take the same type and mean the same thing. Add a
    fourth collision and this fails here rather than in somebody's pipeline.
    """
    longs, _ = _cli_options()
    clashes = {
        opt: uses
        for opt, uses in longs.items()
        if len({(kind, param) for _, kind, param in uses}) > 1
    }
    assert not clashes, "\n".join(
        f"{opt}: " + ", ".join(f"m {c} takes {k} -> {p}" for c, k, p in uses)
        for opt, uses in clashes.items()
    )


def test_a_short_flag_is_the_same_idea_everywhere():
    """-o is the one short flag that names different long options, and deliberately: it
    is "where this command's output goes" in all three. Every other short flag must map
    to exactly one long name, or muscle memory from one command misfires in the next."""
    _, shorts = _cli_options()
    strays = {
        short: sorted({opt for _, opt in uses})
        for short, uses in shorts.items()
        if len({opt for _, opt in uses}) > 1
    }
    assert strays == {"-o": ["--dest", "--out", "--rttm"]}, strays


def test_a_config_key_reaches_every_command_that_has_the_flag():
    """A flag's parameter name has to match its spelling, or a bare key splits.

    `--interim` bound to a parameter called `first` in tui/live/cadence and `interim` in
    dictate, so `first = 0.9` in a config file set three of the four and skipped the
    fourth -- accepted, plausible, and wrong.
    """
    longs, _ = _cli_options()
    split = {
        opt: sorted({param for _, _, param in uses})
        for opt, uses in longs.items()
        if opt.startswith("--") and len({param for _, _, param in uses}) > 1
    }
    assert not split, f"one flag, two parameter names: {split}"


def test_a_new_command_inherits_settings_without_being_listed():
    """The property the inversion buys, stated so it cannot quietly go away."""
    from mouth import config as cfg

    assert cfg.reaches("some-future-command")


def test_nothing_is_excluded_because_nothing_collides():
    """EXCLUDED is empty, and that is a claim about the CLI rather than an oversight:
    every flag name now means one thing. The escape hatch stays for the day one doesn't."""
    from mouth import config as cfg

    assert cfg.EXCLUDED == {}


def test_the_escape_hatch_still_works_if_a_collision_appears(monkeypatch):
    """Empty is not the same as gone. Exercised against a synthetic collision, so the
    mechanism keeps being tested after the real ones were renamed away.

    --speakers is the right shape to test with: two commands have it, so excluding one
    leaves a bare key with somewhere legitimate to land."""
    from mouth import config as cfg

    monkeypatch.setattr(
        cfg, "EXCLUDED", {"diarize": "--speakers would mean something else"}
    )
    assert not cfg.reaches("diarize")

    bare = _map("speakers = 2\n")
    assert bare["transcribe"] == {"num_speakers": 2}, "still reaches the others"
    assert "diarize" not in bare, "and stops at the excluded one"
    # Naming the table is how you mean it anyway.
    assert _map("[diarize]\nspeakers = 2\n") == {"diarize": {"num_speakers": 2}}


def test_the_renamed_diarize_options_land_where_they_say():
    """The three collisions, gone by renaming rather than by exception."""
    # --threshold is the RMS gate, and reaches every session command. Diarize has none.
    assert "diarize" not in _map("threshold = 0.02\n")
    assert _map("[diarize]\nvoice-distance = 0.7\n") == {"diarize": {"voice_distance": 0.7}}
    # --out is a directory; diarize writes one RTTM file and quantize one checkpoint.
    assert "diarize" not in _map('out = "~/notes"\n')
    assert "quantize" not in _map('out = "~/notes"\n')


def test_a_table_overrides_the_bare_key():
    m = _map("record = true\n\n[dictate]\nrecord = false\n")
    assert m["tui"]["record"] is True
    assert m["dictate"]["record"] is False


def test_an_option_belonging_elsewhere_says_where_it_goes(monkeypatch):
    """When a name really is scoped to one command, the error says which -- "unknown
    option" would be a lie and the fix is a table, not a spelling."""
    monkeypatch.setattr(cfgfile, "EXCLUDED", {"quantize": "--bits is a quantiser width"})
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
    # And where the two differ -- `m models --dir` sets model_dir -- both still land.
    assert _map('[models]\ndir = "/ckpts"\n') == {"models": {"model_dir": "/ckpts"}}


def test_home_relative_paths_expand():
    """Click casts strings to Paths for us, but it does not expand `~`, and a config
    file is exactly where someone writes one."""
    out = _map('out = "~/notes"\n')["live"]["out"]
    assert out == str(Path.home() / "notes")


def test_a_missing_file_is_not_a_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgfile.paths, "config_file", lambda: tmp_path / "nope.toml")
    assert cfgfile.locate() is None
    # But one named by hand was named for a reason.
    with pytest.raises(cfgfile.ConfigError):
        cfgfile.locate(tmp_path / "nope.toml")


def _invoke(tmp_path, text, *args):
    from typer.testing import CliRunner

    from mouth.app import app

    cfg = tmp_path / "config.toml"
    cfg.write_text(text)
    return CliRunner().invoke(app, ["--config", str(cfg), *args])


def test_a_flag_still_beats_the_file(tmp_path):
    """The whole claim of the feature: the file moves defaults, it doesn't pin values."""
    from_file = _invoke(tmp_path, "[cadence]\ninterim = 2.0\n", "cadence", "10")
    assert from_file.exit_code == 0
    assert "first text after 2s" in from_file.stdout

    typed = _invoke(
        tmp_path, "[cadence]\ninterim = 2.0\n", "cadence", "10", "--interim", "0.5"
    )
    assert typed.exit_code == 0
    assert "first text after 0.5s" in typed.stdout


def test_a_broken_config_stops_every_command_but_reports_itself(tmp_path):
    """`m config` is how you find out what's wrong with the file, so it survives one."""
    broken = 'backend = "mlx"\nlanguage =\n'
    assert _invoke(tmp_path, broken, "cadence", "10").exit_code == 2

    shown = _invoke(tmp_path, broken, "config")
    assert shown.exit_code == 1
    assert "line 2" in shown.stdout


# --------------------------------------------- drafted partials (--partials x-draft)


def _drafter(prev, paying=True):
    """A draft decoder holding a previous answer, with no model behind it.

    The draft policy is pure sequence logic, so it tests without weights -- which matters,
    because the parts that need a model are the parts a CPU test can never reach.
    """
    from mouth.backends import _MlxDraftDecoder

    d = _MlxDraftDecoder(backend=None, language="English")
    d._prev = list(prev)
    d._paying = paying
    d._index()
    return d


def test_draft_continues_the_previous_answer():
    d = _drafter([10, 11, 12, 13, 14, 15])
    assert d._draft([10, 11, 12])[:3] == [13, 14, 15]


def test_draft_realigns_after_a_revision():
    """The whole reason the draft is found by text and not by index.

    A partial that inserts one token near the start shifts every later token by one. Under
    index alignment the draft never matches again and acceptance collapses for the rest of
    the utterance -- which is precisely what happens when a partial heals.
    """
    prev = [1, 2, 3, 4, 5, 6, 7, 8]
    generated = [1, 2, 99, 3, 4, 5]  # 99 inserted; 3,4,5 are prev[2:5]
    assert _drafter(prev)._draft(generated)[:3] == [6, 7, 8]


def test_draft_prefers_the_most_recent_occurrence():
    """A repeated phrase must resolve to where we are now, not where we were."""
    d = _drafter([7, 8, 9, 1, 2, 7, 8, 9, 3, 4])
    assert d._draft([7, 8, 9])[:2] == [3, 4]


def test_draft_will_not_place_itself_on_a_one_token_match():
    """The floor under the key length, and it is worth real time.

    Every partial ends by decoding a tail the previous answer does not have. There, keys
    of two or more tokens all contain a new one and miss -- but a single token still hits
    somewhere unrelated, and the 64-token draft that follows is a verification that
    accepts nothing. Measured over the new tail: a draft issued 58% of the time, 37
    tokens long, 0.34 accepted. Roughly 20-30% of what drafting saves, spent proving the
    previous answer does not continue here.
    """
    d = _drafter([1, 2, 3, 4, 5])
    assert d._draft([9, 9, 3]) == [], "a one-token tail must not place us"
    assert d._draft([1, 2, 3])[:2] == [4, 5], "a real key still does"


def test_draft_is_empty_when_it_stops_paying():
    """The break-even guard: a verification costs ~3.5 steps and must earn them back."""
    assert _drafter([1, 2, 3], paying=False)._draft([1]) == []
    assert _drafter([])._draft([1, 2]) == []


def test_draft_runs_dry_past_the_end_of_the_previous_answer():
    """The tail a partial exists to add has no draft, and must decode normally."""
    assert _drafter([1, 2, 3])._draft([1, 2, 3]) == []


class _CountingDraft:
    """Stands in for the drafted decoder so the engine's use of it is observable."""

    def __init__(self):
        self.calls, self.resets = [], 0

    mode = "x-draft"
    incremental = False
    stable = ""

    def text(self, pcm):
        self.calls.append(len(pcm))
        return "drafted"

    def reset(self):
        self.resets += 1


def test_finals_never_take_the_drafted_path():
    """Finals run the aligner and are the artifact that gets saved.

    Drafted decode is verified against a batched argmax, which can differ from the
    sequential one in the last bits on a near tie -- fine for provisional text that a
    final overwrites, not for the transcript. So the final must stay on the library's own
    path no matter what.
    """
    from mouth.engine import Transcriber
    from mouth.vad import Chunk

    draft = _CountingDraft()
    segs, interims = [], []
    w = Transcriber(
        _FakeBackend(),
        "English",
        on_segment=segs.append,
        on_interim=interims.append,
        partials=draft,
        timestamps=False,
    )
    w.start()
    audio = np.zeros(FRAME_LEN, dtype=np.float32)
    w.submit(Chunk(audio, 1.0, final=False))
    # Let it drain before queueing the final: a partial with anything newer behind it is
    # deliberately dropped as stale, which would make this test pass for the wrong reason.
    for _ in range(500):
        if interims:
            break
        time.sleep(0.01)
    w.submit(Chunk(audio, 1.0, final=True))
    assert w.close(timeout=5)

    assert len(draft.calls) == 1, "the partial should have been drafted"
    assert [s.text for s in interims] == ["drafted"]
    assert [s.text for s in segs] == ["hi"], "the final must come from the backend"
    assert draft.resets == 1, "a final ends the utterance and must drop its draft"


def test_drafted_decode_upstream_symbols_still_exist():
    """Pin what _MlxDraftDecoder drives below transcribe().

    It reimplements the single-chunk path -- features, prompt, prefill, verify, decode --
    because that is the only level at which the previous answer can be reused. A rename
    upstream would otherwise surface mid-session as an AttributeError behind a spinner.
    """
    pytest.importorskip("mlx_qwen3_asr")
    from mlx_qwen3_asr import audio, generate, tokenizer

    missing = [
        n
        for mod, n in (
            (audio, "compute_features"),
            (generate, "GenerationConfig"),
            (generate, "resolve_max_new_tokens"),
            (generate, "_detect_repetition"),
            (tokenizer, "parse_asr_output"),
        )
        if not hasattr(mod, n)
    ]
    assert not missing, (
        f"mlx_qwen3_asr no longer exports {missing}; backends._MlxDraftDecoder needs them."
    )


# ------------------------------------------------------------------ m tune


def test_voiced_frames_are_not_clip_length():
    """The gate counts loud frames, and a clip is mostly not loud.

    Pre-roll and the trailing tail are part of every clip and part of none of this
    measurement -- which is exactly why a length-based gate would pass a cough.
    """
    from mouth import tune as tn

    audio = np.concatenate(frames([(False, 0.3), (True, 0.2), (False, 0.5)]))
    voiced = tn.count_voiced(audio, 0.01)
    assert 5 <= voiced <= 8, voiced  # ~0.2s of speech, not the 1.0s clip
    assert tn.count_voiced(audio, 10.0) == 0  # nothing clears an absurd threshold
    assert tn.count_voiced(np.zeros(10, dtype=np.float32), 0.01) == 0


def test_min_speech_sits_under_your_shortest_word():
    """Tuning must never make a machine deafer to short words than an untuned one."""
    from mouth import tune as tn
    from mouth.vad import MIN_SPEECH_SEC

    def sample(voiced_sec):
        return tn.Sample(
            np.zeros(16, dtype=np.float32), int(voiced_sec * SAMPLE_RATE / FRAME_LEN)
        )

    got = tn.suggest_min_speech([sample(0.2), sample(0.9)])
    assert 0.05 <= got < 0.2, got
    # A speaker who never says anything short still gets the shipped default, not a
    # gate raised above it.
    assert tn.suggest_min_speech([sample(2.0)]) == MIN_SPEECH_SEC
    assert tn.suggest_min_speech([]) == MIN_SPEECH_SEC
    # And a floor, because the gate is still what keeps key clicks out of the model.
    assert tn.suggest_min_speech([sample(0.01)]) >= 0.05


def test_cost_model_separates_fixed_from_proportional():
    from mouth import tune as tn

    model = tn.fit([(2.0, 0.20), (4.0, 0.30), (8.0, 0.50)])  # 0.1 + 0.05x
    assert model.fixed == pytest.approx(0.1, abs=0.01)
    assert model.per_sec == pytest.approx(0.05, abs=0.01)
    assert model.at(20.0) == pytest.approx(1.1, abs=0.02)


def test_cost_model_survives_a_degenerate_run():
    """One measurement should give a worse model, not a traceback."""
    from mouth import tune as tn

    assert tn.fit([]).at(5.0) == 0.0
    flat = tn.fit([(3.0, 0.4)])
    assert flat.at(1.0) == flat.at(30.0) == pytest.approx(0.4)
    same_x = tn.fit([(3.0, 0.4), (3.0, 0.6)])
    assert same_x.at(10.0) == pytest.approx(0.5)


def test_a_tighter_profile_is_fresher_and_costs_more():
    from mouth import tune as tn

    cheap = tn.CostModel(0.02, 0.01)
    got = [tn.evaluate(p, cheap, cheap, 20.0) for p in tn.PROFILES]
    assert [v.profile.name for v in got] == ["calm", "balanced", "instant"]
    assert got[0].partials < got[1].partials < got[2].partials
    assert got[0].stale_max > got[1].stale_max > got[2].stale_max
    assert got[0].compute < got[1].compute < got[2].compute


def test_recommendation_prefers_a_profile_that_survives_losing_drafting():
    """The guard switches drafting off on unstable audio, and a profile that only fits
    with it doesn't degrade -- it stops keeping up and the text stops moving."""
    from mouth import tune as tn

    # Drafted cost fits everything; undrafted only fits the relaxed schedule
    # (0.30x against snappy's 0.85x and aggressive's 1.57x).
    drafted, full = tn.CostModel(0.01, 0.004), tn.CostModel(0.08, 0.08)
    verdicts = [tn.evaluate(p, drafted, full, 20.0) for p in tn.PROFILES]
    assert all(v.load <= tn.SUSTAINABLE for v in verdicts), "all affordable when drafted"
    assert tn.recommend(verdicts).profile.name == "calm"

    # When everything degrades gracefully, take the freshest.
    verdicts = [tn.evaluate(p, drafted, drafted, 20.0) for p in tn.PROFILES]
    assert tn.recommend(verdicts).profile.name == "instant"


def test_recommendation_never_returns_nothing():
    """A machine that cannot keep up with any profile still needs an answer."""
    from mouth import tune as tn

    hopeless = tn.CostModel(5.0, 5.0)
    got = tn.recommend([tn.evaluate(p, hopeless, hopeless, 20.0) for p in tn.PROFILES])
    assert got.profile.name == "calm", "the cheapest is the least bad"


def test_tuned_config_is_valid_and_says_what_it_set():
    """`m tune --write` writes something `m config` can read back."""
    import tomllib

    from mouth import tune as tn

    text = tn.render(
        backend="mlx",
        model="q8",
        aligner="al",
        min_speech=0.12,
        profile=tn.PROFILES[1],
        drafting=True,
        note="M3 Max",
    )
    data = tomllib.loads(text)
    assert data["backend"] == "mlx" and data["min-speech"] == 0.12
    assert data["partials"] == "x-draft"
    # Bare, not under [tui]: `m live` draws partials too and `m cadence` prints what
    # the schedule costs, so a table would leave both on the shipped defaults.
    assert (data["interim"], data["growth"], data["max-gap"]) == (0.15, 1.25, 1.2)
    # And it round-trips through the real validator, against the real CLI.
    mapped = _map(text)
    assert mapped["tui"]["backend"] == "mlx"
    assert mapped["cadence"]["interim"] == 0.15, "the schedule printer must see it too"
    assert mapped["transcribe"]["min_speech"] == 0.12, "and so must the file path"


def test_bench_lengths_span_the_schedule():
    from mouth import tune as tn

    got = tn.bench_lengths(30.0)
    assert got[0] < 1.0 and got[-1] == 30.0
    assert all(b > a for a, b in itertools.pairwise(got)), got
    assert tn.bench_lengths(0.8)[-1] == 0.8


def test_profile_descriptions_claim_nothing_about_this_machine():
    """A blurb describes the experience; whether the machine can afford it is measured.

    An earlier version said "needs drafting to be affordable at all" in the text of the
    option itself, which hard-coded one machine's answer into every machine's menu.
    """
    from mouth import tune as tn

    banned = ("draft", "machine", "afford", "keep up", "cheap", "fast", "slow")
    for p in tn.PROFILES:
        assert not any(w in p.blurb.lower() for w in banned), p
        # And nothing a first-time user would have to look up.
        assert not any(w in p.blurb.lower() for w in ("partial", "cadence", "decode")), p


def test_effort_is_described_in_words():
    from mouth import tune as tn

    cheap, dear = tn.CostModel(0.001, 0.0005), tn.CostModel(2.0, 2.0)
    assert tn.evaluate(tn.PROFILES[0], cheap, cheap, 20.0).headroom == "easy"
    assert tn.evaluate(tn.PROFILES[2], dear, dear, 20.0).headroom == "too much"


def test_a_capability_cannot_be_claimed_without_the_method():
    """Optional capabilities are protocols, not flags beside the method.

    A flag is a second thing that has to stay true: a backend can carry it without the
    method, or grow the method and forget it, and either way the failure lands mid-session
    as an AttributeError. Matching on the method itself cannot come apart.
    """
    from mouth.backends import (
        Backend,
        Drafting,
        Streaming,
        open_partials,
    )

    class Liar:
        """Claims both capabilities the old way, and has neither."""

        name = detail = "liar"
        streaming = True
        drafting = True

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            return Transcription(text="x")

    liar = Liar()
    assert isinstance(liar, Backend)
    assert not isinstance(liar, Streaming) and not isinstance(liar, Drafting)
    # And the factory hands back a re-encoding decoder rather than taking their word.
    for asked in ("x-draft", "stream"):
        assert open_partials(liar, mode=asked, language="English").mode == "reencode"


def test_context_is_a_capability_and_the_tui_can_edit_it():
    """`--context` is session state a front end can change, not a per-call argument.

    Off the Backend protocol, which is one method wide and which every fake in this file
    implements without knowing anything about context.
    """
    from mouth.backends import Backend, Biasable
    from mouth.engine import Config
    from mouth.tui import build_tui

    class Plain:
        name = detail = "plain"

        def transcribe(self, audio, sample_rate, *, language, timestamps):
            return Transcription(text="x")

    class Biased(Plain):
        context = "Aristotle, the Lyceum"

    assert isinstance(Plain(), Backend) and not isinstance(Plain(), Biasable)
    assert isinstance(Biased(), Biasable)

    # The TUI offers a field for it, out of the way until asked for. Reading `.value`
    # needs a running app (it is a Textual reactive), so this checks what can be checked
    # without one: the binding exists and the field starts hidden.
    ui = build_tui(Config(), Biased())
    assert "k" in [b[0] for b in ui.BINDINGS]
    assert not ui.context.has_class("open")
    # And a backend that cannot take context still builds a UI rather than crashing.
    assert build_tui(Config(), Plain()) is not None


# ------------------------------------------------- calibrating a file vs a microphone


def test_a_file_is_calibrated_from_all_of_itself(tmp_path):
    """A recording that opens on speech must not be judged by its opening second.

    The microphone has to commit to a threshold from one second taken before anyone
    talks, so the median of that second is the noise floor. A file is already here, and
    assuming it starts quiet is how a 28-minute interview with the pauses edited out
    calibrated at 0.146 against a floor of 0.004 -- and lost 36% of its words.
    """
    import soundfile as sf

    from mouth.sources import make_source

    # Loud for a second, quiet for nine: exactly the shape that fools an opening sample.
    loud = rng.normal(0, 0.3, SAMPLE_RATE).astype(np.float32)
    quiet = rng.normal(0, 0.001, SAMPLE_RATE * 9).astype(np.float32)
    path = tmp_path / "starts-loud.wav"
    sf.write(path, np.concatenate([loud, quiet]), SAMPLE_RATE)

    got = make_source(None, path).open().calibrate()
    assert got < 0.05, f"calibrated {got:.4f} off the loud opening"
    # And it is a real floor, not just something small: the quiet part sits at ~0.001.
    assert 0.005 <= got <= 0.02


def test_the_threshold_formula_still_agrees_with_the_microphone():
    """Both paths keep the same 3x and the same 0.005 minimum.

    Only which percentile counts as the floor differs, so speech-over-silence -- what a
    microphone actually hears -- lands in the same place either way.
    """
    from mouth.sources import ambient_threshold

    # Mostly silence with occasional speech, i.e. a room with someone in it.
    levels = np.concatenate([np.full(90, 0.002), np.full(10, 0.3)])
    assert ambient_threshold(levels) == ambient_threshold(levels, quiet=10.0)
    assert ambient_threshold(np.array([])) == 0.01


def test_speech_at_the_median_is_what_breaks_the_median():
    """The failure the percentile exists for, in three lines."""
    from mouth.sources import ambient_threshold

    # A quarter room tone, three quarters speech -- cutting the pauses does not cut the
    # gaps between words. The real interview measured p10=0.004 against a median of 0.061.
    edited = np.concatenate([np.full(25, 0.004), np.full(75, 0.2)])
    assert ambient_threshold(edited) > 0.5, "3x median lands above the speech"
    assert ambient_threshold(edited, quiet=10.0) < 0.05, "the floor is still there"


# --------------------------------------------- cutting on speaker changes, not just silence


def test_a_cut_ends_an_utterance_with_no_silence_to_end_it():
    """The whole point: continuous speech, two speakers, one boundary the audio lacks.

    A recording with the pauses edited out gives the VAD nothing to close on, so one
    utterance runs to the 30s cap holding several people. Measured on a 28-minute
    interview: 58 utterances before, 117 after, and the same word count -- the cuts move
    boundaries, they do not drop audio.
    """
    continuous = [(False, 1), (True, 6), (False, 2)]
    assert len(run(continuous)) == 1
    cut = run(continuous, cuts=[3.0, 5.0])
    assert len(cut) == 3, [round(c.start, 2) for c in cut]
    # Still the same audio, minus the pre-roll each new utterance no longer needs.
    assert sum(len(c.audio) for c in cut) <= sum(len(c.audio) for c in run(continuous))


def test_a_cut_landing_in_silence_changes_nothing():
    """The boundary it marks is already there."""
    spec = [(False, 1), (True, 1), (False, 1.5), (True, 1), (False, 1.5)]
    assert len(run(spec, cuts=[2.6])) == len(run(spec)) == 2


def test_cuts_do_not_disturb_the_utterances_they_do_not_touch():
    spec = [(False, 1), (True, 2), (False, 1.5)]
    plain, cut = run(spec), run(spec, cuts=[90.0])
    assert len(plain) == len(cut) == 1
    assert len(plain[0].audio) == len(cut[0].audio)


def test_only_a_change_of_speaker_is_a_cut():
    """A diarizer emits several turns for one person talking through their own pauses.
    Cutting on those would chop a sentence for nothing."""
    from mouth.diarize import speaker_changes
    from mouth.diarize.offline import Turn

    turns = [Turn(0, 1, 0), Turn(1, 2, 0), Turn(2, 3, 1), Turn(3, 4, 0)]
    assert speaker_changes(turns) == (2.0, 3.0)
    assert speaker_changes([Turn(0, 5, 0)]) == ()
    assert speaker_changes([]) == ()
    # Order of arrival does not matter; the boundaries are in time order.
    assert speaker_changes(list(reversed(turns))) == (2.0, 3.0)
