# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

import uairt
from uairt import _generate

FAKE = os.environ.get("UAIRT_TEST_SESSION_PLUGIN")  # the fake-session plugin built by CMake; the `python_sessions` ctest (shared build) sets these variables
MODEL = os.environ.get("UAIRT_LLAMACPP_TEST_MODEL")
LLAMA_PLUGIN = os.environ.get("UAIRT_TEST_LLAMACPP_PLUGIN")

needs_fake = pytest.mark.skipif(not FAKE, reason="set UAIRT_TEST_SESSION_PLUGIN to the fake-session plugin")
needs_llama = pytest.mark.skipif(not (MODEL and LLAMA_PLUGIN),
                                 reason="set UAIRT_LLAMACPP_TEST_MODEL and UAIRT_TEST_LLAMACPP_PLUGIN")


@pytest.fixture
def fake_model():
    if "fake-session" not in uairt.backends():
        uairt.load_backend_library(FAKE)
    with uairt.Engine("fake-session") as engine, engine.load_model("unused") as model:
        yield model


# --- sampling needs no plugin -------------------------------------------------------------------------------------

def test_greedy_picks_the_largest_logit():
    assert _generate.sample(np.array([0.1, 3.0, 2.0], np.float32)) == 1


def test_top_k_one_is_greedy_at_any_temperature():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert _generate.sample(np.array([0.1, 3.0, 2.9], np.float32), temperature=5.0, top_k=1, rng=rng) == 1


def test_a_seed_makes_sampling_reproducible():
    logits = np.linspace(0, 1, 50).astype(np.float32)
    first = [_generate.sample(logits, 1.0, rng=np.random.default_rng(7)) for _ in range(5)]
    second = [_generate.sample(logits, 1.0, rng=np.random.default_rng(7)) for _ in range(5)]
    assert first == second


def test_a_tiny_top_p_keeps_only_the_best_token():
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert _generate.sample(np.array([0.0, 4.0, 3.9, 0.0], np.float32), 1.0, top_p=0.01, rng=rng) == 1


@pytest.mark.parametrize("kwargs", [{"temperature": -1.0}, {"temperature": float("nan")}, {"temperature": float("inf")}, {"top_k": -1}, {"top_p": 0.0}, {"top_p": 1.5}])
def test_invalid_sampling_arguments(kwargs):
    with pytest.raises(uairt.InvalidArgument):
        _generate.sample(np.array([1.0, 2.0], np.float32), **kwargs)


def test_non_finite_logits_are_rejected():
    with pytest.raises(uairt.InvalidArgument):
        _generate.sample(np.array([1.0, np.nan], np.float32))


# --- the session API through the fake plugin ----------------------------------------------------------------------

@needs_fake
def test_tokenize_rejects_lone_surrogates(fake_model):
    with pytest.raises(uairt.InvalidArgument, match="lone surrogate"):
        fake_model.tokenize("a\ud800b")


@needs_fake
def test_vocabulary_and_tokenizing(fake_model):
    assert fake_model.vocab_size == 257
    tokens = fake_model.tokenize("hi")
    assert tokens.dtype == np.int32 and tokens.tolist() == [256, 104, 105]
    assert fake_model.tokenize("hi", add_special=False).tolist() == [104, 105]
    assert fake_model.tokenize("", add_special=False).tolist() == []
    assert fake_model.detokenize(tokens) == "hi"
    assert fake_model.detokenize([]) == ""


@needs_fake
def test_multibyte_text_round_trips(fake_model):
    text = "héllo 日本"
    assert fake_model.detokenize(fake_model.tokenize(text, add_special=False)) == text


@needs_fake
def test_a_single_token_can_be_part_of_a_character(fake_model):
    first_byte = fake_model.tokenize("日", add_special=False)[:1]
    assert fake_model.detokenize_bytes(first_byte) == "日".encode()[:1]
    assert fake_model.detokenize(first_byte) == "�"


@needs_fake
def test_token_arrays_are_validated(fake_model):
    with pytest.raises(uairt.InvalidArgument):
        fake_model.detokenize(np.array([1.5, 2.5]))
    with pytest.raises(uairt.InvalidArgument):
        fake_model.detokenize(np.array([2 ** 40], dtype=np.int64))
    with pytest.raises(uairt.InvalidArgument):
        fake_model.detokenize([5000])


@needs_fake
def test_session_state(fake_model):
    with fake_model.session(n_ctx=4) as session:
        assert session.position == 0
        with pytest.raises(uairt.InvalidArgument):
            session.logits()
        session.append(fake_model.tokenize("ab"))
        assert session.position == 3
        logits = session.logits()
        assert logits.dtype == np.float32 and logits.shape == (257,) and int(np.argmax(logits)) == ord("b") + 1
        with pytest.raises(uairt.InvalidArgument, match="context full"):
            session.append([1, 2])
        assert session.position == 3
        session.append([])
        session.reset()
        assert session.position == 0


@needs_fake
def test_bad_session_option(fake_model):
    with pytest.raises(uairt.InvalidArgument):
        fake_model.session(nope=1)


@needs_fake
def test_run_is_unsupported_on_a_session_only_model(fake_model):
    with pytest.raises(uairt.Unsupported):
        fake_model.run()


@needs_fake
def test_generate_counts_upward(fake_model):
    assert "".join(uairt.generate(fake_model, "a", max_tokens=3)) == "bcd"
    assert "".join(uairt.generate(fake_model, "a", max_tokens=0)) == ""
    assert "".join(uairt.generate(fake_model, "a", max_tokens=5, stop_tokens=[100])) == "bc"


@needs_fake
def test_generate_validates_its_arguments(fake_model):
    with pytest.raises(uairt.InvalidArgument):
        list(uairt.generate(fake_model, "a", max_tokens=-1))
    with pytest.raises(uairt.InvalidArgument):
        list(uairt.generate(fake_model, "a", temperature=-1))


@needs_fake
def test_generate_yields_valid_text_when_tokens_split_a_character(fake_model):
    # the fake model emits bytes 0x80, 0x81, ...: continuation bytes with no lead byte
    pieces = list(uairt.generate(fake_model, "\x7f", max_tokens=3, temperature=0.0))
    assert all(isinstance(piece, str) for piece in pieces)


@needs_fake
def test_a_session_given_to_generate_is_left_open(fake_model):
    with fake_model.session() as session:
        list(uairt.generate(fake_model, "a", max_tokens=2, session=session))
        assert session.position == 3


# --- the real backend ---------------------------------------------------------------------------------------------

@needs_llama
def test_gguf_through_automodel():
    if "llamacpp" not in uairt.backends():
        uairt.load_backend_library(LLAMA_PLUGIN)
    with uairt.AutoModel.from_file(MODEL, backend="llamacpp", device="cpu") as model:
        assert model.backend == "llamacpp" and model.options == {"n_gpu_layers": "0"}
        assert model.vocab_size > 1000
        text = "héllo 日本"
        assert model.detokenize(model.tokenize(text, add_special=False)) == text
        out = "".join(model.generate("The capital of France is", max_tokens=8))
        assert "Paris" in out
        with pytest.raises(uairt.Unsupported):
            model.run()


# --- a model closes its sessions ----------------------------------------------------------------------------------

def _fake_engine():
    if "fake-session" not in uairt.backends():
        uairt.load_backend_library(FAKE)
    return uairt.Engine("fake-session")


@needs_fake
def test_closing_the_model_closes_its_sessions():
    with _fake_engine() as engine:
        model = engine.load_model("unused")
        session = model.session()
        session.append([1, 2])
        model.close()
        with pytest.raises(uairt.InvalidArgument, match="the session is closed"):
            session.append([1])
        with pytest.raises(uairt.InvalidArgument, match="the session is closed"):
            session.logits()
        with pytest.raises(uairt.InvalidArgument, match="the session is closed"):
            session.position
        with pytest.raises(uairt.InvalidArgument, match="the session is closed"):
            session.reset()


@needs_fake
def test_a_session_collected_after_its_model_closed_is_harmless():
    import gc
    with _fake_engine() as engine:
        model = engine.load_model("unused")
        session = model.session()
        model.close()
        session.close()
        session.close()
        del session
        gc.collect()
        model.close()


@needs_fake
def test_a_session_left_open_in_a_with_block_is_collected_safely():
    import gc
    with _fake_engine() as engine:
        with engine.load_model("unused") as model:
            session = model.session()
        del session
        gc.collect()


@needs_fake
def test_closing_the_session_first_still_works(fake_model):
    session = fake_model.session()
    session.close()
    session.close()
    assert "".join(uairt.generate(fake_model, "a", max_tokens=3)) == "bcd"
