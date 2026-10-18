# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
import pathlib

import pytest

import uairt

PLUGIN = os.environ.get("UAIRT_TEST_LLMSERVER_PLUGIN")
needs_plugin = pytest.mark.skipif(not PLUGIN, reason="set UAIRT_TEST_LLMSERVER_PLUGIN to the llmserver plugin")

# The fake OpenAI-compatible server of the C test, which echoes what it received.
_path = pathlib.Path(__file__).resolve().parents[3] / "tests" / "test_llmserver.py"
_spec = importlib.util.spec_from_file_location("fake_llm_server", _path)
fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake)


@pytest.fixture(scope="module")
def servers():
    plain, keyed = fake.serve(None), fake.serve("secret")
    yield "http://127.0.0.1:%d" % plain.server_address[1], "http://127.0.0.1:%d" % keyed.server_address[1]
    plain.shutdown()
    keyed.shutdown()


@pytest.fixture
def model(servers):
    if "llmserver" not in uairt.backends():
        uairt.load_backend_library(PLUGIN)
    with uairt.Engine("llmserver", base_url=servers[0]) as engine, engine.load_model("fake-model") as model:
        yield model


@needs_plugin
def test_complete_returns_the_text_and_sends_the_options(model):
    text = model.complete("hi", max_tokens=16, temperature=0.5, top_p=1, seed=7, system="be brief")
    assert text == "model=fake-model;sys=be brief;max=16;temp=0.5;top_p=1;seed=7;key=;prompt=hi"


@needs_plugin
def test_options_can_be_a_dict(model):
    assert "max=9;" in model.complete("hi", {"max_tokens": 9})


@needs_plugin
def test_non_ascii_text_round_trips(model):
    prompt = "say \"hi\"\n héllo ✓ \U0001F600"
    assert model.complete(prompt).endswith("prompt=" + prompt)


@needs_plugin
def test_on_text_streams_pieces_in_order(model):
    pieces = []
    text = model.complete("hi", on_text=pieces.append)
    assert len(pieces) > 1 and "".join(pieces) == text


@needs_plugin
def test_on_text_can_stop_the_generation(model):
    pieces = []

    def stop_after_two(piece):
        pieces.append(piece)
        return len(pieces) >= 2

    text = model.complete("__count__", on_text=stop_after_two)
    assert len(pieces) == 2 and text == "".join(pieces)


@needs_plugin
def test_an_exception_in_on_text_is_raised_here(model):
    def explode(piece):
        raise ValueError("from the callback")

    with pytest.raises(ValueError, match="from the callback"):
        model.complete("hi", on_text=explode)


@needs_plugin
def test_server_errors_map_to_exceptions(model):
    with pytest.raises(uairt.RuntimeFailure, match="boom"):
        model.complete("__error500__")
    with pytest.raises(uairt.InvalidArgument):
        model.complete("hi", temperature=3)
    with pytest.raises(uairt.InvalidArgument):
        model.complete("hi", mirostat=1)


@needs_plugin
def test_api_key_and_missing_model(servers):
    if "llmserver" not in uairt.backends():
        uairt.load_backend_library(PLUGIN)
    with uairt.Engine("llmserver", base_url=servers[1], api_key="secret") as engine:
        with engine.load_model("fake-model") as model:
            assert ";key=secret;" in model.complete("hi")
        with pytest.raises(uairt.NotFound):
            engine.load_model("nope")
    with uairt.Engine("llmserver", base_url=servers[1]) as engine:
        with pytest.raises(uairt.InvalidArgument, match="bad key"):
            engine.load_model("fake-model")


@needs_plugin
def test_closed_model_is_refused(model):
    model.close()
    with pytest.raises(uairt.InvalidArgument):
        model.complete("hi")


def test_a_backend_without_text_support_is_unsupported():
    fake_plugin = os.environ.get("UAIRT_TEST_SESSION_PLUGIN")
    if not fake_plugin:
        pytest.skip("set UAIRT_TEST_SESSION_PLUGIN")
    if "fake-session" not in uairt.backends():
        uairt.load_backend_library(fake_plugin)
    with uairt.Engine("fake-session") as engine, engine.load_model("unused") as model:
        with pytest.raises(uairt.Unsupported):
            model.complete("hi")


MODEL = os.environ.get("UAIRT_LLAMACPP_TEST_MODEL")
LLAMA_PLUGIN = os.environ.get("UAIRT_TEST_LLAMACPP_PLUGIN")


@pytest.mark.skipif(not (MODEL and LLAMA_PLUGIN), reason="set UAIRT_LLAMACPP_TEST_MODEL and UAIRT_TEST_LLAMACPP_PLUGIN")
def test_llamacpp_complete_is_greedy_and_repeatable():
    if "llamacpp" not in uairt.backends():
        uairt.load_backend_library(LLAMA_PLUGIN)
    with uairt.Engine("llamacpp", n_gpu_layers=0) as engine, engine.load_model(MODEL) as model:
        first = model.complete("Name three colors.", max_tokens=24)
        assert first and first == model.complete("Name three colors.", max_tokens=24, temperature=0)
        pieces = []
        model.complete("Write a long story about a dragon.", max_tokens=30, on_text=lambda p: pieces.append(p) or len(pieces) >= 3)
        assert len(pieces) == 3
