from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from mlflow.genai.optimize import get_optimized_model
from mlflow.genai.optimize.model_selection import _set_optimized_models


def test_get_optimized_model_returns_default_when_not_optimizing():
    assert get_optimized_model("gen", default="openai:/gpt-4o-mini") == "openai:/gpt-4o-mini"


def test_get_optimized_model_returns_selection_inside_context():
    with _set_optimized_models({"gen": "openai:/gpt-4o"}):
        assert get_optimized_model("gen", default="openai:/gpt-4o-mini") == "openai:/gpt-4o"
        # A slot that is not being optimized still falls back to the default.
        assert get_optimized_model("other", default="fallback") == "fallback"


def test_selection_context_is_reset_on_exit():
    with _set_optimized_models({"gen": "openai:/gpt-4o"}):
        pass
    assert get_optimized_model("gen", default="openai:/gpt-4o-mini") == "openai:/gpt-4o-mini"


def test_selection_does_not_leak_across_threads_without_copy():
    # A raw ThreadPoolExecutor thread does not inherit the caller's context, so
    # the default is returned. This documents why optimize.py copies the context
    # explicitly into worker threads.
    with _set_optimized_models({"gen": "openai:/gpt-4o"}):
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(get_optimized_model, "gen", "openai:/gpt-4o-mini").result()
    assert result == "openai:/gpt-4o-mini"


def test_selection_propagates_across_threads_with_copy_context():
    # With copy_context().run (the mechanism optimize.py uses), the selection is
    # visible inside the worker thread.
    with _set_optimized_models({"gen": "openai:/gpt-4o"}):
        with ThreadPoolExecutor(max_workers=1) as executor:
            ctx = copy_context()
            result = executor.submit(
                ctx.run, get_optimized_model, "gen", "openai:/gpt-4o-mini"
            ).result()
    assert result == "openai:/gpt-4o"
