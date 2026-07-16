"""Model selection support for joint prompt + model optimization.

When an optimizer explores model choices (see ``GepaPromptOptimizer``'s
``model_candidates`` argument), it needs a provider-agnostic way to tell the
``predict_fn`` which model to use for the candidate currently being evaluated.

This mirrors how prompts are optimized: a ``predict_fn`` loads its prompt via
``mlflow.genai.load_prompt(...)`` and the optimizer transparently swaps the
template underneath it. Here, a ``predict_fn`` reads its model via
:py:func:`get_optimized_model` and the optimizer transparently swaps the model
name underneath it. Outside of an optimization run the accessor simply returns
the ``default``, so production code paths are unaffected.
"""

import contextvars
from contextlib import contextmanager

# Maps a model slot name -> the model name selected for the candidate currently
# being evaluated. ``None`` when no optimization is in progress. A ContextVar is
# used so the selection is scoped to the evaluation and propagates into the
# worker threads that run ``predict_fn`` (the optimizer copies the context per
# task; see ``mlflow.genai.optimize.optimize._build_eval_fn``).
_OPTIMIZED_MODELS: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "mlflow_optimized_models", default=None
)


def get_optimized_model(slot: str, default: str) -> str:
    """Return the model chosen by the optimizer for ``slot``, or ``default``.

    Call this inside your ``predict_fn`` to make a model choice optimizable.
    During an optimization run that lists ``slot`` in its ``model_candidates``,
    this returns the model name for the candidate currently being evaluated.
    In every other context (production, or a slot that is not being optimized)
    it returns ``default`` unchanged.

    Args:
        slot: A stable name identifying this model choice (e.g. ``"generator"``).
            Use the same name as the key in the optimizer's ``model_candidates``.
        default: The model to use when no optimization is selecting this slot,
            typically your current production model (e.g. ``"openai:/gpt-4o-mini"``).

    Returns:
        The model name to use for this call.

    Example:

        .. code-block:: python

            import mlflow
            from mlflow.genai.optimize import get_optimized_model


            def predict_fn(inputs: dict) -> str:
                prompt = mlflow.genai.load_prompt("prompts:/qa@production")
                model = get_optimized_model("generator", default="openai:/gpt-4o-mini")
                provider, name = model.split(":/", 1) if ":/" in model else ("openai", model)
                completion = openai.OpenAI().chat.completions.create(
                    model=name,
                    messages=[{"role": "user", "content": prompt.format(**inputs)}],
                )
                return completion.choices[0].message.content
    """
    models = _OPTIMIZED_MODELS.get()
    if models is not None and slot in models:
        return models[slot]
    return default


@contextmanager
def _set_optimized_models(models: dict[str, str]):
    """Scope a model selection for the duration of a candidate evaluation."""
    token = _OPTIMIZED_MODELS.set(models)
    try:
        yield
    finally:
        _OPTIMIZED_MODELS.reset(token)
