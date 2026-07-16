import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize.model_selection import _set_optimized_models
from mlflow.genai.optimize.optimizers.base import BasePromptOptimizer, _EvalFunc
from mlflow.genai.optimize.types import EvaluationResultRecord, PromptOptimizerOutput

if TYPE_CHECKING:
    import gepa

_logger = logging.getLogger(__name__)

# Artifact path and file name constants
PROMPT_CANDIDATES_DIR = "prompt_candidates"
EVAL_RESULTS_FILE = "eval_results.json"
SCORES_FILE = "scores.json"

# Prefix that namespaces a model-selection component inside a GEPA candidate so
# it never collides with a prompt name. GEPA treats a candidate as a flat
# ``{component_name: text}`` dict; here prompt components are keyed by prompt name
# and model components by ``<prefix><slot>``.
_MODEL_COMPONENT_PREFIX = "__model__:"

# Reflection template that steers GEPA's reflection LM to pick a model from a
# fixed candidate list, rather than write free-form text (as it does for
# prompts). GEPA requires the ``<curr_param>`` and ``<side_info>`` placeholders
# and extracts the LM's answer from the last fenced ``` block.
_MODEL_SELECTION_TEMPLATE = """The current model is:
```
<curr_param>
```

Evaluation results with this model:
```
<side_info>
```

Select the best model from this list:
{candidates}

Rules:
- You MUST pick one of the exact names listed above. Do NOT invent names.
- Always try a new model.
- Prefer cheaper/faster models when quality is similar.

Provide your chosen model name within ``` blocks."""


def _is_model_component(component_name: str) -> bool:
    """Return True if a GEPA component name refers to a model slot."""
    return component_name.startswith(_MODEL_COMPONENT_PREFIX)


def _model_component_name(slot: str) -> str:
    """Return the GEPA component name for a model slot."""
    return f"{_MODEL_COMPONENT_PREFIX}{slot}"


def _model_slot(component_name: str) -> str:
    """Return the model slot for a model component name (inverse of the above)."""
    return component_name[len(_MODEL_COMPONENT_PREFIX) :]


def _split_candidate(candidate: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Split a GEPA candidate into (prompt components, model selection by slot)."""
    prompts = {k: v for k, v in candidate.items() if not _is_model_component(k)}
    models = {_model_slot(k): v for k, v in candidate.items() if _is_model_component(k)}
    return prompts, models


class GepaPromptOptimizer(BasePromptOptimizer):
    """
    A prompt adapter that uses GEPA (Genetic-Pareto) optimization algorithm
    to optimize prompts.

    GEPA uses iterative mutation, reflection, and Pareto-aware candidate selection
    to improve text components like prompts. It leverages large language models to
    reflect on system behavior and propose improvements.

    Args:
        reflection_model: Name of the model to use for reflection and optimization.
            Format: "<provider>:/<model>"
            (e.g., "openai:/gpt-4o", "anthropic:/claude-3-5-sonnet-20241022").
        max_metric_calls: Maximum number of evaluation calls during optimization.
            Higher values may lead to better results but increase optimization time.
            Default: 100
        display_progress_bar: Whether to show a progress bar during optimization.
            Default: False
        model_candidates: Optional mapping of model slot name -> list of candidate
            model names to jointly optimize alongside the prompts. When provided,
            GEPA also selects, for each slot, the best-performing model from its
            candidate list (using the same reflection + Pareto search it uses for
            prompts). To make a model choice optimizable, read it in your
            ``predict_fn`` via
            :py:func:`mlflow.genai.optimize.get_optimized_model`. Model names are
            opaque strings passed through to your ``predict_fn`` unchanged, so any
            naming scheme your code understands works (e.g. ``"openai:/gpt-4o"``).
            Default: None (prompts only, unchanged behavior).
        gepa_kwargs: Additional keyword arguments to pass directly to
            gepa.optimize <https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py>.
            Useful for accessing advanced GEPA features not directly exposed
            through MLflow's GEPA interface.

            Note: Parameters already handled by MLflow's GEPA class will be overridden by the direct
            parameters and should not be passed through gepa_kwargs. List of predefined params:

            - max_metric_calls
            - display_progress_bar
            - seed_candidate
            - trainset
            - adapter
            - reflection_lm
            - reflection_prompt_template
            - use_mlflow

    Example:

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

            prompt = mlflow.genai.register_prompt(
                name="qa",
                template="Answer the following question: {{question}}",
            )


            def predict_fn(question: str) -> str:
                completion = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt.format(question=question)}],
                )
                return completion.choices[0].message.content


            dataset = [
                {"inputs": {"question": "What is the capital of France?"}, "outputs": "Paris"},
                {"inputs": {"question": "What is the capital of Germany?"}, "outputs": "Berlin"},
            ]

            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                prompt_uris=[prompt.uri],
                optimizer=GepaPromptOptimizer(
                    reflection_model="openai:/gpt-4o",
                    display_progress_bar=True,
                ),
            )

            print(result.optimized_prompts[0].template)

    Example: Jointly optimizing the model choice with the prompt

        .. code-block:: python

            import mlflow
            import openai
            from mlflow.genai.optimize import get_optimized_model
            from mlflow.genai.optimize.optimizers import GepaPromptOptimizer


            def predict_fn(question: str) -> str:
                prompt = mlflow.genai.load_prompt("prompts:/qa@production")
                # Reads "openai:/gpt-4o-mini" in production; the optimizer swaps in
                # a candidate during optimization.
                model = get_optimized_model("qa_model", default="openai:/gpt-4o-mini")
                completion = openai.OpenAI().chat.completions.create(
                    model=model.split(":/", 1)[-1],
                    messages=[{"role": "user", "content": prompt.format(question=question)}],
                )
                return completion.choices[0].message.content


            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=dataset,
                prompt_uris=["prompts:/qa@production"],
                optimizer=GepaPromptOptimizer(
                    reflection_model="openai:/gpt-4o",
                    model_candidates={
                        "qa_model": ["openai:/gpt-4o-mini", "openai:/gpt-4o"],
                    },
                ),
            )

            print(result.optimized_prompts[0].template)
            print(result.optimized_models)  # e.g. {"qa_model": "openai:/gpt-4o"}
    """

    def __init__(
        self,
        reflection_model: str,
        max_metric_calls: int = 100,
        display_progress_bar: bool = False,
        model_candidates: dict[str, list[str]] | None = None,
        gepa_kwargs: dict[str, Any] | None = None,
    ):
        self.reflection_model = reflection_model
        self.max_metric_calls = max_metric_calls
        self.display_progress_bar = display_progress_bar
        self.model_candidates = self._validate_model_candidates(model_candidates)
        self.gepa_kwargs = gepa_kwargs or {}

    @staticmethod
    def _validate_model_candidates(
        model_candidates: dict[str, list[str]] | None,
    ) -> dict[str, list[str]]:
        """Validate and normalize the ``model_candidates`` argument."""
        if not model_candidates:
            return {}
        normalized: dict[str, list[str]] = {}
        for slot, candidates in model_candidates.items():
            if not isinstance(slot, str) or not slot:
                raise MlflowException.invalid_parameter_value(
                    "`model_candidates` keys must be non-empty strings naming a model slot, "
                    f"got {slot!r}."
                )
            if _is_model_component(slot):
                raise MlflowException.invalid_parameter_value(
                    f"`model_candidates` slot names must not start with "
                    f"{_MODEL_COMPONENT_PREFIX!r} (reserved for internal use), got {slot!r}."
                )
            if not candidates or not isinstance(candidates, (list, tuple)):
                raise MlflowException.invalid_parameter_value(
                    f"`model_candidates[{slot!r}]` must be a non-empty list of model names."
                )
            if not all(isinstance(m, str) and m for m in candidates):
                raise MlflowException.invalid_parameter_value(
                    f"`model_candidates[{slot!r}]` must contain only non-empty model name strings."
                )
            # Preserve order while dropping duplicates so the seed (first entry) is stable.
            seen: set[str] = set()
            deduped = [m for m in candidates if not (m in seen or seen.add(m))]
            normalized[slot] = deduped
        return normalized

    def optimize(
        self,
        eval_fn: _EvalFunc,
        train_data: list[dict[str, Any]],
        target_prompts: dict[str, str],
        enable_tracking: bool = True,
    ) -> PromptOptimizerOutput:
        """
        Optimize the target prompts using GEPA algorithm.

        Args:
            eval_fn: The evaluation function that takes candidate prompts as a dict
                (prompt template name -> prompt template) and a dataset as a list of dicts,
                and returns a list of EvaluationResultRecord.
            train_data: The dataset to use for optimization. Each record should
                include the inputs and outputs fields with dict values.
            target_prompts: The target prompt templates to use. The key is the prompt template
                name and the value is the prompt template.
            enable_tracking: If True (default), automatically log optimization progress.

        Returns:
            The outputs of the prompt optimizer that includes the optimized prompts
            as a dict (prompt template name -> prompt template).
        """
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        if not train_data:
            raise MlflowException.invalid_parameter_value(
                "GEPA optimizer requires `train_data` to be provided."
            )

        try:
            import gepa
        except ImportError as e:
            raise ImportError(
                "GEPA >= 0.0.26 is required. Please install it with: `pip install 'gepa>=0.0.26'`"
            ) from e

        if self.model_candidates:
            # Reject prompt names that collide with the reserved model-component
            # namespace, otherwise _split_candidate would drop them from the
            # prompt dict passed to eval_fn.
            reserved = [name for name in target_prompts if _is_model_component(name)]
            if reserved:
                raise MlflowException.invalid_parameter_value(
                    f"Prompt names must not start with {_MODEL_COMPONENT_PREFIX!r} when "
                    f"`model_candidates` is set (reserved for model components): {reserved}."
                )
            # Per-component reflection templates require a GEPA version that
            # accepts a dict for `reflection_prompt_template` (a str-only version
            # would reject the model-slot templates below).
            self._check_gepa_supports_template_map(gepa)

        provider, model = _parse_model_uri(self.reflection_model)

        class MlflowGEPAAdapter(gepa.GEPAAdapter):
            """
            MLflow optimization adapter for GEPA optimization

            Args:
                eval_function: Function that evaluates candidate prompts on a dataset.
                prompts_dict: Dictionary mapping prompt names to their templates.
                tracking_enabled: Whether to log traces/metrics/params/artifacts during
                    optimization.
                full_dataset_size: Size of the full training dataset, used to distinguish
                    full validation passes from minibatch evaluations.
            """

            def __init__(
                self,
                eval_function,
                prompts_dict,
                tracking_enabled,
                full_dataset_size,
                model_candidates,
            ):
                self.eval_function = eval_function
                self.prompts_dict = prompts_dict
                self.prompt_names = list(prompts_dict.keys())
                self.tracking_enabled = tracking_enabled
                self.full_dataset_size = full_dataset_size
                self.validation_iteration = 0
                # slot -> ordered list of allowed model names. Used both to reject
                # reflection LM hallucinations before they reach predict_fn and to
                # list not-yet-evaluated candidates in the reflection dataset.
                self.model_candidates = model_candidates
                # slot -> {model_name -> [scores]}, feeds the model-selection reflection
                # so the reflection LM can see how each candidate model has performed.
                self.model_score_history: dict[str, dict[str, list[float]]] = {}

            def evaluate(
                self,
                batch: list[dict[str, Any]],
                candidate: dict[str, str],
                capture_traces: bool = False,
            ) -> "gepa.EvaluationBatch":
                """
                Evaluate a candidate (prompts and, optionally, model choices) using
                the MLflow eval function.

                Args:
                    batch: List of data instances to evaluate
                    candidate: Proposed text components (prompts and model selections)
                    capture_traces: Whether to capture execution traces

                Returns:
                    EvaluationBatch with outputs, scores, and optional trajectories
                """
                prompt_candidate, model_selection = _split_candidate(candidate)

                # Reject reflection-LM hallucinations: if a chosen model is not in
                # its slot's candidate list, score the candidate 0 so GEPA learns to
                # avoid it, rather than passing an unknown model to predict_fn.
                invalid = [
                    f"{slot}={name!r}"
                    for slot, name in model_selection.items()
                    if slot in self.model_candidates and name not in self.model_candidates[slot]
                ]
                if invalid:
                    _logger.debug(
                        "Rejecting candidate with out-of-list model selection(s): %s",
                        ", ".join(invalid),
                    )
                    return gepa.EvaluationBatch(
                        outputs=[None] * len(batch),
                        scores=[0.0] * len(batch),
                        trajectories=None,
                        objective_scores=None,
                    )

                # The eval function only knows about prompt components; the model
                # selection is delivered to predict_fn out of band via a ContextVar.
                if model_selection:
                    with _set_optimized_models(model_selection):
                        eval_results = self.eval_function(prompt_candidate, batch)
                else:
                    eval_results = self.eval_function(prompt_candidate, batch)

                outputs = [result.outputs for result in eval_results]
                scores = [result.score for result in eval_results]
                trajectories = eval_results if capture_traces else None
                objective_scores = [result.individual_scores for result in eval_results]

                # Record per-model scores so make_reflective_dataset can summarize
                # how each candidate model has performed across iterations. Only
                # numeric scores are kept (score is None in zero-shot mode), so a
                # recorded model always has at least one score.
                numeric_scores = [s for s in scores if s is not None]
                if numeric_scores:
                    for slot, model_name in model_selection.items():
                        slot_history = self.model_score_history.setdefault(slot, {})
                        slot_history.setdefault(model_name, []).extend(numeric_scores)

                # Track validation candidates only during full dataset validation
                # (not during minibatch evaluation in reflective mutation)
                is_full_validation = not capture_traces and len(batch) == self.full_dataset_size
                if is_full_validation and self.tracking_enabled:
                    self._log_validation_candidate(candidate, eval_results)

                return gepa.EvaluationBatch(
                    outputs=outputs,
                    scores=scores,
                    trajectories=trajectories,
                    objective_scores=objective_scores if any(objective_scores) else None,
                )

            def _log_validation_candidate(
                self,
                candidate: dict[str, str],
                eval_results: list[EvaluationResultRecord],
            ) -> None:
                """
                Log validation candidate prompts and scores as MLflow artifacts.

                Args:
                    candidate: The candidate prompts being validated
                    eval_results: Evaluation results containing scores
                """
                if not self.tracking_enabled:
                    return

                iteration = self.validation_iteration
                self.validation_iteration += 1

                # Compute aggregate score across all records
                aggregate_score = (
                    sum(r.score for r in eval_results) / len(eval_results) if eval_results else 0.0
                )

                # Collect all scorer names
                scorer_names = set()
                for result in eval_results:
                    scorer_names |= result.individual_scores.keys()

                # Build the evaluation results table and log to MLflow as a table artifact
                eval_results_table = {
                    "inputs": [r.inputs for r in eval_results],
                    "output": [r.outputs for r in eval_results],
                    "expectation": [r.expectations for r in eval_results],
                    "aggregate_score": [r.score for r in eval_results],
                }
                for scorer_name in scorer_names:
                    eval_results_table[scorer_name] = [
                        r.individual_scores.get(scorer_name) for r in eval_results
                    ]

                iteration_dir = f"{PROMPT_CANDIDATES_DIR}/iteration_{iteration}"
                mlflow.log_table(
                    data=eval_results_table,
                    artifact_file=f"{iteration_dir}/{EVAL_RESULTS_FILE}",
                )

                # Compute per-scorer average scores
                per_scorer_scores = {}
                for scorer_name in scorer_names:
                    scores = [
                        r.individual_scores[scorer_name]
                        for r in eval_results
                        if scorer_name in r.individual_scores
                    ]
                    if scores:
                        per_scorer_scores[scorer_name] = sum(scores) / len(scores)

                # Log per-scorer metrics for time progression visualization
                mlflow.log_metrics(
                    {"eval_score": aggregate_score}
                    | {f"eval_score.{name}": score for name, score in per_scorer_scores.items()},
                    step=iteration,
                )

                # Log scores summary as JSON artifact
                scores_data = {
                    "aggregate": aggregate_score,
                    "per_scorer": per_scorer_scores,
                }
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    scores_path = tmp_path / SCORES_FILE
                    with open(scores_path, "w") as f:
                        json.dump(scores_data, f, indent=2)
                    mlflow.log_artifact(scores_path, artifact_path=iteration_dir)

                    # Write each component as a separate text file. Model
                    # components carry a ``__model__:`` prefix that is not a valid
                    # filename on all platforms, so normalize it to ``model.<slot>``.
                    for component_name, component_text in candidate.items():
                        if _is_model_component(component_name):
                            file_stem = f"model.{_model_slot(component_name)}"
                        else:
                            file_stem = component_name
                        component_path = tmp_path / f"{file_stem}.txt"
                        with open(component_path, "w") as f:
                            f.write(component_text)
                        mlflow.log_artifact(component_path, artifact_path=iteration_dir)

            def make_reflective_dataset(
                self,
                candidate: dict[str, str],
                eval_batch: "gepa.EvaluationBatch[EvaluationResultRecord, Any]",
                components_to_update: list[str],
            ) -> dict[str, list[dict[str, Any]]]:
                """
                Build a reflective dataset for instruction refinement.

                Args:
                    candidate: The evaluated candidate
                    eval_batch: Result of evaluate with capture_traces=True
                    components_to_update: Component names to update

                Returns:
                    Dict of reflective dataset per component
                """
                reflective_datasets = {}

                for component_name in components_to_update:
                    component_data = []
                    is_model = _is_model_component(component_name)
                    model_history = (
                        self._model_history_str(_model_slot(component_name)) if is_model else None
                    )

                    trajectories = eval_batch.trajectories

                    for i, (trajectory, score) in enumerate(zip(trajectories, eval_batch.scores)):
                        trace = trajectory.trace
                        spans = []
                        if trace:
                            spans = [
                                {
                                    "name": span.name,
                                    "inputs": span.inputs,
                                    "outputs": span.outputs,
                                }
                                for span in trace.data.spans
                            ]

                        record = {
                            "component_name": component_name,
                            "current_text": candidate.get(component_name, ""),
                            "trace": spans,
                            "score": score,
                            "inputs": trajectory.inputs,
                            "outputs": trajectory.outputs,
                            "expectations": trajectory.expectations,
                            "rationales": trajectory.rationales,
                            "index": i,
                        }
                        if model_history:
                            record["model_performance_so_far"] = model_history
                        component_data.append(record)

                    reflective_datasets[component_name] = component_data

                return reflective_datasets

            def _model_history_str(self, slot: str) -> str:
                """Summarize how each candidate model has scored for ``slot``."""
                history = self.model_score_history.get(slot, {})
                lines = []
                for model_name, scores in sorted(history.items()):
                    mean = sum(scores) / len(scores)
                    lines.append(
                        f"- {model_name}: mean={mean:.3f} "
                        f"range=[{min(scores):.3f}, {max(scores):.3f}] trials={len(scores)}"
                    )
                lines.extend(
                    f"- {model_name}: not yet evaluated"
                    for model_name in self.model_candidates.get(slot, [])
                    if model_name not in history
                )
                return "\n".join(lines) if lines else "(none yet)"

        adapter = MlflowGEPAAdapter(
            eval_fn,
            target_prompts,
            enable_tracking,
            full_dataset_size=len(train_data),
            model_candidates=self.model_candidates,
        )

        # Seed the search with the prompts plus, for each model slot, its first
        # candidate as the starting model.
        seed_candidate = dict(target_prompts)
        for slot, candidates in self.model_candidates.items():
            seed_candidate[_model_component_name(slot)] = candidates[0]

        kwargs = self.gepa_kwargs | {
            "seed_candidate": seed_candidate,
            "trainset": train_data,
            "adapter": adapter,
            "reflection_lm": f"{provider}/{model}",
            "max_metric_calls": self.max_metric_calls,
            "display_progress_bar": self.display_progress_bar,
            "use_mlflow": enable_tracking,
        }

        # Model components must be chosen from a fixed candidate list, so give
        # them a discrete-choice reflection template instead of GEPA's default
        # free-form instruction-writing one. Prompt components fall back to
        # GEPA's default template (they are simply absent from the dict).
        if self.model_candidates:
            model_templates = self._build_model_reflection_templates()
            caller_templates = kwargs.get("reflection_prompt_template")
            if caller_templates is None:
                kwargs["reflection_prompt_template"] = model_templates
            elif isinstance(caller_templates, dict):
                # Merge caller-supplied per-prompt templates with the model-slot
                # templates. The caller must not override a model slot's template.
                conflicts = [k for k in caller_templates if _is_model_component(k)]
                if conflicts:
                    raise MlflowException.invalid_parameter_value(
                        "`reflection_prompt_template` in `gepa_kwargs` must not set templates for "
                        f"model components (these are managed by `model_candidates`): {conflicts}."
                    )
                kwargs["reflection_prompt_template"] = {**caller_templates, **model_templates}
            else:
                # A single string template would apply to every component,
                # including the model slots, defeating discrete model selection.
                raise MlflowException.invalid_parameter_value(
                    "When `model_candidates` is set, a `reflection_prompt_template` passed via "
                    "`gepa_kwargs` must be a dict mapping prompt names to templates (model slots "
                    "get their own templates automatically), not a single string."
                )

        gepa_result = gepa.optimize(**kwargs)

        optimized_prompts, optimized_models = _split_candidate(gepa_result.best_candidate)
        (
            initial_eval_score,
            final_eval_score,
            initial_eval_score_per_scorer,
            final_eval_score_per_scorer,
        ) = self._extract_eval_scores(gepa_result)

        return PromptOptimizerOutput(
            optimized_prompts=optimized_prompts,
            optimized_models=optimized_models,
            initial_eval_score=initial_eval_score,
            final_eval_score=final_eval_score,
            initial_eval_score_per_scorer=initial_eval_score_per_scorer,
            final_eval_score_per_scorer=final_eval_score_per_scorer,
        )

    def _build_model_reflection_templates(self) -> dict[str, str]:
        """Build per-slot discrete-choice reflection templates for model components."""
        templates: dict[str, str] = {}
        for slot, candidates in self.model_candidates.items():
            candidate_list = "\n".join(f"- {name}" for name in candidates)
            templates[_model_component_name(slot)] = _MODEL_SELECTION_TEMPLATE.format(
                candidates=candidate_list
            )
        return templates

    @staticmethod
    def _check_gepa_supports_template_map(gepa) -> None:
        """Ensure the installed GEPA accepts a per-component reflection template map.

        ``model_candidates`` relies on passing ``reflection_prompt_template`` as a
        ``dict[str, str]``; older GEPA versions only accept a single ``str`` and
        would raise a confusing error deep inside ``gepa.optimize``.
        """
        import inspect

        try:
            annotation = (
                inspect.signature(gepa.optimize).parameters["reflection_prompt_template"].annotation
            )
        except (ValueError, KeyError):
            # Can't introspect; let GEPA validate at call time rather than block.
            return
        if "dict" not in str(annotation):
            raise MlflowException.invalid_parameter_value(
                "`model_candidates` requires a GEPA version whose `reflection_prompt_template` "
                "accepts a per-component dict (e.g. `pip install 'gepa>=0.1.0'`). The installed "
                f"version exposes `reflection_prompt_template: {annotation}`."
            )

    def _extract_eval_scores(
        self, result: "gepa.GEPAResult"
    ) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
        """
        Extract initial and final evaluation scores from GEPA result.

        Args:
            result: GEPA optimization result

        Returns:
            Tuple of (initial_eval_score, final_eval_score,
                      initial_eval_score_per_scorer, final_eval_score_per_scorer).
            Aggregated scores can be None if unavailable.
        """
        final_eval_score = None
        initial_eval_score = None
        initial_eval_score_per_scorer: dict[str, float] = {}
        final_eval_score_per_scorer: dict[str, float] = {}

        scores = result.val_aggregate_scores
        if scores and len(scores) > 0:
            # The first score is the initial baseline score
            initial_eval_score = scores[0]
            # The highest score is the final optimized score
            final_eval_score = max(scores)

        # Extract per-scorer scores from val_aggregate_subscores
        subscores = getattr(result, "val_aggregate_subscores", None)
        if subscores and len(subscores) > 0:
            # The first subscore dict is the initial baseline per-scorer scores
            initial_eval_score_per_scorer = subscores[0] or {}
            # Find the per-scorer scores corresponding to the best aggregate score
            if scores and len(scores) > 0:
                best_idx = scores.index(max(scores))
                if best_idx < len(subscores) and subscores[best_idx]:
                    final_eval_score_per_scorer = subscores[best_idx]

        return (
            initial_eval_score,
            final_eval_score,
            initial_eval_score_per_scorer,
            final_eval_score_per_scorer,
        )
