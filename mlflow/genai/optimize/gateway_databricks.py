"""Apply an optimized model choice to a Databricks AI Gateway (V2) endpoint.

The optimizer (:py:func:`mlflow.genai.optimize_prompts`) is gateway-agnostic: it
returns the best model per slot. Applying that winner to a live endpoint is a
per-gateway step. This module is the Databricks AI Gateway backend for that step
-- the managed analog of the OSS-gateway :py:func:`mlflow.gateway.update_endpoint_model`.

It reuses MLflow's Databricks auth and HTTP machinery (``get_databricks_host_creds``
+ ``http_request``), so it authenticates the same way as every other Databricks
call in MLflow (PAT, OAuth, CLI profile, or in-notebook credentials) rather than
handling tokens itself. The model is swapped transparently behind the endpoint
name via a partial update, so clients calling the endpoint need no changes.
"""

import posixpath
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request

# AI Gateway V2 endpoints live under this REST base, distinct from Model Serving's
# /api/2.0/serving-endpoints. A pay-per-token foundation model destination points
# at the ``system.ai.<model>`` Unity Catalog name.
_AI_GATEWAY_V2_BASE = "/api/ai-gateway/v2/endpoints"
_PAY_PER_TOKEN_FOUNDATION_MODEL = "PAY_PER_TOKEN_FOUNDATION_MODEL"


def update_databricks_endpoint_model(
    endpoint_name: str,
    model_name: str,
    *,
    destination_type: str = _PAY_PER_TOKEN_FOUNDATION_MODEL,
    databricks_uri: str = "databricks",
) -> dict[str, Any]:
    """Swap the model behind a Databricks AI Gateway (V2) endpoint, in place.

    Repoints ``endpoint_name`` to serve ``model_name`` via a single-destination,
    100%-traffic update, leaving the endpoint name (and therefore all client code)
    unchanged. This is the Databricks backend of the optimizer's "promote the
    winner" step; the OSS-gateway backend is
    :py:func:`mlflow.gateway.update_endpoint_model`.

    Args:
        endpoint_name: Name of the existing AI Gateway V2 endpoint to update.
        model_name: New model to serve. For pay-per-token foundation models this
            is the model name; it is resolved to the ``system.ai.<model_name>``
            destination unless already prefixed with ``system.ai.``.
        destination_type: AI Gateway V2 destination type. Defaults to
            ``"PAY_PER_TOKEN_FOUNDATION_MODEL"``.
        databricks_uri: MLflow Databricks target URI used for authentication
            (e.g. ``"databricks"`` or ``"databricks://<profile>"``). Auth is
            resolved by MLflow's standard Databricks credential chain.

    Returns:
        The updated endpoint as returned by the AI Gateway V2 API.

    Raises:
        MlflowException: If the update request fails.
    """
    if destination_type == _PAY_PER_TOKEN_FOUNDATION_MODEL and not model_name.startswith(
        "system.ai."
    ):
        destination_name = f"system.ai.{model_name}"
    else:
        destination_name = model_name

    config = {
        "destinations": [
            {"name": destination_name, "type": destination_type, "traffic_percentage": 100}
        ],
        "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
    }

    response = http_request(
        host_creds=get_databricks_host_creds(databricks_uri),
        endpoint=posixpath.join(_AI_GATEWAY_V2_BASE, endpoint_name),
        method="PATCH",
        params={"update_mask": "config.destinations,config.routing_strategy"},
        json={"config": config},
        raise_on_status=False,
    )
    augmented_raise_for_status(response)
    try:
        return response.json()
    except ValueError as e:
        raise MlflowException(
            f"AI Gateway returned a non-JSON response updating endpoint {endpoint_name!r}: "
            f"{response.text[:500]}"
        ) from e
