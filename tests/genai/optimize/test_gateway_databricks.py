from unittest import mock

import pytest

import mlflow.genai.optimize.gateway_databricks as gwdb
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize import update_databricks_endpoint_model


class _FakeResponse:
    def __init__(self, body=None, text="{}"):
        self.status_code = 200
        self.text = text
        self._body = body if body is not None else {}

    def json(self):
        if self._body is _INVALID_JSON:
            raise ValueError("no json")
        return self._body


_INVALID_JSON = object()


@pytest.fixture
def capture_request():
    captured = {}

    def fake_http_request(**kwargs):
        captured.update(kwargs)
        return _FakeResponse({"name": "sup"})

    with (
        mock.patch.object(gwdb, "http_request", fake_http_request),
        mock.patch.object(gwdb, "get_databricks_host_creds", lambda uri: f"creds<{uri}>"),
        mock.patch.object(gwdb, "augmented_raise_for_status", lambda r: None),
    ):
        yield captured


def test_update_databricks_endpoint_model_builds_ppt_patch(capture_request):
    update_databricks_endpoint_model("sup", "databricks-claude-sonnet-4")

    assert capture_request["endpoint"] == "/api/ai-gateway/v2/endpoints/sup"
    assert capture_request["method"] == "PATCH"
    assert capture_request["params"] == {
        "update_mask": "config.destinations,config.routing_strategy"
    }
    config = capture_request["json"]["config"]
    assert config["routing_strategy"] == "REQUEST_BASED_TRAFFIC_SPLIT"
    assert config["destinations"] == [
        {
            "name": "system.ai.databricks-claude-sonnet-4",
            "type": "PAY_PER_TOKEN_FOUNDATION_MODEL",
            "traffic_percentage": 100,
        }
    ]


def test_update_databricks_endpoint_model_does_not_double_prefix(capture_request):
    update_databricks_endpoint_model("sup", "system.ai.databricks-gpt-5")
    assert capture_request["json"]["config"]["destinations"][0]["name"] == (
        "system.ai.databricks-gpt-5"
    )


def test_update_databricks_endpoint_model_passes_databricks_uri(capture_request):
    update_databricks_endpoint_model("sup", "m", databricks_uri="databricks://profile")
    assert capture_request["host_creds"] == "creds<databricks://profile>"


def test_update_databricks_endpoint_model_non_ppt_uses_name_verbatim(capture_request):
    update_databricks_endpoint_model(
        "sup", "my-external-model", destination_type="EXTERNAL_FOUNDATION_MODEL"
    )
    dest = capture_request["json"]["config"]["destinations"][0]
    assert dest["name"] == "my-external-model"
    assert dest["type"] == "EXTERNAL_FOUNDATION_MODEL"


def test_update_databricks_endpoint_model_raises_on_non_json_response():
    def fake_http_request(**kwargs):
        return _FakeResponse(_INVALID_JSON, text="<html>gateway error</html>")

    with (
        mock.patch.object(gwdb, "http_request", fake_http_request),
        mock.patch.object(gwdb, "get_databricks_host_creds", lambda uri: "creds"),
        mock.patch.object(gwdb, "augmented_raise_for_status", lambda r: None),
    ):
        with pytest.raises(MlflowException, match="non-JSON response"):
            update_databricks_endpoint_model("sup", "m")
