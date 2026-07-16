import re

import pytest
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import (
    AnthropicConfig,
    EndpointConfig,
    LiteLLMConfig,
    OpenAIConfig,
    _load_gateway_config,
    _resolve_api_key_from_input,
    _save_route_config,
    update_endpoint_model,
)
from mlflow.gateway.utils import assemble_uri_path


@pytest.fixture
def basic_config_dict():
    return {
        "endpoints": [
            {
                "name": "completions-gpt4",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-10",
                        "openai_api_type": "openai",
                        "openai_organization": "my_company",
                    },
                },
            },
            {
                "name": "chat-gpt4",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "sk-openai"},
                },
            },
            {
                "name": "claude-chat",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "claude-v1",
                    "provider": "anthropic",
                    "config": {
                        "anthropic_api_key": "api_key",
                    },
                },
            },
        ]
    }


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["gateway", "/routes/", "/chat"], "/gateway/routes/chat"),
        (["/gateway/", "/routes", "chat"], "/gateway/routes/chat"),
        (["gateway/routes/", "chat"], "/gateway/routes/chat"),
        (["gateway/", "routes/chat"], "/gateway/routes/chat"),
        (["/gateway/routes", "/chat/"], "/gateway/routes/chat"),
        (["/gateway", "/routes/", "chat/"], "/gateway/routes/chat"),
        (["/"], "/"),
        (["gateway", "", "/routes/", "", "/chat", ""], "/gateway/routes/chat"),
    ],
)
def test_assemble_uri_path(paths, expected):
    assert assemble_uri_path(paths) == expected


def test_api_key_parsing_env(tmp_path, monkeypatch):
    # Env-var resolution requires the flag to be enabled
    monkeypatch.setenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_ENV", "true")
    monkeypatch.setenv("KEY_AS_ENV", "my_key")

    assert _resolve_api_key_from_input("$KEY_AS_ENV") == "my_key"
    monkeypatch.delenv("KEY_AS_ENV", raising=False)
    with pytest.raises(MlflowException, match="Environment variable 'KEY_AS_ENV' is not set"):
        _resolve_api_key_from_input("$KEY_AS_ENV")

    string_key = "my_key_as_a_string"

    assert _resolve_api_key_from_input(string_key) == string_key

    # File-based resolution requires the flag to be enabled
    monkeypatch.setenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_FILE", "true")

    conf_path = tmp_path.joinpath("mykey.conf")
    file_key = "Here is my key that sits safely in a file"

    conf_path.write_text(file_key)

    assert _resolve_api_key_from_input(str(conf_path)) == file_key


def test_api_key_env_resolution_blocked_without_flag(monkeypatch):
    monkeypatch.setenv("KEY_AS_ENV", "my_key")
    monkeypatch.delenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_ENV", raising=False)

    # Without the flag, $-prefixed values are returned as literal strings
    assert _resolve_api_key_from_input("$KEY_AS_ENV") == "$KEY_AS_ENV"


def test_api_key_input_exceeding_maximum_filename_length():
    assert _resolve_api_key_from_input("a" * 256) == "a" * 256


def test_api_key_parsing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_FILE", "true")
    key_path = tmp_path.joinpath("api.key")
    config = {
        "endpoints": [
            {
                "name": "claude-chat",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "claude-v1",
                    "provider": "anthropic",
                    "config": {
                        "anthropic_api_key": str(key_path),
                    },
                },
            },
        ]
    }

    key_path.write_text("abc")
    config_path = tmp_path.joinpath("config.yaml")
    config_path.write_text(yaml.safe_dump(config))
    loaded_config = _load_gateway_config(config_path)

    assert isinstance(loaded_config.endpoints[0].model.config, AnthropicConfig)
    assert loaded_config.endpoints[0].model.config.anthropic_api_key == "abc"


def test_route_configuration_parsing(basic_config_dict, tmp_path, monkeypatch):
    conf_path = tmp_path.joinpath("config.yaml")

    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    loaded_config = _load_gateway_config(conf_path)

    save_path = tmp_path.joinpath("config2.yaml")
    _save_route_config(loaded_config, save_path)
    loaded_from_save = _load_gateway_config(save_path)

    completions_gpt4 = loaded_from_save.endpoints[0]
    assert completions_gpt4.name == "completions-gpt4"
    assert completions_gpt4.endpoint_type == "llm/v1/completions"
    assert completions_gpt4.model.name == "gpt-4"
    assert completions_gpt4.model.provider == "openai"
    completions_conf = completions_gpt4.model.config
    assert isinstance(completions_conf, OpenAIConfig)
    assert completions_conf.openai_api_key == "mykey"
    assert completions_conf.openai_api_base == "https://api.openai.com/v1"
    assert completions_conf.openai_api_version == "2023-05-10"
    assert completions_conf.openai_api_type == "openai"
    assert completions_conf.openai_organization == "my_company"

    chat_gpt4 = loaded_from_save.endpoints[1]
    assert chat_gpt4.name == "chat-gpt4"
    assert chat_gpt4.endpoint_type == "llm/v1/chat"
    assert chat_gpt4.model.name == "gpt-4"
    assert chat_gpt4.model.provider == "openai"
    chat_conf = chat_gpt4.model.config
    assert isinstance(chat_conf, OpenAIConfig)
    assert chat_conf.openai_api_key == "sk-openai"
    assert chat_conf.openai_api_base == "https://api.openai.com/v1"
    assert chat_conf.openai_api_type == "openai"
    assert chat_conf.openai_api_version is None
    assert chat_conf.openai_organization is None

    claude = loaded_from_save.endpoints[2]
    assert isinstance(claude.model.config, AnthropicConfig)
    assert claude.name == "claude-chat"
    assert claude.endpoint_type == "llm/v1/chat"
    assert claude.model.name == "claude-v1"
    assert claude.model.provider == "anthropic"
    claude_conf = claude.model.config
    assert claude_conf.anthropic_api_key == "api_key"


def test_convert_route_config_to_routes_payload(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))
    loaded = _load_gateway_config(conf_path)

    assert all(isinstance(route, EndpointConfig) for route in loaded.endpoints)

    routes = [r.to_endpoint() for r in loaded.endpoints]

    for config in loaded.endpoints:
        route = next(x for x in routes if x.name == config.name)
        assert route.endpoint_type == config.endpoint_type
        assert route.model.name == config.model.name
        assert route.model.provider == config.model.provider
        # Pydantic doesn't allow undefined elements to be a part of its serialized object.
        # This test is a guard for devs only in case we inadvertently add sensitive keys to the
        # Route definition that would be returned via the GetRoute or SearchRoutes APIs
        assert not hasattr(route.model, "config")


def test_invalid_route_definition(tmp_path):
    invalid_conf = {
        "endpoints": [
            {
                "name": "invalid_route",
                "endpoint_type": "invalid/route",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-10",
                        "openai_api_type": "openai/v1/chat/completions",
                        "openai_organization": "my_company",
                    },
                },
            }
        ]
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_conf))

    with pytest.raises(MlflowException, match=r"The route_type 'invalid/route' is not supported."):
        _load_gateway_config(conf_path)


def test_invalid_provider(tmp_path):
    invalid_conf = {
        "endpoints": [
            {
                "name": "invalid_route",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "my_provider",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-10",
                        "openai_api_type": "openai/v1/chat/completions",
                        "openai_organization": "my_company",
                    },
                },
            }
        ]
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_conf))

    with pytest.raises(MlflowException, match=r"The provider 'my_provider' is not supported."):
        _load_gateway_config(conf_path)


def test_invalid_model_definition(tmp_path):
    invalid_partial_config = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "invalid",
                    "provider": "openai",
                    "config": {"openai_api_type": "openai"},
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_partial_config))

    with pytest.raises(
        MlflowException, match=re.compile(r"validation error.+openai_api_key", re.DOTALL)
    ):
        _load_gateway_config(conf_path)

    invalid_format_config_key_is_not_string = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "invalid",
                    "provider": "openai",
                    "config": {"openai_api_type": "openai", "openai_api_key": [42]},
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_format_config_key_is_not_string))

    with pytest.raises(
        MlflowException,
        match="The api key provided is not a string",
    ):
        _load_gateway_config(conf_path)

    invalid_format_config_key_invalid_path = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/embeddings",
                "model": {
                    "name": "invalid",
                    "provider": "openai",
                    "config": {"openai_api_type": "openai", "openai_api_key": "/not/a/real/path"},
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_format_config_key_invalid_path))

    assert (
        _load_gateway_config(conf_path).endpoints[0].model.config.openai_api_key
        == "/not/a/real/path"  # pylint: disable=line-too-long
    )

    invalid_no_config = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/embeddings",
                "model": {
                    "name": "invalid",
                    "provider": "anthropic",
                },
            }
        ]
    }
    conf_path = tmp_path.joinpath("config2.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_no_config))

    with pytest.raises(
        MlflowException,
        match="A config must be supplied when setting a provider. The provider entry",
    ):
        _load_gateway_config(conf_path)


@pytest.mark.parametrize(
    "route_name", ["Space Name", "bang!name", "query?name", "redirect#name", "bracket[]name"]
)
def test_invalid_route_name(tmp_path, route_name):
    bad_name = {
        "endpoints": [
            {
                "name": route_name,
                "endpoint_type": "bad/naming",
                "model": {
                    "name": "claude-v1",
                    "provider": "anthropic",
                    "config": {
                        "anthropic_api_key": "claudekey",
                    },
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(bad_name))

    with pytest.raises(
        MlflowException, match="The route name provided contains disallowed characters"
    ):
        _load_gateway_config(conf_path)


def test_default_base_api(tmp_path):
    route_no_base = {
        "endpoints": [
            {
                "name": "chat-gpt4",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "sk-openai"},
                },
            },
        ]
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(route_no_base))
    loaded_conf = _load_gateway_config(conf_path)

    assert loaded_conf.endpoints[0].model.config.openai_api_base == "https://api.openai.com/v1"


def test_duplicate_routes_in_config(tmp_path):
    route = {
        "endpoints": [
            {
                "name": "classifier",
                "endpoint_type": "llm/v1/classifier",
                "model": {
                    "name": "serving-endpoints/document-classifier/Production/invocations",
                    "provider": "databricks-model-serving",
                    "config": {
                        "databricks_api_token": "MY_TOKEN",
                        "databricks_api_base": "https://my-shard-001/",
                    },
                },
            },
            {
                "name": "classifier",
                "endpoint_type": "llm/v1/classifier",
                "model": {
                    "name": "serving-endpoints/document-classifier/Production/invocations",
                    "provider": "databricks_serving_endpoint",
                    "config": {
                        "databricks_api_token": "MY_TOKEN",
                        "databricks_api_base": "https://my-shard-001/",
                    },
                },
            },
        ]
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(route))
    with pytest.raises(
        MlflowException, match="Duplicate names found in endpoint / route configurations"
    ):
        _load_gateway_config(conf_path)


def test_litellm_config_removes_auth_mode():
    config = LiteLLMConfig(
        litellm_provider="bedrock",
        litellm_auth_config={
            "auth_mode": "access_keys",
            "aws_region_name": "us-west-2",
            "api_key": "test-key",
        },
    )
    assert "auth_mode" not in config.litellm_auth_config
    assert config.litellm_auth_config["aws_region_name"] == "us-west-2"
    assert config.litellm_auth_config["api_key"] == "test-key"


def test_update_endpoint_model_same_provider(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    update_endpoint_model(conf_path, "completions-gpt4", "gpt-4o")

    reloaded = _load_gateway_config(conf_path)
    swapped = next(e for e in reloaded.endpoints if e.name == "completions-gpt4")
    # Model name is swapped, provider config (incl. API key) is preserved.
    assert swapped.model.name == "gpt-4o"
    assert swapped.model.provider == "openai"
    assert swapped.model.config.openai_api_key == "mykey"
    assert swapped.model.config.openai_organization == "my_company"
    # Other endpoints are untouched.
    others = {e.name: e.model.name for e in reloaded.endpoints if e.name != "completions-gpt4"}
    assert others == {"chat-gpt4": "gpt-4", "claude-chat": "claude-v1"}


def test_update_endpoint_model_leaves_no_temp_file(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    update_endpoint_model(conf_path, "chat-gpt4", "gpt-4o")

    # The atomic write must not leave the temp file behind.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_update_endpoint_model_unknown_endpoint_raises(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    with pytest.raises(MlflowException, match="not found"):
        update_endpoint_model(conf_path, "does-not-exist", "gpt-4o")


def test_update_endpoint_model_missing_config_raises(tmp_path):
    with pytest.raises(MlflowException, match="does not exist"):
        update_endpoint_model(tmp_path.joinpath("nope.yaml"), "any", "gpt-4o")


def test_update_endpoint_model_provider_change_requires_config(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    with pytest.raises(MlflowException, match="requires new provider credentials"):
        update_endpoint_model(conf_path, "chat-gpt4", "claude-v1", provider="anthropic")


def test_update_endpoint_model_provider_change_with_config(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    update_endpoint_model(
        conf_path,
        "chat-gpt4",
        "claude-3-5-sonnet-20241022",
        provider="anthropic",
        config={"anthropic_api_key": "ak-new"},
    )

    reloaded = _load_gateway_config(conf_path)
    swapped = next(e for e in reloaded.endpoints if e.name == "chat-gpt4")
    assert swapped.model.provider == "anthropic"
    assert swapped.model.name == "claude-3-5-sonnet-20241022"
    assert isinstance(swapped.model.config, AnthropicConfig)
    assert swapped.model.config.anthropic_api_key == "ak-new"


def test_update_endpoint_model_preserves_api_key_reference(tmp_path, monkeypatch):
    # In a running gateway, key-resolution flags are enabled, so loading through
    # the config parser resolves `$ENV_VAR` references into literal secrets. The
    # swap must NOT write the resolved secret back to disk.
    monkeypatch.setenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_ENV", "true")
    monkeypatch.setenv("MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_FILE", "true")
    monkeypatch.setenv("MY_OPENAI_KEY", "sk-the-real-secret")
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(
        yaml.safe_dump({
            "endpoints": [
                {
                    "name": "chat",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4o-mini",
                        "provider": "openai",
                        "config": {"openai_api_key": "$MY_OPENAI_KEY"},
                    },
                }
            ]
        })
    )

    update_endpoint_model(conf_path, "chat", "gpt-4o")

    on_disk = yaml.safe_load(conf_path.read_text())
    key = on_disk["endpoints"][0]["model"]["config"]["openai_api_key"]
    assert key == "$MY_OPENAI_KEY", "swap leaked the resolved secret into the config file"
    assert on_disk["endpoints"][0]["model"]["name"] == "gpt-4o"


def test_update_endpoint_model_writes_the_given_path(basic_config_dict, tmp_path):
    # The swap must write the exact path it was given -- that is the path the
    # gateway's config watcher monitors for reloads. (For a symlinked config, the
    # atomic replace lands on that path so the watcher still fires.)
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    update_endpoint_model(conf_path, "chat-gpt4", "gpt-4o")

    on_disk = yaml.safe_load(conf_path.read_text())
    swapped = next(e for e in on_disk["endpoints"] if e["name"] == "chat-gpt4")
    assert swapped["model"]["name"] == "gpt-4o"


def test_update_endpoint_model_preserves_traffic_routes(tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(
        yaml.safe_dump({
            "endpoints": [
                {
                    "name": "chat-a",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4o-mini",
                        "provider": "openai",
                        "config": {"openai_api_key": "k"},
                    },
                },
                {
                    "name": "chat-b",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4o",
                        "provider": "openai",
                        "config": {"openai_api_key": "k"},
                    },
                },
            ],
            "routes": [
                {
                    "name": "split",
                    "task_type": "llm/v1/chat",
                    "destinations": [
                        {"name": "chat-a", "traffic_percentage": 50},
                        {"name": "chat-b", "traffic_percentage": 50},
                    ],
                }
            ],
        })
    )

    update_endpoint_model(conf_path, "chat-a", "gpt-4o")

    reloaded = _load_gateway_config(conf_path)
    assert reloaded.routes is not None
    assert reloaded.routes[0].name == "split"
    assert [d.name for d in reloaded.routes[0].destinations] == ["chat-a", "chat-b"]
