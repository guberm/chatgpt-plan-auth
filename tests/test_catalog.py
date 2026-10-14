"""Test the plan catalog, request shaping and subscription error messages.

Port of the matching cases from Home Assistant core's
``test_chatgpt_auth.py`` (module ``requests`` cases included), with the
``hass`` http client replaced by a fake httpx transport.
"""

from typing import Any

import httpx
import pytest
from conftest import CLIENT_ID, HOST_ID, FakeHTTP, credentials, token_response

from chatgpt_plan_auth import (
    MODELS_URL,
    SUBSCRIPTION_MODEL_SUGGESTIONS,
    ChatGPTAuth,
    ChatGPTPlanAuthError,
    ChatGPTPlanAuthFatalError,
    async_get_subscription_models,
    async_get_subscription_models_for_credentials,
    credentials_from_token,
    shape_request_for_subscription,
    subscription_error_message,
    subscription_model_options,
)
from chatgpt_plan_auth.const import SUBSCRIPTION_INFERENCE_ERROR_HINTS


async def test_get_subscription_models() -> None:
    """Test that the model catalog slugs are returned in order."""
    http = FakeHTTP(
        (
            200,
            {
                "models": [
                    {"slug": "gpt-6.1-sol", "title": "Sol"},
                    {"slug": "gpt-6-astra"},
                    {"no_slug": "ignored"},
                ]
            },
        )
    )
    auth = ChatGPTAuth(credentials(), client=http.client)

    assert await async_get_subscription_models(http.client, auth) == [
        "gpt-6.1-sol",
        "gpt-6-astra",
    ]
    assert str(http.requests[0].url) == MODELS_URL
    assert http.authorization() == "Bearer access-1"


async def test_get_subscription_models_for_credentials() -> None:
    """Test the catalog of an account which is not stored yet."""
    http = FakeHTTP((200, {"models": [{"slug": "gpt-6.1-sol"}]}))
    creds = credentials_from_token(
        token_response(), client_id=CLIENT_ID, ext_agent_host_id=HOST_ID
    )

    assert await async_get_subscription_models_for_credentials(http.client, creds) == [
        "gpt-6.1-sol"
    ]
    assert http.authorization() == "Bearer access-1"


async def test_get_subscription_models_malformed_body() -> None:
    """Test that a catalog body without a models list is rejected."""
    http = FakeHTTP((200, {"data": []}))
    auth = ChatGPTAuth(credentials(), client=http.client)

    with pytest.raises(ChatGPTPlanAuthError, match="did not contain a models list"):
        await async_get_subscription_models(http.client, auth)


async def test_get_subscription_models_invalid_json() -> None:
    """Test an unparsable catalog body is reported."""
    http = FakeHTTP((200, None))
    auth = ChatGPTAuth(credentials(), client=http.client)

    with pytest.raises(ChatGPTPlanAuthError, match="Invalid model catalog response"):
        await async_get_subscription_models(http.client, auth)


async def test_get_subscription_models_http_error() -> None:
    """Test that a failing catalog request is reported."""
    http = FakeHTTP((500, {"error": "boom"}))
    auth = ChatGPTAuth(credentials(), client=http.client)

    with pytest.raises(ChatGPTPlanAuthError, match="Error listing OpenAI models"):
        await async_get_subscription_models(http.client, auth)


@pytest.mark.parametrize("status", [401, 403])
async def test_get_subscription_models_auth_failed(status: int) -> None:
    """Test that a rejected token fails the auth."""
    http = FakeHTTP((status, {"error": "nope"}))
    auth = ChatGPTAuth(credentials(), client=http.client)

    with pytest.raises(ChatGPTPlanAuthFatalError):
        await async_get_subscription_models(http.client, auth)


async def test_get_subscription_models_connection_error() -> None:
    """Test a connection error while listing the account models."""
    http = FakeHTTP(httpx.ConnectError("boom"))
    auth = ChatGPTAuth(credentials(), client=http.client)

    with pytest.raises(ChatGPTPlanAuthError, match="Error talking to OpenAI"):
        await async_get_subscription_models(http.client, auth)


def test_subscription_model_options() -> None:
    """Test the catalog falls back to the known suggestions."""
    assert subscription_model_options(["gpt-6.1-sol"]) == ["gpt-6.1-sol"]
    assert subscription_model_options([]) == SUBSCRIPTION_MODEL_SUGGESTIONS


def test_shape_request_for_subscription() -> None:
    """Test the request shaping for the subscription route."""
    model_args: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "input": [],
        "max_output_tokens": 3000,
        "safety_identifier": "conv-1",
        "prompt_cache_key": "subentry-1",
        "prompt_cache_retention": "24h",
        "temperature": 1.0,
        "top_p": 1.0,
        "service_tier": "auto",
        "store": True,
        "stream": False,
        "tools": [
            {"type": "function", "name": "turn_on"},
            {"type": "web_search"},
            {"type": "image_generation"},
        ],
        "tool_choice": {"type": "image_generation"},
    }
    messages: list[Any] = [
        {"type": "message", "role": "developer", "content": "You are this app"},
        {"type": "message", "role": "user", "content": "Turn on the light"},
        {"type": "message", "role": "system", "content": "Stay brief"},
        {"type": "function_call", "name": "turn_on", "arguments": "{}", "call_id": "1"},
    ]

    shape_request_for_subscription(model_args, messages)

    for forbidden in (
        "max_output_tokens",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "temperature",
        "top_p",
        "service_tier",
        "tool_choice",
    ):
        assert forbidden not in model_args

    # The subscription route requires unsaved, streamed responses.
    assert model_args["store"] is False
    assert model_args["stream"] is True
    # System content is sent as instructions instead of an input item.
    assert model_args["instructions"] == "You are this app"
    # A replayed function call stays flat: the route has no namespaces.
    assert messages == [
        {"type": "message", "role": "user", "content": "Turn on the light"},
        {"type": "message", "role": "developer", "content": "Stay brief"},
        {
            "type": "function_call",
            "name": "turn_on",
            "arguments": "{}",
            "call_id": "1",
        },
    ]
    # Function tools keep their flat shape; the hosted image generation tool is
    # dropped and web search is kept.
    assert model_args["tools"] == [
        {"type": "function", "name": "turn_on"},
        {"type": "web_search"},
    ]
    # Parameters which are supported by both routes are kept.
    assert model_args["model"] == "gpt-4o-mini"


def test_shape_request_for_subscription_removes_all_tools() -> None:
    """Test the tools argument is dropped when no tool survives."""
    model_args: dict[str, Any] = {
        "tools": [{"type": "image_generation"}, {"type": "code_interpreter"}]
    }

    shape_request_for_subscription(model_args, [])

    assert "tools" not in model_args


def test_shape_request_for_subscription_appends_instructions() -> None:
    """Test system prompts are appended to existing instructions."""
    model_args: dict[str, Any] = {"instructions": "Be brief"}
    messages: list[Any] = [
        {"type": "message", "role": "system", "content": "Use tools"}
    ]

    shape_request_for_subscription(model_args, messages)

    assert model_args["instructions"] == "Be brief\n\nUse tools"
    assert messages == []


def test_subscription_error_message() -> None:
    """Test the messages of the subscription route inference errors."""
    openai = pytest.importorskip("openai")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    known = openai.RateLimitError(
        message="limit",
        response=httpx.Response(429, request=request),
        body={"code": "subscription_sharing_usage_limit_exceeded"},
    )
    assert subscription_error_message(known) == SUBSCRIPTION_INFERENCE_ERROR_HINTS[
        "subscription_sharing_usage_limit_exceeded"
    ]

    # Any other rejected token means the sign in is no longer valid.
    rejected = openai.AuthenticationError(
        message="nope",
        response=httpx.Response(401, request=request),
        body={"code": "something_else"},
    )
    assert subscription_error_message(rejected) is not None
    assert "sign-in is no longer valid" in subscription_error_message(rejected)

    # An unrelated error is not rewritten.
    assert subscription_error_message(openai.OpenAIError("plain")) is None
