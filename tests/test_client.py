"""Test the OpenAI SDK client which talks to the plan usage route."""

import httpx
import pytest
from conftest import FakeHTTP, credentials, token_response

from chatgpt_plan_auth import ChatGPTAuth

pytest.importorskip("openai")

from chatgpt_plan_auth.client import (  # noqa: E402
    ChatGPTSubscriptionAuth,
    ChatGPTSubscriptionClient,
)


async def test_subscription_client_custom_auth() -> None:
    """Test that the subscription client attaches and retries the token."""
    class RecordingAuth(ChatGPTAuth):
        """Count authentication errors so the retry can be observed."""

        errors = 0

        async def async_handle_authentication_error(self) -> None:
            """Record the call and refresh the token."""
            type(self).errors += 1
            await super().async_handle_authentication_error()

    fake = FakeHTTP(
        (200, token_response(access_token="access-2", refresh_token="refresh-2"))
    )
    auth = RecordingAuth(credentials(), client=fake.client)

    flow = ChatGPTSubscriptionAuth(auth).async_auth_flow(
        httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    sent = await anext(flow)
    assert sent.headers["Authorization"] == "Bearer access-1"

    # A 401 refreshes the token and replays the request once.
    retried = await flow.asend(httpx.Response(401, request=sent))
    assert retried.headers["Authorization"] == "Bearer access-2"
    assert RecordingAuth.errors == 1
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=sent))

    await fake.client.aclose()


async def test_subscription_client_request_carries_access_token() -> None:
    """Test that a request through the client authenticates with the plan token."""
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"object": "list", "data": []})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ChatGPTSubscriptionClient(
        ChatGPTAuth(credentials(), client=http_client), http_client=http_client
    )
    try:
        models = await client.with_options(timeout=10.0).models.list()
    finally:
        await http_client.aclose()

    assert models.object == "list"
    # A placeholder API key must never reach the wire.
    assert seen["authorization"] == "Bearer access-1"


async def test_subscription_client_bounds_retries() -> None:
    """Test that a degraded provider edge is not retried for minutes.

    The route answers a broken provider edge with a 5xx plus a `retry_after` of
    a minute, which the SDK honors while backing off; the client bounds that so
    an interactive request fails with the status instead of hanging.
    """
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    try:
        client = ChatGPTSubscriptionClient(
            ChatGPTAuth(credentials(), client=http_client), http_client=http_client
        )
        assert client.max_retries == 1

        override = ChatGPTSubscriptionClient(
            ChatGPTAuth(credentials(), client=http_client),
            http_client=http_client,
            max_retries=5,
        )
        assert override.max_retries == 5
    finally:
        await http_client.aclose()
