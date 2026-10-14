"""Test the httpx flavour handling of the auth flow and error classes.

Home Assistant's development branch hands the SDK an ``httpx2`` client while
older releases use ``httpx``; an auth flow or an error tuple of the wrong
flavour breaks at runtime, which no mocked test would catch. These tests pin the
selection logic itself.
"""

from types import SimpleNamespace

import httpx
import pytest

from chatgpt_plan_auth._compat import (
    HTTP_ERRORS,
    create_auth_flow,
    create_auth_flow_class,
)
from chatgpt_plan_auth.client import ChatGPTSubscriptionAuth

try:
    import httpx2
except ImportError:  # pragma: no cover - httpx2 is optional
    httpx2 = None


class _FakeAuth:
    """Minimal credential store used by the flow."""

    def __init__(self) -> None:
        """Start with a token which the provider rejects once."""
        self.credentials = SimpleNamespace(access_token="access-2")
        self.refreshes = 0

    async def async_get_access_token(self) -> str:
        """Return the token of the current credentials."""
        return "access-1"

    async def async_handle_authentication_error(self) -> None:
        """Record that the flow refreshed after a 401."""
        self.refreshes += 1


def test_httpx_flow_matches_httpx() -> None:
    """Test that the httpx flow derives from the httpx auth base."""
    assert issubclass(create_auth_flow_class(httpx.Auth), httpx.Auth)
    assert issubclass(ChatGPTSubscriptionAuth, httpx.Auth)


def test_error_classes_cover_httpx() -> None:
    """Test that httpx errors are part of the caught families."""
    assert httpx.HTTPError in HTTP_ERRORS


async def test_httpx_flow_attaches_and_refreshes_once() -> None:
    """Test the flow behaviour on a plain httpx client."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    auth = _FakeAuth()
    flow = create_auth_flow(client, auth).async_auth_flow(
        httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    try:
        sent = await anext(flow)
        assert sent.headers["Authorization"] == "Bearer access-1"

        retried = await flow.asend(httpx.Response(401, request=sent))
        assert retried.headers["Authorization"] == "Bearer access-2"
        assert auth.refreshes == 1
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx.Response(200, request=sent))
    finally:
        await client.aclose()


@pytest.mark.skipif(httpx2 is None, reason="httpx2 is not installed")
def test_httpx2_client_gets_an_httpx2_flow() -> None:
    """Test that an httpx2 client receives a flow of its own flavour."""
    client = httpx2.AsyncClient()
    flow = create_auth_flow(client, _FakeAuth())
    assert isinstance(flow, httpx2.Auth)
    assert not isinstance(flow, httpx.Auth)
    assert httpx2.HTTPError in HTTP_ERRORS


@pytest.mark.skipif(httpx2 is None, reason="httpx2 is not installed")
async def test_httpx2_flow_attaches_token() -> None:
    """Test that the httpx2 flow attaches the access token."""
    client = httpx2.AsyncClient()
    auth = _FakeAuth()
    try:
        flow = create_auth_flow(client, auth).async_auth_flow(
            httpx2.Request("POST", "https://api.openai.com/v1/responses")
        )
        sent = await anext(flow)
        assert sent.headers["Authorization"] == "Bearer access-1"
    finally:
        await client.aclose()
