"""OpenAI SDK client which authenticates with a ChatGPT plan.

Requires the ``openai`` extra: ``pip install chatgpt-plan-auth[openai]``. Nothing
else in this package imports the OpenAI SDK.

The ported overrides carry no ``typing.override`` decorator: that is Python
3.12+ and this package supports 3.10.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from .auth import ChatGPTAuth
from .const import PLACEHOLDER_API_KEY


class ChatGPTSubscriptionAuth(httpx.Auth):
    """Attach the SIWC access token and refresh it once on a 401."""

    def __init__(self, auth: ChatGPTAuth) -> None:
        """Initialize the auth flow."""
        self._auth = auth

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Attach the access token, refreshing it when it was rejected."""
        request.headers["Authorization"] = (
            f"Bearer {await self._auth.async_get_access_token()}"
        )
        response = yield request
        if response.status_code == 401:
            await self._auth.async_handle_authentication_error()
            request.headers["Authorization"] = (
                f"Bearer {self._auth.credentials.access_token}"
            )
            yield request


class ChatGPTSubscriptionClient(AsyncOpenAI):
    """OpenAI client authenticating with a ChatGPT plan subscription."""

    def __init__(
        self,
        auth: ChatGPTAuth,
        http_client: httpx.AsyncClient,
        **options: Any,
    ) -> None:
        """Initialize the client."""
        self.auth = auth
        # A ChatGPT plan has no API key: the bearer token is attached per request
        # by `custom_auth`, which also refreshes it. The SDK still requires a
        # non-empty `api_key` to build a client, and it does not support a
        # callable one (that raises inside the request).
        options.pop("api_key", None)
        # The plan usage route answers a degraded provider edge with a 5xx and a
        # `retry_after` (60 seconds when this was written), which the SDK honors
        # while backing off. The default of two retries therefore leaves an
        # interactive request waiting minutes before the caller sees the error,
        # so bound it to a single retry and let the caller report the status.
        options.setdefault("max_retries", 1)
        super().__init__(api_key=PLACEHOLDER_API_KEY, http_client=http_client, **options)

    def with_options(self, **options: Any) -> ChatGPTSubscriptionClient:
        """Return a copy of this client which keeps the ChatGPT plan sign in.

        The base implementation copies a client with `type(self)(**options)`,
        which would not carry the sign in of this instance.
        """
        options.setdefault("http_client", self._client)
        return ChatGPTSubscriptionClient(self.auth, **options)

    @property
    def custom_auth(self) -> httpx.Auth | None:
        """Return the token refreshing auth flow."""
        return ChatGPTSubscriptionAuth(self.auth)


__all__ = ["ChatGPTSubscriptionAuth", "ChatGPTSubscriptionClient"]
