"""Support both httpx flavours a caller may hand in.

Home Assistant's development branch uses ``httpx2``: its
``helpers.httpx_client.get_async_client`` returns an ``httpx2.AsyncClient`` and
the OpenAI SDK it ships validates the auth flow against ``httpx2``. Older
releases (and a plain ``httpx`` caller) use ``httpx``. The two packages are
separate types, so an auth flow which subclasses the wrong one is rejected with
``TypeError: Invalid "auth" argument`` and error classes of the other flavour
are not caught. Everything here picks the flavour from the client in use so that
neither caller has to care.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx

try:  # pragma: no cover - depends on the caller's environment
    import httpx2
except ImportError:  # pragma: no cover
    httpx2 = None  # type: ignore[assignment]

# Error classes of the flavour in use: a caller may hand in either client, so
# both families have to be caught.
HTTP_ERRORS: tuple[type[BaseException], ...] = (
    ((httpx2.HTTPError,) if httpx2 is not None else ()) + (httpx.HTTPError,)
)

# One flow class per auth base, built on first use.
_FLOW_CLASSES: dict[type, type] = {}


def _auth_base(client: Any) -> type:
    """Return the ``Auth`` base class matching the client in use."""
    if httpx2 is not None and isinstance(client, httpx2.AsyncClient):
        return httpx2.Auth
    return httpx.Auth


def create_auth_flow(client: Any, auth: Any) -> Any:
    """Return an auth flow of the same flavour as ``client``.

    The flow attaches the access token of ``auth`` to every request and, when
    the provider rejects it with a 401, refreshes it once and replays the
    request.
    """
    base = _auth_base(client)
    flow_class = _FLOW_CLASSES.get(base)
    if flow_class is None:
        flow_class = create_auth_flow_class(base)
        _FLOW_CLASSES[base] = flow_class
    return flow_class(auth)


def create_auth_flow_class(base: type) -> type:
    """Build an auth flow class on top of one httpx ``Auth`` implementation."""

    class _SubscriptionAuth(base):  # type: ignore[misc, valid-type]
        """Attach the plan access token, refreshing it once on a 401."""

        def __init__(self, auth: Any) -> None:
            """Initialize the flow for one credential store."""
            self._auth = auth

        async def async_auth_flow(
            self, request: Any
        ) -> AsyncGenerator[Any, Any]:
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

    _SubscriptionAuth.__name__ = "ChatGPTSubscriptionAuth"
    _SubscriptionAuth.__qualname__ = "ChatGPTSubscriptionAuth"
    return _SubscriptionAuth


__all__ = ["HTTP_ERRORS", "create_auth_flow", "create_auth_flow_class"]
