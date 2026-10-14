"""Shared test helpers: a fake httpx transport and canned SIWC payloads."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl

import httpx

from chatgpt_plan_auth import ChatGPTCredentials, credentials_from_token
from chatgpt_plan_auth.const import CLIENT_ID_PREFIX, ISSUER, PLAN_SCOPE

CLIENT_ID = f"{CLIENT_ID_PREFIX}abc123"
HOST_ID = "urn:uuid:00000000-0000-4000-8000-000000000000"


class FakeHTTP:
    """An httpx.AsyncClient over a MockTransport with a canned reply queue.

    Replies are ``(status, json_body)`` tuples, an ``httpx.Response`` or an
    exception to raise. Every request is recorded for assertions.
    """

    def __init__(self, *replies: Any) -> None:
        """Initialize the fake transport."""
        self.requests: list[httpx.Request] = []
        self._replies = list(replies)
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Answer one request from the queue."""
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        status, body = reply
        if body is None:
            return httpx.Response(status, content=b"<not json>")
        return httpx.Response(status, json=body)

    @property
    def call_count(self) -> int:
        """Return how many requests were made."""
        return len(self.requests)

    def form(self, index: int = 0) -> dict[str, str]:
        """Return the form encoded body of one request."""
        return dict(parse_qsl(self.requests[index].content.decode()))

    def authorization(self, index: int = 0) -> str | None:
        """Return the Authorization header of one request."""
        return self.requests[index].headers.get("authorization")


def token_response(**kwargs: Any) -> dict[str, Any]:
    """Create a token endpoint response body."""
    token = {
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "id_token": "id-1",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": f"{PLAN_SCOPE} email offline_access openid profile resource.invoke",
    }
    token.update(kwargs)
    return token


def credentials(**overrides: Any) -> ChatGPTCredentials:
    """Create credentials, with optional token fields and attribute overrides."""
    token = token_response(**overrides.pop("token", {}))
    result = credentials_from_token(
        token,
        client_id=CLIENT_ID,
        ext_agent_host_id=HOST_ID,
        sub="sub-value",
    )
    for key, value in overrides.items():
        if key in ("expires_at", "earliest_refresh_at") and isinstance(value, str):
            value = datetime.fromisoformat(value)
        setattr(result, key, value)
    return result


def make_id_token(**claims: Any) -> str:
    """Create an ID token carrying ``claims`` with a bogus signature.

    The signature is never verified (see the module docstring of
    ``chatgpt_plan_auth.auth``), which is what makes this helper possible.
    """
    payload = {
        "iss": ISSUER,
        "sub": "chatgpt-user-1",
        "email": "user@example.com",
        "aud": CLIENT_ID,
        "nonce": "nonce-1",
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        **claims,
    }

    def segment(value: dict[str, Any]) -> str:
        raw = json.dumps(value).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{segment({'alg': 'RS256', 'kid': 'test'})}.{segment(payload)}.signature"


def expires_in(seconds: int) -> str:
    """Return an ISO 8601 timestamp ``seconds`` from now."""
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def parse_params(url: str) -> dict[str, str]:
    """Return the query parameters of a URL."""
    return dict(parse_qsl(httpx.URL(url).query.decode()))


def slugs(*values: Iterable[str]) -> dict[str, Any]:
    """Return a non-standard catalog body carrying slugs."""
    return {"models": [{"slug": value} for value in values]}
