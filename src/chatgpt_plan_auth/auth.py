"""Sign in with ChatGPT (ChatGPT plan usage) authorization.

This implements the OAuth flow documented at
https://developers.openai.com/siwc/token-sharing-open-source/ which lets an
open-source, locally hosted app use an eligible ChatGPT plan for eligible
Responses API requests. The access token obtained here is used as the bearer
token against ``https://api.openai.com/v1`` instead of an API key.

HISTORY AND SCOPE
-----------------
This module was extracted from Home Assistant core, where the same code made
the reviewer's point for them: it is deep domain knowledge about an external
system and belongs in a library, not in core. Six modules are ported verbatim
apart from ``hass`` becoming a plain :class:`httpx.AsyncClient`:

``const`` / ``auth`` / ``catalog`` / ``requests`` (this package, ``httpx`` only)
and ``client`` (the OpenAI SDK wrapper, behind the ``openai`` extra).

DELIBERATE SIMPLIFICATION: NO ID TOKEN SIGNATURE VERIFICATION
-------------------------------------------------------------
``/.well-known/jwks.json`` is never fetched, RSA keys are not parsed and no
crypto dependency is carried. The ID token is received directly from the
provider's TLS-authenticated token endpoint in the same response as the code
exchange, which is the trust model Home Assistant's own OAuth2 flow uses, so
the signature adds nothing an attacker who already controls the token response
could not forge. Issuer, audience, expiry and the per-attempt nonce are still
verified in :func:`validate_id_token`, and the callback ``state`` is checked in
:func:`async_exchange_code`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import secrets
from typing import Any
import uuid
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx

from .const import (
    AGENT_NAME_HINT,
    AUTH_METHOD_CHATGPT,
    AUTHORIZE_URL,
    CONF_ACCESS_TOKEN,
    CONF_AUTH_METHOD,
    CONF_CLIENT_ID,
    CONF_EARLIEST_REFRESH_AT,
    CONF_EMAIL,
    CONF_EXPIRES_AT,
    CONF_EXT_AGENT_HOST_ID,
    CONF_ID_TOKEN,
    CONF_REFRESH_TOKEN,
    CONF_SCOPES,
    CONF_SUB,
    DYNAMIC_AGENT_CLIENT,
    EXT_AGENT_HOST_ID_SCHEME,
    FATAL_REFRESH_ERRORS,
    HTTP_TIMEOUT,
    ISSUER,
    REDIRECT_URI,
    RESOURCE,
    SCOPE,
    TOKEN_EXPIRY_MARGIN,
    TOKEN_URL,
)
from .errors import ChatGPTPlanAuthError, ChatGPTPlanAuthFatalError

_LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def _parse_optional_datetime(value: Any) -> datetime | None:
    """Parse an optional ISO 8601 timestamp."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


@dataclass
class ChatGPTCredentials:
    """Credentials of one Sign in with ChatGPT registration."""

    client_id: str
    ext_agent_host_id: str
    sub: str
    email: str | None
    access_token: str
    refresh_token: str
    id_token: str
    expires_at: datetime
    scopes: list[str]
    earliest_refresh_at: datetime | None = None

    @property
    def unique_id(self) -> str:
        """Return the account identity scoped to this registration.

        The subject identifies the ChatGPT account, the issued client id the
        registration (one account can have several workspaces registered).
        """
        return f"{self.sub}:{self.client_id}"

    def as_entry_data(self) -> dict[str, Any]:
        """Return the credentials as a plain mapping, ready to be persisted."""
        data: dict[str, Any] = {
            CONF_AUTH_METHOD: AUTH_METHOD_CHATGPT,
            CONF_CLIENT_ID: self.client_id,
            CONF_EXT_AGENT_HOST_ID: self.ext_agent_host_id,
            CONF_SUB: self.sub,
            CONF_ACCESS_TOKEN: self.access_token,
            CONF_REFRESH_TOKEN: self.refresh_token,
            CONF_ID_TOKEN: self.id_token,
            CONF_EXPIRES_AT: self.expires_at.isoformat(),
            CONF_SCOPES: self.scopes,
        }
        if self.email is not None:
            data[CONF_EMAIL] = self.email
        if self.earliest_refresh_at is not None:
            data[CONF_EARLIEST_REFRESH_AT] = self.earliest_refresh_at.isoformat()
        return data

    @classmethod
    def from_entry_data(cls, data: Mapping[str, Any]) -> ChatGPTCredentials:
        """Build credentials from persisted data."""
        return cls(
            client_id=data[CONF_CLIENT_ID],
            ext_agent_host_id=data.get(CONF_EXT_AGENT_HOST_ID, ""),
            sub=data.get(CONF_SUB, ""),
            email=data.get(CONF_EMAIL),
            access_token=data[CONF_ACCESS_TOKEN],
            refresh_token=data[CONF_REFRESH_TOKEN],
            id_token=data.get(CONF_ID_TOKEN, ""),
            expires_at=_parse_optional_datetime(data[CONF_EXPIRES_AT]) or _utcnow(),
            scopes=list(data.get(CONF_SCOPES, [])),
            earliest_refresh_at=_parse_optional_datetime(
                data.get(CONF_EARLIEST_REFRESH_AT, "")
            ),
        )


def create_ext_agent_host_id() -> str:
    """Return a new stable ``ext_agent_host_id`` for this installation.

    The host id identifies the host the app runs on, not the signed in account,
    and every registration of one instance shares it. A UUID based host id is
    documented as a supported alternative to a JWK thumbprint URI and needs no
    key management; callers generate it once and store it themselves.
    """
    return f"{EXT_AGENT_HOST_ID_SCHEME}{uuid.uuid4()}"


@dataclass
class AuthorizationRequest:
    """A pending SIWC authorization attempt."""

    url: str
    state: str
    nonce: str
    code_verifier: str
    redirect_uri: str
    ext_agent_host_id: str
    # The client id to exchange the code with: `dynamic_agent_client` until the
    # callback returns the issued client id.
    client_id: str
    registration: bool


def _create_pkce_pair() -> tuple[str, str]:
    """Create a PKCE verifier and its S256 challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def create_authorization_request(
    *,
    client_id: str | None,
    ext_agent_host_id: str,
    id_token_hint: str | None = None,
    agent_name_hint: str = AGENT_NAME_HINT,
) -> AuthorizationRequest:
    """Build the authorization URL for a new or a returning ChatGPT account.

    This is the first half of the flow; :func:`async_exchange_code` finishes it.
    A fresh state, nonce and PKCE verifier are generated for every attempt.
    """
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier, challenge = _create_pkce_pair()

    registration = client_id is None
    params = {
        "response_type": "code",
        "client_id": DYNAMIC_AGENT_CLIENT if registration else client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "ext_agent_host_id": ext_agent_host_id,
    }
    if registration:
        # agent_name_hint is display metadata and is only sent for the first
        # (dynamic client registration) authorization request.
        params["agent_name_hint"] = agent_name_hint
    if id_token_hint:
        params["id_token_hint"] = id_token_hint

    return AuthorizationRequest(
        url=f"{AUTHORIZE_URL}?{urlencode(params)}",
        state=state,
        nonce=nonce,
        code_verifier=verifier,
        redirect_uri=REDIRECT_URI,
        ext_agent_host_id=ext_agent_host_id,
        client_id=client_id or DYNAMIC_AGENT_CLIENT,
        registration=registration,
    )


def parse_callback_url(url: str) -> dict[str, str]:
    """Parse the query of the URL the user's browser ended up on.

    The loopback callback is normally on a different machine than the app, so
    the page itself cannot be loaded. The address bar still holds the OAuth
    parameters, which is all that is needed.
    """
    return dict(parse_qsl(urlparse(url.strip()).query))


def _error_code(body: Any) -> str | None:
    """Return the OAuth error code of a token endpoint response body."""
    match body:
        case {"error": {"code": str() as code}}:
            return code
        case {"error": str() as code}:
            return code
        case {"code": str() as code}:
            return code
    return None


def _error_message(body: Any) -> str:
    """Return the message of a token endpoint error response body."""
    match body:
        case {"error": {"message": str() as message}}:
            return message
        case {"error_description": str() as description}:
            return description
        case {"error": {"error_description": str() as description}}:
            return description
    return "unknown error"


async def async_token_request(
    client: httpx.AsyncClient,
    data: dict[str, str],
    *,
    fatal_errors: bool = False,
) -> dict[str, Any]:
    """POST a form encoded grant to the SIWC token endpoint.

    ``fatal_errors`` maps the errors which invalidate the stored refresh token
    to :class:`ChatGPTPlanAuthFatalError` so the caller can start reauth.
    """
    try:
        response = await client.post(TOKEN_URL, data=data, timeout=HTTP_TIMEOUT)
    except httpx.HTTPError as err:
        raise ChatGPTPlanAuthError(f"Error talking to OpenAI: {err}") from err

    if response.status_code == 200:
        try:
            return dict(response.json())
        except ValueError as err:
            raise ChatGPTPlanAuthError("Invalid token response from OpenAI") from err

    try:
        body = response.json()
    except ValueError:
        body = None
    code = _error_code(body)
    message = _error_message(body)

    if fatal_errors and code in FATAL_REFRESH_ERRORS:
        raise ChatGPTPlanAuthFatalError(
            f"Sign in with ChatGPT authentication failed ({code}): {message}"
        )

    raise ChatGPTPlanAuthError(f"Sign in with ChatGPT failed ({code}): {message}")


def _decode_jwt_claims(id_token: str) -> Mapping[str, Any]:
    """Return the claims of an ID token without verifying its signature.

    See the module docstring: the token arrives from the TLS-authenticated
    token endpoint, so the signature is not checked and no key material or
    crypto dependency is carried.
    """
    parts = id_token.split(".")
    if len(parts) != 3:
        raise ChatGPTPlanAuthError("Invalid ID token: not a JWT")
    padding = "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
    except (binascii.Error, ValueError, UnicodeDecodeError) as err:
        raise ChatGPTPlanAuthError(f"Invalid ID token: {err}") from err
    if not isinstance(claims, Mapping):
        raise ChatGPTPlanAuthError("Invalid ID token: claims are not an object")
    return claims


def validate_id_token(
    id_token: str, *, client_id: str, nonce: str
) -> Mapping[str, Any]:
    """Validate an ID token and return its claims.

    The issuer, the audience (the issued client id), the expiry and the nonce
    of this authorization attempt are all verified. The signature is not; see
    the module docstring.
    """
    claims = _decode_jwt_claims(id_token)

    if claims.get("iss") != ISSUER:
        raise ChatGPTPlanAuthError(
            f"Invalid ID token: unexpected issuer {claims.get('iss')!r}"
        )

    audience = claims.get("aud")
    audiences = [audience] if isinstance(audience, str) else list(audience or [])
    if client_id not in audiences:
        raise ChatGPTPlanAuthError("Invalid ID token: audience mismatch")

    expires_at = claims.get("exp")
    if not isinstance(expires_at, (int, float)) or expires_at <= _utcnow().timestamp():
        raise ChatGPTPlanAuthError("Invalid ID token: expired")

    if claims.get("nonce") != nonce:
        raise ChatGPTPlanAuthError("Invalid ID token: nonce mismatch")

    return claims


def _parse_optional_string(value: Any) -> str | None:
    """Return a string claim value, or None."""
    return value if isinstance(value, str) else None


async def async_exchange_code(
    client: httpx.AsyncClient,
    code: str,
    *,
    request: AuthorizationRequest,
    callback_client_id: str | None = None,
    callback_state: str | None = None,
) -> ChatGPTCredentials:
    """Exchange the authorization code of a callback for credentials.

    ``callback_state`` is compared against the state of ``request`` (the CSRF
    check) and the nonce of the ID token is verified by
    :func:`validate_id_token`. A dynamic client registration returns the issued
    client id in the callback as ``client_id``; pass it as
    ``callback_client_id`` so the code is exchanged with the registered client.
    """
    if callback_state is not None and callback_state != request.state:
        raise ChatGPTPlanAuthError("Sign in with ChatGPT failed: state mismatch")

    client_id = callback_client_id or request.client_id

    token = await async_token_request(
        client,
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": request.code_verifier,
            "redirect_uri": request.redirect_uri,
            "resource": RESOURCE,
        },
        fatal_errors=True,
    )

    sub: str | None = None
    email: str | None = None
    if id_token := token.get("id_token"):
        claims = validate_id_token(id_token, client_id=client_id, nonce=request.nonce)
        sub = _parse_optional_string(claims.get("sub"))
        email = _parse_optional_string(claims.get("email"))

    return credentials_from_token(
        token,
        client_id=client_id,
        ext_agent_host_id=request.ext_agent_host_id,
        sub=sub,
        email=email,
    )


def credentials_from_token(
    token: Mapping[str, Any],
    *,
    client_id: str,
    ext_agent_host_id: str,
    previous: ChatGPTCredentials | None = None,
    sub: str | None = None,
    email: str | None = None,
) -> ChatGPTCredentials:
    """Build credentials from a token endpoint response.

    Refresh tokens rotate on every refresh, so the response always replaces the
    stored credentials. Fields the response does not repeat are kept from
    ``previous``.
    """
    scopes = str(token.get("scope", "")).split()
    expires_in = token.get("expires_in")
    return ChatGPTCredentials(
        client_id=client_id,
        ext_agent_host_id=ext_agent_host_id,
        sub=sub if sub is not None else (previous.sub if previous else ""),
        email=email if email is not None else (previous.email if previous else None),
        access_token=token["access_token"],
        refresh_token=token.get("refresh_token")
        or (previous.refresh_token if previous else ""),
        id_token=token.get("id_token") or (previous.id_token if previous else ""),
        expires_at=_utcnow() + timedelta(seconds=int(expires_in) if expires_in else 3600),
        scopes=scopes or (previous.scopes if previous else []),
        earliest_refresh_at=_parse_optional_datetime(token.get("earliest_refresh_at")),
    )


class ChatGPTAuth:
    """Manage the SIWC token lifecycle of one set of credentials.

    The manager is storage agnostic: it refreshes through the given
    :class:`httpx.AsyncClient` and reports every rotation through ``on_update``
    so the caller can persist it. A dead refresh token raises
    :class:`ChatGPTPlanAuthFatalError`; starting the reauth flow is the
    caller's job.
    """

    def __init__(
        self,
        credentials: ChatGPTCredentials,
        *,
        client: httpx.AsyncClient,
        on_update: Callable[[ChatGPTCredentials], None] | None = None,
    ) -> None:
        """Initialize the token manager."""
        self.credentials = credentials
        self.client = client
        self.on_update = on_update
        self._lock = asyncio.Lock()

    @classmethod
    def from_entry_data(
        cls,
        data: Mapping[str, Any],
        *,
        client: httpx.AsyncClient,
        on_update: Callable[[ChatGPTCredentials], None] | None = None,
    ) -> ChatGPTAuth:
        """Build the token manager from persisted credentials."""
        return cls(
            ChatGPTCredentials.from_entry_data(data), client=client, on_update=on_update
        )

    async def async_get_access_token(self) -> str:
        """Return a valid access token, refreshing it when it is about to expire.

        This is used as the async `api_key` provider of the OpenAI client, so
        the token is checked (and refreshed) before every request.
        """
        async with self._lock:
            if self._needs_refresh():
                await self._async_refresh()
        return self.credentials.access_token

    def _needs_refresh(self) -> bool:
        """Return if the access token should be refreshed."""
        now = _utcnow()
        if (
            self.credentials.earliest_refresh_at is not None
            and now < self.credentials.earliest_refresh_at
        ):
            # The token endpoint rejects refresh attempts before this time.
            return False
        return now >= self.credentials.expires_at - TOKEN_EXPIRY_MARGIN

    async def async_handle_authentication_error(self) -> None:
        """Refresh the token after the API rejected the access token."""
        async with self._lock:
            if (
                self.credentials.earliest_refresh_at is not None
                and _utcnow() < self.credentials.earliest_refresh_at
            ):
                # The token endpoint refuses refreshes before this time, so the
                # rejected request was not a token expiry problem.
                return
            await self._async_refresh()

    async def _async_refresh(self) -> None:
        """Exchange the refresh token for a new set of tokens."""
        token = await async_token_request(
            self.client,
            {
                "grant_type": "refresh_token",
                "client_id": self.credentials.client_id,
                "refresh_token": self.credentials.refresh_token,
                "resource": RESOURCE,
            },
            fatal_errors=True,
        )

        self._store(
            credentials_from_token(
                token,
                client_id=self.credentials.client_id,
                ext_agent_host_id=self.credentials.ext_agent_host_id,
                previous=self.credentials,
            )
        )

    def _store(self, credentials: ChatGPTCredentials) -> None:
        """Replace the (rotated) credentials and hand them to the caller."""
        self.credentials = credentials
        if self.on_update is not None:
            self.on_update(credentials)


__all__ = [
    "AuthorizationRequest",
    "ChatGPTCredentials",
    "ChatGPTAuth",
    "async_exchange_code",
    "async_token_request",
    "create_authorization_request",
    "create_ext_agent_host_id",
    "credentials_from_token",
    "parse_callback_url",
    "validate_id_token",
]
