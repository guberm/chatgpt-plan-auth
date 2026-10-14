"""Test the Sign in with ChatGPT authorization and token lifecycle.

Port of Home Assistant core's
``tests/components/openai_conversation/test_chatgpt_auth.py`` with every
JWKS/RSA/signing case dropped (the library deliberately does not verify the ID
token signature) and ``hass`` replaced by a fake httpx transport.
"""

import asyncio
import time
from typing import Any

import httpx
import pytest
from conftest import (
    CLIENT_ID,
    HOST_ID,
    FakeHTTP,
    credentials,
    expires_in,
    make_id_token,
    parse_params,
    token_response,
)

from chatgpt_plan_auth import (
    AUTHORIZE_URL,
    DYNAMIC_AGENT_CLIENT,
    PLAN_SCOPE,
    REDIRECT_URI,
    RESOURCE,
    SCOPE,
    TOKEN_URL,
    ChatGPTCredentials,
    ChatGPTAuth,
    ChatGPTPlanAuthError,
    ChatGPTPlanAuthFatalError,
    async_exchange_code,
    async_token_request,
    create_authorization_request,
    create_ext_agent_host_id,
    credentials_from_token,
    parse_callback_url,
    validate_id_token,
)
from chatgpt_plan_auth.auth import AGENT_NAME_HINT


def test_authorization_request_registration() -> None:
    """Test the first authorization request registers a dynamic client."""
    request = create_authorization_request(client_id=None, ext_agent_host_id=HOST_ID)

    assert request.url.startswith(f"{AUTHORIZE_URL}?")
    assert request.registration
    assert request.client_id == DYNAMIC_AGENT_CLIENT
    assert request.redirect_uri == REDIRECT_URI

    params = parse_params(request.url)
    assert params == {
        "response_type": "code",
        "client_id": DYNAMIC_AGENT_CLIENT,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "resource": RESOURCE,
        "state": request.state,
        "nonce": request.nonce,
        "code_challenge": params["code_challenge"],
        "code_challenge_method": "S256",
        "ext_agent_host_id": HOST_ID,
        "agent_name_hint": AGENT_NAME_HINT,
    }
    assert params["code_challenge"] != request.code_verifier
    assert "=" not in params["code_challenge"]


def test_authorization_request_reauthorization() -> None:
    """Test a returning sign in reuses the issued client id."""
    request = create_authorization_request(
        client_id=CLIENT_ID,
        ext_agent_host_id=HOST_ID,
        id_token_hint="retained-id-token",
        agent_name_hint="Home Assistant",
    )

    assert not request.registration
    assert request.client_id == CLIENT_ID

    params = parse_params(request.url)
    assert params["client_id"] == CLIENT_ID
    assert params["id_token_hint"] == "retained-id-token"
    # agent_name_hint is only sent for the initial registration.
    assert "agent_name_hint" not in params


def test_create_ext_agent_host_id() -> None:
    """Test the generated host id is a UUID URN and a fresh one each time."""
    host_id = create_ext_agent_host_id()
    assert host_id.startswith("urn:uuid:")
    assert host_id != create_ext_agent_host_id()


def test_parse_callback_url() -> None:
    """Test parsing of the URL the browser ended up on."""
    url = (
        f"{REDIRECT_URI}?code=the-code&scope=chatgpt.tokens.use.direct+email"
        f"&state=the-state&client_id={CLIENT_ID}"
    )
    assert parse_callback_url(url) == {
        "code": "the-code",
        "scope": "chatgpt.tokens.use.direct email",
        "state": "the-state",
        "client_id": CLIENT_ID,
    }


async def test_token_request_success() -> None:
    """Test a successful token endpoint request."""
    http = FakeHTTP((200, token_response()))

    assert await async_token_request(http.client, {"grant_type": "refresh_token"}) == (
        token_response()
    )
    assert str(http.requests[0].url) == TOKEN_URL
    assert http.form() == {"grant_type": "refresh_token"}


@pytest.mark.parametrize(
    "error",
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
        "invalid_client",
    ],
)
async def test_token_request_fatal_errors(error: str) -> None:
    """Test errors which invalidate the refresh token."""
    http = FakeHTTP((400, {"error": error}), (400, {"error": error}))

    with pytest.raises(ChatGPTPlanAuthFatalError):
        await async_token_request(
            http.client, {"grant_type": "refresh_token"}, fatal_errors=True
        )

    # Without fatal_errors the caller has to recover itself.
    with pytest.raises(ChatGPTPlanAuthError):
        await async_token_request(http.client, {"grant_type": "refresh_token"})


async def test_token_request_connection_error() -> None:
    """Test a connection error while talking to the token endpoint."""
    http = FakeHTTP(httpx.ConnectError("boom"))

    with pytest.raises(ChatGPTPlanAuthError, match="Error talking to OpenAI"):
        await async_token_request(http.client, {"grant_type": "refresh_token"})


@pytest.mark.parametrize(
    ("body", "code", "message"),
    [
        (
            {"error": {"code": "code-1", "message": "message-1"}},
            "code-1",
            "message-1",
        ),
        ({"code": "code-2"}, "code-2", "unknown error"),
        ({"error_description": "description-3"}, None, "description-3"),
        ({"error": {"error_description": "description-4"}}, None, "description-4"),
    ],
)
async def test_token_request_error_body_variants(
    body: Any, code: str | None, message: str
) -> None:
    """Test the error code and message of the known token endpoint bodies."""
    http = FakeHTTP((400, body))

    with pytest.raises(ChatGPTPlanAuthError) as err:
        await async_token_request(http.client, {"grant_type": "refresh_token"})

    assert str(err.value) == f"Sign in with ChatGPT failed ({code}): {message}"


async def test_token_request_invalid_json_success() -> None:
    """Test a successful status with a non-JSON body is reported."""
    http = FakeHTTP((200, None))

    with pytest.raises(ChatGPTPlanAuthError, match="Invalid token response"):
        await async_token_request(http.client, {"grant_type": "refresh_token"})


async def test_token_request_error_without_json_body() -> None:
    """Test an error status with a non-JSON body falls back to no error code."""
    http = FakeHTTP((500, None))

    with pytest.raises(ChatGPTPlanAuthError) as err:
        await async_token_request(
            http.client, {"grant_type": "refresh_token"}, fatal_errors=True
        )

    assert str(err.value) == "Sign in with ChatGPT failed (None): unknown error"


def test_validate_id_token() -> None:
    """Test ID token claim validation, without a signature check."""
    claims = validate_id_token(
        make_id_token(nonce="nonce-1"), client_id=CLIENT_ID, nonce="nonce-1"
    )

    assert claims["sub"] == "chatgpt-user-1"
    assert claims["email"] == "user@example.com"


@pytest.mark.parametrize(
    ("client_id", "nonce"),
    [
        ("oaiapp_other", "nonce-1"),
        (CLIENT_ID, "nonce-other"),
    ],
)
def test_validate_id_token_rejects(client_id: str, nonce: str) -> None:
    """Test that a wrong audience or nonce is rejected."""
    with pytest.raises(ChatGPTPlanAuthError):
        validate_id_token(
            make_id_token(client_id=CLIENT_ID, nonce="nonce-1"),
            client_id=client_id,
            nonce=nonce,
        )


def test_validate_id_token_expired() -> None:
    """Test that an expired ID token is rejected."""
    expired = int(time.time()) - 60

    with pytest.raises(ChatGPTPlanAuthError, match="expired"):
        validate_id_token(
            make_id_token(exp=expired), client_id=CLIENT_ID, nonce="nonce-1"
        )


def test_validate_id_token_bad_issuer() -> None:
    """Test that an ID token from another issuer is rejected."""
    with pytest.raises(ChatGPTPlanAuthError, match="issuer"):
        validate_id_token(
            make_id_token(iss="https://evil.example"), client_id=CLIENT_ID, nonce="nonce-1"
        )


def test_validate_id_token_not_a_jwt() -> None:
    """Test an unparsable ID token is reported."""
    with pytest.raises(ChatGPTPlanAuthError, match="Invalid ID token"):
        validate_id_token("not-a-jwt", client_id=CLIENT_ID, nonce="nonce-1")


async def test_exchange_code() -> None:
    """Test a code exchange returns credentials built from the ID token claims."""
    request = create_authorization_request(client_id=None, ext_agent_host_id=HOST_ID)
    http = FakeHTTP(
        (
            200,
            token_response(
                id_token=make_id_token(
                    client_id=DYNAMIC_AGENT_CLIENT, nonce=request.nonce
                )
            ),
        )
    )

    result = await async_exchange_code(
        http.client,
        "the-code",
        request=request,
        callback_client_id=CLIENT_ID,
        callback_state=request.state,
    )

    assert http.form() == {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": "the-code",
        "code_verifier": request.code_verifier,
        "redirect_uri": REDIRECT_URI,
        "resource": RESOURCE,
    }
    assert result.client_id == CLIENT_ID
    assert result.ext_agent_host_id == HOST_ID
    assert result.sub == "chatgpt-user-1"
    assert result.email == "user@example.com"
    assert result.refresh_token == "refresh-1"
    assert PLAN_SCOPE in result.scopes
    assert result.earliest_refresh_at is None


async def test_exchange_code_state_mismatch() -> None:
    """Test a callback with the wrong state is rejected before any request."""
    request = create_authorization_request(client_id=CLIENT_ID, ext_agent_host_id=HOST_ID)
    http = FakeHTTP()

    with pytest.raises(ChatGPTPlanAuthError, match="state mismatch"):
        await async_exchange_code(
            http.client, "the-code", request=request, callback_state="other-state"
        )

    assert http.call_count == 0


async def test_exchange_code_nonce_mismatch() -> None:
    """Test a nonce which does not belong to this attempt is rejected."""
    request = create_authorization_request(client_id=CLIENT_ID, ext_agent_host_id=HOST_ID)
    http = FakeHTTP((200, token_response(id_token=make_id_token(nonce="other-nonce"))))

    with pytest.raises(ChatGPTPlanAuthError, match="nonce mismatch"):
        await async_exchange_code(
            http.client, "the-code", request=request, callback_state=request.state
        )


async def test_exchange_code_fatal_error() -> None:
    """Test a rejected code exchange is fatal."""
    request = create_authorization_request(client_id=CLIENT_ID, ext_agent_host_id=HOST_ID)
    http = FakeHTTP((400, {"error": "invalid_grant"}))

    with pytest.raises(ChatGPTPlanAuthFatalError):
        await async_exchange_code(http.client, "the-code", request=request)


def test_credentials_roundtrip() -> None:
    """Test that credentials survive a persistence round trip."""
    token = token_response(earliest_refresh_at="2030-01-01T00:00:00+00:00")
    from_token = credentials_from_token(
        token,
        client_id=CLIENT_ID,
        ext_agent_host_id=HOST_ID,
        sub="chatgpt-user-1",
        email="user@example.com",
    )

    assert from_token.unique_id == f"chatgpt-user-1:{CLIENT_ID}"
    assert from_token.scopes == [
        PLAN_SCOPE,
        "email",
        "offline_access",
        "openid",
        "profile",
        "resource.invoke",
    ]
    assert from_token.earliest_refresh_at is not None

    data = from_token.as_entry_data()
    assert data["client_id"] == CLIENT_ID
    assert data["ext_agent_host_id"] == HOST_ID
    assert data["email"] == "user@example.com"

    assert ChatGPTCredentials.from_entry_data(data) == from_token


async def test_access_token_is_refreshed_before_expiry() -> None:
    """Test that an expiring access token is refreshed and reported."""
    http = FakeHTTP(
        (
            200,
            token_response(
                access_token="access-2", refresh_token="refresh-2", id_token="id-2"
            ),
        )
    )
    updated: list[ChatGPTCredentials] = []
    auth = ChatGPTAuth(
        credentials(expires_at=expires_in(-1)), client=http.client, on_update=updated.append
    )

    assert await auth.async_get_access_token() == "access-2"
    # The refresh token rotates on every refresh, and the caller is told.
    assert auth.credentials.refresh_token == "refresh-2"
    assert auth.credentials.access_token == "access-2"
    assert auth.credentials.id_token == "id-2"
    assert updated == [auth.credentials]
    assert http.form() == {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": "refresh-1",
        "resource": RESOURCE,
    }


async def test_access_token_not_refreshed_while_valid() -> None:
    """Test that a valid access token is reused."""
    http = FakeHTTP()
    auth = ChatGPTAuth(credentials(), client=http.client)

    assert await auth.async_get_access_token() == "access-1"
    assert http.call_count == 0


async def test_access_token_respects_earliest_refresh_at() -> None:
    """Test that the token endpoint is not polled before earliest_refresh_at."""
    http = FakeHTTP()
    auth = ChatGPTAuth(
        credentials(expires_at=expires_in(-1), earliest_refresh_at=expires_in(300)),
        client=http.client,
    )

    assert await auth.async_get_access_token() == "access-1"
    assert http.call_count == 0


async def test_concurrent_requests_refresh_once() -> None:
    """Test that concurrent requests only refresh the token once."""
    http = FakeHTTP(
        (200, token_response(access_token="access-2", refresh_token="refresh-2"))
    )
    auth = ChatGPTAuth(credentials(expires_at=expires_in(-1)), client=http.client)

    await asyncio.gather(
        auth.async_get_access_token(),
        auth.async_get_access_token(),
        auth.async_get_access_token(),
    )

    assert http.call_count == 1


async def test_refresh_failure_raises_fatal() -> None:
    """Test that a dead refresh token raises a fatal error."""
    http = FakeHTTP((400, {"error": "invalid_grant"}))
    auth = ChatGPTAuth(credentials(expires_at=expires_in(-1)), client=http.client)

    with pytest.raises(ChatGPTPlanAuthFatalError):
        await auth.async_get_access_token()

    # The stored credentials are left alone so the caller can start reauth.
    assert auth.credentials.access_token == "access-1"


async def test_authentication_error_forces_refresh() -> None:
    """Test that a rejected access token is refreshed even when not expired."""
    http = FakeHTTP(
        (200, token_response(access_token="access-2", refresh_token="refresh-2"))
    )
    auth = ChatGPTAuth(credentials(), client=http.client)

    await auth.async_handle_authentication_error()

    assert auth.credentials.access_token == "access-2"


async def test_authentication_error_respects_earliest_refresh_at() -> None:
    """Test a rejected token is not refreshed before earliest_refresh_at."""
    http = FakeHTTP()
    auth = ChatGPTAuth(
        credentials(earliest_refresh_at=expires_in(300)), client=http.client
    )

    await auth.async_handle_authentication_error()

    assert http.call_count == 0


async def test_refresh_keeps_identity_and_old_refresh_token() -> None:
    """Test fields missing from a refresh response are retained."""
    http = FakeHTTP(
        (200, {"access_token": "access-2", "expires_in": 3600, "scope": ""})
    )
    auth = ChatGPTAuth(credentials(scopes=[PLAN_SCOPE]), client=http.client)

    await auth.async_handle_authentication_error()

    assert auth.credentials.access_token == "access-2"
    assert auth.credentials.refresh_token == "refresh-1"
    assert auth.credentials.id_token == "id-1"
    assert auth.credentials.sub == "sub-value"
    assert auth.credentials.scopes == [PLAN_SCOPE]


async def test_from_entry_data_builds_manager() -> None:
    """Test that a manager can be built from persisted credentials."""
    http = FakeHTTP()
    data = credentials(expires_at=expires_in(3600)).as_entry_data()

    auth = ChatGPTAuth.from_entry_data(data, client=http.client)

    assert await auth.async_get_access_token() == "access-1"
    assert http.call_count == 0


async def test_token_expiry_margin_is_respected() -> None:
    """Test a token inside the refresh margin is refreshed."""
    http = FakeHTTP((200, token_response(access_token="access-2")))
    auth = ChatGPTAuth(credentials(expires_at=expires_in(60)), client=http.client)

    assert await auth.async_get_access_token() == "access-2"
    assert http.call_count == 1

    # A token outside the margin is left alone.
    fresh = ChatGPTAuth(credentials(expires_at=expires_in(3600)), client=FakeHTTP().client)
    assert await fresh.async_get_access_token() == "access-1"
