# chatgpt-plan-auth

Sign in with ChatGPT (ChatGPT plan usage) for locally hosted, open-source apps.

Implements the SIWC OAuth flow documented at
<https://developers.openai.com/siwc/token-sharing-open-source/>: register a
dynamic client, authorize with PKCE, exchange the code for tokens, keep them
refreshed and rotated, read the plan's model catalog, and shape Responses API
requests for the plan usage route. The access token is used as the bearer token
against `https://api.openai.com/v1` instead of an API key.

The library is a straight extraction of Home Assistant core's
`openai_conversation/chatgpt_auth.py` (the reviewer's point: deep knowledge of
an external system belongs in a library, not in core). It carries no Home
Assistant imports, no `hass`, no config-entry or storage knowledge, and no
crypto dependency.

```bash
pip install chatgpt-plan-auth          # protocol only: httpx
pip install chatgpt-plan-auth[openai]  # + the OpenAI SDK client wrapper
```

## What lives where

| Module | Contents |
| --- | --- |
| `chatgpt_plan_auth.const` | endpoints, `dynamic_agent_client`, scopes, the `ext_agent_host_id` scheme, rejected params/tools lists, model suggestions |
| `chatgpt_plan_auth.auth` | authorization URL + PKCE/state/nonce, callback parsing, token requests, code exchange, ID token claim validation, credentials, token lifecycle |
| `chatgpt_plan_auth.catalog` | plan model catalog fetch/parse and the suggestion fallback |
| `chatgpt_plan_auth.requests` | `shape_request_for_subscription`, `subscription_error_message` |
| `chatgpt_plan_auth.client` | httpx auth flow and `openai.AsyncOpenAI` subclass (extra: `openai`) |

Everything is caller-driven: no Home Assistant, no global session, no storage.

```python
import httpx
from chatgpt_plan_auth import (
    ChatGPTCredentials, ChatGPTAuth, async_exchange_code, create_authorization_request,
    create_ext_agent_host_id, parse_callback_url,
)

host_id = create_ext_agent_host_id()          # generate once, persist it yourself
request = create_authorization_request(client_id=None, ext_agent_host_id=host_id)
# send the user to request.url, then read the loopback address bar (any machine):
params = parse_callback_url(callback_url)
async with httpx.AsyncClient() as client:
    credentials = await async_exchange_code(
        client, params["code"], request=request,
        callback_client_id=params.get("client_id"), callback_state=params.get("state"),
    )

# Later, per request:
auth = ChatGPTAuth(credentials, client=client, on_update=save)  # `save` persists rotation
token = await auth.async_get_access_token()
```

Errors: `ChatGPTPlanAuthFatalError` means the stored refresh token can never be
used again (start your re-authentication flow); any other
`ChatGPTPlanAuthError` is transient. With the `openai` extra,
`ChatGPTSubscriptionClient(auth, http_client=...)` attaches the token per
request and retries once on a 401.

## Deliberate simplification: no ID token signature verification

`/.well-known/jwks.json` is never fetched, RSA keys are never parsed and no JWT
or crypto dependency is carried — the JWKS helpers and their cache from core are
gone.

Rationale: the ID token arrives directly from the provider's
TLS-authenticated token endpoint in the same response as the code exchange. That
is the trust model Home Assistant's own OAuth2 flow uses, and it is exactly the
dependency-heavy part the core reviewer objected to. A signature check adds
nothing there: anyone able to forge the token response could equally forge the
signature.

What is still verified on every code exchange:

- `iss` is `https://auth.openai.com`
- `aud` contains the client id the code was exchanged with
- `exp` is in the future
- `nonce` equals the nonce of that authorization attempt
- the callback `state` equals the state of that authorization attempt

Token endpoint errors (`invalid_grant`, `invalid_client`, `token_expired`, ...)
still surface as fatal errors and never invalidate good credentials.

If you need signature verification — or you cannot trust your network path to
the token endpoint — verify the ID token against the JWKS in your application
before calling `validate_id_token`.

## Out of scope (stays in the consuming application)

- Home Assistant: config-entry inspection/storage, `Store` for the host id, the
  HA http client, config-flow and reauth steps.
- Application branding: pass `agent_name_hint` (defaults to
  `chatgpt-plan-auth`).
- Persisting credentials: the library returns plain mappings
  (`ChatGPTCredentials.as_entry_data()`), nothing is written to disk.

## Tests

```bash
uv run --with pytest-asyncio --with openai python -m pytest tests -q
```

All network access is a fake `httpx.MockTransport`; no test touches the network
and no test needs key material.
