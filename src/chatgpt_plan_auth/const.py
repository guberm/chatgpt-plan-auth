"""Constants of the Sign in with ChatGPT (ChatGPT plan usage) protocol.

Ports the protocol constants out of Home Assistant core's
``openai_conversation/chatgpt_auth.py``. Endpoint URLs, the dynamic client
registration id, scopes, the ``ext_agent_host_id`` scheme, the parameters and
tools the plan usage route rejects, and the model suggestions live here.
"""

from datetime import timedelta
from typing import Any

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
MODELS_URL = "https://api.openai.com/v1/models"

# The callback URI must be an HTTP loopback address on 127.0.0.1; only the port
# may vary. A fixed port keeps the URI stable between authorization attempts.
REDIRECT_URI = "http://127.0.0.1:1455/auth/callback"

# `dynamic_agent_client` is the first-time registration entrypoint only. It must
# never be stored: the callback returns the issued client id (oaiapp_...) which
# is used for the code exchange and all later authorizations.
DYNAMIC_AGENT_CLIENT = "dynamic_agent_client"
CLIENT_ID_PREFIX = "oaiapp_"

# Display metadata of the first (dynamic client registration) authorization
# request. Callers pass their own product name.
AGENT_NAME_HINT = "chatgpt-plan-auth"

# The `ext_agent_host_id` scheme: a UUID based host id is documented as a
# supported alternative to a JWK thumbprint URI and needs no key management.
EXT_AGENT_HOST_ID_SCHEME = "urn:uuid:"

SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
# Granted scope required before the ChatGPT plan may be used.
PLAN_SCOPE = "chatgpt.tokens.use.direct"

# Refresh the access token this long before it expires. Access tokens live one
# hour and refresh tokens 30 days.
TOKEN_EXPIRY_MARGIN = timedelta(minutes=5)
HTTP_TIMEOUT = 30

# Keys of the credential mapping (`ChatGPTCredentials.as_entry_data`). They are
# plain strings so a caller can persist them wherever it wants; Home Assistant
# stores them as config entry data.
CONF_AUTH_METHOD = "auth_method"
CONF_ACCESS_TOKEN = "access_token"
CONF_CLIENT_ID = "client_id"
CONF_EMAIL = "email"
CONF_EARLIEST_REFRESH_AT = "earliest_refresh_at"
CONF_EXPIRES_AT = "expires_at"
CONF_EXT_AGENT_HOST_ID = "ext_agent_host_id"
CONF_ID_TOKEN = "id_token"
CONF_REFRESH_TOKEN = "refresh_token"
CONF_SCOPES = "scopes"
CONF_SUB = "sub"

AUTH_METHOD_API_KEY = "api_key"
AUTH_METHOD_CHATGPT = "chatgpt"

# Token endpoint errors which mean the refresh token can never be used again.
FATAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_client",
        "invalid_refresh_token",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
        "token_expired",
    }
)

# Errors returned by the subscription route itself (inference requests).
SUBSCRIPTION_INFERENCE_ERROR_HINTS = {
    "subscription_sharing_user_not_eligible": (
        "This ChatGPT account is not eligible to use the ChatGPT plan with "
        "this app. Check your plan in ChatGPT settings."
    ),
    "subscription_sharing_usage_limit_exceeded": (
        "The ChatGPT plan usage limit was reached. Check ChatGPT settings -> "
        "Usage to see the remaining plan usage."
    ),
    "subscription_sharing_usage_unavailable": (
        "ChatGPT plan usage is currently unavailable, please retry later."
    ),
    "subscription_sharing_route_not_supported": (
        "This request is not supported by ChatGPT plan usage."
    ),
    "subscription_sharing_unsupported_capability": (
        "This request uses a capability which ChatGPT plan usage does not support."
    ),
    "subscription_sharing_invalid_user": (
        "The ChatGPT sign-in is no longer valid. Please sign in with ChatGPT again."
    ),
}

# Parameters the ChatGPT plan usage route rejects. Everything else in the
# request builder is either required (model/input) or accepted.
SUBSCRIPTION_UNSUPPORTED_PARAMS = (
    "background",
    "conversation",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "moderation",
    "multi_agent",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "temperature",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
)

# Hosted tools which the subscription route does not provide. Web search is
# still offered by the route, subject to the account's plan and policy.
SUBSCRIPTION_UNSUPPORTED_TOOL_TYPES = (
    "code_interpreter",
    "file_search",
    "image_generation",
    "mcp",
    "tool_search",
)

# The subscription route requires unsaved, streamed responses.
SUBSCRIPTION_STORE = False
SUBSCRIPTION_STREAM = True

# Placeholder value the OpenAI SDK insists on; the real credential is the OAuth
# access token attached by `ChatGPTSubscriptionClient.custom_auth`.
PLACEHOLDER_API_KEY = "chatgpt-plan"

# Models which were observed in the catalog of a ChatGPT plan account. The
# catalog is authoritative and is used whenever it can be read, but it can be
# unavailable (OpenAI answered a transient HTTP 520 while this was written) and
# the plan usage route rejects the regular API default model, so these are
# offered as suggestions in that case. The model field stays editable, so a
# model outside this list can still be used.
SUBSCRIPTION_MODEL_SUGGESTIONS: list[str] = [
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-reserve",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-daybreak-blue-latest",
    "gpt-5.5",
    "codex-auto-review",
]


def is_chatgpt_data(data: dict[str, Any]) -> bool:
    """Return if stored data authenticates with a ChatGPT plan.

    API key data keeps the historical layout and has no auth method key.
    """
    return data.get(CONF_AUTH_METHOD) == AUTH_METHOD_CHATGPT
