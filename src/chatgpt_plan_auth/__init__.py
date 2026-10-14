"""Sign in with ChatGPT (ChatGPT plan usage) for locally hosted apps.

Implements the OAuth flow documented at
https://developers.openai.com/siwc/token-sharing-open-source/: register a
dynamic client, authorize with PKCE, exchange the code for tokens, refresh and
rotate them, read the plan's model catalog, and shape Responses API requests
for the plan usage route.

The protocol code needs only :mod:`httpx`. The OpenAI SDK wrapper
(:mod:`chatgpt_plan_auth.client`) needs the ``openai`` extra.
"""

from .auth import (
    AuthorizationRequest,
    ChatGPTCredentials,
    ChatGPTAuth,
    async_exchange_code,
    async_token_request,
    create_authorization_request,
    create_ext_agent_host_id,
    credentials_from_token,
    parse_callback_url,
    validate_id_token,
)
from .catalog import (
    async_get_subscription_models,
    async_get_subscription_models_for_credentials,
    subscription_model_options,
)
from .const import (
    AUTH_METHOD_CHATGPT,
    AUTHORIZE_URL,
    CLIENT_ID_PREFIX,
    DYNAMIC_AGENT_CLIENT,
    MODELS_URL,
    PLAN_SCOPE,
    REDIRECT_URI,
    RESOURCE,
    SCOPE,
    SUBSCRIPTION_MODEL_SUGGESTIONS,
    TOKEN_URL,
)
from .errors import ChatGPTPlanAuthError, ChatGPTPlanAuthFatalError
from .requests import shape_request_for_subscription, subscription_error_message

__version__ = "0.1.0"

__all__ = [
    "AUTH_METHOD_CHATGPT",
    "AUTHORIZE_URL",
    "CLIENT_ID_PREFIX",
    "DYNAMIC_AGENT_CLIENT",
    "MODELS_URL",
    "PLAN_SCOPE",
    "REDIRECT_URI",
    "RESOURCE",
    "SCOPE",
    "SUBSCRIPTION_MODEL_SUGGESTIONS",
    "TOKEN_URL",
    "AuthorizationRequest",
    "ChatGPTCredentials",
    "ChatGPTAuth",
    "ChatGPTPlanAuthError",
    "ChatGPTPlanAuthFatalError",
    "async_exchange_code",
    "async_get_subscription_models",
    "async_get_subscription_models_for_credentials",
    "async_token_request",
    "create_authorization_request",
    "create_ext_agent_host_id",
    "credentials_from_token",
    "parse_callback_url",
    "shape_request_for_subscription",
    "subscription_error_message",
    "subscription_model_options",
    "validate_id_token",
]
