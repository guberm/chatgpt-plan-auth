"""Fetch the ChatGPT plan model catalog.

The plan usage route serves a non-standard catalog body (a ``models`` list of
objects carrying a ``slug``) which the OpenAI SDK cannot parse, so the request
is made directly with :mod:`httpx`.
"""

from __future__ import annotations

import httpx

from ._compat import HTTP_ERRORS
from .auth import ChatGPTCredentials, ChatGPTAuth
from .const import (
    HTTP_TIMEOUT,
    MODELS_URL,
    SUBSCRIPTION_MODEL_SUGGESTIONS,
)
from .errors import ChatGPTPlanAuthError, ChatGPTPlanAuthFatalError


def subscription_model_options(models: list[str]) -> list[str]:
    """Return the models to offer, falling back to the known suggestions."""
    return models or SUBSCRIPTION_MODEL_SUGGESTIONS


async def async_get_subscription_models(
    client: httpx.AsyncClient, auth: ChatGPTAuth
) -> list[str]:
    """Return the model slugs offered to a ChatGPT plan account."""
    return await _async_fetch_subscription_models(
        client, await auth.async_get_access_token()
    )


async def async_get_subscription_models_for_credentials(
    client: httpx.AsyncClient, credentials: ChatGPTCredentials
) -> list[str]:
    """Return the model slugs of an account which is not stored yet.

    Used while creating or converting an entry, when there is no
    :class:`ChatGPTAuth` (and therefore no stored token) to supply the access
    token.
    """
    return await _async_fetch_subscription_models(client, credentials.access_token)


async def _async_fetch_subscription_models(
    client: httpx.AsyncClient, access_token: str
) -> list[str]:
    """Return the model slugs offered to a ChatGPT plan account."""
    try:
        response = await client.get(
            MODELS_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=HTTP_TIMEOUT,
        )
    except HTTP_ERRORS as err:
        raise ChatGPTPlanAuthError(f"Error talking to OpenAI: {err}") from err

    if response.status_code in (401, 403):
        raise ChatGPTPlanAuthFatalError(
            "Sign in with ChatGPT authentication failed while listing models"
        )

    try:
        response.raise_for_status()
    except HTTP_ERRORS as err:
        raise ChatGPTPlanAuthError(f"Error listing OpenAI models: {err}") from err

    try:
        payload = response.json()
    except ValueError as err:
        raise ChatGPTPlanAuthError(
            "Invalid model catalog response from OpenAI"
        ) from err

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ChatGPTPlanAuthError(
            "The model catalog response from OpenAI did not contain a models list"
        )

    return [
        item["slug"]
        for item in models
        if isinstance(item, dict) and isinstance(item.get("slug"), str)
    ]


__all__ = [
    "async_get_subscription_models",
    "async_get_subscription_models_for_credentials",
    "subscription_model_options",
]
