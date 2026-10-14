"""Adapt Responses API requests for the ChatGPT plan usage route.

The plan usage route accepts a subset of the Responses API: ``store: false``
with ``stream: true``, no system input items, no hosted tools beyond web
search, and none of the parameters listed in ``SUBSCRIPTION_UNSUPPORTED_PARAMS``.
"""

from __future__ import annotations

from typing import Any

from .const import (
    SUBSCRIPTION_INFERENCE_ERROR_HINTS,
    SUBSCRIPTION_STORE,
    SUBSCRIPTION_STREAM,
    SUBSCRIPTION_UNSUPPORTED_PARAMS,
    SUBSCRIPTION_UNSUPPORTED_TOOL_TYPES,
)


def shape_request_for_subscription(
    model_args: dict[str, Any],
    messages: list[Any],
) -> None:
    """Adapt a Responses API request for the ChatGPT plan usage route.

    The subscription route only accepts `store: false` together with
    `stream: true` and rejects a documented list of parameters. It also does
    not accept system input items: the system prompt is sent as `instructions`
    instead. Function tools are accepted flat.
    """
    for param in SUBSCRIPTION_UNSUPPORTED_PARAMS:
        model_args.pop(param, None)

    model_args["store"] = SUBSCRIPTION_STORE
    model_args["stream"] = SUBSCRIPTION_STREAM

    # System/developer messages are not accepted as input items.
    instructions = model_args.get("instructions")
    while messages and (
        isinstance(messages[0], dict)
        and messages[0].get("type") == "message"
        and messages[0].get("role") in ("system", "developer")
    ):
        content = messages.pop(0).get("content")
        if isinstance(content, str):
            instructions = f"{instructions}\n\n{content}" if instructions else content
    if instructions:
        model_args["instructions"] = instructions

    # The route rejects explicit `system` input items.
    for item in messages:
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "system"
        ):
            item["role"] = "developer"

    # Hosted tools which the subscription route does not provide are removed;
    # function tools are sent flat and `web_search` is kept.
    if tools := model_args.get("tools"):
        kept = [
            tool
            for tool in tools
            if tool.get("type") == "function"
            or tool.get("type") not in SUBSCRIPTION_UNSUPPORTED_TOOL_TYPES
        ]
        if kept:
            model_args["tools"] = kept
        else:
            model_args.pop("tools")

    # Naming a single function as `tool_choice` is not supported.
    if isinstance(model_args.get("tool_choice"), dict):
        model_args.pop("tool_choice")


def subscription_error_message(err: Exception) -> str | None:
    """Return a helpful message for a subscription route error.

    Duck types on ``code`` and only imports the OpenAI SDK when the error is not
    recognised by code, so this module works without the ``openai`` extra.
    """
    code: Any = getattr(err, "code", None)
    if isinstance(code, str) and (
        hint := SUBSCRIPTION_INFERENCE_ERROR_HINTS.get(code)
    ):
        return hint

    try:
        import openai
    except ImportError:
        return None

    if isinstance(err, openai.AuthenticationError):
        return SUBSCRIPTION_INFERENCE_ERROR_HINTS["subscription_sharing_invalid_user"]
    return None


__all__ = ["shape_request_for_subscription", "subscription_error_message"]
