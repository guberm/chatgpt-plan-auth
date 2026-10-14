"""Errors of the Sign in with ChatGPT flow.

Callers map these onto their own error types: :class:`ChatGPTPlanAuthFatalError`
means the stored refresh token can never be used again and the user has to sign
in again (Home Assistant starts its reauth flow), every other
:class:`ChatGPTPlanAuthError` is a transient failure.
"""


class ChatGPTPlanAuthError(Exception):
    """Error of the Sign in with ChatGPT flow."""


class ChatGPTPlanAuthFatalError(ChatGPTPlanAuthError):
    """The stored credentials are dead and the user has to sign in again."""
