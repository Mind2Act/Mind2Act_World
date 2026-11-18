"""Compatibility entry point for roboprobe client."""
from .roboprobe.client import LoggedNativeClient, create, rate_limit_wait

__all__ = ['LoggedNativeClient', 'create', 'rate_limit_wait']
