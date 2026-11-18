"""Compatibility entry point for roboprobe eef."""
from .roboprobe.eef import create, policy_defaults

__all__ = ['create', 'policy_defaults']
