"""Compatibility entry point for roboprobe planner service."""
from .roboprobe.planner_service import main, verify_sources

__all__ = ['main', 'verify_sources']


if __name__ == "__main__":
    main()
