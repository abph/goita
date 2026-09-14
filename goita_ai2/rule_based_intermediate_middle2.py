"""Compatibility entry point for the frozen intermediate-middle-2 AI.

The implementation is isolated in goita_ai2.intermediate_middle2 so later
changes to the developing AI cannot alter this saved profile.
"""

from goita_ai2.intermediate_middle2 import RuleBasedAgent

__all__ = ["RuleBasedAgent"]
