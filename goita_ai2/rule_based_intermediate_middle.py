"""Compatibility entry point for the frozen intermediate-middle AI.

The implementation is isolated in goita_ai2.intermediate_middle so later
changes to the developing AI cannot alter this saved profile.
"""

from goita_ai2.intermediate_middle import RuleBasedAgent

__all__ = ["RuleBasedAgent"]
