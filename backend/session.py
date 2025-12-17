# backend/session.py
from typing import Dict
from goita_ai2.state import GoitaState
from goita_ai2.agents.rule_based import RuleBasedAgent

class GameSession:
    def __init__(self, state: GoitaState, agents: Dict[str, RuleBasedAgent]):
        self.state = state
        self.agents = agents
