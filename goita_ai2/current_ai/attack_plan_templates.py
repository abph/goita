"""Catalogs representative Goita attack shapes as reusable templates.
It translates established kyosha, big-piece, middle-pair, shi, royal, and
fourth-piece principles into preferred sequences for branched plan generation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from goita_ai2.constants import POINTS
from goita_ai2.current_ai.branched_attack_plan import Action, BranchedAttackPlan


class AttackPlanTemplateFamily(str, Enum):
    """Broad strategy family used by the representative catalog."""

    KYOSHA_BIG = "kyosha_big"
    MIDDLE_PAIR = "middle_pair"
    SHI_PRESSURE = "shi_pressure"
    ROYAL_FOURTH = "royal_fourth"


@dataclass(frozen=True)
class RepresentativeAttackTemplate:
    """One attack order plus pieces that must survive until their role."""

    template_id: str
    family: AttackPlanTemplateFamily
    label: str
    attack_sequence: Tuple[str, ...]
    block_sequence: Tuple[Optional[str], ...] = tuple()
    protected_pieces: Tuple[str, ...] = tuple()
    priority: int = 100
    rationale: str = ""

    def expected_attack(self, attack_number: int) -> Optional[str]:
        index = int(attack_number) - 1
        if index < 0 or index >= len(self.attack_sequence):
            return None
        return self.attack_sequence[index]

    def expected_block(self, attack_number: int) -> Optional[str]:
        index = int(attack_number) - 1
        if index < 0 or index >= len(self.block_sequence):
            return None
        return self.block_sequence[index]

    def as_dict(self) -> Dict[str, object]:
        return {
            "template_id": self.template_id,
            "family": self.family.value,
            "label": self.label,
            "attack_sequence": list(self.attack_sequence),
            "block_sequence": list(self.block_sequence),
            "protected_pieces": list(self.protected_pieces),
            "priority": self.priority,
            "rationale": self.rationale,
        }


class AttackPlanTemplateMixin:
    """Matches known hand shapes and emits template-steered plan candidates."""

    @staticmethod
    def _attack_template_family(label: str) -> AttackPlanTemplateFamily:
        if "two_kyosha" in label:
            if "middle_pair" in label:
                return AttackPlanTemplateFamily.MIDDLE_PAIR
            return AttackPlanTemplateFamily.KYOSHA_BIG
        if "middle_pair" in label:
            return AttackPlanTemplateFamily.MIDDLE_PAIR
        return AttackPlanTemplateFamily.ROYAL_FOURTH

    def _fixed_sequence_attack_template(
        self,
        state,
        player: str,
        counts: Counter,
    ) -> Optional[RepresentativeAttackTemplate]:
        info = self._special_attack_sequence_plan(counts)
        if not isinstance(info, dict):
            return None
        label = str(info.get("label", "fixed_sequence"))
        if label == "dealer_three_shi_two_single_bigs_royal" and state.dealer != player:
            return None
        sequence = tuple(str(piece) for piece in info.get("sequence", ()))
        if not sequence:
            return None
        raw_blocks = tuple(info.get("block_sequence", ()))
        blocks = tuple(None if piece is None else str(piece) for piece in raw_blocks)
        family = self._attack_template_family(label)
        return RepresentativeAttackTemplate(
            template_id=label,
            family=family,
            label=label.replace("_", " "),
            attack_sequence=sequence,
            block_sequence=blocks,
            priority=400,
            rationale=(
                "Use the established hand-shape order; a public tactic such as "
                "kakarigotae still has higher priority."
            ),
        )

    def _four_shi_attack_template(
        self,
        counts: Counter,
    ) -> Optional[RepresentativeAttackTemplate]:
        if counts.get("1", 0) < 4:
            return None
        return RepresentativeAttackTemplate(
            template_id="four_shi_pressure",
            family=AttackPlanTemplateFamily.SHI_PRESSURE,
            label="four shi pressure",
            attack_sequence=("1", "1", "1"),
            block_sequence=("1",),
            priority=360,
            rationale=(
                "Show four shi by returning shi and keep enough shi to continue "
                "the first three attacks."
            ),
        )

    def _fourth_middle_attack_templates(
        self,
        state,
        player: str,
        counts: Counter,
    ) -> Tuple[RepresentativeAttackTemplate, ...]:
        tr = self._track.get(id(state))
        attack_number = int(tr.get("my_attack_count", 0)) + 1 if tr else 1
        if attack_number > 3:
            return tuple()

        fourth = [
            piece
            for piece in ("2", "3", "4", "5")
            if counts.get(piece, 0) > 0
            and self._is_fourth_middle_attack(state, player, piece)
        ]
        templates: List[RepresentativeAttackTemplate] = []
        for finisher in fourth:
            support = [
                piece
                for piece in ("1", "2", "3", "4", "5", "6", "7")
                if piece != finisher and counts.get(piece, 0) >= 2
            ]
            if not support:
                continue
            support.sort(
                key=lambda piece: (counts.get(piece, 0), POINTS.get(piece, 0), piece),
                reverse=True,
            )
            lead = support[0]
            sequence = (lead, lead, finisher)
            templates.append(RepresentativeAttackTemplate(
                template_id=f"fourth_middle_finisher_{finisher}",
                family=AttackPlanTemplateFamily.ROYAL_FOURTH,
                label=f"preserve fourth middle {finisher} for attack three",
                attack_sequence=sequence,
                protected_pieces=(finisher,),
                priority=440,
                rationale=(
                    "A publicly identified fourth middle piece is held as the "
                    "third attack, when opponents have the least pass freedom."
                ),
            ))
        return tuple(templates)

    def _royal_preservation_attack_template(
        self,
        counts: Counter,
    ) -> Optional[RepresentativeAttackTemplate]:
        royals = tuple(piece for piece in ("8", "9") if counts.get(piece, 0) > 0)
        if not royals:
            return None
        repeated = [
            piece
            for piece in ("1", "2", "3", "4", "5", "6", "7")
            if counts.get(piece, 0) >= 2
        ]
        if not repeated:
            return None
        repeated.sort(
            key=lambda piece: (counts.get(piece, 0), POINTS.get(piece, 0), piece),
            reverse=True,
        )
        lead = repeated[0]
        return RepresentativeAttackTemplate(
            template_id="royal_receive_width",
            family=AttackPlanTemplateFamily.ROYAL_FOURTH,
            label="preserve royals for receive width",
            attack_sequence=(lead, lead),
            protected_pieces=royals,
            priority=260,
            rationale=(
                "Keep royal pieces out of early hidden blocks and attacks so they "
                "remain available for receiving and the finishing bridge."
            ),
        )

    def _representative_attack_templates(
        self,
        state,
        player: str,
    ) -> Tuple[RepresentativeAttackTemplate, ...]:
        """Return every representative shape supported by current public facts."""
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        initial = Counter(tr.get("my_init_count", ())) if tr else Counter()
        current = Counter(str(piece) for piece in state.hands[player])
        templates: List[RepresentativeAttackTemplate] = []

        fixed = self._fixed_sequence_attack_template(state, player, initial or current)
        if fixed is not None:
            templates.append(fixed)
        four_shi = self._four_shi_attack_template(initial or current)
        if four_shi is not None:
            templates.append(four_shi)
        templates.extend(self._fourth_middle_attack_templates(
            state,
            player,
            current,
        ))
        royal = self._royal_preservation_attack_template(current)
        if royal is not None:
            templates.append(royal)

        unique: Dict[str, RepresentativeAttackTemplate] = {}
        for template in templates:
            previous = unique.get(template.template_id)
            if previous is None or template.priority > previous.priority:
                unique[template.template_id] = template
        return tuple(sorted(
            unique.values(),
            key=lambda item: (item.priority, item.template_id),
            reverse=True,
        ))

    @staticmethod
    def _template_preserves_blocks(
        hand: Sequence[str],
        action: Action,
        template: RepresentativeAttackTemplate,
    ) -> bool:
        block = action[1]
        if block is None or block not in template.protected_pieces:
            return True
        # A duplicate may be hidden only when another copy remains reserved.
        return list(hand).count(block) >= template.protected_pieces.count(block) + 1

    def _template_root_actions(
        self,
        state,
        player: str,
        actions: Sequence[Action],
        template: RepresentativeAttackTemplate,
        attack_number: int,
    ) -> Tuple[Action, ...]:
        expected_attack = template.expected_attack(attack_number)
        if expected_attack is None:
            return tuple()
        expected_block = template.expected_block(attack_number)
        future_required = Counter(template.attack_sequence[attack_number:])
        candidates = []
        for action in self._branched_root_attack_candidates(actions):
            if action[2] != expected_attack:
                continue
            if expected_block is not None and expected_block in state.hands[player]:
                if action[1] != expected_block:
                    continue
            if not self._template_preserves_blocks(
                state.hands[player],
                action,
                template,
            ):
                continue
            remaining = self._branched_remove_action(state.hands[player], action)
            if remaining is None or any(
                remaining.count(piece) < count
                for piece, count in future_required.items()
            ):
                continue
            candidates.append(action)
        return tuple(candidates)

    def _generate_representative_attack_plans(
        self,
        state,
        player: str,
        actions: Sequence[Action],
        *,
        max_plans: Optional[int] = None,
    ) -> Tuple[BranchedAttackPlan, ...]:
        """Convert all currently applicable representative shapes to graphs."""
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        revision = int(tr.get("piece_inference_revision", 0)) if tr else 0
        attack_number = int(tr.get("my_attack_count", 0)) + 1 if tr else 1
        plans: List[BranchedAttackPlan] = []
        for template in self._representative_attack_templates(state, player):
            roots = self._template_root_actions(
                state,
                player,
                actions,
                template,
                attack_number,
            )
            for index, root in enumerate(roots, start=1):
                plans.append(self._build_branched_attack_plan(
                    state,
                    player,
                    root,
                    attack_number=attack_number,
                    revision=revision,
                    candidate_label=f"{template.template_id}-{index}",
                    preferred_attacks=template.attack_sequence,
                    permanently_protected_pieces=(
                        template.protected_pieces
                        if template.template_id == "royal_receive_width"
                        else tuple()
                    ),
                    source=f"representative:{template.template_id}",
                    assumptions=(
                        template.rationale,
                        "proved wins and public tactics remain higher priority",
                        "the route is rebuilt when a public response breaks the shape",
                    ),
                ))
                if max_plans is not None and len(plans) >= max(0, int(max_plans)):
                    return tuple(plans)
        return tuple(plans)


__all__ = [
    "AttackPlanTemplateFamily",
    "AttackPlanTemplateMixin",
    "RepresentativeAttackTemplate",
]
