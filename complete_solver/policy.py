"""Policy helpers for solved Complete subgames."""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

from .actions import NTPAction, TPAction
from .finite_horizon import StatePolicy

T = TypeVar("T")


def sample_tp_action(policy: StatePolicy, rng: random.Random | None = None) -> TPAction:
    return _sample(policy.tp_actions, policy.tp_policy, rng)


def sample_ntp_action(policy: StatePolicy, rng: random.Random | None = None) -> NTPAction:
    return _sample(policy.ntp_actions, policy.ntp_policy, rng)


def policy_mass_by_skill(policy: StatePolicy) -> dict[str, float]:
    mass: dict[str, float] = {}
    for action, probability in zip(policy.tp_actions, policy.tp_policy):
        key = "数字" if isinstance(action.skill, int) else str(action.skill)
        mass[key] = mass.get(key, 0.0) + probability
    return dict(sorted(mass.items()))


def reaction_mass(policy: StatePolicy) -> dict[str, float]:
    mass: dict[str, float] = {}
    for action, probability in zip(policy.ntp_actions, policy.ntp_policy):
        mass[action.reaction] = mass.get(action.reaction, 0.0) + probability
    return dict(sorted(mass.items()))


def _sample(actions: Sequence[T], probabilities: Sequence[float], rng: random.Random | None) -> T:
    if not actions:
        raise ValueError("cannot sample from an empty policy")
    generator = rng or random
    threshold = generator.random()
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return action
    return actions[-1]
