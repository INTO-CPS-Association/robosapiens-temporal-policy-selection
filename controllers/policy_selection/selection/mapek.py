"""Health-aware temporal MAPE-K selector (no Webots or oracle dependencies)."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

from .modalities import POLICY_INPUTS, PREFERENCE, observation_for


@dataclass
class Health:
    faulty: bool = False
    positive_streak: int = 0
    negative_streak: int = 0

    def update(self, anomaly: bool, onset: int, recovery: int) -> bool:
        if anomaly:
            self.positive_streak += 1
            self.negative_streak = 0
        else:
            self.negative_streak += 1
            self.positive_streak = 0
        if not self.faulty and self.positive_streak >= onset:
            self.faulty = True
        elif self.faulty and self.negative_streak >= recovery:
            self.faulty = False
        return self.faulty


@dataclass
class MAPEKState:
    image: Health = field(default_factory=Health)
    lidar: Health = field(default_factory=Health)
    active: str | None = None
    timestep: int = 0
    last_switch: int = -10**9
    switches: int = 0


class TemporalSelector:
    """Generic requirement-based selector with modality dependency and fallback.

    Dwell gates *preference-driven* switchbacks only. Inadmissible policies are
    replaced immediately after confirmed onset, irrespective of dwell.
    """

    def __init__(self, policy_requirements: Mapping[str, Sequence[str]] | None = None,
                 preference: Sequence[str] = PREFERENCE, onset: int = 5, recovery: int = 10,
                 dwell: int = 10, world_depends_on_lidar: bool = True):
        self.requirements = {name: frozenset(keys) for name, keys in
                             (POLICY_INPUTS if policy_requirements is None else policy_requirements).items()}
        self.preference = tuple(preference)
        if set(self.requirements) != set(self.preference):
            raise ValueError("preference must list every policy exactly once")
        if len(self.preference) != len(set(self.preference)):
            raise ValueError("preference contains duplicate policies")
        if onset < 1 or recovery < 1 or dwell < 0:
            raise ValueError("onset/recovery >=1 and dwell >=0 required")
        self.onset, self.recovery, self.dwell = onset, recovery, dwell
        self.dependent = world_depends_on_lidar
        self.state = MAPEKState()

    def reset(self):
        self.state = MAPEKState()

    def update(self, image_raw: bool, lidar_raw: bool) -> dict:
        s = self.state
        s.timestep += 1
        img_fault = s.image.update(bool(image_raw), self.onset, self.recovery)
        lidar_fault = s.lidar.update(bool(lidar_raw), self.onset, self.recovery)
        effective = {
            "image": not img_fault,
            "lidar": not lidar_fault,
            "world_info": not lidar_fault if self.dependent else True,
        }
        reliable = {k for k, healthy in effective.items() if healthy}
        admissible = [p for p in self.preference if self.requirements[p] <= reliable]
        candidate = admissible[0] if admissible else "safe"
        previous = s.active
        if previous is None or previous == "safe":
            chosen = candidate
        elif previous not in admissible:
            # Hard safety/health constraint overrides dwell.
            chosen = candidate
        elif candidate != previous and s.timestep - s.last_switch >= self.dwell:
            chosen = candidate
        else:
            chosen = previous
        if chosen != s.active:
            if s.active is not None:
                s.switches += 1
            s.active = chosen
            s.last_switch = s.timestep
        return dict(active_policy=chosen, previous_policy=previous,
                    requested_policy=candidate, admissible_policies=admissible,
                    health=effective, image_fault=img_fault,
                    lidar_fault=lidar_fault, switch_count=s.switches,
                    switched=previous is not None and previous != chosen,
                    timestep=s.timestep)


class MAPEKController:
    """Connect a learned monitor and policy objects exposing .predict()."""

    def __init__(self, monitor, policies: Mapping, selector: TemporalSelector,
                 safe_action: int = 5, deterministic: bool = False):
        if set(policies) != set(selector.requirements):
            raise ValueError("Loaded policies must match selector library")
        self.monitor, self.policies, self.selector = monitor, policies, selector
        self.safe_action, self.deterministic = safe_action, deterministic

    def reset(self):
        self.selector.reset()

    def act(self, observation: dict):
        evidence = self.monitor.analyze(observation)
        decision = self.selector.update(evidence["image_raw"], evidence["lidar_raw"])
        policy = decision["active_policy"]
        if policy == "safe":
            action = self.safe_action
        else:
            action, _ = self.policies[policy].predict(
                observation_for(policy, observation), deterministic=self.deterministic)
            action = int(action)
        return action, {**evidence, **decision, "action": action}
