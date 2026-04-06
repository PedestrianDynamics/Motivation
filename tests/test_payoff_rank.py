"""Tests for payoff/rank update behavior."""

from __future__ import annotations

import math

from src.inifile_parser import parse_motivation_strategy, parse_payoff_update_interval
from src.motivation_model import EVCStrategy


def _build_strategy() -> EVCStrategy:
    strategy = EVCStrategy(
        motivation_door_center=(0.0, 0.0),
        agent_ids=[0, 1, 2],
        agent_positions=[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        width=2.0,
        height=0.5,
        max_reward=3,
        seed=1,
        min_value_high=1.0,
        max_value_high=1.0,
        min_value_low=1.0,
        max_value_low=1.0,
        number_high_value=0,
        nagents=3,
        motivation_mode="PVE",
        payoff_k=8.0,
        payoff_q0=0.5,
        rank_tie_tolerance_m=1e-3,
        payoff_update_interval_s=1.0,
    )
    return strategy


def test_update_interval_conversion() -> None:
    strategy = _build_strategy()
    strategy.configure_payoff_update_interval(time_step=0.01)
    assert strategy.payoff_update_interval_steps == 100
    strategy.configure_payoff_update_interval(time_step=2.0)
    assert strategy.payoff_update_interval_steps == 1


def test_rank_updates_only_on_schedule() -> None:
    strategy = _build_strategy()
    strategy.configure_payoff_update_interval(time_step=0.5)  # 2 steps
    positions = {0: (1.0, 0.0), 1: (2.0, 0.0), 2: (3.0, 0.0)}
    changed = strategy.update_payoff_cache(0, positions, number_agents_in_simulation=3)
    assert changed is True
    r0, _, p0 = strategy.get_rank_payoff(0)
    assert r0 == 1
    assert p0 > 0.5

    # Not scheduled: cache should not update.
    positions2 = {0: (3.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 0.0)}
    changed = strategy.update_payoff_cache(1, positions2, number_agents_in_simulation=3)
    assert changed is False
    r0_after, _, p0_after = strategy.get_rank_payoff(0)
    assert r0_after == r0
    assert math.isclose(p0_after, p0)

    # Scheduled again: now cache updates.
    changed = strategy.update_payoff_cache(2, positions2, number_agents_in_simulation=3)
    assert changed is True
    r0_new, _, _ = strategy.get_rank_payoff(0)
    assert r0_new == 3


def test_parse_payoff_update_interval() -> None:
    cfg = {"motivation_parameters": {"payoff": {"update_interval_s": 2.0}}}
    assert math.isclose(parse_payoff_update_interval(cfg), 2.0)


def test_parse_motivation_mode_no_motivation() -> None:
    """NO_MOTIVATION is mapped to BASE_MODEL for backward compatibility."""
    cfg = {"motivation_parameters": {"motivation_mode": "NO_MOTIVATION"}}
    assert parse_motivation_strategy(cfg) == "BASE_MODEL"
