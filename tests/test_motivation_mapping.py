"""Tests for motivation mapping utilities."""

from __future__ import annotations

import math
import pytest

from src.motivation_mapping import (
    MOTIVATION_HIGH,
    MOTIVATION_LOW,
    MOTIVATION_NORMAL,
    MotivationParameterMapper,
    fit_logistic_from_anchors,
    evaluate_logistic,
    AnchorValues,
    clamp_motivation,
)


def test_fit_logistic_tracks_anchor_points() -> None:
    anchors = AnchorValues(low=0.5, normal=1.2, high=3.6)
    params = fit_logistic_from_anchors(anchors, inflection_target=1.5)
    assert math.isclose(evaluate_logistic(MOTIVATION_LOW, params), 0.5, abs_tol=1e-6)
    assert math.isclose(
        evaluate_logistic(MOTIVATION_NORMAL, params), 1.2, abs_tol=0.15
    )
    assert math.isclose(evaluate_logistic(MOTIVATION_HIGH, params), 3.6, abs_tol=1e-6)


def test_clamp_motivation_dynamic_upper_bound() -> None:
    assert math.isclose(clamp_motivation(10.0, normal_v_0=1.2, motivation_min=0.1), 3.0)
    assert math.isclose(clamp_motivation(10.0, normal_v_0=0.9, motivation_min=0.1), 4.0)
    assert math.isclose(
        clamp_motivation(0.01, normal_v_0=1.2, motivation_min=0.1), 0.1
    )


def test_mapper_anchor_hits_and_constant_range() -> None:
    mapper = MotivationParameterMapper(
        mapping_block={
            "mapping_function": "logistic",
            "motivation_min": 0.1,
            "desired_speed_anchors": {"low": 0.5, "normal": 1.2, "high": 3.6},
            "time_gap_anchors": {"low": 2.0, "normal": 1.0, "high": 0.01},
            "buffer_anchors": {"low": 1.0, "normal": 0.1, "high": 0.0},
            "strength_neighbor_repulsion_anchors": {
                "low": 0.1,
                "normal": 0.6,
                "high": 0.9,
            },
        },
        normal_v_0=1.2,
        range_default=0.4,
    )

    assert math.isclose(mapper.max_motivation(), 3.0, rel_tol=0.0, abs_tol=1e-12)

    assert math.isclose(mapper.desired_speed(MOTIVATION_LOW), 0.5, abs_tol=1e-6)
    assert math.isclose(mapper.desired_speed(MOTIVATION_NORMAL), 1.2, abs_tol=1e-6)
    assert math.isclose(mapper.desired_speed(MOTIVATION_HIGH), 3.6, abs_tol=1e-6)

    assert math.isclose(mapper.time_gap(MOTIVATION_LOW, 1.0), 2.0, abs_tol=1e-6)
    assert math.isclose(mapper.time_gap(MOTIVATION_NORMAL, 1.0), 1.0, abs_tol=1e-6)
    assert math.isclose(mapper.time_gap(MOTIVATION_HIGH, 1.0), 0.01, abs_tol=1e-6)

    assert math.isclose(mapper.buffer(MOTIVATION_LOW), 1.0, abs_tol=1e-6)
    assert math.isclose(mapper.buffer(MOTIVATION_NORMAL), 0.1, abs_tol=1e-6)
    assert math.isclose(mapper.buffer(MOTIVATION_HIGH), 0.0, abs_tol=1e-9)

    assert math.isclose(
        mapper.strength_neighbor_repulsion(MOTIVATION_LOW), 0.1, abs_tol=1e-6
    )
    assert math.isclose(
        mapper.strength_neighbor_repulsion(MOTIVATION_NORMAL), 0.6, abs_tol=1e-6
    )
    assert math.isclose(
        mapper.strength_neighbor_repulsion(MOTIVATION_HIGH), 0.9, abs_tol=1e-6
    )

    for motivation in [0.1, 0.5, 1.0, 2.5, 3.0]:
        assert math.isclose(
            mapper.range_neighbor_repulsion(motivation), 0.4, abs_tol=1e-12
        )


def test_mapper_monotonicity_for_default_anchors() -> None:
    mapper = MotivationParameterMapper(
        mapping_block={
            "mapping_function": "logistic",
            "motivation_min": 0.1,
            "desired_speed_anchors": {"low": 0.5, "normal": 1.2, "high": 3.6},
            "time_gap_anchors": {"low": 2.0, "normal": 1.0, "high": 0.01},
            "buffer_anchors": {"low": 1.0, "normal": 0.1, "high": 0.0},
            "strength_neighbor_repulsion_anchors": {
                "low": 0.1,
                "normal": 0.6,
                "high": 0.9,
            },
        },
        normal_v_0=1.2,
        range_default=0.4,
    )

    motivations = [0.1, 0.5, 1.0, 2.0, 3.0]
    speeds = [mapper.desired_speed(m) for m in motivations]
    time_gaps = [mapper.time_gap(m, 1.0) for m in motivations]
    buffers = [mapper.buffer(m) for m in motivations]
    strengths = [mapper.strength_neighbor_repulsion(m) for m in motivations]

    assert speeds == sorted(speeds)
    assert strengths == sorted(strengths)
    assert time_gaps == sorted(time_gaps, reverse=True)
    assert buffers == sorted(buffers, reverse=True)


def test_non_monotonic_strength_anchors_raise_explicit_error() -> None:
    with pytest.raises(ValueError, match="must be monotonic"):
        MotivationParameterMapper(
            mapping_block={
                "mapping_function": "logistic",
                "motivation_min": 0.1,
                "desired_speed_anchors": {"low": 0.5, "normal": 1.2, "high": 3.6},
                "time_gap_anchors": {"low": 2.0, "normal": 1.0, "high": 0.01},
                "buffer_anchors": {"low": 1.0, "normal": 0.1, "high": 0.0},
                "strength_neighbor_repulsion_anchors": {
                    "low": 0.9,
                    "normal": 0.6,
                    "high": 0.9,
                },
            },
            normal_v_0=1.2,
            range_default=0.4,
        )


def test_logistic_mapping_keeps_monotonicity() -> None:
    mapper = MotivationParameterMapper(
        mapping_block={
            "mapping_function": "logistic",
            "inflection_target": 1.5,
            "motivation_min": 0.1,
            "desired_speed_anchors": {"low": 0.5, "normal": 1.2, "high": 3.6},
            "time_gap_anchors": {"low": 2.0, "normal": 1.0, "high": 0.01},
            "buffer_anchors": {"low": 1.0, "normal": 0.1, "high": 0.0},
            "strength_neighbor_repulsion_anchors": {
                "low": 0.1,
                "normal": 0.6,
                "high": 0.9,
            },
        },
        normal_v_0=1.2,
        range_default=0.4,
    )
    values = [mapper.desired_speed(m) for m in [0.1, 1.0, 2.0, 3.0]]
    assert values == sorted(values)


def test_manual_logistic_k_is_applied_per_parameter() -> None:
    mapper = MotivationParameterMapper(
        mapping_block={
            "mapping_function": "logistic",
            "inflection_target": 1.0,
            "motivation_min": 0.1,
            "desired_speed_anchors": {"low": 0.5, "normal": 1.2, "high": 3.6},
            "time_gap_anchors": {"low": 2.0, "normal": 1.0, "high": 0.01},
            "buffer_anchors": {"low": 1.0, "normal": 0.1, "high": 0.0},
            "strength_neighbor_repulsion_anchors": {
                "low": 0.1,
                "normal": 0.6,
                "high": 0.9,
            },
            "use_manual_logistic_k": True,
            "logistic_k": {
                "desired_speed": 2.5,
                "time_gap": -1.25,
                "buffer": -2.0,
                "strength_neighbor_repulsion": 1.75,
            },
        },
        normal_v_0=1.2,
        range_default=0.4,
    )
    assert math.isclose(mapper.logistic_parameters["desired_speed"].k, 2.5)
    assert math.isclose(mapper.logistic_parameters["time_gap"].k, -1.25)
    assert math.isclose(mapper.logistic_parameters["buffer"].k, -2.0)
    assert math.isclose(
        mapper.logistic_parameters["strength_neighbor_repulsion"].k, 1.75
    )
