"""Motivation-to-parameter mappings."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

MOTIVATION_LOW = 0.1
MOTIVATION_NORMAL = 1.0
MOTIVATION_HIGH = 3.0
FIT_EPSILON = 1e-6
DEFAULT_MOTIVATION_MIN = MOTIVATION_LOW

DEFAULT_MAPPING_BLOCK: Dict[str, Any] = {
    "mapping_function": "logistic",
    "motivation_min": DEFAULT_MOTIVATION_MIN,
    "inflection_target": MOTIVATION_NORMAL,
    "desired_speed_anchors": {"low": 0.5, "normal": 1.2, "high": 3.6},
    "time_gap_anchors": {"low": 2.0, "normal": 1.0, "high": 0.01},
    "buffer_anchors": {"low": 1.0, "normal": 0.1, "high": 0.0},
    "strength_neighbor_repulsion_anchors": {"low": 0.0, "normal": 0.1, "high": 0.9},
    "use_manual_logistic_k": False,
    "logistic_k": {
        "desired_speed": 10.0,
        "time_gap": 10.0,
        "buffer": 10.0,
        "strength_neighbor_repulsion": 10.0,
    },
}


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value into [min_value, max_value]."""
    return max(min_value, min(value, max_value))


def clamp_motivation(
    motivation: float,
    normal_v_0: float,
    motivation_min: float = DEFAULT_MOTIVATION_MIN,
) -> float:
    """Clamp motivation to the supported mapping domain."""
    _ = normal_v_0
    return clamp(motivation, motivation_min, MOTIVATION_HIGH)


@dataclass(frozen=True)
class LogisticParams:
    """Parameters for y = y_min + (y_max-y_min)/(1+exp(-k*(x-m0)))."""

    y_min: float
    y_max: float
    k: float
    m0: float


@dataclass(frozen=True)
class AnchorValues:
    """Low/normal/high values."""

    low: float
    normal: float
    high: float

    @classmethod
    def from_dict(
        cls, values: Dict[str, Any], defaults: Dict[str, float]
    ) -> "AnchorValues":
        """Create anchor values from a dict and defaults."""
        return cls(
            low=float(values.get("low", defaults["low"])),
            normal=float(values.get("normal", defaults["normal"])),
            high=float(values.get("high", defaults["high"])),
        )

    def as_dict(self) -> Dict[str, float]:
        """Serialize to plain dictionary."""
        return {"low": self.low, "normal": self.normal, "high": self.high}

    def is_monotonic(self) -> bool:
        """Check if low->normal->high is monotonic."""
        return (self.low <= self.normal <= self.high) or (
            self.low >= self.normal >= self.high
        )


def _logistic_fraction(x: float, k: float, m0: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - m0)))


def _predict_y2_from_k(k: float, anchors: AnchorValues, m0: float) -> float:
    x1, x2, x3 = MOTIVATION_LOW, MOTIVATION_NORMAL, MOTIVATION_HIGH
    s1 = _logistic_fraction(x1, k, m0)
    s2 = _logistic_fraction(x2, k, m0)
    s3 = _logistic_fraction(x3, k, m0)
    denom = s3 - s1
    if abs(denom) < 1e-14:
        raise ValueError("Degenerate logistic fit: s3 == s1.")
    delta = (anchors.high - anchors.low) / denom
    y_min = anchors.low - delta * s1
    return y_min + delta * s2


def fit_logistic_from_anchors(
    anchors: AnchorValues, inflection_target: float
) -> LogisticParams:
    """Fit logistic parameters from three anchors on the active motivation scale."""
    if not anchors.is_monotonic():
        raise ValueError(
            "Anchor values must be monotonic for logistic mapping. "
            f"Got low={anchors.low}, normal={anchors.normal}, high={anchors.high}."
        )

    if not (MOTIVATION_LOW <= inflection_target <= MOTIVATION_HIGH):
        raise ValueError(
            f"inflection_target must be in [{MOTIVATION_LOW}, {MOTIVATION_HIGH}]. "
            f"Got {inflection_target}."
        )

    increasing = anchors.high >= anchors.low
    k_candidates = [i * 0.05 for i in range(1, 4001)]
    if not increasing:
        k_candidates = [-k for k in k_candidates]
    m0_candidates = [
        MOTIVATION_LOW + i * 0.01
        for i in range(
            int(round((MOTIVATION_HIGH - MOTIVATION_LOW) / 0.01)) + 1
        )
    ]

    best_k = k_candidates[0]
    best_m0 = inflection_target
    best_err = float("inf")
    for m0 in m0_candidates:
        for k in k_candidates:
            y2 = _predict_y2_from_k(k, anchors, m0)
            err = abs(y2 - anchors.normal)
            err += 1e-4 * abs(m0 - inflection_target)
            if err < best_err:
                best_err = err
                best_k = k
                best_m0 = m0

    s1 = _logistic_fraction(MOTIVATION_LOW, best_k, best_m0)
    s3 = _logistic_fraction(MOTIVATION_HIGH, best_k, best_m0)
    denom = s3 - s1
    delta = (anchors.high - anchors.low) / denom
    y_min = anchors.low - delta * s1
    y_max = y_min + delta
    return LogisticParams(y_min=y_min, y_max=y_max, k=best_k, m0=best_m0)


def logistic_from_endpoints_and_k(
    low: float, high: float, k: float, m0: float
) -> LogisticParams:
    """Build logistic params from low/high anchor endpoints and chosen k."""
    s1 = _logistic_fraction(MOTIVATION_LOW, k, m0)
    s3 = _logistic_fraction(MOTIVATION_HIGH, k, m0)
    denom = s3 - s1
    if abs(denom) < 1e-12:
        raise ValueError("Manual k is too close to zero and causes degenerate mapping.")
    delta = (high - low) / denom
    y_min = low - delta * s1
    y_max = y_min + delta
    return LogisticParams(y_min=y_min, y_max=y_max, k=k, m0=m0)


def evaluate_logistic(x: float, params: LogisticParams) -> float:
    """Evaluate logistic function."""
    s = _logistic_fraction(x, params.k, params.m0)
    return params.y_min + (params.y_max - params.y_min) * s


@dataclass
class LogisticCurve:
    """Logistic curve with endpoint clamping."""

    anchors: AnchorValues
    x_low: float = MOTIVATION_LOW
    x_normal: float = MOTIVATION_NORMAL
    x_high: float = MOTIVATION_HIGH
    params: LogisticParams | None = None
    _params: LogisticParams = field(init=False, repr=False)
    inflection_target: float = 1.5

    def __post_init__(self) -> None:
        if self.params is not None:
            self._params = self.params
            return

        self._params = fit_logistic_from_anchors(self.anchors, self.inflection_target)

    def evaluate(self, x: float) -> float:
        """Evaluate with hard endpoint clamping."""
        if x <= self.x_low:
            return self.anchors.low
        if x >= self.x_high:
            return self.anchors.high

        y = evaluate_logistic(x, self._params)
        y_min = min(self.anchors.low, self.anchors.normal, self.anchors.high)
        y_max = max(self.anchors.low, self.anchors.normal, self.anchors.high)
        return clamp(y, y_min, y_max)


def merge_mapping_block(motivation_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Merge mapping settings with defaults."""
    merged = copy.deepcopy(DEFAULT_MAPPING_BLOCK)
    for key in (
        "mapping_function",
        "motivation_min",
        "inflection_target",
        "use_manual_logistic_k",
    ):
        if key in motivation_parameters:
            merged[key] = motivation_parameters[key]

    for key in (
        "desired_speed_anchors",
        "time_gap_anchors",
        "buffer_anchors",
        "strength_neighbor_repulsion_anchors",
    ):
        if isinstance(motivation_parameters.get(key), dict):
            merged[key].update(motivation_parameters[key])
    if isinstance(motivation_parameters.get("logistic_k"), dict):
        merged["logistic_k"] = copy.deepcopy(motivation_parameters["logistic_k"])

    return merged


def ensure_mapping_block(motivation_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure mapping defaults exist in config and return merged block."""
    merged = merge_mapping_block(motivation_parameters)
    motivation_parameters.update(merged)
    return merged


@dataclass
class MotivationParameterMapper:
    """Map motivation to CFSM parameters."""

    mapping_block: Dict[str, Any]
    normal_v_0: float
    range_default: float

    def __post_init__(self) -> None:
        block = merge_mapping_block(self.mapping_block)
        self.mapping_function = str(block["mapping_function"])
        if self.mapping_function != "logistic":
            raise ValueError(
                f"Unsupported mapping_function '{self.mapping_function}'. Use 'logistic'."
            )
        self.motivation_min = float(block["motivation_min"])
        self.inflection_target = float(block["inflection_target"])
        self.use_manual_logistic_k = bool(block["use_manual_logistic_k"])

        self.desired_speed_anchors = AnchorValues.from_dict(
            block["desired_speed_anchors"], DEFAULT_MAPPING_BLOCK["desired_speed_anchors"]
        )
        self.time_gap_anchors = AnchorValues.from_dict(
            block["time_gap_anchors"], DEFAULT_MAPPING_BLOCK["time_gap_anchors"]
        )
        self.buffer_anchors = AnchorValues.from_dict(
            block["buffer_anchors"], DEFAULT_MAPPING_BLOCK["buffer_anchors"]
        )
        self.strength_anchors = AnchorValues.from_dict(
            block["strength_neighbor_repulsion_anchors"],
            DEFAULT_MAPPING_BLOCK["strength_neighbor_repulsion_anchors"],
        )

        self.logistic_parameters = {
            "desired_speed": fit_logistic_from_anchors(
                self.desired_speed_anchors, self.inflection_target
            ),
            "time_gap": fit_logistic_from_anchors(
                self.time_gap_anchors, self.inflection_target
            ),
            "buffer": fit_logistic_from_anchors(
                self.buffer_anchors, self.inflection_target
            ),
            "strength_neighbor_repulsion": fit_logistic_from_anchors(
                self.strength_anchors,
                self.inflection_target,
            ),
        }
        if self.use_manual_logistic_k:
            manual_k = block.get("logistic_k", {})
            self.logistic_parameters = self._build_manual_k_params(manual_k)

        self.mapping_block["inflection_target"] = self.inflection_target
        self.mapping_block["use_manual_logistic_k"] = self.use_manual_logistic_k
        self.mapping_block["logistic_k"] = {
            name: params.k for name, params in self.logistic_parameters.items()
        }

        self.speed_curve = LogisticCurve(
            self.desired_speed_anchors,
            params=self.logistic_parameters["desired_speed"],
            inflection_target=self.inflection_target,
        )
        self.time_gap_curve = LogisticCurve(
            self.time_gap_anchors,
            params=self.logistic_parameters["time_gap"],
            inflection_target=self.inflection_target,
        )
        self.buffer_curve = LogisticCurve(
            self.buffer_anchors,
            params=self.logistic_parameters["buffer"],
            inflection_target=self.inflection_target,
        )
        self.strength_curve = LogisticCurve(
            self.strength_anchors,
            params=self.logistic_parameters["strength_neighbor_repulsion"],
            inflection_target=self.inflection_target,
        )

    def _build_manual_k_params(
        self, manual_k: Dict[str, Any]
    ) -> Dict[str, LogisticParams]:
        keys = (
            "desired_speed",
            "time_gap",
            "buffer",
            "strength_neighbor_repulsion",
        )
        anchors_by_key = {
            "desired_speed": self.desired_speed_anchors,
            "time_gap": self.time_gap_anchors,
            "buffer": self.buffer_anchors,
            "strength_neighbor_repulsion": self.strength_anchors,
        }
        params: Dict[str, LogisticParams] = {}
        for key in keys:
            if key not in manual_k:
                raise ValueError(
                    f"Missing manual logistic k for '{key}'. "
                    "Disable manual mode or provide all k values."
                )
            k = float(manual_k[key])
            params[key] = logistic_from_endpoints_and_k(
                low=anchors_by_key[key].low,
                high=anchors_by_key[key].high,
                k=k,
                m0=self.inflection_target,
            )
        return params

    def clamp_motivation(self, motivation: float) -> float:
        """Clamp motivation according to configured lower bound and physical upper bound."""
        return clamp_motivation(
            motivation,
            normal_v_0=self.normal_v_0,
            motivation_min=self.motivation_min,
        )

    def max_motivation(self) -> float:
        """Return normalized maximum motivation."""
        return MOTIVATION_HIGH

    def desired_speed(self, motivation: float) -> float:
        """Desired speed mapping."""
        m = self.clamp_motivation(motivation)
        return self.speed_curve.evaluate(m)

    def time_gap(self, motivation: float, normal_time_gap: float) -> float:
        """Time gap mapping."""
        m = self.clamp_motivation(motivation)
        return self.time_gap_curve.evaluate(m)

    def buffer(self, motivation: float) -> float:
        """Buffer mapping."""
        m = self.clamp_motivation(motivation)
        return self.buffer_curve.evaluate(m)

    def strength_neighbor_repulsion(self, motivation: float) -> float:
        """Neighbor repulsion strength mapping."""
        m = self.clamp_motivation(motivation)
        return self.strength_curve.evaluate(m)

    def range_neighbor_repulsion(self, motivation: float) -> float:
        """Neighbor repulsion range mapping."""
        _ = motivation
        return self.range_default

    def sample_curve_data(
        self, normal_time_gap: float, num_points: int = 200
    ) -> Dict[str, List[float]]:
        """Sample mapping curves for plotting."""
        lower = self.motivation_min
        upper = max(MOTIVATION_HIGH, self.max_motivation())
        step = (upper - lower) / (num_points - 1)
        motivation_values = [lower + step * i for i in range(num_points)]
        return {
            "motivation": motivation_values,
            "desired_speed": [self.desired_speed(float(m)) for m in motivation_values],
            "time_gap": [
                self.time_gap(float(m), normal_time_gap) for m in motivation_values
            ],
            "buffer": [self.buffer(float(m)) for m in motivation_values],
            "strength_neighbor_repulsion": [
                self.strength_neighbor_repulsion(float(m)) for m in motivation_values
            ],
            "range_neighbor_repulsion": [
                self.range_neighbor_repulsion(float(m)) for m in motivation_values
            ],
        }


def plot_parameter_mappings(
    mapper: MotivationParameterMapper, normal_time_gap: float
) -> Any:
    """Plot active motivation-to-parameter mappings."""
    import matplotlib.pyplot as plt

    data = mapper.sample_curve_data(normal_time_gap=normal_time_gap, num_points=300)
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    plots = [
        ("desired_speed", "Desired speed", "m/s"),
        ("time_gap", "Time gap", "s"),
        ("buffer", "Buffer", "m"),
        ("strength_neighbor_repulsion", "Strength neighbor repulsion", "-"),
        ("range_neighbor_repulsion", "Range neighbor repulsion", "m"),
    ]
    anchors_by_key = {
        "desired_speed": mapper.desired_speed_anchors,
        "time_gap": mapper.time_gap_anchors,
        "buffer": mapper.buffer_anchors,
        "strength_neighbor_repulsion": mapper.strength_anchors,
        "range_neighbor_repulsion": AnchorValues(
            low=mapper.range_default,
            normal=mapper.range_default,
            high=mapper.range_default,
        ),
    }

    axes_flat = axes.flatten()
    for idx, (key, title, unit) in enumerate(plots):
        ax = axes_flat[idx]
        ax.plot(data["motivation"], data[key], lw=2)
        ax.set_title(title)
        ax.set_xlabel("Motivation")
        ax.set_ylabel(unit)
        ax.axvline(MOTIVATION_LOW, color="gray", ls="--", lw=0.8)
        ax.axvline(MOTIVATION_NORMAL, color="gray", ls="--", lw=0.8)
        ax.axvline(MOTIVATION_HIGH, color="gray", ls="--", lw=0.8)
        anchors = anchors_by_key[key]
        for y in (anchors.low, anchors.normal, anchors.high):
            ax.axhline(y, color="gray", ls=":", lw=0.8, alpha=0.8)
        if key in mapper.logistic_parameters:
            m0 = mapper.logistic_parameters[key].m0
            if key == "desired_speed":
                y0 = mapper.desired_speed(m0)
            elif key == "time_gap":
                y0 = mapper.time_gap(m0, normal_time_gap)
            elif key == "buffer":
                y0 = mapper.buffer(m0)
            else:
                y0 = mapper.strength_neighbor_repulsion(m0)
            ax.axvline(m0, color="tab:red", ls=":", lw=1.0, alpha=0.9)
            ax.scatter([m0], [y0], color="tab:red", s=28, zorder=5)
            ax.text(
                m0,
                y0,
                f"  m0={m0:.2f}",
                fontsize=8,
                va="bottom",
                color="tab:red",
            )
            y_norm = anchors.normal
            ax.scatter([MOTIVATION_NORMAL], [y_norm], color="tab:green", s=28, zorder=5)
            ax.text(
                MOTIVATION_NORMAL,
                y_norm,
                "  normal anchor",
                fontsize=8,
                va="bottom",
                color="tab:green",
            )

    axes_flat[-1].axis("off")
    fig.suptitle("Active Motivation Parameter Mapping")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig
