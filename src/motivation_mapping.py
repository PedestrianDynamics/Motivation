"""Motivation-to-parameter mappings."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

MOTIVATION_LOW = 0.1
MOTIVATION_NORMAL = 1.0
MOTIVATION_HIGH = 3.0
FIT_EPSILON = 1e-6

DEFAULT_MAPPING_BLOCK: Dict[str, Any] = {
    "mapping_function": "gompertz",
    "motivation_min": MOTIVATION_LOW,
    "fit_mode": "exact_anchors",
    "inflection_target": 1.5,
    "desired_speed_anchors": {"low": 0.5, "normal": 1.2, "high": 3.6},
    "time_gap_anchors": {"low": 2.0, "normal": 1.0, "high": 0.01},
    "buffer_anchors": {"low": 1.0, "normal": 0.1, "high": 0.0},
    "repulsion_strength_mode": "config_bounds",
    "range_neighbor_repulsion_mode": "constant_d_ped",
    "use_manual_gompertz_parameters": False,
    "gompertz_parameters": {},
}


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value into [min_value, max_value]."""
    return max(min_value, min(value, max_value))


def clamp_motivation(
    motivation: float, normal_v_0: float, motivation_min: float = MOTIVATION_LOW
) -> float:
    """Clamp motivation with dynamic physical upper limit."""
    if normal_v_0 <= 0:
        raise ValueError(f"normal_v_0 must be positive. Got {normal_v_0}")

    max_motivation = 3.6 / normal_v_0
    return clamp(motivation, motivation_min, max_motivation)


@dataclass(frozen=True)
class GompertzParams:
    """Parameters for y = A*exp(-B*exp(-C*x))."""

    a: float
    b: float
    c: float


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


def _gompertz_difference(
    asymptote: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> float:
    """Return C12-C23 for a given asymptote A."""
    s1 = -math.log(y1 / asymptote)
    s2 = -math.log(y2 / asymptote)
    s3 = -math.log(y3 / asymptote)

    c12 = math.log(s1 / s2) / (x2 - x1)
    c23 = math.log(s2 / s3) / (x3 - x2)
    return c12 - c23


def fit_gompertz_exact(points: Iterable[Tuple[float, float]]) -> GompertzParams:
    """Fit Gompertz parameters exactly through three positive points."""
    sorted_points = sorted((float(x), float(y)) for x, y in points)
    if len(sorted_points) != 3:
        raise ValueError("fit_gompertz_exact requires exactly 3 points.")

    x1, y1 = sorted_points[0]
    x2, y2 = sorted_points[1]
    x3, y3 = sorted_points[2]

    if x1 == x2 or x2 == x3 or x1 == x3:
        raise ValueError("x values must be distinct.")

    if y1 <= 0 or y2 <= 0 or y3 <= 0:
        raise ValueError("y values must be strictly positive.")

    a_min = max(y1, y2, y3) * (1 + 1e-9)
    a_low = a_min
    f_low = _gompertz_difference(a_low, x1, y1, x2, y2, x3, y3)

    if abs(f_low) < 1e-12:
        a_star = a_low
    else:
        a_high = a_low * 2.0
        bracketed = False
        for _ in range(120):
            f_high = _gompertz_difference(a_high, x1, y1, x2, y2, x3, y3)
            if abs(f_high) < 1e-12:
                a_star = a_high
                bracketed = True
                break

            if f_low * f_high < 0:
                bracketed = True
                break

            a_high *= 2.0

        if not bracketed:
            raise ValueError("Could not bracket Gompertz asymptote for exact fit.")

        if "a_star" not in locals():
            for _ in range(200):
                a_mid = 0.5 * (a_low + a_high)
                f_mid = _gompertz_difference(a_mid, x1, y1, x2, y2, x3, y3)
                if abs(f_mid) < 1e-14:
                    a_low = a_high = a_mid
                    break
                if f_low * f_mid < 0:
                    a_high = a_mid
                else:
                    a_low = a_mid
                    f_low = f_mid

            a_star = 0.5 * (a_low + a_high)

    s1 = -math.log(y1 / a_star)
    s2 = -math.log(y2 / a_star)
    s3 = -math.log(y3 / a_star)

    c12 = math.log(s1 / s2) / (x2 - x1)
    c23 = math.log(s2 / s3) / (x3 - x2)
    c_star = 0.5 * (c12 + c23)
    b_star = s1 * math.exp(c_star * x1)

    return GompertzParams(a=a_star, b=b_star, c=c_star)


def evaluate_gompertz(x: float, params: GompertzParams) -> float:
    """Evaluate Gompertz function."""
    return params.a * math.exp(-params.b * math.exp(-params.c * x))


@dataclass
class GompertzCurve:
    """Gompertz curve with exact endpoint clamping."""

    anchors: AnchorValues
    x_low: float = MOTIVATION_LOW
    x_normal: float = MOTIVATION_NORMAL
    x_high: float = MOTIVATION_HIGH
    params: GompertzParams | None = None
    _params: GompertzParams = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.params is not None:
            self._params = self.params
            return

        self._params = estimate_gompertz_from_anchors(self.anchors)

    def evaluate(self, x: float) -> float:
        """Evaluate with hard endpoint clamping."""
        if x <= self.x_low:
            return self.anchors.low
        if x >= self.x_high:
            return self.anchors.high

        y = evaluate_gompertz(x, self._params)
        y_min = min(self.anchors.low, self.anchors.high)
        y_max = max(self.anchors.low, self.anchors.high)
        return clamp(y, y_min, y_max)


def estimate_gompertz_from_anchors(anchors: AnchorValues) -> GompertzParams:
    """Estimate Gompertz params from low/normal/high anchors."""
    if not anchors.is_monotonic():
        raise ValueError(
            "Anchor values must be monotonic for Gompertz mapping. "
            f"Got low={anchors.low}, normal={anchors.normal}, high={anchors.high}."
        )

    fit_points = [
        (MOTIVATION_LOW, max(anchors.low, FIT_EPSILON)),
        (MOTIVATION_NORMAL, max(anchors.normal, FIT_EPSILON)),
        (MOTIVATION_HIGH, max(anchors.high, FIT_EPSILON)),
    ]
    return fit_gompertz_exact(fit_points)


def estimate_gompertz_sigmoid_preferred(
    anchors: AnchorValues, inflection_target: float
) -> GompertzParams:
    """Approximate Gompertz fit with an inflection target in the plotted range."""
    if not anchors.is_monotonic():
        raise ValueError(
            "Anchor values must be monotonic for Gompertz mapping. "
            f"Got low={anchors.low}, normal={anchors.normal}, high={anchors.high}."
        )

    x_vals = [MOTIVATION_LOW, MOTIVATION_NORMAL, MOTIVATION_HIGH]
    y_vals = [anchors.low, anchors.normal, anchors.high]

    increasing = anchors.high >= anchors.low
    c_sign = 1.0 if increasing else -1.0
    best: GompertzParams | None = None
    best_err = float("inf")
    max_y = max(y_vals)

    for c_mag in [0.05 + i * 0.01 for i in range(1, 501)]:
        c = c_sign * c_mag
        # x_inflection = ln(B) / C => choose B from target inflection.
        b = math.exp(c * inflection_target)
        k_vals = [math.exp(-b * math.exp(-c * x)) for x in x_vals]
        denom = sum(k * k for k in k_vals)
        if denom <= 0:
            continue
        a = sum(k * y for k, y in zip(k_vals, y_vals)) / denom
        a = max(a, max_y * (1 + 1e-9))

        preds = [a * k for k in k_vals]
        err = sum((p - y) ** 2 for p, y in zip(preds, y_vals))
        if err < best_err:
            best_err = err
            best = GompertzParams(a=a, b=b, c=c)

    if best is None:
        raise ValueError("Could not estimate sigmoid-preferred Gompertz parameters.")
    return best


def merge_mapping_block(motivation_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Merge mapping settings with defaults."""
    merged = copy.deepcopy(DEFAULT_MAPPING_BLOCK)
    for key in (
        "mapping_function",
        "motivation_min",
        "fit_mode",
        "inflection_target",
        "repulsion_strength_mode",
        "range_neighbor_repulsion_mode",
        "use_manual_gompertz_parameters",
    ):
        if key in motivation_parameters:
            merged[key] = motivation_parameters[key]

    for key in ("desired_speed_anchors", "time_gap_anchors", "buffer_anchors"):
        if isinstance(motivation_parameters.get(key), dict):
            merged[key].update(motivation_parameters[key])

    if isinstance(motivation_parameters.get("gompertz_parameters"), dict):
        merged["gompertz_parameters"] = copy.deepcopy(
            motivation_parameters["gompertz_parameters"]
        )

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
    strength_default: float
    strength_min: float
    strength_max: float
    range_default: float

    def __post_init__(self) -> None:
        block = merge_mapping_block(self.mapping_block)
        self.mapping_function = str(block["mapping_function"])
        self.motivation_min = float(block["motivation_min"])
        self.fit_mode = str(block["fit_mode"])
        self.inflection_target = float(block["inflection_target"])
        self.repulsion_strength_mode = str(block["repulsion_strength_mode"])
        self.range_neighbor_repulsion_mode = str(
            block["range_neighbor_repulsion_mode"]
        )
        self.use_manual_gompertz_parameters = bool(
            block["use_manual_gompertz_parameters"]
        )

        self.desired_speed_anchors = AnchorValues.from_dict(
            block["desired_speed_anchors"], DEFAULT_MAPPING_BLOCK["desired_speed_anchors"]
        )
        self.time_gap_anchors = AnchorValues.from_dict(
            block["time_gap_anchors"], DEFAULT_MAPPING_BLOCK["time_gap_anchors"]
        )
        self.buffer_anchors = AnchorValues.from_dict(
            block["buffer_anchors"], DEFAULT_MAPPING_BLOCK["buffer_anchors"]
        )
        self.strength_anchors = AnchorValues(
            low=float(self.strength_min),
            normal=float(self.strength_default),
            high=float(self.strength_max),
        )

        self.estimated_gompertz_parameters = {
            "desired_speed": self._estimate_curve_params(self.desired_speed_anchors),
            "time_gap": self._estimate_curve_params(self.time_gap_anchors),
            "buffer": self._estimate_curve_params(self.buffer_anchors),
            "strength_neighbor_repulsion": self._estimate_curve_params(
                self.strength_anchors
            ),
        }

        manual_parameters = block.get("gompertz_parameters", {})
        self.gompertz_parameters = self._resolve_active_parameters(manual_parameters)
        self.mapping_block["gompertz_parameters"] = self.gompertz_parameters_as_dict()
        self.mapping_block["use_manual_gompertz_parameters"] = (
            self.use_manual_gompertz_parameters
        )
        self.mapping_block["fit_mode"] = self.fit_mode
        self.mapping_block["inflection_target"] = self.inflection_target

        self.speed_curve = GompertzCurve(
            self.desired_speed_anchors, params=self.gompertz_parameters["desired_speed"]
        )
        self.time_gap_curve = GompertzCurve(
            self.time_gap_anchors, params=self.gompertz_parameters["time_gap"]
        )
        self.buffer_curve = GompertzCurve(
            self.buffer_anchors, params=self.gompertz_parameters["buffer"]
        )
        self.strength_curve = GompertzCurve(
            self.strength_anchors,
            params=self.gompertz_parameters["strength_neighbor_repulsion"],
        )

    def _resolve_active_parameters(
        self, manual_parameters: Dict[str, Any]
    ) -> Dict[str, GompertzParams]:
        """Use manual parameters when enabled, otherwise use estimated values."""
        if not self.use_manual_gompertz_parameters:
            return copy.deepcopy(self.estimated_gompertz_parameters)

        expected = (
            "desired_speed",
            "time_gap",
            "buffer",
            "strength_neighbor_repulsion",
        )
        active: Dict[str, GompertzParams] = {}
        for name in expected:
            params = manual_parameters.get(name)
            if not isinstance(params, dict):
                raise ValueError(
                    f"Missing manual Gompertz parameters for '{name}'. "
                    "Disable manual mode or provide a,b,c."
                )
            try:
                active[name] = GompertzParams(
                    a=float(params["a"]),
                    b=float(params["b"]),
                    c=float(params["c"]),
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid manual Gompertz parameters for '{name}'. Expected numeric a,b,c."
                ) from exc
        return active

    def _estimate_curve_params(self, anchors: AnchorValues) -> GompertzParams:
        """Estimate Gompertz params according to fit mode."""
        if self.fit_mode == "exact_anchors":
            return estimate_gompertz_from_anchors(anchors)
        if self.fit_mode == "sigmoid_preferred":
            return estimate_gompertz_sigmoid_preferred(
                anchors, inflection_target=self.inflection_target
            )
        raise ValueError(
            f"Unknown fit_mode '{self.fit_mode}'. Use 'exact_anchors' or 'sigmoid_preferred'."
        )

    def gompertz_parameters_as_dict(self) -> Dict[str, Dict[str, float]]:
        """Serialize active Gompertz parameters."""
        return {
            name: {"a": p.a, "b": p.b, "c": p.c}
            for name, p in self.gompertz_parameters.items()
        }

    def estimated_gompertz_parameters_as_dict(self) -> Dict[str, Dict[str, float]]:
        """Serialize estimated Gompertz parameters."""
        return {
            name: {"a": p.a, "b": p.b, "c": p.c}
            for name, p in self.estimated_gompertz_parameters.items()
        }

    def clamp_motivation(self, motivation: float) -> float:
        """Clamp motivation according to configured lower bound and physical upper bound."""
        return clamp_motivation(
            motivation,
            normal_v_0=self.normal_v_0,
            motivation_min=self.motivation_min,
        )

    def max_motivation(self) -> float:
        """Return dynamic maximum motivation."""
        return 3.6 / self.normal_v_0

    def desired_speed(self, motivation: float) -> float:
        """Desired speed mapping."""
        m = self.clamp_motivation(motivation)
        if self.mapping_function == "gompertz":
            return self.speed_curve.evaluate(m)
        return self.normal_v_0 * m

    def time_gap(self, motivation: float, normal_time_gap: float) -> float:
        """Time gap mapping."""
        m = self.clamp_motivation(motivation)
        if self.mapping_function == "gompertz":
            return self.time_gap_curve.evaluate(m)
        return normal_time_gap / m

    def buffer(self, motivation: float) -> float:
        """Buffer mapping."""
        m = self.clamp_motivation(motivation)
        if self.mapping_function == "gompertz":
            return self.buffer_curve.evaluate(m)
        t = (m - 0.5) / (3.0 - 0.5)
        t = clamp(t, 0.0, 1.0)
        return 1.5 - (1.5 - 0.1) * t

    def strength_neighbor_repulsion(self, motivation: float) -> float:
        """Neighbor repulsion strength mapping."""
        m = self.clamp_motivation(motivation)
        if self.mapping_function == "gompertz":
            if self.repulsion_strength_mode == "config_bounds":
                return self.strength_curve.evaluate(m)
            return self.strength_default
        return self.strength_min + (self.strength_max - self.strength_min) * 0.5 * (
            m - 1.0
        )

    def range_neighbor_repulsion(self, motivation: float) -> float:
        """Neighbor repulsion range mapping."""
        if self.range_neighbor_repulsion_mode == "constant_d_ped":
            return self.range_default
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
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)

    plots = [
        ("desired_speed", "Desired speed", "m/s"),
        ("time_gap", "Time gap", "s"),
        ("buffer", "Buffer", "m"),
        ("strength_neighbor_repulsion", "Strength neighbor repulsion", "-"),
        ("range_neighbor_repulsion", "Range neighbor repulsion", "m"),
    ]

    axes_flat = axes.flatten()
    for idx, (key, title, unit) in enumerate(plots):
        ax = axes_flat[idx]
        ax.plot(data["motivation"], data[key], lw=2)
        ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Motivation")
        ax.set_ylabel(unit)
        ax.axvline(MOTIVATION_LOW, color="gray", ls="--", lw=0.8)
        ax.axvline(MOTIVATION_NORMAL, color="gray", ls="--", lw=0.8)
        ax.axvline(MOTIVATION_HIGH, color="gray", ls="--", lw=0.8)

    axes_flat[-1].axis("off")
    fig.suptitle("Active Motivation Parameter Mapping")
    return fig
