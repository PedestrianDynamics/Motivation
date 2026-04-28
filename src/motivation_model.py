"""Module for motivational model."""

import hashlib
import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .logger_config import log_debug
from .motivation_mapping import MotivationParameterMapper

Point: TypeAlias = Tuple[float, float]


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp 'value' to be within [min_val, max_val]."""
    return max(min_val, min(value, max_val))


class SeedOperation(Enum):
    """Enumeration of different seeding operations for tracking purposes."""

    AGENT_VALUE_ASSIGNMENT = auto()
    GENERAL_RANDOMIZATION = auto()


@dataclass
class SeedManager:
    """Manages seeds for reproducible randomization across different operations."""

    base_seed: int
    _operation_seeds: Dict[SeedOperation, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize operation-specific seeds based on the base seed."""
        for operation in SeedOperation:
            # Create unique seeds for each operation using hash function
            operation_string = f"{self.base_seed}_{operation.name}"
            hash_value = hashlib.md5(operation_string.encode()).hexdigest()
            # Convert first 8 characters of hash to integer
            self._operation_seeds[operation] = int(hash_value[:8], 16)

    def get_operation_seed(
        self, operation: SeedOperation, sub_id: Optional[int] = None
    ) -> int:
        """Get a deterministic seed for a specific operation and optional sub-operation."""
        operation_base = self._operation_seeds[operation]
        if sub_id is not None:
            # Combine operation seed with sub_id for unique but deterministic result
            return operation_base + sub_id * 1000
        return operation_base

    def set_seed_for_operation(
        self, operation: SeedOperation, sub_id: Optional[int] = None
    ) -> None:
        """Set the random seed for a specific operation."""
        random.seed(self.get_operation_seed(operation, sub_id))


class MotivationStrategy(ABC):
    """Abstract class for strategy model."""

    @abstractmethod
    def motivation(self, params: dict[str, Any]) -> float:
        """Return the motivation value."""
        pass

    @abstractmethod
    def plot(self) -> List[Figure]:
        """Plot internal functions for debugging purpose."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return name of the model."""
        pass

    @abstractmethod
    def get_value(self, **kwargs: Any) -> float:
        """Get value of agents."""
        raise NotImplementedError("Subclasses should implement this!")


@dataclass
class DefaultMotivationStrategy(MotivationStrategy):
    """Default strategy for motivation calculation based on distance."""

    width: float = 1.0
    height: float = 1.0
    alpha: float = 1.0

    @staticmethod
    def name() -> str:
        """Give back name of strategy."""
        return "DefaultStrategy"

    def get_value(self, **kwargs: Any) -> float:
        """Random value in interval."""
        return 1.0

    def motivation(self, params: dict[str, Any]) -> float:
        """Motivation based on distance to entrance."""
        distance = params["distance"]
        if distance >= self.width:
            return 0.0

        expr = 1 / ((distance / self.width) ** 2 - 1)
        if np.isinf(expr):
            return 0.0

        return float(np.exp(expr) * np.e * self.height)

    def plot(self) -> List[Figure]:
        """Plot functions of the strategy model."""
        fig = plt.figure()
        distances = np.linspace(0, 10, 100)
        m = []
        for dist in distances:
            m.append(self.motivation({"distance": dist}))

        plt.plot(distances, m)
        plt.grid(alpha=0.3)
        plt.ylim([-0.1, 3])
        plt.xlim([-0.1, 4])
        plt.ylabel("Motivation")
        plt.xlabel("Distance / m")
        plt.title(f"{self.name()} -  M(width, height)")
        return [fig]


@dataclass
class EVPStrategy(MotivationStrategy):
    """Motivation theory based on E.V.C (model4)."""

    VALUE_SCALE_ALPHA: ClassVar[float] = 14.0 / 3.0

    motivation_door_center: Point
    agent_ids: List[int] = field(default_factory=list)
    agent_positions: List[Point] = field(default_factory=list)
    pedestrian_value: Dict[int, float] = field(default_factory=dict)
    width: float = 1.0
    height: float = 1.0
    max_reward: int = 0
    percent: float = 1
    competition_max: float = 1
    competition_decay_reward: float = 5
    seed: int = 1
    min_value: float = 1.0
    max_value: float = 7.0
    nagents: int = 10
    evc: bool = True
    normal_v_0: float = 1.2
    normal_time_gap: float = 1.0
    motivation_change = 1.0  # to strengthen the change of the state. Not yet used!
    seed_manager: Optional[SeedManager] = None
    motivation_min: float = 0.1
    motivation_mode: str = "PVE"
    payoff_k: float = 8.0
    payoff_q0: float = 0.5
    rank_tie_tolerance_m: float = 1e-3
    payoff_update_interval_s: float = 1.0
    payoff_update_interval_steps: int = 1
    payoff_cache: Dict[int, float] = field(default_factory=dict)
    rank_abs_cache: Dict[int, int] = field(default_factory=dict)
    rank_q_cache: Dict[int, float] = field(default_factory=dict)
    _last_rank_update_iteration: int = -1

    def calculate_exit_distance(self, pos: Point) -> float:
        """Calculate distance from a position to the door center.

        Returns absolute distance in spatial units.
        """
        dx = pos[0] - self.motivation_door_center[0]
        dy = pos[1] - self.motivation_door_center[1]
        return math.sqrt(dx * dx + dy * dy)

    def calculate_exit_distance_squared(self, pos: Point) -> float:
        """Squared distance to door center (used for faster ranking)."""
        dx = pos[0] - self.motivation_door_center[0]
        dy = pos[1] - self.motivation_door_center[1]
        return dx * dx + dy * dy

    @staticmethod
    def get_derived_seed(base_seed: int, operation_id: int) -> int:
        """Create a new seed based on the base seed and operation type."""
        return base_seed * 1000 + operation_id

    def __post_init__(self) -> None:
        """Assign each agent an intrinsic value drawn uniformly from [min_value, max_value]."""
        logging.info(f"EVPStrategy post_init: Seed = {self.seed}")
        if not self.agent_positions:
            logging.critical(
                f"Agent positions are not correctly initialised {self.agent_positions = } "
            )
            logging.critical(f"{self.agent_ids = } ")

        if self.seed is not None and self.seed_manager is None:
            self.seed_manager = SeedManager(self.seed)

        logging.info(f"Number of agents {len(self.agent_ids) = }")
        logging.info(f"Number of positions {len(self.agent_positions) = }")
        logging.info(f"Value range: [{self.min_value}, {self.max_value}]")

        for n in self.agent_ids:
            if self.seed_manager:
                self.seed_manager.set_seed_for_operation(
                    SeedOperation.AGENT_VALUE_ASSIGNMENT, sub_id=n
                )
            self.pedestrian_value[n] = self.value(
                self.min_value,
                self.max_value,
                self.get_derived_seed(self.seed, n),
            )

        self.motivation_mode = str(self.motivation_mode).upper()
        if self.motivation_mode not in {"E", "SE", "V", "P", "PVE", "BASE_MODEL"}:
            raise ValueError(
                f"Unknown motivation_mode '{self.motivation_mode}'. "
                "Use one of: E, SE, V, P, PVE, BASE_MODEL."
            )
        self.value_min = self.min_value
        self.value_max = self.max_value

    def configure_payoff_update_interval(self, time_step: float) -> None:
        """Configure payoff rank update cadence from seconds to simulation steps."""
        if time_step <= 0:
            raise ValueError(f"time_step must be > 0. Got {time_step}.")
        self.payoff_update_interval_steps = max(
            1, int(round(self.payoff_update_interval_s / time_step))
        )

    def _rank_tolerance_squared(self, distances_sq: List[float]) -> float:
        """Convert linear tie tolerance (m) to squared-distance tolerance."""
        if not distances_sq:
            return self.rank_tie_tolerance_m * self.rank_tie_tolerance_m
        mean_d = sum(math.sqrt(max(d2, 0.0)) for d2 in distances_sq) / len(distances_sq)
        eps = self.rank_tie_tolerance_m
        return 2.0 * mean_d * eps + eps * eps

    def _compute_rank_and_payoff(
        self, agent_positions: Dict[int, Point], number_agents_in_simulation: int
    ) -> None:
        """Compute rank and payoff for active agents."""
        if number_agents_in_simulation <= 0:
            self.payoff_cache.clear()
            self.rank_abs_cache.clear()
            self.rank_q_cache.clear()
            return

        items = [
            (agent_id, self.calculate_exit_distance_squared(pos))
            for agent_id, pos in agent_positions.items()
        ]
        items.sort(key=lambda x: x[1])
        distances_sq = [d2 for _, d2 in items]
        eps_sq = self._rank_tolerance_squared(distances_sq)

        n_left = max(0, self.max_reward - number_agents_in_simulation)
        new_rank_abs: Dict[int, int] = {}
        new_rank_q: Dict[int, float] = {}
        new_payoff: Dict[int, float] = {}

        if not items:
            self.payoff_cache = new_payoff
            self.rank_abs_cache = new_rank_abs
            self.rank_q_cache = new_rank_q
            return

        current_rank_in_room = 1
        group_start_idx = 0
        prev_d2 = items[0][1]

        for idx, (agent_id, d2) in enumerate(items):
            if abs(d2 - prev_d2) > eps_sq:
                group_start_idx = idx
                current_rank_in_room = group_start_idx + 1
            rank_abs = n_left + current_rank_in_room
            q = (rank_abs - 1) / max(1, self.max_reward - 1)
            payoff = 1.0 / (1.0 + math.exp(self.payoff_k * (q - self.payoff_q0)))
            new_rank_abs[agent_id] = rank_abs
            new_rank_q[agent_id] = q
            new_payoff[agent_id] = payoff
            prev_d2 = d2

        self.payoff_cache = new_payoff
        self.rank_abs_cache = new_rank_abs
        self.rank_q_cache = new_rank_q

    def update_payoff_cache(
        self,
        iteration_count: int,
        agent_positions: Dict[int, Point],
        number_agents_in_simulation: int,
    ) -> bool:
        """Update rank/payoff cache on configured schedule."""
        if iteration_count % self.payoff_update_interval_steps != 0:
            return False
        self._compute_rank_and_payoff(agent_positions, number_agents_in_simulation)
        self._last_rank_update_iteration = iteration_count
        return True

    def get_rank_payoff(self, agent_id: int) -> Tuple[int, float, float]:
        """Return cached rank_abs, rank_q, payoff for an agent."""
        rank_abs = int(self.rank_abs_cache.get(agent_id, self.max_reward))
        rank_q = float(self.rank_q_cache.get(agent_id, 1.0))
        payoff = float(self.payoff_cache.get(agent_id, 0.0))
        return rank_abs, rank_q, payoff

    @staticmethod
    def name() -> str:
        """Give back name of strategy."""
        return "EVPStrategy"

    @staticmethod
    def expectancy(_distance: float, _width: float, _height: float) -> float:
        """Calculate scaled spatial expectancy ES in [0.1, 1.0]."""
        if _height <= 0:
            return 1.0
        if _distance >= _width:
            return 0.1

        expr = 1 / ((_distance / _width) ** 2 - 1)
        if np.isinf(expr):
            return 1.0

        raw = float(1 + np.exp(expr) * np.e * _height)
        # At d=0: expr=-1 => raw_max = 1 + height.
        max_raw = float(1 + _height)
        normalized = float(clamp((raw - 1.0) / (max_raw - 1.0), 0.0, 1.0))
        return float(clamp(0.1 + 0.9 * normalized, 0.1, 1.0))

    @staticmethod
    def competition(N: int, c0: float, N0: float, percent: float, Nmax: float) -> float:
        """
        Compute the value of a function that starts with a constant value c0.

        Then linearly decays to zero starting from N0 and reaching zero at Nmax.

        Parameters:
        - N (int): input value for which the function is evaluated.
        - c0 (float): The initial constant value of the function.
        - N0 (float): The point at which the function starts to linearly decay.
        - percent*Nmax: The value of N at which the function reaches zero.

        Returns:
        - float: The function value corresponding to input value in N.
        """
        max_reward = percent * Nmax
        slope = (c0 - 1) / (max_reward - N0)
        if N <= N0:
            return c0
        elif N < max_reward:
            return c0 - slope * (N - N0)
        else:
            return 1

    def get_value(self, **kwargs: Any) -> float:
        """Random value in interval."""
        ped_id = kwargs.get("agent_id")

        if isinstance(ped_id, int):
            return self.pedestrian_value[ped_id]
        else:
            logging.error("Something went south in get_value: returning 0.0!")
            return 0.0

    @staticmethod
    def value(min_v: float, max_v: float, seed: Optional[float] = None) -> float:
        """Random value in interval. seed is optional."""
        if seed is not None:
            random.seed(seed)

        return random.uniform(min_v, max_v)

    def motivation(self, params: dict[str, Any]) -> float:
        """Define mode-based EVP model."""
        distance = params["distance"]
        agent_id = params["agent_id"]
        if "seed" not in params:
            params["seed"] = None

        value = float(self.pedestrian_value[agent_id])
        V_unit = value / self.VALUE_SCALE_ALPHA
        SE_unit = EVPStrategy.expectancy(
            distance,
            self.width,
            self.height,
        )
        P_unit = float(self.payoff_cache.get(agent_id, 1.0))
        E_unit = SE_unit + P_unit

        if self.motivation_mode == "BASE_MODEL":
            M = 1.0
        elif self.motivation_mode in {"E", "SE"}:
            M = SE_unit
        elif self.motivation_mode == "V":
            M = V_unit
        elif self.motivation_mode == "P":
            M = P_unit
        else:
            M = V_unit * E_unit

        return clamp(M, self.motivation_min, 3.0)

    def plot(self) -> List[Figure]:
        """Plot functions for inspection."""
        fig0, ax0 = plt.subplots(ncols=1, nrows=1)
        fig1, ax1 = plt.subplots(ncols=1, nrows=1)
        fig2, ax2 = plt.subplots(ncols=1, nrows=1)
        fig3, ax3 = plt.subplots(ncols=1, nrows=1)
        fig4, ax4 = plt.subplots(ncols=1, nrows=1)
        distances = np.linspace(0, 10, 100)
        # ES
        E = []
        for dist in distances:
            E.append(self.expectancy(dist, self.width, self.height))

        ax0.plot(distances, E)
        ax0.grid(alpha=0.3)
        ax0.set_ylim((-0.1, 1.2))
        ax0.set_xlim((-0.1, self.width + 1))
        # ax0.set_title(f"{self.name()} - ES (width, height)")
        ax0.set_xlabel("$d$ / m", size=14)
        ax0.set_ylabel(r"$ES(d)$", size=14)
        fig0.savefig("expectancy.pdf")
        # V
        V_abs = [self.get_value(agent_id=s) for s in self.agent_ids]

        x_vals = np.array(self.agent_ids, dtype=float)
        ax1.scatter(
            x_vals,
            np.array(V_abs),
            facecolors="tab:blue",
            edgecolors="tab:blue",
            s=45,
        )
        ax1.axhspan(self.min_value, self.max_value, color="tab:blue", alpha=0.08)
        ax1.grid(alpha=0.3)
        ax1.set_ylim((-0.1, max(V_abs, default=1.0) + 1.0))
        ax1.set_xlim((-0.1, self.max_reward + 1))
        ax1.set_xlabel("# Agents", size=14)
        ax1.set_ylabel("Value V", size=14)
        fig1.savefig("value.pdf")
        # P
        q_vals = np.linspace(0.0, 1.0, 200)
        p_vals = [
            1.0 / (1.0 + math.exp(self.payoff_k * (q - self.payoff_q0))) for q in q_vals
        ]
        ax2.plot(q_vals, p_vals, "-")
        p0 = 1.0 / (1.0 + math.exp(self.payoff_k * (self.payoff_q0 - self.payoff_q0)))
        ax2.axvline(self.payoff_q0, color="red", ls="--", lw=1.2)
        ax2.scatter([self.payoff_q0], [p0], color="red", s=35, zorder=5)
        ax2.grid(alpha=0.3)
        ax2.set_xlim((0, 1))
        ax2.set_ylim((-0.1, 1.1))
        ax2.set_xlabel("Normalized rank q", size=14)
        ax2.set_ylabel(r"$P(q)$", size=14)
        fig2.savefig("payoff.pdf")
        # M
        max_value_id = max(
            self.pedestrian_value, key=lambda k: self.pedestrian_value[k]
        )
        min_value_id = min(
            self.pedestrian_value, key=lambda k: self.pedestrian_value[k]
        )
        # logging.info(
        #     f"id max value {max_value_id}: {self.pedestrian_value[max_value_id]}"
        # )
        # logging.info(
        #     f"id min value {min_value_id}: {self.pedestrian_value[min_value_id]}"
        # )
        N_three = np.linspace(10, self.max_reward, 3)
        symbols = ["-", "--", "-."]

        for i, n in enumerate(N_three[1:2]):
            n = int(n)
            for id_ in self.agent_ids:
                m_max = []
                for dist in distances:
                    params = {
                        "distance": dist,
                        "number_agents_in_simulation": n,
                        "seed": self.seed,
                        "agent_id": id_,
                    }
                    m_max.append(self.motivation(params))
                if id_ == max_value_id:
                    ax3.plot(
                        distances,
                        m_max,
                        linestyle=symbols[i % len(symbols)],
                        color="blue",
                        lw=2,
                        label="max value",
                    )
                else:
                    ax3.plot(distances, m_max, linestyle="-", color="gray", lw=0.08)
        for i, n in enumerate(N_three[1:2]):
            n = int(n)
            m_min = []
            for dist in distances:
                params = {
                    "distance": dist,
                    "number_agents_in_simulation": n,
                    "seed": self.seed,
                    "agent_id": min_value_id,
                }
                m_min.append(self.motivation(params))
            ax3.plot(
                distances,
                m_min,
                linestyle=symbols[i % len(symbols)],
                color="red",
                lw=2,
                label="min value",
            )

        ax3.grid(alpha=0.3)
        # ax3.set_ylim([-0.1, 3])
        ax3.set_xlim((-0.1, self.width + 1))
        ax3.legend()
        title = (
            f"{self.name()} - {self.motivation_mode} "
            f"(N={self.max_reward:.0f}, seed={self.seed:.0f})"
        )
        # ax3.set_title(title)
        ax3.set_xlabel(r"$d$ / m", size=14)
        ax3.set_ylabel("$M$", size=14)
        fig3.savefig("motivation.pdf")
        #
        rank_m = []
        for idx, agent_id in enumerate(self.agent_ids):
            if idx < len(self.agent_positions):
                distance = math.sqrt(
                    self.calculate_exit_distance_squared(self.agent_positions[idx])
                )
            else:
                distance = self.width / 2
            params = {
                "distance": distance,
                "number_agents_in_simulation": self.max_reward,
                "seed": self.seed,
                "agent_id": agent_id,
            }
            rank_abs, _, _ = self.get_rank_payoff(agent_id)
            rank_m.append((int(rank_abs), int(agent_id), self.motivation(params)))

        rank_m.sort(key=lambda item: (item[0], item[1]))
        x_rank = [item[0] for item in rank_m]
        m_sorted = [item[2] for item in rank_m]

        ax4.plot(x_rank, m_sorted, ".-")
        ax4.grid(alpha=0.3)
        # ax3.set_ylim([-0.1, 3])
        # ax3.set_xlim([-0.1, 4])
        ax4.set_title(
            f"{self.name()} | mode={self.motivation_mode} | M sorted by current rank"
        )
        ax4.set_xlabel("Rank (absolute)")
        ax4.set_ylabel("Motivation")

        return [fig0, fig1, fig2, fig3, fig4]


@dataclass
class MotivationModel:
    """Class defining the motivation model."""

    door_point1: Point = (60, 101)
    door_point2: Point = (60, 102)
    normal_v_0: float = 1.2
    normal_time_gap: float = 1.0
    motivation_strategy: Any = None
    parameter_mapper: Optional[MotivationParameterMapper] = None

    def print_details(self) -> None:
        """Print member variables for debugging."""
        log_debug(f"Motivation Model: {self.motivation_strategy.name()}")
        log_debug(f">>  Door Point 1: {self.door_point1}")
        log_debug(f">>  Door Point 2: {self.door_point2}")
        log_debug(f">  Normal Velocity 0: {self.normal_v_0}")
        log_debug(f">>  Normal Time Gap: {self.normal_time_gap}")

    def __post_init__(self) -> None:
        """Init v0 and time gap."""
        if self.normal_v_0 is None:
            self.normal_v_0 = 1.2  # Default value if parsing returns None

        if self.normal_time_gap is None:
            self.normal_time_gap = 1.0  # Default value if parsing returns None

    def calculate_motivation_state(
        self, motivation_i: float, agent_id: int
    ) -> Tuple[float, float]:
        """Return v0, T tuples depending on Motivation. (v0,T)=(1.2,1)."""

        if self.parameter_mapper is not None:
            motivation_i = self.parameter_mapper.clamp_motivation(motivation_i)
            v_0_new = self.parameter_mapper.desired_speed(motivation_i)
            time_gap_new = self.parameter_mapper.time_gap(
                motivation_i, self.normal_time_gap
            )
            return v_0_new, time_gap_new

        v_0_new = self.normal_v_0 * motivation_i
        time_gap_new = self.normal_time_gap / motivation_i
        return v_0_new, time_gap_new

    def plot(self) -> Tuple[Figure, Figure]:
        """Plot model."""
        fig, ax = plt.subplots(ncols=1, nrows=1)
        fig1, ax1 = plt.subplots(ncols=1, nrows=1)
        N_three = np.linspace(10, self.motivation_strategy.max_reward, 3)
        min_value_id = min(
            self.motivation_strategy.pedestrian_value,
            key=lambda k: self.motivation_strategy.pedestrian_value[k],
        )
        max_value_id = max(
            self.motivation_strategy.pedestrian_value,
            key=lambda k: self.motivation_strategy.pedestrian_value[k],
        )
        distances = np.linspace(0, 10, 4)
        symbols = ["-.", "-", "--"]
        for i, n in enumerate(N_three):
            n = int(n)
            for id_ in self.motivation_strategy.agent_ids:
                v0_list = []
                T_list = []
                for dist in distances:
                    params = {
                        "distance": dist,
                        "number_agents_in_simulation": n,
                        "seed": self.motivation_strategy.seed,
                        "agent_id": id_,
                    }
                    motiv = self.motivation_strategy.motivation(params)
                    v_0, time_gap = self.calculate_motivation_state(motiv, id_)
                    # logging.info(f"{n = }, {dist = }")
                    # logging.info(f"{id_ = }, {motiv = }, {v_0 = }, {time_gap = }")
                    v0_list.append(v_0)
                    T_list.append(time_gap)

                if id_ == max_value_id:
                    ax.plot(
                        distances,
                        v0_list,
                        linestyle=symbols[i % len(symbols)],
                        color="blue",
                        label=f"max value, Nmax={n}",
                    )
                    ax1.plot(
                        distances,
                        T_list,
                        linestyle=symbols[i % len(symbols)],
                        color="blue",
                        label=f"max value, Nmax={n}",
                    )
                else:
                    ax.plot(distances, v0_list, linestyle="-", color="gray", lw=0.03)
                    ax1.plot(distances, T_list, linestyle="-", color="gray", lw=0.03)

        for i, n in enumerate(N_three):
            n = int(n)
            # v01 = v0 * self.motivation_strategy.get_value(agent_id=i)
            v0_list = []
            T_list = []
            for dist in distances:
                params = {
                    "distance": dist,
                    "number_agents_in_simulation": n,
                    "seed": self.motivation_strategy.seed,
                    "agent_id": min_value_id,
                }
                motiv = self.motivation_strategy.motivation(params)
                v_0, time_gap = self.calculate_motivation_state(motiv, min_value_id)
                v0_list.append(v_0)
                T_list.append(time_gap)

            ax.plot(
                distances,
                v0_list,
                linestyle=symbols[i % len(symbols)],
                color="red",
                label=f"min value, Nmax={n}",
            )
            ax1.plot(
                distances,
                T_list,
                linestyle=symbols[i % len(symbols)],
                color="red",
                label=f"min value, Nmax={n}",
            )

        ax.grid(alpha=0.3)
        ax.set_title(
            rf"{self.motivation_strategy.name()} - new $v_0$. id at different distance"
        )
        ax.set_xlabel("Distance / m")
        ax.set_ylabel(r"$\tilde v_0$ / m/s")
        ax.legend()
        # ====
        ax1.grid(alpha=0.3)
        ax1.set_title(
            rf"{self.motivation_strategy.name()} - new $T$. id at different distance"
        )
        ax1.set_xlabel("Distance / m")
        ax1.set_ylabel(r"$\tilde T$ / s")
        ax1.legend()

        return fig, fig1
