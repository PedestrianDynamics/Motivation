"""Module for motivational model."""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .logger_config import log_debug

Point: TypeAlias = Tuple[float, float]


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
        pass


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
class EVCStrategy(MotivationStrategy):
    """Motivation theory based on E.V.C (model4)."""

    agent_ids: List[int] = field(default_factory=list)
    pedestrian_value: Dict[int, float] = field(default_factory=dict)
    width: float = 1.0
    height: float = 1.0
    max_reward: int = 0
    percent: float = 1
    competition_max: float = 1
    competition_decay_reward: float = 5
    seed: int = 0
    min_value_high: float = 0.5
    max_value_high: float = 1
    min_value_low: float = 0
    max_value_low: float = 0.5
    number_high_value: int = 10
    nagents: int = 10
    evc: bool = True
    alpha = 1.0

    def __post_init__(self) -> None:
        """Initialize array pedestrian_value with random values in min max interval."""
        if self.seed is not None:
            random.seed(self.seed)

        if self.number_high_value > self.nagents:
            logging.warning(
                f"Configuration error: {self.number_high_value} > {self.nagents}. Change value to match!"
            )
            self.number_high_value = self.nagents

        high_value_agents = set(random.sample(self.agent_ids, self.number_high_value))
        for n in self.agent_ids:
            if n in high_value_agents:
                # This agent gets a high value
                self.pedestrian_value[n] = self.value(
                    self.min_value_high, self.max_value_high
                )
            else:
                # This agent gets a low value
                self.pedestrian_value[n] = self.value(
                    self.min_value_low, self.max_value_low
                )

    @staticmethod
    def name() -> str:
        """Give back name of strategy."""
        return "EVCStrategy"

    @staticmethod
    def expectancy(_distance: float, _width: float, _height: float) -> float:
        """Calculate Expectancy depending on the distance to the entrance."""
        if _distance >= _width:
            return 0.0

        expr = 1 / ((_distance / _width) ** 2 - 1)
        if np.isinf(expr):
            return 0.0

        return float(np.exp(expr) * np.e * _height)

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
        slope = c0 / (max_reward - N0)
        if N <= N0:
            return c0
        elif N < max_reward:
            return c0 - slope * (N - N0)
        else:
            return 0

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
        """Define EVC model."""
        number_agents_in_simulation = params["number_agents_in_simulation"]
        distance = params["distance"]
        agent_id = params["agent_id"]
        got_reward = self.max_reward - number_agents_in_simulation
        if "seed" not in params:
            params["seed"] = None

        value = self.pedestrian_value[agent_id] if self.evc else 1.0
        V_unit = value / self.max_value_high
        C_unit = EVCStrategy.competition(
            N=got_reward,
            c0=self.competition_max,
            N0=self.competition_decay_reward,
            percent=self.percent,
            Nmax=self.max_reward,
        )
        E_unit = EVCStrategy.expectancy(
            distance,
            self.width,
            self.height,
        )

        M = V_unit * E_unit * C_unit
        return M

    def plot(self) -> List[Figure]:
        """Plot functions for inspection."""
        fig0, ax0 = plt.subplots(ncols=1, nrows=1)
        fig1, ax1 = plt.subplots(ncols=1, nrows=1)
        fig2, ax2 = plt.subplots(ncols=1, nrows=1)
        fig3, ax3 = plt.subplots(ncols=1, nrows=1)
        fig4, ax4 = plt.subplots(ncols=1, nrows=1)
        distances = np.linspace(0, 10, 100)
        # E
        E = []
        for dist in distances:
            E.append(self.expectancy(dist, self.width, self.height))

        ax0.plot(distances, E)
        ax0.grid(alpha=0.3)
        ax0.set_ylim((-0.1, 3))
        ax0.set_xlim((-0.1, 4))
        ax0.set_title(f"{self.name()} - E (width, height)")
        ax0.set_xlabel("Distance / m")
        ax0.set_ylabel("Expectancy")
        # V
        V = []
        for s in self.agent_ids:
            V.append(self.get_value(agent_id=s))

        ax1.plot(self.agent_ids, V, "o")
        ax1.grid(alpha=0.3)
        ax1.set_ylim((-0.1, 5))
        ax1.set_xlim((-0.1, self.max_reward + 1))
        ax1.set_title(f"{self.name()} - V (seed = {self.seed:.0f})")
        ax1.set_xlabel("# Agents")
        ax1.set_ylabel("Value")
        # C
        C = []
        Nrange = np.arange(0, self.max_reward + 1)

        for n in Nrange:
            C.append(
                self.competition(
                    N=n,
                    c0=self.competition_max,
                    N0=self.competition_decay_reward,
                    percent=self.percent,
                    Nmax=self.max_reward,
                )
            )

        ax2.plot(Nrange, C, ".-")
        ax2.grid(alpha=0.3)
        ax2.set_xlim((0, self.max_reward + 1))
        ax2.set_ylim((-0.1, self.competition_max + 1))
        ax2.set_xlabel("#agents left simulation")
        ax2.set_ylabel("Competition")
        ax2.set_xticks(
            [0, self.competition_decay_reward, self.max_reward * self.percent],
            labels=[
                "0",
                f"N0={self.competition_decay_reward}",
                f"Nmax={ self.max_reward * self.percent:.0f}",
            ],
        )
        ax2.set_yticks(
            [0, self.competition_max], labels=["0", f"Max={self.competition_max:.0f}"]
        )
        ax2.set_title(
            f"{self.name()} - C ({self.percent*100:.0f}% of max reward {self.max_reward:.0f})"
        )
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
        N_three = np.linspace(10, self.max_reward * self.percent, 3)
        symbols = ["--", "-", "-."]
        for i, n in enumerate(N_three):
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
                        label=f"max value, Nmax={n}",
                    )
                else:
                    ax3.plot(distances, m_max, linestyle="-", color="gray", lw=0.08)
        for i, n in enumerate(N_three):
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
                label=f"min value, Nmax={n}",
            )

        ax3.grid(alpha=0.3)
        # ax3.set_ylim([-0.1, 3])
        ax3.set_xlim((-0.1, 4))
        ax3.legend()
        if self.evc:
            title = (
                f"{self.name()} - E.V.C (N={self.max_reward:.0f}, seed={self.seed:.0f})"
            )
        else:
            title = f"EC-V -  E.C-V (N={self.max_reward:.0f}, seed={self.seed:.0f})"
        ax3.set_title(title)
        ax3.set_xlabel("Distance / m")
        ax3.set_ylabel("Motivation")
        #
        M = []
        distance = self.width / 2
        for agent_id, distance in zip(self.agent_ids, distances):
            params = {
                "distance": distance,
                "number_agents_in_simulation": self.max_reward,
                "seed": self.seed,
                "agent_id": agent_id,
            }
            M.append(self.motivation(params))

        ax4.plot(self.agent_ids, M, ".-")
        ax4.grid(alpha=0.3)
        # ax3.set_ylim([-0.1, 3])
        # ax3.set_xlim([-0.1, 4])
        ax4.set_title(
            f"{self.name()} - E.V.C (N={self.max_reward:.0f}). Each id at different distance"
        )
        ax4.set_xlabel("Agent ids")
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
        v_0 = self.normal_v_0 * self.motivation_strategy.get_value(agent_id=agent_id)

        time_gap = self.normal_time_gap
        v_0_new = v_0 * (1 + self.motivation_strategy.alpha * motivation_i)
        time_gap_new = time_gap / (1 + self.motivation_strategy.alpha * motivation_i)
        return v_0_new, time_gap_new

    def plot(self) -> Tuple[Figure, Figure]:
        """Plot model."""
        fig, ax = plt.subplots(ncols=1, nrows=1)
        fig1, ax1 = plt.subplots(ncols=1, nrows=1)
        N_three = np.linspace(
            10,
            self.motivation_strategy.max_reward * self.motivation_strategy.percent,
            3,
        )
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
