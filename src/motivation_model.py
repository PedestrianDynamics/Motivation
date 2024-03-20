"""Module for motivational model."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, TypeAlias, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .logger_config import log_debug
import logging

Point: TypeAlias = Tuple[float, float]


class MotivationStrategy(ABC):
    @abstractmethod
    def motivation(self, params: dict[str, Any]) -> float:
        pass

    @abstractmethod
    def plot(self) -> List[Figure]:
        pass

    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def get_value(self, **kwargs) -> float:
        pass


@dataclass
class DefaultMotivationStrategy(MotivationStrategy):
    """Default strategy for motivation calculation based on distance."""

    width: float = 1.0
    height: float = 1.0

    @staticmethod
    def name() -> str:
        """Give back name of strategy."""
        return "DefaultStrategy"

    def get_value(self, **kwargs) -> float:
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

    agent_ids: List[str] = field(default_factory=list)
    pedestrian_value: Dict[int, float] = field(default_factory=dict)
    width: float = 1.0
    height: float = 1.0
    max_reward: int = 0
    percent: float = 1
    competition_max: float = 1
    competition_decay_reward: float = 5
    seed: int = 0
    min_value: float = 0
    max_value: float = 1
    nagents: int = 10
    evc: bool = True

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)

        for n in self.agent_ids:
            self.pedestrian_value[n] = self.value(self.min_value, self.max_value)

    @staticmethod
    def name() -> str:
        """Give back name of strategy."""
        return "EVCStrategy"

    @staticmethod
    def expectancy(_distance: float, _width: float, _height: float) -> float:
        """Calculate Expectancy depends on the distance to the entrance."""
        if _distance >= _width:
            return 0.0

        expr = 1 / ((_distance / _width) ** 2 - 1)
        if np.isinf(expr):
            return 0.0

        return float(np.exp(expr) * np.e * _height)

    @staticmethod
    def competition(N: int, c0: float, N0: float, percent: float, Nmax: float) -> float:
        """
        Computes the value of a function that starts with a constant value c0,
        then linearly decays to zero starting from N0 and reaching zero at Nmax.

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

    def get_value(self, **kwargs) -> float:
        """Random value in interval."""
        ped_id = kwargs.get("agent_id")
        return self.pedestrian_value[ped_id]

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
        return float(
            value
            * EVCStrategy.competition(
                N=got_reward,
                c0=self.competition_max,
                N0=self.competition_decay_reward,
                percent=self.percent,
                Nmax=self.max_reward,
            )
            * EVCStrategy.expectancy(
                distance,
                self.width,
                self.height,
            )
        )

    def plot(self) -> List[Figure]:
        """Plot functions for inspection."""
        fig0, ax0 = plt.subplots(ncols=1, nrows=1)
        fig1, ax1 = plt.subplots(ncols=1, nrows=1)
        fig2, ax2 = plt.subplots(ncols=1, nrows=1)
        fig3, ax3 = plt.subplots(ncols=1, nrows=1)
        distances = np.linspace(0, 10, 100)
        # E
        E = []
        for dist in distances:
            E.append(self.expectancy(dist, self.width, self.height))

        ax0.plot(distances, E)
        ax0.grid(alpha=0.3)
        ax0.set_ylim([-0.1, 3])
        ax0.set_xlim([-0.1, 4])
        ax0.set_title(f"{self.name()} - E (width, height)")
        ax0.set_xlabel("Distance / m")
        ax0.set_ylabel("Expectancy")
        # V
        V = []
        for s in self.agent_ids:
            V.append(self.get_value(agent_id=s))

        ax1.plot(self.agent_ids, V, "o")
        ax1.grid(alpha=0.3)
        ax1.set_ylim([-0.1, 5])
        ax1.set_xlim([-0.1, self.max_reward + 1])
        ax1.set_title(f"{self.name()} - V (seed = {self.seed:.0f})")
        ax1.set_xlabel("# Agents")
        ax1.set_ylabel("Value")
        # C
        C = []
        N = np.arange(0, self.max_reward + 1)

        for n in N:
            C.append(
                self.competition(
                    N=n,
                    c0=self.competition_max,
                    N0=self.competition_decay_reward,
                    percent=self.percent,
                    Nmax=self.max_reward,
                )
            )

        ax2.plot(N, C, ".-")
        ax2.grid(alpha=0.3)
        ax2.set_xlim([0, self.max_reward + 1])
        ax2.set_ylim([-0.1, self.competition_max + 1])
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
        max_value_id = max(self.pedestrian_value, key=self.pedestrian_value.get)
        min_value_id = min(self.pedestrian_value, key=self.pedestrian_value.get)
        logging.info(
            f"id max value {max_value_id}: {self.pedestrian_value[max_value_id]}"
        )
        logging.info(
            f"id min value {min_value_id}: {self.pedestrian_value[min_value_id]}"
        )
        N = np.linspace(10, self.max_reward * self.percent, 3)
        symbols = ["-", "--", "-."]
        for i, n in enumerate(N):
            n = int(n)
            m_max = []
            print(int(N[-1]))
            for dist in distances:
                params = {
                    "distance": dist,
                    "number_agents_in_simulation": n,
                    "seed": self.seed,
                    "agent_id": max_value_id,
                }
                m_max.append(self.motivation(params))
            ax3.plot(
                distances,
                m_max,
                linestyle=symbols[i % len(symbols)],
                color="blue",
                label=f"max value, Nmax={n}",
            )
        for i, n in enumerate(N):
            n = int(n)
            m_min = []
            print(int(N[-1]))
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
        ax3.set_xlim([-0.1, 4])
        ax3.legend()
        if self.evc:
            title = (
                f"{self.name()} - E.V.C (N={self.max_reward:.0f}, seed={self.seed:.0f})"
            )
        else:
            title = f"EC-V -  E.C-V (N={self.max_reward:.0f}, seed={self.seed:.0f})"
        ax3.set_title(title)
        ax3.set_xlabel("Agent ids")
        ax3.set_ylabel("Motivation")

        return [fig0, fig1, fig2, fig3]


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
            self.normal_time_gap = 1  # Default value if parsing returns None

    def calculate_motivation_state(self, motivation_i: float) -> Tuple[float, float]:
        """Return v0, T tuples depending on Motivation. (v0,T)=(1.2,1)."""
        v_0 = self.normal_v_0
        time_gap = self.normal_time_gap
        v_0_new = (1 + motivation_i) * v_0
        time_gap_new = time_gap / (1 + motivation_i)

        return v_0_new, time_gap_new
