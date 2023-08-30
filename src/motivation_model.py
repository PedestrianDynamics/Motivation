"""
Module for motivational model
"""
from typing import Tuple, TypeAlias, Protocol, Optional, Any
from dataclasses import dataclass
import numpy as np
import random
from .profiles import ParameterGrid
from .logger_config import log_debug

Point: TypeAlias = Tuple[float, float]


@dataclass
class DefaultMotivationStrategy:
    """Default strategy for motivation calculation based on distance."""

    width: float = 1.0
    height: float = 1.0

    def motivation(self, params: dict[str, Any]) -> float:
        """Motivation based on distance to entrance."""
        distance = params["distance"]
        if distance >= self.width:
            return 0.0

        expr = 1 / ((distance / self.width) ** 2 - 1)
        if np.isinf(expr):
            return 0.0

        return float(np.exp(expr) * np.e * self.height)


@dataclass
class EVCStrategy:
    """Motivation theory based on E.V.C (model4)"""

    width: float = 1.0
    height: float = 1.0
    max_reward: int = 0
    seed: int = 0
    min_value: int = 0
    max_value: int = 1

    @staticmethod
    def expectancy(_distance: float, _width: float, _height: float) -> float:
        """Expectancy depends on the distance to the entrance."""
        if _distance >= _width:
            return 0.0

        expr = 1 / ((_distance / _width) ** 2 - 1)
        if np.isinf(expr):
            return 0.0

        return float(np.exp(expr) * np.e * _height)

    @staticmethod
    def competition(got_reward: int, max_reward: int) -> float:
        """Competition is a question of rewards.

        got_reward: How many got reward
        max_reward: hom many max can get reward
        """
        comp = 0
        if got_reward <= max_reward:
            comp = 1 - got_reward / max_reward

        return comp

    @staticmethod
    def value(min_v: float, max_v: float, seed: Optional[float] = None):
        """Random value in interval. seed is optional."""
        if seed is not None:
            random.seed(seed)
        return random.uniform(min_v, max_v)

    def motivation(self, params: dict[str, Any]) -> float:
        """EVC model."""

        number_agents_in_simulation = params["number_agents_in_simulation"]
        distance = params["distance"]
        got_reward = self.max_reward - number_agents_in_simulation
        return (
            EVCStrategy.value(self.min_value, self.max_value, self.seed)
            * EVCStrategy.competition(got_reward, self.max_reward)
            * EVCStrategy.expectancy(
                distance,
                self.width,
                self.height,
            )
        )


@dataclass
class MotivationModel:
    """Class defining the motivation model"""

    door_point1: Point = (60, 101)
    door_point2: Point = (60, 102)
    normal_v_0: float = 1.2
    normal_time_gap: float = 1.0
    active: int = 1
    motivation_strategy: Any = None

    def print_details(self) -> None:
        """Print member variables for debugging."""

        log_debug("Motivation Model:")
        log_debug(f">>  Door Point 1: {self.door_point1}")
        log_debug(f">>  Door Point 2: {self.door_point2}")
        log_debug(f">  Normal Velocity 0: {self.normal_v_0}")
        log_debug(f">>  Normal Time Gap: {self.normal_time_gap}")
        log_debug(f">>  Active: {self.active}")

    def __post_init__(self) -> None:
        if self.normal_v_0 is None:
            self.normal_v_0 = 1.2  # Default value if parsing returns None

        if self.normal_time_gap is None:
            self.normal_time_gap = 1  # Default value if parsing returns None

    def calculate_motivation_state(self, motivation_i: float) -> Tuple[float, float]:
        """return v0, T tuples depending on Motivation. (v0,T)=(1.2,1)."""

        v_0 = self.normal_v_0
        time_gap = self.normal_time_gap
        v_0_new = (1 + 0 * motivation_i) * v_0  # TODO
        time_gap_new = time_gap / (1 + motivation_i)

        return v_0_new, time_gap_new

    def get_profile_number(
        self, position: Point, number_agents_in_simulation: int, grid: ParameterGrid
    ) -> Tuple[int, float, float, float, float]:
        """Calculate the profile num from grid based on motivation related (v0,T)."""

        def calculate_distance():
            x_door = 0.5 * (self.door_point1[0] + self.door_point2[0])
            y_door = 0.5 * (self.door_point1[1] + self.door_point2[1])
            door = [x_door, y_door]
            distance = (
                (position[0] - door[0]) ** 2 + (position[1] - door[1]) ** 2
            ) ** 0.5
            return distance

        distance = calculate_distance()
        params = {
            "distance": distance,
            "number_agents_in_simulation": number_agents_in_simulation,
        }
        motivation_i = self.motivation_strategy.motivation(params)

        v_0, time_gap = self.calculate_motivation_state(motivation_i)
        number = int(grid.get_profile_number(v_0_i=v_0, time_gap_i=time_gap))
        return number, motivation_i, v_0, time_gap, distance
