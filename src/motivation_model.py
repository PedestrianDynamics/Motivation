"""
Module for motivational model
"""
from typing import Tuple, TypeAlias, Protocol, Optional
from dataclasses import dataclass
import numpy as np
import random
from .profiles import ParameterGrid
from .logger_config import log_debug

Point: TypeAlias = Tuple[float, float]


@dataclass
class MotivationParameters:
    """Class with several parameters for different MotivationStrategies."""

    width: float = 1.0
    height: float = 1.0
    distance: float = 0.0
    got_reward: int = 0
    max_reward: int = 0
    seed: int = 0
    min_value: int = 0
    max_value: int = 1


class MotivationStrategy(Protocol):
    """Protocol for defining the motivation calculation strategy."""

    def motivation(motivation_parameters: MotivationParameters) -> float:
        """Flexible."""


class DefaultMotivationStrategy:
    """Default strategy for motivation calculation based on distance."""

    def motivation(motivation_parameters: MotivationParameters) -> float:
        """Motivation based on distance to entrance."""
        _width = motivation_parameters.width
        _height = motivation_parameters.height
        _distance = motivation_parameters.distance
        if _distance >= _width:
            return 0.0

        expr = 1 / ((_distance / _width) ** 2 - 1)
        if np.isinf(expr):
            return 0.0

        return float(np.exp(expr) * np.e * _height)


class EzelMotivationStrategy:
    """Motivation theory based on E.V.C (model4)"""

    def expectancy(_distance: float, _width: float, _height: float) -> float:
        """Expectancy depends on the distance to the entrance."""
        if _distance >= _width:
            return 0.0

        expr = 1 / ((_distance / _width) ** 2 - 1)
        if np.isinf(expr):
            return 0.0

        return float(np.exp(expr) * np.e * _height)

    def competition(got_reward: int, max_reward: int) -> float:
        """Competition is a question of rewards.

        got_reward: How many got reward
        max_reward: hom many max can get reward
        """
        comp = 0
        if got_reward <= max_reward:
            comp = 1 - got_reward / max_reward

        return comp

    def value(min_v: float, max_v: float, seed: Optional[float] = None):
        """Random value in interval. seed is optional."""
        if seed is not None:
            random.seed(seed)
        return random.uniform(min_v, max_v)

    def motivation(motivation_parameters: MotivationParameters) -> float:
        return (
            self.value(
                motivation_parameters.min_value,
                motivation_parameters.max_value,
                motivation_parameters.seed,
            )
            * self.competition(
                motivation_parameters.got_reward, motivation_parameters.max_reward
            )
            * self.expectancy(
                motivation_parameters.distance,
                motivation_parameters.width,
                motivation_parameters.height,
            )
        )


@dataclass
class MotivationModel:
    """Class defining the motivation model"""

    motivation_parameters: MotivationParameters
    door_point1: Point = (60, 101)
    door_point2: Point = (60, 102)
    normal_v_0: float = 1.2
    normal_time_gap: float = 1.0
    active: int = 1
    motivation_strategy: MotivationStrategy = DefaultMotivationStrategy()

    def print_details(self) -> None:
        """Print member variables for debugging"""

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
        """return v0, T tuples depending on Motivation. (v0,T)=(1.2,1)"""

        v_0 = self.normal_v_0
        time_gap = self.normal_time_gap
        v_0_new = (1 + motivation_i) * v_0
        time_gap_new = time_gap / (1 + motivation_i)

        return v_0_new, time_gap_new

    def get_profile_number(
        self, position: Point, number_agents_in_simulation: int, grid: ParameterGrid
    ) -> Tuple[int, float, float, float, float]:
        """Calculate the profile num from grid based on motivation related (v0,T)."""

        def update_motivation_parameters():
            x_door = 0.5 * (self.door_point1[0] + self.door_point2[0])
            y_door = 0.5 * (self.door_point1[1] + self.door_point2[1])
            door = [x_door, y_door]
            distance = (
                (position[0] - door[0]) ** 2 + (position[1] - door[1]) ** 2
            ) ** 0.5
            self.motivation_parameters.distance = distance
            self.motivation_parameters.got_reward = (
                self.motivation_parameters.max_reward - number_agents_in_simulation
            )
            return distance

        distance = update_motivation_parameters()
        motivation_i = self.motivation_strategy.motivation(
            motivation_parameters=self.motivation_parameters
        )
        v_0, time_gap = self.calculate_motivation_state(motivation_i)
        number = int(grid.get_profile_number(v_0_i=v_0, time_gap_i=time_gap))
        return number, motivation_i, v_0, time_gap, distance
