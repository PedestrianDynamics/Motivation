"""
Module for motivational model
"""
from typing import Tuple, TypeAlias
from dataclasses import dataclass
import numpy as np

from .profiles import ParameterGrid

Point: TypeAlias = Tuple[float, float]


@dataclass
class MotivationModel:
    """Class defining the motivation model"""

    door_point1: Point = (60, 101)
    door_point2: Point = (60, 102)
    normal_v_0: float = 1.2
    normal_time_gap: float = 1
    active: int = 1

    def print_details(self):
        print(f"Motivation Model:")
        print(f"  Door Point 1: {self.door_point1}")
        print(f"  Door Point 2: {self.door_point2}")
        print(f"  Normal Velocity 0: {self.normal_v_0}")
        print(f"  Normal Time Gap: {self.normal_time_gap}")
        print(f"  Active: {self.active}")

    def __post_init__(self):
        if self.normal_v_0 is None:
            self.normal_v_0 = 1.2  # Default value if parsing returns None

        if self.normal_time_gap is None:
            self.normal_time_gap = 1  # Default value if parsing returns None

    def motivation(self, _distance: float, _width: float, _height: float) -> float:
        """Exponential motivation with compact support depending on distance"""

        motivation_y = float(
            np.exp(1 / ((_distance / _width) ** 2 - 1)) * np.e * _height
        )
        if _distance >= _width:
            motivation_y = 0.0

        return motivation_y

    def calculate_motivation_state(self, motivation_i: float) -> Tuple[float, float]:
        """return v0, T tuples depending on Motivation. (v0,T)=(1.2,1)"""

        v_0 = self.normal_v_0
        time_gap = self.normal_time_gap
        v_0_new = (1 + motivation_i) * v_0
        time_gap_new = time_gap / (1 + motivation_i)

        return v_0_new, time_gap_new

    def expectancy(
        self, position: float, slope: float, start: float, end: float
    ) -> float:
        """pieacwise constant to start, then linearly decreasing to end."""

        if position <= start:
            return slope * (end - start)
        if position <= end:
            return slope * (-position + end)

        return 0

    def competition(self, got_reward: int, max_reward: int) -> float:
        """Competition

        got_reward: How many got reward
        max_reward: hom many max can get reward
        """
        if got_reward <= max_reward:
            comp = 1 - got_reward / max_reward
        else:
            comp = 0

        return comp

    def get_profile_number(
        self, position: Point, grid: ParameterGrid
    ) -> Tuple[int, float, float, float, float]:
        """Calculate the profile num from grid based on motivation related (v0,T)"""

        height = 1
        width = 1
        x_door = 0.5 * (self.door_point1[0] + self.door_point2[0])
        y_door = 0.5 * (self.door_point1[1] + self.door_point2[1])
        door = [x_door, y_door]  # [62, (102.6 + 101.4) / 2]  # TODO: parse from json
        distance = ((position[0] - door[0]) ** 2 + (position[1] - door[1]) ** 2) ** 0.5
        motivation_i = self.motivation(distance, width, height)
        v_0, time_gap = self.calculate_motivation_state(motivation_i)
        number = int(grid.get_profile_number(v_0_i=v_0, time_gap_i=time_gap))
        return number, motivation_i, v_0, time_gap, distance
