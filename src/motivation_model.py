"""
Module for motivational model
"""
from typing import Tuple, TypeAlias
import numpy as np
from .profiles import ParameterGrid

Point: TypeAlias = Tuple[float, float]


def motivation(distance: float, _width: float, _height: float) -> float:
    """Exponential motivation with compact support depending on distance"""

    motivation_y = float(np.exp(1 / ((distance / _width) ** 2 - 1)) * np.e * _height)
    if distance >= _width:
        motivation_y = 0.0

    return motivation_y


def calculate_motivation_state(motivation_i: float) -> Tuple[float, float]:
    """return v0, T tuples depending on Motivation. (v0,T)=(1.2,1)"""

    v_0 = 1.2
    time_gap = 1
    v_0_new = (1 + motivation_i) * v_0
    time_gap_new = time_gap / (1 + motivation_i)

    return v_0_new, time_gap_new


# def motivation_trying(x: float, slope: float, start: float, end: float) -> float:
#     """pieacwise linear function from 0 to start and from end to 0 and constant inbetween"""

#     if x <= start:
#         return slope * x
#     elif x <= end:
#         return slope * start
#     elif x <= (start + end):
#         return -slope * x + slope * (start + end)
#     else:
#         return 0


# def f(x: float, slope: float, start: float, end: float) -> float:
#     """pieacwise constant to start, then linearly decreasing to end. Then 0 else"""

#     if x <= start:
#         return slope * (end - start)
#     elif x <= end:
#         return slope * (-x + end)
#     else:
#         return 0


# def expectancy(x: float, slope: float, start: float, end: float) -> float:
#     """pieacwise constant to start, then linearly decreasing to end. Then 0 else"""

#     if x <= start:
#         return slope * (end - start)
#     elif x <= end:
#         return slope * (-x + end)
#     else:
#         return 0


# def competition(got_reward: int, max_reward: int) -> float:
#     """Competition

#     got_reward: How many got reward
#     max_reward: hom many max can get reward
#     """
#     if got_reward <= max_reward:
#         c = 1 - got_reward / max_reward
#     else:
#         c = 0
#     return c


def get_profile_number(
    position: Point, grid: ParameterGrid
) -> Tuple[int, float, float, float, float]:
    """Calculate the profile num from grid based on motivation related (v0,T)"""

    height = 1
    width = 1
    # [60, 102.6],
    # [60, 101.4]
    door = [62, (102.6 + 101.4) / 2]  # TODO: parse from json
    distance = ((position[0] - door[0]) ** 2 + (position[1] - door[1]) ** 2) ** 0.5
    motivation_i = motivation(distance, width, height)
    v_0, time_gap = calculate_motivation_state(motivation_i)
    number = int(grid.get_profile_number(v_0_i=v_0, time_gap_i=time_gap))
    return number, motivation_i, v_0, time_gap, distance


if "__main__" == __name__:
    pass
