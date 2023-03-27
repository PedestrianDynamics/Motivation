"""
Define velocityModel parameter profiles and fonctionalities, especially
returning a profile id for given (v0, T)
This is necessary, since we need to create a grid of parameter values beforehands and pass it to
the simulation module.
With the profile id we can select a profile that contains
*approaximately* the same parameter values as (v0, T)
"""

from dataclasses import dataclass, field
from itertools import product
from typing import List

import numpy as np
import numpy.typing as npt


@dataclass
class VelocityModelParameterProfile:
    """Parameter profile definition"""

    time_gap: float
    v_0: float
    tau: float
    radius: float
    number: int


@dataclass
class ParameterGrid:
    """Grid of parameter profiles"""

    min_v_0: float
    max_v_0: float
    v_0_step: float
    min_time_gap: float
    max_time_gap: float
    time_gap_step: float
    profiles: List[VelocityModelParameterProfile] = field(init=False)

    def __post_init__(self) -> None:
        """Init profile list and profiles numbers"""

        self.profiles = []
        profile_number = 0
        for v_0 in self.get_v_0_range():
            for time_gap in self.get_time_gap_range():
                self.profiles.append(
                    VelocityModelParameterProfile(
                        time_gap=time_gap,
                        v_0=v_0,
                        tau=0.5,
                        radius=0.2,
                        number=profile_number,
                    )
                )
                profile_number += 1

    def get_v_0_range(self) -> npt.NDArray[np.float32]:
        """distretise v0 in interval"""
        return np.arange(self.min_v_0, self.max_v_0 + self.v_0_step, self.v_0_step)

    def get_time_gap_range(self) -> npt.NDArray[np.float32]:
        """discretize time_gap in interval"""

        return np.arange(
            self.min_time_gap,
            self.max_time_gap + self.time_gap_step,
            self.time_gap_step,
        )

    def get_profile_number(self, v_0_i: float, time_gap_i: float) -> int:
        """Return profile number of tuple of variables"""

        array_v_0 = self.get_v_0_range()
        array_time_gap = self.get_time_gap_range()
        grid = list(product(array_v_0, array_time_gap))
        closest_point = min(
            grid, key=lambda p: abs(p[0] - v_0_i) + abs(p[1] - time_gap_i)
        )  # type: ignore
        return int(grid.index(closest_point))
