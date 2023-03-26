from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ParameterProfile:
    v0: float
    T: float
    number: int


@dataclass
class ParameterGrid:
    min_v0: float
    max_v0: float
    v0_step: float
    min_T: float
    max_T: float
    T_step: float
    profiles: List[ParameterProfile] = field(init=False)

    def __post_init__(self):
        self.profiles = []
        profile_number = 0
        for v0 in self.get_v0_range():
            for T in self.get_T_range():
                self.profiles.append(
                    ParameterProfile(v0=v0, T=T, number=profile_number)
                )
                profile_number += 1

    def get_v0_range(self):
        return [i for i in np.arange(self.min_v0, self.max_v0 + 1, self.v0_step)]

    def get_T_range(self):
        return [i for i in np.arange(self.min_T, self.max_T + 1, self.T_step)]

    def get_profile_number(self, v0_i, T_i):
        v0_index = int((v0_i - self.min_v0) / self.v0_step)
        T_index = int((T_i - self.min_T) / self.T_step)
        return T_index * len(self.get_v0_range()) + v0_index


if __name__ == "__main__":
    grid = ParameterGrid(min_v0=0, max_v0=1, v0_step=0.5, min_T=0, max_T=1, T_step=0.5)
    print(grid.profiles)

    print(grid.get_profile_number(v0_i=0, T_i=0.5))

    print(grid.get_profile_number(v0_i=0.5, T_i=0))

    print(grid.profiles[1].number)

    print(grid.profiles[2].v0, grid.profiles[2].T)
