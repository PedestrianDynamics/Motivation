import numpy as np
import matplotlib.pyplot as plt
from typing import TypeAlias, Tuple

Point: TypeAlias = Tuple[float, float]

# from profiles import ParameterGrid


def motivation(distance: float, width: float, height: float) -> float:
    """Exponential motivation with compact support depending on distance"""

    y = float(np.exp(1 / ((x / width) ** 2 - 1)) * np.e * height)
    if distance >= width:
        y = 0.0

    return y


def calculate_motivation_state(motivation: float) -> Tuple[float, float]:
    """return v0, T tuples depending on Motivation"""

    v_0 = 1.2
    T = 1
    v_0_tilde = (1 + motivation) * v_0
    T_tilde = T / (1 + motivation)

    return v_0_tilde, T_tilde


def motivation_trying(x: float, slope: float, start: float, end: float) -> float:
    """pieacwise linear function from 0 to start and from end to 0 and constant inbetween"""

    if x <= start:
        return slope * x
    elif x <= end:
        return slope * start
    elif x <= (start + end):
        return -slope * x + slope * (start + end)
    else:
        return 0


def f(x: float, slope: float, start: float, end: float) -> float:
    """pieacwise constant to start, then linearly decreasing to end. Then 0 else"""

    if x <= start:
        return slope * (end - start)
    elif x <= end:
        return slope * (-x + end)
    else:
        return 0


def expectancy(x: float, slope: float, start: float, end: float) -> float:
    """pieacwise constant to start, then linearly decreasing to end. Then 0 else"""

    if x <= start:
        return slope * (end - start)
    elif x <= end:
        return slope * (-x + end)
    else:
        return 0


def competition(got_reward: int, max_reward: int) -> float:
    """Competition

    got_reward: How many got reward
    max_reward: hom many max can get reward
    """
    if got_reward <= max_reward:
        c = 1 - got_reward / max_reward
    else:
        c = 0
    return c


def get_profile_number(position: Point, grid) -> int:
    """Calculate the profile number from grid based on motivation related (v0,T)"""

    center = 2
    width = 0.5
    # [60, 102.6],
    # [60, 101.4]
    door = [62, (102.6 + 101.4) / 2]  # TODO: parse from json
    distance = ((position[0] - door[0]) ** 2 + (position[1] - door[1]) ** 2) ** 0.5
    motivation_i = motivation(distance, center, width)
    v_0, time_gap = calculate_motivation_state(motivation_i)
    number = grid.get_profile_number(v_0_i=v_0, time_gap_i=time_gap)
    return number


if "__main__" == __name__:
    # Parameters
    m = 1
    a = 1
    b = 3
    n_p = a + b

    # Generate x and y values
    x = np.linspace(0, b + a + 1, 100)

    # fig, ax = plt.subplots(2, 2)
    # fig.tight_layout()
    center = 2
    x2 = np.linspace(0, 10, 1000)
    for width in [0.5, 1]:
        y = motivation(x2, center, width)
        plt.plot(x2, y, label=f"{width=}")
        plt.ylabel("M")
        plt.grid(alpha=0.3)
        plt.legend()

    # for m in [1, 2, 3]:
    #     y = np.array([motivation_trying(xi, m, a, b) for xi in x])
    #     ax[0][0].plot(x, y, label=f"{m=}")
    #     ax[0][0].set_ylabel("M")
    #     ax[0][0].grid(alpha=0.3)
    #     ax[0][0].legend()

    # x2 = np.linspace(0, 10, 100)

    # plt.subplot(2, 2, 3)
    # F = np.array([f(gi, m, a, b) for gi in x])
    # E = np.array([15 + -expectancy(xi, m, a, b) for xi in x])  # F / (C + 1)
    # ax[1][0].plot(x, E)
    # ax[1][0].grid(alpha=0.3)
    # ax[1][0].set_ylabel("C2")

    # E2 = np.array([expectancy(xi, m, a, b) for xi in x])
    # ax[1][1].plot(x, E2)
    # ax[1][1].grid(alpha=0.3)
    # ax[1][1].set_ylabel("E2")

    # C = np.array([competition(gi, n_p) for gi in x])
    # ax[0][1].plot(x, C)
    # ax[0][1].grid(alpha=0.3)
    # ax[0][1].set_ylabel("C*E")

    # # for m in [1, 2, 3]:
    # #     M = np.array([motivation_trying(xi, m, a, b) for xi in x])
    # #     ax[1][1].plot(x, E * C)
    # #     ax[1][1].grid(alpha=0.3)
    # #     ax[1][1].set_ylabel("E.C")

    plt.savefig("plot2.png")
