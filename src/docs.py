import matplotlib.pyplot as plt
import streamlit as st

from .motivation_mapping import (
    AnchorValues,
    evaluate_logistic,
    fit_logistic_from_anchors,
)


def _logistic_runtime_plot() -> None:
    """Render active logistic mapping with anchor points."""
    anchors = AnchorValues(low=0.5, normal=1.2, high=3.6)
    params = fit_logistic_from_anchors(anchors, inflection_target=0.5)
    m_values = [0.0 + i * (1.0 - 0.0) / 200 for i in range(201)]
    y_values = [evaluate_logistic(m, params) for m in m_values]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(m_values, y_values, lw=2, label="Logistic fit")
    ax.scatter([0.0, 0.5, 1.0], [0.5, 1.2, 3.6], c="red", zorder=3, label="anchors")
    ax.set_xlabel("Motivation m")
    ax.set_ylabel("Desired speed (m/s)")
    # ax.set_title("Logistic runtime mapping: desired speed")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def _logistic_example_plot() -> None:
    """Render illustrative logistic-family example."""
    lower = 0.5
    upper = 3.6
    x0 = 0.5
    k = 2.4

    m_values = [0.0 + i * (1.0 - 0.0) / 300 for i in range(301)]
    y_values = [
        lower + (upper - lower) / (1.0 + pow(2.718281828, -k * (m - x0)))
        for m in m_values
    ]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(m_values, y_values, lw=2, color="#2ca02c", label="logistic example")
    ax.scatter(
        [0.0, 0.5, 1.0], [0.5, 1.2, 3.6], c="red", zorder=3, label="reference anchors"
    )
    ax.set_xlabel("Motivation m")
    ax.set_ylabel("Desired speed (m/s)")
    ax.set_title("Logistic family example")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def main() -> None:
    """Documentation tab for current PVE + logistic model."""
    st.markdown("## Motivation Model (PVE + Logistic)")
    st.markdown(
        "This app uses normalized `E`, `V`, `P` components, a weighted combination in `PVE` mode, and logistic parameter mapping."
    )
    st.markdown("### Core equations")
    st.latex(r"V_i^{\mathrm{unit}} = \mathrm{clip}\left(\frac{V_i - V_{\min}}{V_{\max} - V_{\min}},\;0,\;1\right)")
    st.latex(r"E_i^{\mathrm{unit}} = \mathrm{clip}\left(\frac{E(d_i) - E_{\min}}{E_{\max} - E_{\min}},\;0,\;1\right)")
    st.latex(r"r_i = n_{\mathrm{left}} + r_i^{\mathrm{room}}")
    st.latex(r"q_i = \frac{r_i - 1}{\max\left(1,\;N_{\max} - 1\right)}")
    st.latex(r"P_i^{\mathrm{unit}}(q_i) = \frac{1}{1 + \exp\left(k_p\,(q_i-q_0)\right)}")
    st.latex(r"M_i \in \left\{E_i^{\mathrm{unit}},\;V_i^{\mathrm{unit}},\;P_i^{\mathrm{unit}},\;\frac{w_V V_i^{\mathrm{unit}} + w_E E_i^{\mathrm{unit}} + w_P P_i^{\mathrm{unit}}}{w_V + w_E + w_P}\right\}")
    st.latex(r"m_i^{\mathrm{used}} = \mathrm{clip}\left(m_i,\; m_{\min},\; 1\right)")
    st.latex(
        r"y(m) = y_{\min} + \frac{y_{\max} - y_{\min}}{1 + \exp\left(-k\,(m - m_0)\right)}"
    )
    st.markdown(
        "Here `r_i` is the absolute rank, `q_i` is the normalized rank, `w_V`, `w_E`, `w_P` are the combination weights, and `y(m)` is one mapped operational parameter with `m0 = inflection_target`."
    )
    st.markdown("### Unit Construction")
    st.markdown(
        """
`V_unit`
- Start from each agent's absolute value `V_i`.
- Normalize it with the configured low/high value bounds.
- Clip to `[0, 1]`.

`E_unit`
- Compute the raw expectancy from the current distance to the motivation door.
- Normalize the raw expectancy between its configured minimum and maximum.
- Clip to `[0, 1]`.

`Rank` and `P_unit`
- For each active agent, compute the squared distance to the door center.
- Sort agents by distance; the closest active agent gets in-room rank `1`.
- Agents within the tie tolerance share the same in-room rank.
- Add the number of already-left agents to obtain the absolute rank `r_i`.
- Normalize the absolute rank to `q_i in [0,1]`.
- Evaluate the logistic payoff `P_unit(q_i)`.

`M`
- `E` mode uses `E_unit`.
- `V` mode uses `V_unit`.
- `P` mode uses `P_unit`.
- `PVE` mode uses the weighted average of `V_unit`, `E_unit`, and `P_unit`.
- `weight_v`, `weight_e`, and `weight_p` control the relative influence of the three normalized components.
"""
    )

    st.markdown("### Requirements (Current)")
    st.markdown(
        """
| Parameter | Low motivation | Normal motivation | High motivation | Rule |
|---|---:|---:|---:|---|
| Motivation (m) | 0.0 | 0.5 | 1.0 | Clamped by m in [m_min, 1] |
| Desired speed (v0_tilde) [m/s] | 0.5 | 1.2 | 3.6 | Logistic |
| Time gap (T_tilde) [s] | 2.0 | 1.0 | 0.01 | Logistic |
| Buffer (b_tilde) [m] | 1.0 | 0.1 | 0.0 | Logistic |
| Strength neighbor repulsion (A_tilde) | A_low | A_normal | A_high | Logistic; anchors from `strength_neighbor_repulsion_anchors` |
| Range neighbor repulsion (D_tilde) | d_ped | d_ped | d_ped | Constant |
"""
    )

    st.markdown("### Logistic runtime mapping")
    _logistic_runtime_plot()
    st.markdown("### Logistic-family parameters")
    st.markdown("- `y_min, y_max`: lower and upper asymptotes")
    st.markdown("- `k`: steepness")
    st.markdown("- `m0`: inflection point")
    _logistic_example_plot()
