import matplotlib.pyplot as plt
import streamlit as st

from .motivation_mapping import (
    AnchorValues,
    evaluate_gompertz,
    estimate_gompertz_from_anchors,
)


def _gompertz_example_plot() -> None:
    """Render a Gompertz example with anchor points."""
    anchors = AnchorValues(low=0.5, normal=1.2, high=3.6)
    params = estimate_gompertz_from_anchors(anchors)
    m_values = [0.1 + i * (3.0 - 0.1) / 200 for i in range(201)]
    y_values = [evaluate_gompertz(m, params) for m in m_values]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(m_values, y_values, lw=2, label="Gompertz fit")
    ax.scatter([0.1, 1.0, 3.0], [0.5, 1.2, 3.6], c="red", zorder=3, label="anchors")
    ax.set_xlabel("Motivation m")
    ax.set_ylabel("Desired speed (m/s)")
    ax.set_title("Gompertz Example: desired speed mapping")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def _gompertz_fit_mode_comparison_plot() -> None:
    """Render a side-by-side comparison of fit modes."""
    anchors = AnchorValues(low=0.5, normal=1.2, high=3.6)
    params_exact = estimate_gompertz_from_anchors(anchors)
    params_sig = estimate_gompertz_from_anchors(anchors)

    # Import here to avoid circular app dependencies and keep docs self-contained.
    from .motivation_mapping import estimate_gompertz_sigmoid_preferred

    params_sig = estimate_gompertz_sigmoid_preferred(anchors, inflection_target=1.5)

    m_values = [0.1 + i * (3.0 - 0.1) / 300 for i in range(301)]
    y_exact = [evaluate_gompertz(m, params_exact) for m in m_values]
    y_sig = [evaluate_gompertz(m, params_sig) for m in m_values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(m_values, y_exact, lw=2, color="#1f77b4", label="exact_anchors")
    axes[0].scatter([0.1, 1.0, 3.0], [0.5, 1.2, 3.6], c="red", zorder=3)
    axes[0].set_title("exact_anchors")
    axes[0].set_xlabel("Motivation m")
    axes[0].set_ylabel("Desired speed (m/s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        m_values, y_sig, lw=2, color="#ff7f0e", label="sigmoid_preferred"
    )
    axes[1].scatter([0.1, 1.0, 3.0], [0.5, 1.2, 3.6], c="red", zorder=3)
    axes[1].set_title("sigmoid_preferred")
    axes[1].set_xlabel("Motivation m")
    axes[1].set_ylabel("Desired speed (m/s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    st.pyplot(fig)


def _logistic_example_plot() -> None:
    """Render an illustrative logistic-family example (docs only)."""
    lower = 0.5
    upper = 3.6
    x0 = 1.5
    k = 2.4

    m_values = [0.1 + i * (3.0 - 0.1) / 300 for i in range(301)]
    y_values = [lower + (upper - lower) / (1.0 + pow(2.718281828, -k * (m - x0))) for m in m_values]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(m_values, y_values, lw=2, color="#2ca02c", label="logistic example")
    ax.scatter([0.1, 1.0, 3.0], [0.5, 1.2, 3.6], c="red", zorder=3, label="reference anchors")
    ax.set_xlabel("Motivation m")
    ax.set_ylabel("Desired speed (m/s)")
    ax.set_title("Logistic family example (illustrative only)")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def main() -> None:
    """Documentation tab for current EVC + Gompertz model."""
    st.markdown("## Motivation Model (EVC + Gompertz)")
    st.markdown(
        "This app uses **EVC motivation** and **Gompertz-based parameter mapping** only. "
        "The old **Default Strategy** is intentionally removed from this documentation."
    )
    st.markdown("### Core equations")
    st.latex(r"M_i = E_i \cdot V_i \cdot C_i")
    st.latex(
        r"m_i^{\mathrm{used}} = \mathrm{clip}\left(m_i,\; m_{\min},\; \frac{3.6}{v_0^{\mathrm{normal}}}\right)"
    )
    st.latex(r"y(m) = a \cdot \exp\left(-b \cdot \exp(-c \cdot m)\right)")
    st.markdown(
        "Where `y(m)` is one mapped operational parameter and `(a,b,c)` can be "
        "fitted from anchors or entered manually in the app."
    )
    st.markdown(
        "- `fit_mode = exact_anchors`: exact match at low/normal/high.\n"
        "- `fit_mode = sigmoid_preferred`: approximate anchors, prefers an in-range inflection."
    )

    st.markdown("### Requirements (Current)")
    st.markdown(
        """
| Parameter | Low motivation | Normal motivation | High motivation | Rule |
|---|---:|---:|---:|---|
| Motivation (m) | 0.1 | 1.0 | 3.0 | Clamped by m in [m_min, 3.6 / v0_normal] |
| Desired speed (v0_tilde) [m/s] | 0.5 | 1.2 | 3.6 | Gompertz |
| Time gap (T_tilde) [s] | 2.0 | 1.0 | 0.01 | Gompertz |
| Buffer (b_tilde) [m] | 1.0 | 0.1 | 0.0 | Gompertz |
| Strength neighbor repulsion (A_tilde) | a_ped_min | a_ped | a_ped_max | Gompertz; anchors from config |
| Range neighbor repulsion (D_tilde) | d_ped | d_ped | d_ped | Constant |
"""
    )

    st.markdown("### Gompertz plot")
    _gompertz_example_plot()
    st.markdown("### Fit mode comparison")
    st.markdown(
        "- `exact_anchors`: passes exactly through low/normal/high anchors.\n"
        "- `sigmoid_preferred`: allows small anchor mismatch to keep a stronger S-shape in-range."
    )
    _gompertz_fit_mode_comparison_plot()
    st.markdown("### Logistic-family example (docs only)")
    st.markdown("This is an alternative functional family (not active in runtime code):")
    st.latex(
        r"y(m) = y_{\min} + \frac{y_{\max} - y_{\min}}{1 + \exp\left(-k\,(m - m_0)\right)}"
    )
    st.markdown("- `y_min, y_max`: lower and upper asymptotes")
    st.markdown("- `k`: steepness")
    st.markdown("- `m0`: inflection point")
    _logistic_example_plot()
