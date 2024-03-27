import streamlit as st


def main() -> None:
    """Print some documentation."""
    st.markdown("""## Default Strategy""")
    st.markdown(r"""
        $
        \textbf{motivation}(distance) =
        \begin{cases}
        0 & \text{if\;} distance  \geq \text{width}, \\
        e \cdot \text{height}\cdot\exp\left(\frac{1}{\left(\frac{distance}{\text{width}}\right)^2 - 1}\right) & \text{otherwise}.
        \end{cases}
        $
        
        ---
        ---
        """)
    st.markdown(r"""
        ## EVC
        $\textbf{motivation} = E\cdot V\cdot C,$ where
        - $E$: expectancy
        - $V$: value
        - $C$: competition
        
        ---
        """)

    st.markdown(
        r"""
        $
        \textbf{expectancy}(distance) = 
        \begin{cases}
        0 & \text{if\;} distance  \geq \text{width}, \\
        e \cdot \text{height}\cdot\exp\left(\frac{1}{\left(\frac{distance}{\text{width}}\right)^2 - 1}\right) & \text{otherwise}.
        \end{cases}\\
        $
        
        **Note:** this is the same function like default strategy
        
        ---
        $$
        \textbf{competition} = 
        \begin{cases} 
        c_0 & \text{if } N \leq N_0 \\
        c_0 - \left(\frac{c_0}{\text{percent} \cdot N_{\text{max}} - N_0}\right) \cdot (N - N_0) & \text{if } N_0 < N < \text{percent} \cdot N_{\text{max}} \\
        0 & \text{if } N \geq \text{percent} \cdot N_{\text{max}},
        \end{cases}
        $$
        with $N$  the number of agents still in the simulation. See also table.

        
        ---
        $\textbf{value} = random\_number \in [v_{\min}, v_{\max}].$

        ---
        ## Parameters

        | Parameter    | Meaning| Function|
        |--------------|:-----:|:-----:|
        |$N_0$ | Number of agents at which the decay of the function starts.| Competition|
        |$N_{\max}$ | Initial number of agents in the simulation|Competition|
        |$c_0$ | Maximal competition|Competition|
        |$p$ | Percentage number $\in [0, 1]$.|Competition|
        |
        |$v_{\min}$| Mimimum value | Value|
        |$v_{\max}$| Maximum value | Value|    
        |    
        |width| Range of influence | Expectancy|
        |height| Amplitude of influence | Expectancy|
            
            
        ## Update agents
        For an agent $i$ we calculate $m_i$ by one of the methods above and update its parameters as follows:
        
        $\tilde v_i^0 =  (v_i^0\cdot V_i)(1 + m_i)$ and $\tilde T_i = \frac{T_i}{1 + m_i}$

        The first part of the equation is equalivalent to

        $\tilde v_i^0 =  v_i^0\cdot V_i(1 + E_i\cdot V_i\cdot C_i)$.

        Here we see that the influence of $V_i$ is squared.
        Therefore, the second variation of the model reads
            
        ## EC-V
       $$     
       \begin{cases}     
        \tilde v_i^0 =  v_i^0(1 + E_i\cdot C_i)\cdot V_i, \\
        \tilde T_i = \frac{T_i}{1 + E_i\cdot C_i}.
       \end{cases}
        $$
        """,
        unsafe_allow_html=True,
    )
