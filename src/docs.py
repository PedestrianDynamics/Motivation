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
        ### 1 Expectancy
        A bell-like function with maximum height at distance=0, and it goes to 0 beyond width.        

        - **Local Maximum:**
         Near the door ($distance \le width$), $E$ is relatively larger.
        - **Decay to Zero:**
         Once beyond a certain "influence zone" ($distance > width$), the expectancy drops to 0.
        
        --- 
        
        $
        \textbf{expectancy}(distance) = 
        \begin{cases}
        0 & \text{if\;} distance  \geq \text{width}, \\
        e \cdot \text{height}\cdot\exp\left(\frac{1}{\left(\frac{distance}{\text{width}}\right)^2 - 1}\right) & \text{otherwise}.
        \end{cases}\\
        $
        
        **Note:** this is the same function like default strategy
        
        ---
        ### 2. Competition

        1. Early Competition:
        At the start (up to $N_0$​ departures), everyone competes for a reward or advantage. This phase can mimic a scenario where there is some strong external incentive for the first few people to escape.

        2. Gradual Decline:
        Between $N_0$​ and $\% N_{max⁡}$​, the reward or the "reason to compete" diminishes as more agents leave.

        3. No Competition:
        Once a critical number (or fraction) of people have left, $C$ goes to 0. This suggests that no meaningful benefit remains for being among the next ones out.

        Current implementation:
        - $N$ is the number of agents who have already left the room.
        - $C$ starts at $c_0$​ and remains constant as long as $N<N_0​$.
        - After $N_0$​ agents have left, the competition begins to drop linearly until eventually it hits 0 at $N=\% N_{max}⁡$​.

        ---
        
        $$
        \textbf{competition} = 

        \begin{cases} 
        c_0 & \text{if } N \leq N_0 \\
        c_0 - \left(\frac{c_0}{\text{percent} \cdot N_{\text{max}} - N_0}\right) \cdot (N - N_0) & \text{if } N_0 < N < \text{percent} \cdot N_{\text{max}} \\
        0 & \text{if } N \geq \text{percent} \cdot N_{\text{max}},
        \end{cases}
        $$
       
        ---
        ##### 3 Value function

        - Agents have a parameter $V_i$ that represents their intrinsic "care level" (low or high).
        - Assign low or high values based on distance to exit, or with some probability distribution.
        
        $\textbf{value} = random\_number \in [v_{\min}, v_{\max}].$
        
        We propose a method for assigning values to agents based on their spatial proximity to a designated exit.
        In this approach, the likelihood that an agent receives a high value decays exponentially with distance from the exit.

        ###### Distance Decay Parameter

        A key parameter in this model is the **distance decay**, which regulates the rate at which the probability of being high value decreases with distance. The parameter is defined as:

        $$
        \text{distance\_decay} = -\frac{\text{width}}{\ln(0.01)}
        $$
        This formulation guarantees that the probability of an agent being assigned a high value is approximately 0.01 when the agent is at a distance equal to the defined `width` from the exit.

        ###### Probability Calculation
        ###### Distance Measurement
        For an agent at position \(\mathbf{p} = (x, y)\) and an exit at \(\mathbf{p}_{\text{exit}} = (x_{\text{exit}}, y_{\text{exit}})\), the Euclidean distance is computed as:

        $$
        d(\mathbf{p}, \mathbf{p}_{\text{exit}}) = \sqrt{(x - x_{\text{exit}})^2 + (y - y_{\text{exit}})^2}
        $$
        ###### Exponential Decay Function
       
        The probability \(P(\mathbf{p})\) that an agent at \(\mathbf{p}\) is assigned a high value is given by:

        $$
        P(\mathbf{p}) = \exp\left(-\frac{d(\mathbf{p}, \mathbf{p}_{\text{exit}})}{\text{distance\_decay}}\right)
        $$

        This exponential decay ensures that agents closer to the exit have a higher probability of being designated as high value.

        ###### Seed Management for Reproducibility

        To maintain reproducibility in the randomness inherent in the assignment process, a **SeedManager** is used. Each random operation employs a derived seed, calculated as:

        $$
        \text{derived\_seed} = \text{base\_seed} \times 1000 + \text{operation\_id}
        $$

        This ensures that every operation, including the generation of random values for agents, is deterministic when the same base seed is used.

        ##### Value Assignment Process

        The final assignment of values to agents proceeds through the following steps:

        1. **Probability Computation:**  
        For each agent, the high value probability is calculated using the agent's distance from the exit.

        2. **Random Perturbation:**  
        To prevent strictly deterministic outcomes (especially when agents have similar distances), a small random factor is introduced:
   
        $$
        P' = P(\mathbf{p}) \times \left(1 + U[0, 0.2]\right)
        $$
   
        where \(U[0, 0.2]\) represents a uniformly distributed random variable between 0 and 0.2.

        3. **Sorting and Selection:**  
        Agents are sorted in descending order based on the perturbed probability \(P'\). The top \(N\) agents—where \(N\) is the predefined number of high value agents—are selected.

        4. **Final Value Generation:**  
        - **High Value Agents:**  
        Each agent in the selected set receives a value in the range
        $$ [v_{\min}^{\text{high}}, v_{\max}^{\text{high}}]$$.
        - **Low Value Agents:**  
        All remaining agents are assigned a value in the range $$[v_{\min}^{\text{low}}, v_{\max}^{\text{low}}]$$.
     
        Each value is generated using a random number generator seeded with the agent's derived seed.        
        
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
        $$
        m_i = V_i \cdot E_i \cdot C_i \in [0, 1]
        $$
        
        Then
        $$
        \tilde v_i^0 =  v_i^0\cdot V_i
        $$
        This one-time scaling ensures that agents who "care more" start with a higher desired speed.
        
        and
        $$
        \tilde T_i = \frac{T_i}{\Big(1+m_i\Big)},
        $$
        """,
        unsafe_allow_html=True,
    )
