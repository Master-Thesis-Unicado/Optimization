# Climb Trajectory Optimization Using Energy Height Theory

This script models and visualizes aircraft climb trajectories using a **strategy-based energy framework**. It is rooted in the concept of **specific energy height**, where the aircraft’s total energy is treated as a combination of potential and kinetic components. The framework supports a wide variety of climb strategies, including linear allocation, altitude-biased exponential functions, constant-rate climbs, and eventually variable-energy climbs. This allows for both educational and performance-driven analyses of flight mechanics.

---

## Theory Overview

### 1. **Specific Energy Height**

The total specific energy (per unit mass) of an aircraft is defined as:

$$
E = h + \frac{V^2}{2g_0}
$$

Where:

* $h$: altitude \[m]
* $V$: true airspeed \[m/s]
* $g_0$: standard gravitational acceleration \[m/s]

This representation expresses both potential and kinetic energy in terms of an equivalent height (in meters), allowing a unified way to measure and manage aircraft energy states.

---

### 2. **Rate of Energy Gain**

By differentiating the specific energy expression with respect to time, we obtain:

$$
\frac{dE}{dt} = \frac{dh}{dt} + \frac{V}{g_0} \cdot \frac{dV}{dt}
$$

This shows how energy gain (rate) is split between vertical motion (climb) and forward acceleration. It becomes the foundation for constructing energy-allocation strategies.

---

### 3. **Energy Split Strategy via `altitude_fraction`**

To control how the aircraft divides its energy input between climb and speed, a parameter $r \in [0, 1]$ (named `altitude_fraction`) is introduced:

* $r$: fraction of energy allocated to altitude gain (climb)
* $1 - r$: fraction allocated to speed increase (acceleration)

Under this rule, and assuming constant total specific energy rate $\dot{E}$, we compute rates as:

$$
\frac{dh}{dt} = r \cdot \dot{E}, \quad \frac{dV}{dt} = (1 - r) \cdot \dot{E} \cdot \frac{g_0}{V}
$$

This method generalizes well across strategies by varying $r$ and $\dot{E}$ independently.

---

## 4.1 **Fixed Energy Linear Strategy**

In this simplest case, the energy rate $\dot{E} = E_{\text{DOT}}$ remains **constant**, and the allocation ratio $r$ is also fixed throughout the climb.

Resulting rate formulas are:

$$
\frac{dh}{dt} = r \cdot E_{\text{DOT}}, \quad \frac{dV}{dt} = (1 - r) \cdot E_{\text{DOT}} \cdot \frac{g_0}{V}
$$

The climb and speed rates are constant (with respect to time) for a given $r$, but speed gain will still cause nonlinear effects due to the $\frac{1}{V}$ dependence.

**Implemented in**: `StrategyProfiles.FixedEnergy.Linear`

---

## 4.2 **Exponential Bias Strategy**

This strategy class allows **altitude-dependent** redistribution of energy while still assuming a **constant total specific energy rate** $\dot{E} = E_{\text{DOT}}$.

Unlike the linear case, the preference to climb or accelerate **evolves** with altitude, creating more realistic profiles.

### 4.2.1 General Methodology

Let:

* $r \in (0, 1)$: base climb preference at sea level
* $h$: current altitude
* $h_{\text{target}}$: mission-defined target altitude

We apply **exponential weighting** to adjust climb/speed priorities dynamically:

#### Increasing Climb Bias

$$
\text{climb\_weight} = r \cdot e^{h / h_{\text{target}}}, \quad \text{speed\_weight} = (1 - r) \cdot e^{-h / h_{\text{target}}}
$$

#### Decreasing Climb Bias

$$
\text{climb\_weight} = r \cdot e^{-h / h_{\text{target}}}, \quad \text{speed\_weight} = (1 - r) \cdot e^{h / h_{\text{target}}}
$$

#### Increasing Speed Bias

$$
\text{speed\_weight} = r \cdot e^{h / h_{\text{target}}}, \quad \text{climb\_weight} = (1 - r) \cdot e^{-h / h_{\text{target}}}
$$

#### Decreasing Speed Bias

$$
\text{speed\_weight} = r \cdot e^{-h / h_{\text{target}}}, \quad \text{climb\_weight} = (1 - r) \cdot e^{h / h_{\text{target}}}
$$

Weights are **normalized** to obtain valid fractions:

$$
\text{climb\_fraction} = \frac{\text{climb\_weight}}{\text{climb\_weight} + \text{speed\_weight}}, \quad \text{speed\_fraction} = 1 - \text{climb\_fraction}
$$

And finally, rates are computed as:

$$
\frac{dh}{dt} = \text{climb\_fraction} \cdot E_{\text{DOT}}, \quad \frac{dV}{dt} = \text{speed\_fraction} \cdot E_{\text{DOT}} \cdot \frac{g(h)}{V}
$$

Where $g(h)$ is retrieved from the atmospheric model.

### 4.2.2 Strategy Variants in Code

Each strategy is explicitly defined under `StrategyProfiles.FixedEnergy.Exponential`:

| Strategy Function  | Description                                        |
| ------------------ | -------------------------------------------------- |
| `increasing_climb` | Emphasizes climbing more as altitude increases     |
| `decreasing_climb` | Prioritizes climb early, then transitions to speed |
| `increasing_speed` | Delays acceleration until higher altitudes         |
| `decreasing_speed` | Emphasizes acceleration early in the climb         |

This class enables flexible mission adaptation, particularly useful for designing eco-climb profiles or optimizing fuel burn.

---

## 4.3 **Constant Rate Strategies**

Implemented in `StrategyProfiles.ConstantRates`  these assume fixed speed:

### Constant Speed

$$
\frac{dh}{dt} = g_0, \quad \frac{dV}{dt} = 0
$$
 
 
---

## 5. **Simulation Logic**

The function `simulate_climb_path()` integrates the state equations over time using **Euler's method**:

* At each time step, current altitude and velocity are passed to a strategy function.
* The function returns $\frac{dh}{dt}$, $\frac{dV}{dt}$, which are used to update state variables.
* The climb ends **exactly** when $h = h_{\text{target}}$ without overshooting.

This ensures numerical stability and accuracy of terminal conditions.

---
 

### Possible Extensions:

* Thrust & drag model (based on atmosphere and engine map)
* TSFC-based fuel burn tracking
* Optimization for minimum fuel or time
* Aerodynamic constraints (e.g., max $C_L$, $\alpha$, etc.)


 ---


# Summary of Climb Performance Concepts (Raymer, Chapter 17.3)

This summary consolidates key concepts and equations related to **steady climb and descent** from Daniel Raymer's *Aircraft Design: A Conceptual Approach*, with a focus on climb gradient, best angle/rate of climb, and time/fuel to climb.



## Steady Climbing Flight and Climb Gradient

- **Climb gradient** \( G \) is the ratio of vertical to horizontal distance traveled.
- It is equivalent to \( \sin(\gamma) \), where \( \gamma \) is the climb angle:
  
  \[
  \gamma = \sin^{-1} \left( \frac{T - D}{W} \right) = \sin^{-1} \left( \frac{T}{W} - \frac{1}{L/D} \right) \tag{Eq. 17.38}
  \]

- The **vertical velocity** or **rate of climb** \( V_v \) is:

  \[
  V_v = V \sin(\gamma) = V \sqrt{ \frac{T}{W} - \frac{1}{L/D} } \tag{Eq. 17.39}
  \]

- Force balances used:

  \[
  \sum F_x = T - D - W \sin(\gamma) \tag{Eq. 17.6}
  \]
  \[
  \sum F_z = L - W \cos(\gamma) \tag{Eq. 17.7}
  \]

---

## Graphical Method: Best Angle and Rate of Climb

- **Best rate of climb** maximizes vertical velocity \( V_v \).
- **Best angle of climb** maximizes altitude gain per unit horizontal distance (i.e., max \( \gamma \)).
- Plot \( V_v \) vs airspeed (using Eq. 17.39) and superimpose thrust/drag data to identify:

  - **Peak of the curve**: Best rate of climb.
  - **Tangency from origin**: Best angle of climb.  
    (Refer to Fig. 17.4 in Raymer)

---

## Jet Aircraft: Best Climb Conditions

- For jets, thrust \( T \) is mostly constant with speed.
- Best rate of climb is found by maximizing:

  \[
  V_v = V \left( \frac{T}{W} - \frac{\rho V^2 C_D}{2(W/S)} - \frac{2K}{\rho V} \left( \frac{W}{S} \right) \right) \tag{Eq. 17.42}
  \]

- Setting \( \frac{dV_v}{dV} = 0 \) and solving gives:

  \[
  V = \sqrt{ \frac{W/S}{3 \rho C_{D_0}} \left( \frac{T}{W} + \sqrt{ \left( \frac{T}{W} \right)^2 + 12 C_{D_0} K } \right) } \tag{Eq. 17.43}
  \]

- Example: The B-70 has a best climb speed of 583 kt (≈1080 km/h).

---

## Time and Fuel to Climb

- Time to climb a small height \( dh \):

  \[
  dt = \frac{dh}{V_v} \tag{Eq. 17.46}
  \]

- Fuel burn over that time:

  \[
  dW_f = -C_T T \, dt \tag{Eq. 17.47}
  \]

- Since \( V_v \) varies with altitude, it can be linearly approximated:

  \[
  V_v = V_{v_i} - a(h_{i+1} - h_i) \tag{Eq. 17.48}
  \]
  \[
  a = \frac{V_{v_2} - V_{v_1}}{h_2 - h_1} \tag{Eq. 17.49}
  \]

- Total time and fuel between two altitudes:

  \[
  t_{i+1} - t_i = \frac{1}{a} \ln \left( \frac{V_{v_i}}{V_{v_{i+1}}} \right) \tag{Eq. 17.50}
  \]
  \[
  \Delta W_{\text{fuel}} = -(CT)_{\text{avg}} (t_{i+1} - t_i) \tag{Eq. 17.51}
  \]

- Improved accuracy can be achieved via **iteration**, updating \( W \) after each step.

---

## Reference

Raymer, D. P. (2021). *Aircraft Design: A Conceptual Approach*, 6th ed., AIAA Education Series, Chapter 17.3.

---



## Future Considerations and Open Questions

As development continues, several physical constraints and operational factors need to be addressed:

### 1. **Thrust Limitations**
- The climb energy rate is ultimately limited by the available engine thrust at a given altitude.
- To model this accurately, engine performance data (e.g., thrust vs. altitude) is needed.

### 2. **TSFC Clarification**
- The thrust-specific fuel consumption (TSFC) is often provided in units such as lb fuel / lb thrust / hr.
- A consistent unit system should be defined, and conversions should be handled clearly for modeling fuel flow.

### 3. **Angle of Attack Constraints**
- The maximum achievable angle of attack limits climb steepness and lift.
- For steady climb, the angle of attack can be derived using Raymer's Equation 17.38 (refer to “Aircraft Design: A Conceptual Approach”).

### 4. **Scenario Planning**
We may consider simulating under different mission or design contexts:
- Minimum fuel climb
- Minimum time climb
- Constant Mach  
- Engine-out or degraded thrust condition

### 5. **Boundary Conditions**
We should consider the boundary conditions defined for each segment.

 

