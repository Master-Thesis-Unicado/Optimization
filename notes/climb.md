# Mission Analysis — Notes and Derivations

This note explains the mathematical foundations and implementation details of the Python code used for aircraft mission segment analysis. Each section aligns closely with the structure of the source code and references the International Standard Atmosphere (ISA) model and Raymer’s fuel-burn formulations.

---

## ISA Atmospheric Model

### 1. Temperature Variation with Altitude

The International Standard Atmosphere (ISA) assumes a linear decrease in temperature with altitude within the troposphere (up to \~11 km):

$$
T(h) = T_0 + a \cdot h
$$

Where:

* $T_0 = 288.15 \; \text{K}$: sea-level standard temperature
* $a = -0.0065 \; \text{K/m}$: temperature lapse rate
* $h$: geometric altitude in meters

This is implemented in the `get_temperature()` method of the `Atmosphere` class.

---

### 2. Speed of Sound in the Atmosphere

Speed of sound varies with temperature and is computed using the formula:

$$
a(h) = \sqrt{\gamma \cdot R \cdot T(h)}
$$

Where:

* $\gamma = 1.4$: specific heat ratio of air
* $R = 287.05 \; \text{J/(kg\,K)}$: specific gas constant
* $T(h)$: temperature at altitude $h$

This can be rewritten using the ISA relation for temperature as:

$$
a(h) = a_0 \cdot \sqrt{\Theta}, \quad \text{where } \Theta = \frac{T(h)}{T_0}, \quad a_0 = \sqrt{\gamma R T_0}
$$

The `get_speed_of_sound()` method applies this formulation.

---

## Energy Height Framework

### 3. Definition of Energy Height

The total energy per unit weight, or energy height $E_s$, combines altitude and velocity:

$$
E_s = h + \frac{V^2}{2g_0}
$$

Where:

* $h$: altitude in meters
* $V$: true airspeed (m/s)
* $g_0 = 9.81 \; \text{m/s}^2$: standard gravity

This framework provides a unified measure for potential and kinetic energy.

---

### 4. Time Rate of Change of Energy Height

The rate of change of $E_s$ captures both climbing and accelerating:

$$
\frac{dE_s}{dt} = \frac{dh}{dt} + \frac{V}{g_0} \cdot \frac{dV}{dt}
$$

This expression supports the analysis of time-dependent energy changes in segments.

---

### 5. Power Available and Required

The net specific power available is expressed as:

$$
P_s = \frac{F_n - D}{W} \cdot V
$$

Matching this to the rate of change of energy height:

$$
\frac{dh}{dt} + \frac{V}{g_0} \cdot \frac{dV}{dt} = \frac{F_n - D}{W} \cdot V
$$

Where:

* $F_n$: thrust
* $D$: drag
* $W$: weight of the aircraft

---

## Fuel Burn Models for Mission Segments

### 6. Accelerated Climb (Raymer Eq 17.97)

In segments with both climb and acceleration:

$$
\Delta E_s = \dot{h} \cdot \Delta t + \frac{(V_1 + \dot{V} \cdot \Delta t)^2 - V_1^2}{2g_0}
$$

The weight fraction becomes:

$$
\frac{W_f}{W_i} = \exp\left( -\frac{\text{TSFC} \cdot g_0 \cdot \Delta E_s}{V (1 - D/T)} \right)
$$

Substituting:

* $V = M \cdot a(h) = M \cdot a_0 \cdot \sqrt{\Theta}$
* $\frac{D}{T} = \frac{C_D}{C_L} \cdot \frac{W}{T}$

This is evaluated in the `accelerated_climb()` function.

---

### 7. Subsonic Loiter (Raymer Case 8)

In steady-speed, level flight loiter conditions:

Define Endurance Factor (EF):

$$
EF = \frac{C_L}{C_D \cdot \text{TSFC} \cdot g_0}
$$

Then the weight fraction is:

$$
\frac{W_f}{W_i} = \exp\left( -\frac{\Delta t}{EF} \right)
$$

Used in `subsonic_loiter()`.

---

### 8. Idle Thrust Segment (Raymer Eq 19.7)

For segments with low, fixed thrust:

$$
\frac{W_f}{W_i} = 1 - \text{TSFC} \cdot g_0 \cdot \frac{T \cdot \Delta t}{W}
$$

Where:

* $T$: average thrust during the segment
* $\Delta t$: time duration
* $W$: initial weight

Implemented in `idle_thrust()`.

---

### 9. Constant Altitude Cruise (Breguet Range - Time Form)

Raymer’s time-based Breguet equation gives:

$$
\frac{W_f}{W_i} = \exp\left( -\text{TSFC} \cdot g_0 \cdot \frac{C_D}{C_L} \cdot \Delta t \right)
$$

Valid for steady, level, constant-speed cruise. Handled by `constant_altitude_cruise()`.

---

## TSFC Unit Conversion

### Source Format

Engine TSFC values are commonly tabulated as:

$$
\text{TSFC}_{\text{hb}} = \frac{\text{kg fuel}}{\text{kgf} \cdot \text{hr}}
$$

### Conversion to SI Units

1. Convert time:

$$
\text{TSFC}_{s,\,kgf} = \frac{\text{TSFC}_{hb}}{3600}
$$

2. Convert kgf to Newtons:

$$
\text{TSFC}_{SI} = \text{TSFC}_{s,\,kgf} \cdot g_0
$$

This gives:

$$
\text{TSFC}_{SI} = \frac{\text{kg fuel}}{\text{N} \cdot \text{s}}
$$

---

## References

All equations and modeling approaches are based on:

**Raymer, D.P. (2024). Aircraft Design: A Conceptual Approach (7th ed.)**
American Institute of Aeronautics and Astronautics (AIAA)

---

This document provides a coherent, structured explanation of the physical principles and implementation strategies behind each mission segment in the code.
