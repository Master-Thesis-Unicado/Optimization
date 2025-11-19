# ISA Atmospheric Model

This note explains the formulas used in the `Atmosphere` class.

---

###  Temperature vs Altitude (ISA Model)

The standard temperature lapse rate is:

$$
T(h) = T_0 + a \cdot h
$$

Where:  
- \( T_0 = 288.15 \, K \) (sea-level temperature)  
- \( a = -0.0065 \, K/m \) (lapse rate)  
- \( h \): altitude in meters

---

###  Speed of Sound

The speed of sound at altitude is calculated using:

$$
a = \sqrt{\gamma \cdot R \cdot T}
$$

Where:
- \( \gamma = 1.4 \) (ratio of specific heats for air)  
- \( R = 287.05 \, J/(kg \cdot K) \) (specific gas constant)  
- \( T \): temperature in Kelvin
