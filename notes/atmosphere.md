# Mission Analysis — Atmosphere Model

This note explains the mathematical foundations and implementation details of the Python code used for atmosphere model analysis. Each section aligns closely with the structure of the source code and references the International Standard Atmosphere (ISA) model.

---

## ISA Atmospheric Model

### 1. Temperature Variation with Altitude

The International Standard Atmosphere (ISA) assumes a linear decrease in temperature with altitude within the troposphere (up to approximately 11 km):

$$
T(h) = T_0 + a \cdot h
$$

Where:

- \( T_0 = 288.15 \; \text{K} \): Sea-level standard temperature  
- \( a = -0.0065 \; \text{K/m} \): Temperature lapse rate  
- \( h \): Geometric altitude in meters

This is implemented in the `get_temperature()` method of the `Atmosphere` class.

---

### 2. Speed of Sound in the Atmosphere

The speed of sound varies with temperature and is computed using:

$$
a(h) = \sqrt{\gamma \cdot R \cdot T(h)}
$$

Where:

- \( \gamma = 1.4 \): Ratio of specific heats for air  
- \( R = 287.05 \; \mathrm{J/(kg \cdot K)} \): Specific gas constant for air

- \( T(h) \): Temperature at altitude \( h \)

Alternatively, using a nondimensional temperature ratio:

$$
a(h) = a_0 \cdot \sqrt{\Theta}, \quad \text{where} \quad \Theta = \frac{T(h)}{T_0}, \quad a_0 = \sqrt{\gamma R T_0}
$$

At sea level, \( a_0 \approx 340.3 \; \text{m/s} \).  
This is implemented in the `get_speed_of_sound()` method of the `Atmosphere` class.

---

### 3. Implementation Overview

In Python, the implementation of the ISA temperature and speed of sound follows:

```python
class Atmosphere:
    def get_temperature(self, altitude):
        return T0 + lapse_rate * altitude

    def get_speed_of_sound(self, altitude):
        T = self.get_temperature(altitude)
        return math.sqrt(gamma * R * T)
```

### Sources / References

This chapter lists the primary literature and standards used in the development of the atmospheric model for aircraft mission analysis. The references include both textbook derivations and internationally recognized atmospheric data models.

---

#### [1] Anderson, J. D. (2016). *Introduction to Flight*.  
**8th Edition**, McGraw-Hill Education, New York.  
Relevant Sections: Chapter 4, pp. 177–178  
Provides derivations for speed of sound in compressible flow, ISA assumptions, and Mach number definitions.  
Equations used:  
- \( a = \sqrt{\gamma R T} \) (Eq. 4.54)  
- \( M = \frac{V}{a} \) (Eq. 4.55)


#### [2] U.S. Standard Atmosphere, 1976  
NOAA, NASA, and United States Air Force.  
Washington, D.C.: U.S. Government Printing Office.

Defines standard atmospheric properties such as temperature, pressure, and density profiles as a function of altitude. 
 

**Pages 50–179** provide detailed tabulated values for temperature, pressure, and density across altitude, which can be used to validate numerical implementations of atmospheric models.

---

**Reflection:**  
The current implementation of the atmospheric model doesn't includes **layer-specific behavior**, such as polytropic and isothermal modeling across the troposphere and lower/upper stratosphere.

The layer-specific atmospheric model, is based on:
##### [2.1]  Institute of Flight System Dynamics, Technical University of Munich.  
*Centralized Definition Notes*, Version 3.0, p. 101.  
Edited by Prof. Dr.-Ing. Florian Holzapfel.  


```python
    def atmospheric_properties(self, FL):
        # 1.1 Constants initialization
        g_s = 9.80665  # Gravitational acceleration at sea level (m/s²)
        R = 287.05  # Specific gas constant for dry air (J/(kg·K))
        
        # 1.2 Reference values at Mean Sea Level (MSL)
        T_MSL = 288.15  # Temperature at MSL (K)
        p_MSL = 101325  # Pressure at MSL (Pa)
        rho_MSL = 1.225  # Density at MSL (kg/m³)

        # 1.3 Layer-specific thermodynamic constants
        n_trop = 1.235  # Polytropic index for Troposphere
        n_uStr = 0.001  # Polytropic index for Upper Stratosphere
        
        # 1.4 Atmospheric layer boundary heights (m)
        H_G11 = 11000  # Tropopause height
        H_G20 = 20000  # Upper stratosphere base height

        # 1.5 Standard values at layer boundaries
        T_11 = 216.65  # Temperature at tropopause (K)
        p_11 = 22632  # Pressure at tropopause (Pa)
        rho_11 = 0.364  # Density at tropopause (kg/m³)

        T_20 = 216.65  # Temperature at stratosphere base (K)
        p_20 = 5474.88  # Pressure at stratosphere base (Pa)
        rho_20 = 0.088  # Density at stratosphere base (kg/m³)

        # 1.6 Temperature gradients (K/m)
        gamma_Tropo = -0.0065  # Tropospheric lapse rate
        gamma_UpperStr = 0.001  # Upper stratosphere lapse rate

        H_G = FL * 0.3048  # Convert flight level (feet) to geopotential height (meters)

        # Atmospheric layer calculations
        if H_G <= H_G11:  # Troposphere (0-11 km)
            # Theory: Temperature decreases linearly with height, use polytropic process equations
            T = T_MSL * (1 + (gamma_Tropo / T_MSL) * H_G)
            p = p_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (n_trop / (n_trop - 1))
            rho = rho_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (1 / (n_trop - 1))

        elif H_G <= H_G20:  # Lower Stratosphere (11-20 km)
            # Theory: Isothermal layer, pressure/density follow exponential decay (hydrostatic equilibrium)
            T = T_11
            p = p_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))
            rho = rho_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))

        else:  # Upper Stratosphere (>20 km)
            # Theory: Temperature increases slightly, modified polytropic relationships
            T = T_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20))
            p = p_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20)) ** (n_uStr / (n_uStr - 1))
            rho = rho_20 * (1 - ((n_uStr - 1) / n_uStr) * (g_s / (R * T_20)) * (H_G - H_G20)) ** (1 / (n_uStr - 1))

        # 1.7 Return atmospheric properties tuple (Temperature, Pressure, Density)
        return T, p, rho
```
 