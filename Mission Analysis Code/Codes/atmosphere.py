import math
import numpy as np

class Atmosphere:
    """Layer-based ISA model using polytropic and exponential formulations.
    Returns atmospheric properties as a function of geopotential altitude (in meters),
    and altitude-dependent gravity.
    """
    
    # Atmospheric constants
    G_C = 9.80665  # m/s²
    R = 287.05     # J/(kg·K)
    
    # Sea-level reference values
    T_MSL = 288.15  # K
    P_MSL = 101325  # Pa
    RHO_MSL = 1.225  # kg/m³
    
    # Lapse rates
    GAMMA_TROPO = -0.0065  # K/m
    GAMMA_UPPER_STR = 0.001  # K/m
    
    # Layer boundaries
    H_G11 = 11000  # Tropopause (11 km)
    H_G20 = 20000  # Upper stratosphere base
    
    # Boundary values
    T_11 = 216.65  # K
    P_11 = 22632   # Pa
    RHO_11 = 0.364  # kg/m³
    
    T_20 = 216.65  # K
    P_20 = 5474.88  # Pa
    RHO_20 = 0.088  # kg/m³
    
    # Polytropic indices
    N_TROP = 1.235
    N_USTR = 0.001

    def calculate_atmospheric_properties(self, FL):
        """Calculate atmospheric properties from Flight Level (FL).
        
        Args:
            FL: Flight Level (altitude in hundreds of feet)
            
        Returns:
            tuple: (T [K], p [Pa], rho [kg/m^3])
        """
        # Convert flight level to meters
        altitude_m = FL * 0.3048
        return self.calculate_atmospheric_properties_meters(altitude_m)
    
    def calculate_atmospheric_properties_meters(self, altitude_m: float):
        """Calculate atmospheric properties directly from altitude in meters (no conversion needed)."""
        return self._calculate_properties_from_meters(altitude_m)
    
    def _calculate_properties_from_meters(self, H_G: float):
        """Calculate atmospheric properties from altitude in meters using ISA model.
        
        Args:
            H_G: Geopotential altitude in meters
            
        Returns:
            tuple: (T [K], p [Pa], rho [kg/m^3])
        """
        # Atmospheric layers
        if H_G <= self.H_G11:
            T = self.T_MSL * (1 + (self.GAMMA_TROPO / self.T_MSL) * H_G)
            p = self.P_MSL * (1 + (self.GAMMA_TROPO / self.T_MSL) * H_G) ** (self.N_TROP / (self.N_TROP - 1))
            rho = self.RHO_MSL * (1 + (self.GAMMA_TROPO / self.T_MSL) * H_G) ** (1 / (self.N_TROP - 1))
        elif H_G <= self.H_G20:
            T = self.T_11
            p = self.P_11 * math.exp(-self.G_C / (self.R * self.T_11) * (H_G - self.H_G11))
            rho = self.RHO_11 * math.exp(-self.G_C / (self.R * self.T_11) * (H_G - self.H_G11))
        else:
            T = self.T_20 * (1 + (self.GAMMA_UPPER_STR / self.T_20) * (H_G - self.H_G20))
            p = self.P_20 * (1 + (self.GAMMA_UPPER_STR / self.T_20) * (H_G - self.H_G20)) ** (self.N_USTR / (self.N_USTR - 1))
            rho = self.RHO_20 * (1 - ((self.N_USTR - 1) / self.N_USTR) * (self.G_C / (self.R * self.T_20)) * (H_G - self.H_G20)) ** (1 / (self.N_USTR - 1))

        return T, p, rho

    def get_temperature(self, altitude_m: float) -> float:
        """Wrapper to get temperature only."""
        T, _, _ = self.calculate_atmospheric_properties_meters(altitude_m)
        return T

    def get_speed_of_sound(self, altitude_m: float) -> float:
        """Compute speed of sound using T(h)."""
        T = self.get_temperature(altitude_m)
        
        # Validate temperature
        if T <= 0 or not np.isfinite(T):
            raise ValueError(f"Invalid temperature {T} K at altitude {altitude_m} m")
        
        return math.sqrt(1.4 * self.R * T)

    def get_gravity(self, altitude_m: float) -> float:
        """Compute gravity as a function of altitude using the inverse-square law."""
        R_e = 6371000.0  # Earth radius in meters
        return self.G_C * (R_e / (R_e + altitude_m))**2


