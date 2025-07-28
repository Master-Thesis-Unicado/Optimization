import math

class Atmosphere:
    """Layer-based ISA model using polytropic and exponential formulations.
    Returns atmospheric properties as a function of geopotential altitude (in meters),
    and altitude-dependent gravity.
    """

    def calculate_atmospheric_properties(self, FL):
        # Constants
        g_s = 9.80665  # m/s²
        R = 287.05     # J/(kg·K)

        # Sea-level reference values
        T_MSL = 288.15  # K
        p_MSL = 101325  # Pa
        rho_MSL = 1.225  # kg/m³

        # Lapse rates
        gamma_Tropo = -0.0065  # K/m
        gamma_UpperStr = 0.001  # K/m

        # Layer boundaries
        H_G11 = 11000  # Tropopause (11 km)
        H_G20 = 20000  # Upper stratosphere base

        # Boundary values
        T_11 = 216.65  # K
        p_11 = 22632   # Pa
        rho_11 = 0.364  # kg/m³

        T_20 = 216.65  # K
        p_20 = 5474.88  # Pa
        rho_20 = 0.088  # kg/m³

        # Polytropic indices
        n_trop = 1.235
        n_uStr = 0.001

        # Convert flight level to meters
        H_G = FL * 0.3048

        # Atmospheric layers
        if H_G <= H_G11:
            T = T_MSL * (1 + (gamma_Tropo / T_MSL) * H_G)
            p = p_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (n_trop / (n_trop - 1))
            rho = rho_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (1 / (n_trop - 1))
        elif H_G <= H_G20:
            T = T_11
            p = p_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))
            rho = rho_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))
        else:
            T = T_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20))
            p = p_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20)) ** (n_uStr / (n_uStr - 1))
            rho = rho_20 * (1 - ((n_uStr - 1) / n_uStr) * (g_s / (R * T_20)) * (H_G - H_G20)) ** (1 / (n_uStr - 1))

        return T, p, rho

    def get_temperature(self, altitude_m: float) -> float:
        """Wrapper to get temperature only."""
        T, _, _ = self.calculate_atmospheric_properties(altitude_m / 0.3048)
        return T

    def get_speed_of_sound(self, altitude_m: float) -> float:
        """Compute speed of sound using T(h)."""
        T = self.get_temperature(altitude_m)
        return math.sqrt(1.4 * 287.05 * T)

    def get_gravity(self, altitude_m: float) -> float:
        """Compute gravity as a function of altitude using the inverse-square law."""
        R_e = 6371000.0  # Earth radius in meters
        g0 = 9.80665     # Standard gravity at sea level
        return g0 * (R_e / (R_e + altitude_m))**2


# --- Dummy test cases for standalone testing ---
if __name__ == "__main__":
    atm = Atmosphere()

    for FL in [0, 35000, 60000]:
        altitude_m = FL * 0.3048
        T, p, rho = atm.calculate_atmospheric_properties(FL)
        a = atm.get_speed_of_sound(altitude_m)
        g = atm.get_gravity(altitude_m)
        print(f"FL{FL} → T = {T:.2f} K, p = {p:.2f} Pa, ρ = {rho:.4f} kg/m³, a = {a:.2f} m/s, g = {g:.5f} m/s²")
