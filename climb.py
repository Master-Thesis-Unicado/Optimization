import numpy as np
from atmosphere import Atmosphere
from pathlib import Path
import pyengine as engine

# (1) Engine & aircraft configuration --------------------------------------------------
STUB = Path("D:/Icloud/iCloudDrive/Master Thesis/Mission Analysis Code/lls/stubs/engines/PW1127G-JM")
eng = engine.Engine(str(STUB))
N_ENGINES = 2  # total number of engines

# Aerodynamic/aircraft constants
atm = Atmosphere()  # ISA atmosphere model
S_ref = 122.4   # [m^2] wing reference area
CD0  = 0.02     # [-] zero-lift drag coefficient
AR   = 9.5      # [-] aspect ratio
e    = 0.85     # [-] Oswald efficiency

# Initial condition & target
initial_mass_kg  = 60000.0  # [kg]
initial_altitude = 0        # [m]
initial_speed    = 75       # [m/s]
target_altitude  = 4267.2   # [m] (14,000 ft)
dt               = 0.2      # [s] integration step

# Strategy parameter sweep (used by generate_strategy)
altitude_fractions = np.linspace(0.1, 0.9, 5)

# Engine query envelope
MACH_MIN_FOR_ENGINE   = 0.00
MACH_MAX_FOR_ENGINE   = 0.94
ALT_MIN_FT_FOR_ENGINE = 0.0
ALT_MAX_FT_FOR_ENGINE = 14000.0

# (2) Strategy profiles (chosen BEFORE running the integrator) -------------------------
class StrategyProfiles:
    """
    Strategy functions return raw weights (cw, sw).
    The integrator normalizes them so w_c + w_s = 1 and applies the specific energy magnitude.
    """
    class FixedEnergy:
        class Linear:
            @staticmethod
            def profile(altitude, velocity, altitude_fraction):
                af = float(np.clip(altitude_fraction, 0.0, 1.0))
                cw = af
                sw = 1.0 - af
                return cw, sw

        class Exponential:
            @staticmethod
            def increasing_climb(altitude, velocity, altitude_fraction):
                af = float(np.clip(altitude_fraction, 0.0, 1.0))
                cw = af * np.exp(altitude / target_altitude)
                sw = (1.0 - af) * np.exp(-altitude / target_altitude)
                return cw, sw

            @staticmethod
            def decreasing_climb(altitude, velocity, altitude_fraction):
                af = float(np.clip(altitude_fraction, 0.0, 1.0))
                cw = af * np.exp(-altitude / target_altitude)
                sw = (1.0 - af) * np.exp(altitude / target_altitude)
                return cw, sw

            @staticmethod
            def increasing_speed(altitude, velocity, altitude_fraction):
                af = float(np.clip(altitude_fraction, 0.0, 1.0))
                sw = af * np.exp(altitude / target_altitude)
                cw = (1.0 - af) * np.exp(-altitude / target_altitude)
                return cw, sw

            @staticmethod
            def decreasing_speed(altitude, velocity, altitude_fraction):
                af = float(np.clip(altitude_fraction, 0.0, 1.0))
                sw = af * np.exp(-altitude / target_altitude)
                cw = (1.0 - af) * np.exp(altitude / target_altitude)
                return cw, sw

    class ConstantRates:
        @staticmethod
        def constant_speed(altitude, velocity, altitude_fraction=None):
            # All specific energy to climb; speed held constant by construction after normalization
            cw, sw = 1.0, 0.0
            return cw, sw

        @staticmethod
        def constant_mach():
            # Returns a strategy function that still yields (cw, sw) but flags const-Mach behavior
            def _const_mach(altitude, velocity, altitude_fraction=None):
                # Kinematics in the integrator enforce dM/dt ≈ 0; weights remain (1, 0)
                cw, sw = 1.0, 0.0
                return cw, sw
            _const_mach._const_mach = True
            return _const_mach


def generate_strategy(profile='linear'):
    """
    Build a list of (altitude_fraction, strategy_function) pairs for the requested profile.
    For 'constant_speed' and 'constant_mach', altitude_fraction is None.
    """
    strategies = []
    for af in altitude_fractions:
        if profile == 'linear':
            func = StrategyProfiles.FixedEnergy.Linear.profile
        elif profile == 'exponential_increasing_climb':
            func = StrategyProfiles.FixedEnergy.Exponential.increasing_climb
        elif profile == 'exponential_decreasing_climb':
            func = StrategyProfiles.FixedEnergy.Exponential.decreasing_climb
        elif profile == 'exponential_increasing_speed':
            func = StrategyProfiles.FixedEnergy.Exponential.increasing_speed
        elif profile == 'exponential_decreasing_speed':
            func = StrategyProfiles.FixedEnergy.Exponential.decreasing_speed
        else:
            continue
        
        strategies.append((af, lambda h, V, af=af, f=func: f(h, V, af)))

    if profile == 'constant_speed':
        strategies.append((None, StrategyProfiles.ConstantRates.constant_speed))
    elif profile == 'constant_mach':
        strategies.append((None, StrategyProfiles.ConstantRates.constant_mach()))

    return strategies


# (3) Aerodynamics (used inside the integrator) ---------------------------------
def compute_drag(rho, V, S, CD):
    """D = 0.5 * rho * V^2 * S * CD  [N]"""
    return 0.5 * rho * V**2 * S * CD

def compute_CD(CL, AR, e, CD0):
    """CD = CD0 + CL^2 / (pi * AR * e)"""
    return CD0 + (CL**2) / (np.pi * AR * e)

# (4) Lever solver (FADEC-like) --------------------------------------------------------
def find_lever_for_thrust(required_thrust_total, mach, altitude_ft,
                          lever_grid=None, allow_refine=True):
    """
    Simple FADEC-like lever solver:
      1) sample thrust at a lever grid (0..1)
      2) check idle/max
      3) linear interpolate in the bracketing interval
      4) optional single refine call at the interpolated lever

    Returns: (lever, per_engine_thrust, thrust_limited_flag)
    """
    thrust_limited = False
    T_req = float(required_thrust_total) / float(N_ENGINES)

    
    if lever_grid is None:
        lever_grid = np.linspace(0.0, 1.0, 21)

    def safe_thrust(lv):
        try:
            Tv = eng.get_thrust_with_lever_position(float(lv), float(mach), float(altitude_ft))
            if not np.isfinite(Tv) or Tv < 0:
                return None
            return float(Tv)
        except Exception:
            return None

    thrusts = [safe_thrust(lv) for lv in lever_grid]
    valid_idx = [i for i, Tv in enumerate(thrusts) if Tv is not None]

    if not valid_idx:
        return None, None, thrust_limited

    
    for i in range(1, len(lever_grid)):
        if (thrusts[i] is not None) and (thrusts[i-1] is not None) and (thrusts[i] < thrusts[i-1]):
            thrusts[i] = thrusts[i-1]

    T0 = thrusts[0]
    T1 = thrusts[-1]

    # Idle meets demand
    if (T0 is not None) and (T0 >= T_req):
        return 0.0, T0, thrust_limited

    # Max insufficient -> clamp (thrust-limited)
    if (T1 is None) or (T1 < T_req):
        if T1 is None:
            return None, None, thrust_limited
        thrust_limited = True
        return 1.0, T1, thrust_limited

    # Search for bracket and interpolate
    for i in range(len(lever_grid) - 1):
        Ti   = thrusts[i]
        Tip1 = thrusts[i + 1]
        if (Ti is None) or (Tip1 is None):
            continue
        if (Ti <= T_req) and (T_req <= Tip1) and (Tip1 > Ti):
            li, lj = lever_grid[i], lever_grid[i + 1]
            lv = li + (T_req - Ti) * (lj - li) / (Tip1 - Ti)
            if allow_refine:
                Tstar = safe_thrust(lv)
                if Tstar is not None:
                    return float(lv), float(Tstar), thrust_limited
            # fallback to closer endpoint if refine failed
            if (T_req - Ti) <= (Tip1 - T_req):
                return float(li), float(Ti), thrust_limited
            else:
                return float(lj), float(Tip1), thrust_limited

    # Fallback: closest valid grid point
    diffs = [(abs(thrusts[i] - T_req), lever_grid[i], thrusts[i]) for i in valid_idx]
    diffs.sort(key=lambda x: x[0])
    _, lv_best, Tv_best = diffs[0]
    return float(lv_best), float(Tv_best), thrust_limited

# (5) Main integrator (the core of the "run") -----------------------------------------
def simulate_climb_path(strategy_function, altitude_fraction_input, dt=1.0):
    """
    Integrate climb using a specific-energy split:
      - Strategy provides (cw, sw) → normalized to (w_c, w_s).
      - E_DOT_cmd [m/s] is the commanded specific-energy magnitude.
      - Kinematics:
          dh/dt = w_c * E_DOT_cmd
          dv/dt = (g / V) * (w_s * E_DOT_cmd)  (or const-Mach variant)
      - Power balance:
          F_req = D + (W * E_DOT) / V
    """
    gamma, R = 1.4, 287.05  # for a = sqrt(gamma * R * T)

    # Histories
    h, V, t = [initial_altitude], [initial_speed], [0.0]
    mass_kg = initial_mass_kg
    lever_positions = []
    none_lever_times = []
    limit_times = []

    # Fuel diagnostics
    fuel_flow_kg_s_hist = []  # total (all engines)
    fuel_burn_step_kg   = []
    mass_kg_hist        = [mass_kg]

    # March until target altitude
    while h[-1] < target_altitude:
        altitude, velocity, time_s = h[-1], V[-1], t[-1]

        # Weight
        g = atm.get_gravity(altitude)
        W = mass_kg * g

        # (1) Strategy → normalized shares (w_c + w_s = 1)
        cw, sw = strategy_function(altitude, velocity, altitude_fraction_input)
        s = max(cw + sw, 1e-12)
        w_c = cw / s
        w_s = sw / s

        # (2) Atmosphere
        T, P, rho = atm.calculate_atmospheric_properties(altitude)
        a    = np.sqrt(gamma * R * T)
        mach = velocity / max(a, 1e-9)
        alt_ft = altitude * 3.28084

        # Engine-query-safe state
        mach_eng   = float(np.clip(mach,   MACH_MIN_FOR_ENGINE, MACH_MAX_FOR_ENGINE))
        alt_ft_eng = float(np.clip(alt_ft, ALT_MIN_FT_FOR_ENGINE, ALT_MAX_FT_FOR_ENGINE))

        # (3) Aerodynamics
        CL_dyn = (2 * W) / (max(rho, 1e-12) * max(velocity, 1e-6)**2 * S_ref)
        CD = compute_CD(CL_dyn, AR, e, CD0)
        D  = compute_drag(rho, velocity, S_ref, CD)

        # (4) Commanded specific energy (global magnitude; strategies only split it)
        E_DOT_cmd = 6.5  # [m/s]
 
        if getattr(strategy_function, "_const_mach", False):
            eps = 1.0
            T2, _, _ = atm.calculate_atmospheric_properties(altitude + eps)
            dTdh = (T2 - T) / eps
            dadh = 0.5 * a / max(T, 1e-9) * dTdh

            dh_dt = w_c * E_DOT_cmd
            dv_dt = (velocity / max(a, 1e-9)) * dadh * dh_dt
        else:
            dh_dt = w_c * E_DOT_cmd
            dv_dt = (g / max(velocity, 1e-9)) * (w_s * E_DOT_cmd)

        # (5) Power balance : total aircraft required thrust
        E_DOT = dh_dt + (velocity / g) * dv_dt
        F_required_total = D + (E_DOT * W) / max(velocity, 1e-9)

        # (6) Lever selection (FADEC-like solver; includes idle/max logic)
        lv, real_thrust_per_engine, thrust_limited = find_lever_for_thrust(
            F_required_total, mach_eng, alt_ft_eng, lever_grid=None, allow_refine=True
        )

        lever_positions.append(lv)

        if lv is None or real_thrust_per_engine is None:
            print(f"[WARNING] No valid lever at h={altitude:.1f} m, V={velocity:.1f} m/s "
                  f"(M={mach:.2f}, Alt={alt_ft:.0f} ft)")
            none_lever_times.append(time_s)
        else:
            if thrust_limited:
                limit_times.append(time_s)
            # align engine state to selected lever for TSFC
            real_thrust_per_engine = eng.get_thrust_with_lever_position(float(lv), mach_eng, alt_ft_eng)

        # (7) Fuel burn 
        if lv is not None and real_thrust_per_engine is not None:
            tsfc = eng.get_tsfc()  # per engine at current state
            if tsfc > 1e-3:        
                tsfc /= 3600.0
            fuel_flow_kg_s_per_engine = max(tsfc, 0.0) * max(real_thrust_per_engine, 0.0)
            fuel_flow_kg_s_total = fuel_flow_kg_s_per_engine * N_ENGINES
            burned_kg = fuel_flow_kg_s_total * dt
            mass_kg = max(mass_kg - burned_kg, 0.0)
        else:
            fuel_flow_kg_s_total = np.nan
            burned_kg = 0.0

        fuel_flow_kg_s_hist.append(fuel_flow_kg_s_total)
        fuel_burn_step_kg.append(burned_kg)
        mass_kg_hist.append(mass_kg)

        # (8) Integrate state
        h_new = altitude + dh_dt * dt
        V_new = velocity + dv_dt * dt

        # Terminal condition with partial step
        if h_new >= target_altitude:
            h_new = target_altitude
            dt_last = (target_altitude - altitude) / max(dh_dt, 1e-9)
            t_new = time_s + dt_last
            V_new = velocity + dv_dt * dt_last
            t.append(t_new); h.append(h_new); V.append(V_new)
            break

        h.append(h_new); V.append(V_new); t.append(time_s + dt)

    # Final summary & diagnostics bundle
    final_results = {
        "Final Altitude": h[-1],
        "Final Velocity": V[-1],
        "Total Climb Time": t[-1],
        "Final Lever Position": lever_positions[-1] if lever_positions else None,
        "Final Mass (kg)": mass_kg,
        "Total Fuel Burned (kg)": initial_mass_kg - mass_kg,
        "Engines": N_ENGINES,
    }

    diagnostics = {
        "altitudes": h,
        "velocities": V,
        "times": t,
        "lever_positions": lever_positions,
        "none_lever_times": none_lever_times,
        "limit_times": limit_times,
        "fuel_flow_kg_s": fuel_flow_kg_s_hist,    # total (all engines)
        "fuel_burn_step_kg": fuel_burn_step_kg,
        "mass_kg": mass_kg_hist[:-1],
    }

    return t, h, V, lever_positions, final_results, diagnostics

# (6) Runner stub (entry used by external pipeline) ------------------------------------
def simulate_physics_based_climb():
    print("Physics-based climb simulation placeholder executed.")

# (7) Quick self-test when run directly ------------------------------------------------
if __name__ == "__main__":
    # quick sanity run using a mid AF linear profile
    strategies = generate_strategy("linear")
    af, strat_fn = strategies[2]  # e.g., AF=0.50
    t, h, V, lever_positions, final_results, diagnostics = simulate_climb_path(strat_fn, af, dt=dt)
    print("[INFO] Finished one test run.")
    print("[INFO] Final Altitude (m):", final_results["Final Altitude"])
    print("[INFO] Final Velocity (m/s):", final_results["Final Velocity"])
    print("[INFO] Total Climb Time (s):", final_results["Total Climb Time"])
    print("[INFO] Total Fuel Burned (kg):", final_results["Total Fuel Burned (kg)"])
