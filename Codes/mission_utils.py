# ========================================================================
# MISSION UTILITIES MODULE
# ========================================================================
"""
Shared computational primitives for all mission phases.

Mathematical utilities:
    1. State conversions: V ↔ M via V = M·a(h)
    2. Flight envelope: V_stall = √(2·W/(ρ·S·CL_max)), M_min with safety margin
    3. Propulsion: δ(T_req, M, h), ṁ = TSFC·T, fuel flow integration
    4. Performance: Ps = (T-D)·V/(m·g), a = (T-D)/m, thrust balance
    5. Array operations: padding, indexing, time series construction

All functions handle edge cases (division by zero, invalid inputs) with guards.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Aircraft parameters and atmospheric model
from aircraft_config import (
    a_from_altitude, isa_properties, G_C, S_REF_M2, M_MMO, N_ENGINES, M_MIN_EFFECTIVE
)

# Engine model interface
from pyengine_wrapper import EngineWrapper


# ========================================================================
# SECTION 1: FLIGHT STATE CONVERSIONS
# ========================================================================

def velocity_from_mach(mach: float, altitude_m: float) -> float:
    """
    Convert Mach to true airspeed: V = M·a(h).
    
    Parameters:
        mach: M [-] - Mach number
        altitude_m: h [m] - altitude
        
    Returns:
        V [m/s]: true airspeed
    """
    a = a_from_altitude(altitude_m)  # a(h) [m/s]: speed of sound
    return mach * a


def mach_from_velocity(velocity_mps: float, altitude_m: float) -> float:
    """
    Convert true airspeed to Mach: M = V/a(h).
    
    Division by zero guard: a(h) ≥ 10^-9 m/s.
    
    Parameters:
        velocity_mps: V [m/s] - true airspeed
        altitude_m: h [m] - altitude
        
    Returns:
        M [-]: Mach number
    """
    a = a_from_altitude(altitude_m)  # a(h) [m/s]: speed of sound
    return velocity_mps / max(a, 1e-9)


# ========================================================================
# SECTION 2: FLIGHT ENVELOPE LIMITS
# ========================================================================

def calculate_stall_speed(weight_kg: float, altitude_m: float,
                          cl_max: float, s_ref_m2: Optional[float] = None) -> float:
    """
    Compute stall speed via lift equation.
    
    Stall condition: L = W at minimum dynamic pressure
        q_min = W/(S·CL_max)
        V_stall = √(2·q_min/ρ) = √(2·W/(ρ·S·CL_max))
    
    Parameters:
        weight_kg: m [kg] - aircraft mass
        altitude_m: h [m] - altitude
        cl_max: CL_max [-] - maximum lift coefficient
        s_ref_m2: S [m²] - reference area (default: S_REF_M2)
        
    Returns:
        V_stall [m/s]: stall speed
    """
    if s_ref_m2 is None:
        s_ref_m2 = S_REF_M2
    
    # Atmospheric density: ρ(h)
    T, p, rho = isa_properties(altitude_m)
    
    if rho <= 0:
        raise ValueError(f"Invalid density at altitude {altitude_m:.0f}m: rho={rho}")
    
    # Stall computation
    weight_N = weight_kg * G_C                      # W = m·g [N]
    q_min = weight_N / (s_ref_m2 * cl_max)          # q_min = W/(S·CL_max) [Pa]
    v_stall_mps = np.sqrt(2 * q_min / rho)          # V_stall = √(2·q_min/ρ) [m/s]
    
    return float(v_stall_mps)


def calculate_stall_mach(weight_kg: float, altitude_m: float,
                         cl_max: float, s_ref_m2: Optional[float] = None,
                         safety_margin: float = 1.3) -> float:
    """
    Compute minimum safe Mach with safety margin above stall.
    
    Safety factor: V_min = V_stall × k_safety (k_safety = 1.3 → 30% margin)
    Mach conversion: M_min = V_min/a(h)
    
    Parameters:
        weight_kg: m [kg] - aircraft mass
        altitude_m: h [m] - altitude
        cl_max: CL_max [-] - maximum lift coefficient
        s_ref_m2: S [m²] - reference area (default: S_REF_M2)
        safety_margin: k_safety [-] - safety factor (default: 1.3)
        
    Returns:
        M_min [-]: minimum safe Mach
    """
    v_stall = calculate_stall_speed(weight_kg, altitude_m, cl_max, s_ref_m2)
    v_min = v_stall * safety_margin                  # V_min = V_stall × k_safety
    m_min = mach_from_velocity(v_min, altitude_m)    # M_min = V_min/a(h)
    
    return float(m_min)


# ========================================================================
# SECTION 3: PROPULSION COMPUTATIONS
# ========================================================================

def find_lever_for_thrust(eng: EngineWrapper, required_thrust_total: float,
                          mach: float, altitude_m: float,
                          n_engines: int = N_ENGINES,
                          lever_grid: Optional[np.ndarray] = None,
                          allow_refine: bool = True) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Solve for throttle lever position: δ(T_req, M, h).
    
    Algorithm:
        1. Sample T(δ, M, h) over δ grid (21 points, δ ∈ [0,1])
        2. Enforce monotonicity: T(δ_i+1) ≥ T(δ_i)
        3. Find bracketing interval: T(δ_i) ≤ T_req ≤ T(δ_i+1)
        4. Interpolate: δ = δ_i + α·(δ_i+1 - δ_i) where α ∈ [0,1]
        5. Optional refinement: 11-point sub-grid for precision
    
    Thrust limited: T_max(M, h) < T_req → δ = 1, flag = True
    
    Parameters:
        eng: EngineWrapper - engine model T(δ, M, h)
        required_thrust_total: T_req [N] - required total thrust (all engines)
        mach: M [-] - Mach number
        altitude_m: h [m] - altitude
        n_engines: N_eng - number of engines (default: N_ENGINES)
        lever_grid: δ_grid - sample points (default: linspace(0,1,21))
        allow_refine: bool - enable sub-grid refinement
        
    Returns:
        (δ [-], T_per_engine [N], thrust_limited: bool)
    """
    thrust_limited = False
    T_req = float(required_thrust_total) / float(n_engines)

    if lever_grid is None:
        lever_grid = np.linspace(0.0, 1.0, 21)

    def safe_thrust(lv):
        """Helper function to safely query engine thrust."""
        Tv = eng.thrust_with_lever(float(lv), float(mach), float(altitude_m))
        return Tv

    # Sample thrust at all lever positions
    thrusts = [safe_thrust(lv) for lv in lever_grid]
    valid_idx = [i for i, Tv in enumerate(thrusts) if Tv is not None]

    if not valid_idx:
        return None, None, thrust_limited

    # Enforce weak monotonicity (handle irregular engine data)
    for i in range(1, len(lever_grid)):
        if (thrusts[i] is not None) and (thrusts[i-1] is not None) and (thrusts[i] < thrusts[i-1]):
            thrusts[i] = thrusts[i-1]

    T0 = thrusts[0]
    T1 = thrusts[-1]

    # Check if minimum lever already exceeds requirement
    if (T0 is not None) and (T0 >= T_req):
        return float(lever_grid[0]), T0, thrust_limited

    # Check if maximum lever is insufficient (thrust limited)
    if (T1 is not None) and (T1 <= T_req):
        thrust_limited = True
        return float(lever_grid[-1]), T1, thrust_limited

    # Interpolate to find required lever position
    for i in range(len(lever_grid)-1):
        T_curr = thrusts[i]
        T_next = thrusts[i+1]
        if (T_curr is not None) and (T_next is not None):
            if T_curr <= T_req <= T_next:
                if allow_refine:
                    # Refine with smaller sub-grid for accuracy
                    fine_grid = np.linspace(lever_grid[i], lever_grid[i+1], 11)
                    fine_thrusts = [safe_thrust(lv) for lv in fine_grid]
                    for j in range(len(fine_grid)-1):
                        if (fine_thrusts[j] is not None) and (fine_thrusts[j+1] is not None):
                            if fine_thrusts[j] <= T_req <= fine_thrusts[j+1]:
                                t = (T_req - fine_thrusts[j]) / (fine_thrusts[j+1] - fine_thrusts[j])
                                lever = fine_grid[j] + t * (fine_grid[j+1] - fine_grid[j])
                                T_actual = safe_thrust(lever)
                                return float(lever), T_actual, thrust_limited
                else:
                    # Simple linear interpolation without refinement
                    t = (T_req - T_curr) / (T_next - T_curr)
                    lever = lever_grid[i] + t * (lever_grid[i+1] - lever_grid[i])
                    T_actual = safe_thrust(lever)
                    return float(lever), T_actual, thrust_limited

    return None, None, thrust_limited


def calculate_fuel_flow_rate_safe(tsfc: Optional[float], thrust_per_engine_N: float,
                                  n_engines: int = N_ENGINES) -> float:
    """
    Compute total fuel flow rate: ṁ = TSFC·T_per_engine·N_eng [kg/s].
    
    Guard: Returns 0.0 if TSFC invalid (None, non-finite, negative).
    
    Parameters:
        tsfc: TSFC [kg/(N·s)] - thrust-specific fuel consumption (can be None)
        thrust_per_engine_N: T_per_engine [N] - thrust per engine
        n_engines: N_eng - number of engines (default: N_ENGINES)
        
    Returns:
        ṁ [kg/s]: total fuel flow rate (0.0 if invalid)
    """
    if tsfc is None or not np.isfinite(tsfc) or tsfc < 0:
        return 0.0
    return tsfc * max(thrust_per_engine_N, 0.0) * n_engines


def validate_tsfc(tsfc: Optional[float], fallback: float = 0.0) -> float:
    """
    Sanitize TSFC with fallback for invalid values.
    
    Invalid: None, non-finite, or negative → return fallback.
    
    Parameters:
        tsfc: TSFC [kg/(N·s)] - value to validate
        fallback: fallback value (default: 0.0)
        
    Returns:
        TSFC [kg/(N·s)]: valid value or fallback
    """
    if tsfc is None or not np.isfinite(tsfc) or tsfc < 0:
        return fallback
    return tsfc


def thrust_per_engine_to_total(thrust_per_engine_N: float, 
                               n_engines: int = N_ENGINES) -> float:
    """
    Aggregate thrust: T_total = T_per_engine × N_eng.
    
    Parameters:
        thrust_per_engine_N: T_per_engine [N] - thrust per engine
        n_engines: N_eng - number of engines (default: N_ENGINES)
        
    Returns:
        T_total [N]: total thrust
    """
    return thrust_per_engine_N * n_engines


def calculate_fuel_consumption_step(thrust_total_N: float, tsfc_kg_per_N_s: float,
                                    time_step_s: float) -> float:
    """
    Integrate fuel consumption over time step: Δm = ṁ·Δt = TSFC·T·Δt.
    
    Parameters:
        thrust_total_N: T [N] - total thrust
        tsfc_kg_per_N_s: TSFC [kg/(N·s)] - thrust-specific fuel consumption
        time_step_s: Δt [s] - time step
        
    Returns:
        Δm [kg]: fuel consumed
    """
    fuel_flow_kgps = thrust_total_N * tsfc_kg_per_N_s  # ṁ = TSFC·T [kg/s]
    return fuel_flow_kgps * time_step_s                # Δm = ṁ·Δt [kg]


# ========================================================================
# SECTION 4: MASS EVOLUTION
# ========================================================================

def update_weight_after_burn(current_weight_kg: float,
                            fuel_burned_kg: float) -> float:
    """
    Update mass after fuel consumption: m_new = m_current - Δm_fuel.
    
    Guard: Raises ValueError if m_new ≤ 0.
    
    Parameters:
        current_weight_kg: m_current [kg] - current mass
        fuel_burned_kg: Δm_fuel [kg] - fuel consumed
        
    Returns:
        m_new [kg]: updated mass
    """
    new_weight = current_weight_kg - fuel_burned_kg
    if new_weight <= 0:
        raise ValueError(f"Mass became non-positive: {current_weight_kg:.1f} - {fuel_burned_kg:.1f} = {new_weight:.1f} kg")
    return new_weight


# ========================================================================
# SECTION 5: PERFORMANCE METRICS
# ========================================================================

def calculate_specific_excess_power(thrust_total_N: float, drag_N: float,
                                   weight_kg: float, velocity_mps: float) -> float:
    """
    Compute specific excess power: Ps = (T-D)·V/(m·g) [m/s].
    
    Excess power: P_excess = (T-D)·V [W]
    Specific: Ps = P_excess/W = (T-D)·V/(m·g) [m/s]
    
    Division guards: V ≥ 10^-9, W ≥ 10^-9.
    
    Parameters:
        thrust_total_N: T [N] - total thrust
        drag_N: D [N] - drag force
        weight_kg: m [kg] - aircraft mass
        velocity_mps: V [m/s] - true airspeed
        
    Returns:
        Ps [m/s]: specific excess power
    """
    weight_N = weight_kg * G_C                          # W = m·g [N]
    velocity_safe = max(velocity_mps, 1e-9)             # Guard
    weight_safe = max(weight_N, 1e-9)                   # Guard
    
    return (thrust_total_N - drag_N) * velocity_safe / weight_safe


def calculate_required_thrust_level_flight(drag_N: float) -> float:
    """
    Level flight thrust balance: T = D.
    
    Steady level flight: Ps = 0 → T = D.
    
    Parameters:
        drag_N: D [N] - drag force
        
    Returns:
        T_req [N]: required thrust
    """
    return float(drag_N)


def calculate_acceleration_rate(thrust_total_N: float, drag_N: float, 
                               mass_kg: float) -> float:
    """
    Compute linear acceleration: a = (T-D)/m [m/s²].
    
    Newton's second law: F_net = m·a → a = F_net/m = (T-D)/m.
    
    Division guard: m ≥ 10^-9.
    
    Parameters:
        thrust_total_N: T [N] - total thrust
        drag_N: D [N] - drag force
        mass_kg: m [kg] - aircraft mass
        
    Returns:
        a [m/s²]: acceleration (positive) or deceleration (negative)
    """
    return (thrust_total_N - drag_N) / max(mass_kg, 1e-9)


def calculate_time_from_altitude_change(altitude_change_m: float, 
                                       ps_mps: float) -> float:
    """
    Compute time for altitude change: Δt = Δh/Ps [s].
    
    Energy rate: Ps = dh/dt → Δt = Δh/Ps.
    
    Guard: Returns 0.0 if |Ps| < 10^-6.
    
    Parameters:
        altitude_change_m: Δh [m] - altitude change (signed)
        ps_mps: Ps [m/s] - specific excess power
        
    Returns:
        Δt [s]: time duration (0.0 if Ps negligible)
    """
    if abs(ps_mps) < 1e-6:
        return 0.0
    return abs(altitude_change_m) / abs(ps_mps)


def calculate_time_from_velocity_change(velocity_change_mps: float,
                                       acceleration_mps2: float) -> float:
    """
    Compute time for velocity change: Δt = ΔV/a [s].
    
    Kinematics: a = dV/dt → Δt = ΔV/a.
    
    Guard: Returns 0.0 if |a| < 10^-6.
    
    Parameters:
        velocity_change_mps: ΔV [m/s] - velocity change (signed)
        acceleration_mps2: a [m/s²] - acceleration
        
    Returns:
        Δt [s]: time duration (0.0 if acceleration negligible)
    """
    if abs(acceleration_mps2) < 1e-6:
        return 0.0
    return abs(velocity_change_mps) / abs(acceleration_mps2)


# ========================================================================
# SECTION 6: ARRAY PROCESSING UTILITIES
# ========================================================================

def pad_array_to_length(arr: np.ndarray, target_length: int,
                        pad_value: float) -> np.ndarray:
    """
    Pad array to specified length.
    
    Operation: arr_padded = [arr, pad_value × (N_target - len(arr))]
    No-op if len(arr) ≥ N_target.
    
    Parameters:
        arr: array to pad
        target_length: N_target - desired length
        pad_value: fill value for padding
        
    Returns:
        arr_padded: padded array (length = max(len(arr), N_target))
    """
    if len(arr) >= target_length:
        return arr
    return np.concatenate([arr, np.full(target_length - len(arr), pad_value)])


def find_closest_index(value: float, array: np.ndarray) -> int:
    """
    Find nearest neighbor index: i = argmin|array - value|.
    
    Parameters:
        value: target value
        array: search array
        
    Returns:
        i: index of closest element
    """
    return int(np.argmin(np.abs(array - value)))


def build_time_array_from_segments(dt_segments: np.ndarray, 
                                   n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct cumulative time series from time steps.
    
    Algorithm:
        t[0] = 0
        t[i+1] = t[i] + Δt[i] for i = 0..N-2
        
    Returns both cumulative t and per-point Δt.
    
    Parameters:
        dt_segments: Δt array [s] - time steps (length N-1)
        n_points: N - total trajectory points
        
    Returns:
        (t [s], Δt [s]): cumulative time and per-point deltas
    """
    time_array = np.zeros(n_points)
    
    # Cumulative sum: t[i+1] = t[i] + Δt[i]
    for i in range(min(len(dt_segments), n_points - 1)):
        time_array[i + 1] = time_array[i] + dt_segments[i]
    
    # Per-point deltas: Δt[i] = t[i] - t[i-1]
    dt_array_full = np.zeros(n_points)
    dt_array_full[1:] = np.diff(time_array)
    
    return time_array, dt_array_full

