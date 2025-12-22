# ========================================================================
# SECTION 1: MODULE INITIALIZATION
# ========================================================================
# Standard imports
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any, Tuple

# Aircraft configuration: N_ENGINES
from aircraft_config import N_ENGINES

# Mission parameters: Mach/altitude bounds, cruise climb settings, time step
from mission_config import (
    MIN_CRUISE_MACH,
    MAX_CRUISE_MACH,
    MIN_CRUISE_ALT_M,
    MAX_CRUISE_ALT_M,
    ENABLE_CRUISE_CLIMB,
    CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION,
    CRUISE_CLIMB_ALTITUDE_INCREMENT_M,
    TARGET_MACH_TOLERANCE,
    CRUISE_CLIMB_FALLBACK_LEVER,
    CRUISE_CLIMB_MACH_GRID_MARGIN,
    MIN_CONTINUED_CRUISE_DISTANCE_KM,
    N_MACH_SAMPLES_CLIMB,
    N_ALTITUDE_STEPS_CLIMB,
    N_LEVER_SAMPLES_CLIMB,
    CRUISE_TIME_STEP_S
)

# External model interfaces: aerodynamics D(M,h,m) and propulsion T(δ,M,h)
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper

# Atmospheric model: ISA properties T(h), ρ(h), a(h)
from aircraft_config import isa_properties, a_from_altitude

# Climb optimization for cruise climb segments
from climb import ClimbingCore, ClimbInitialState

# Utility functions: kinematics, energy balance, fuel consumption
from mission_utils import (
    find_lever_for_thrust,
    calculate_specific_excess_power,
    update_weight_after_burn,
    calculate_fuel_consumption_step,
    calculate_required_thrust_level_flight
)

# ========================================================================
# SECTION 2: DATA STRUCTURES
# ========================================================================

@dataclass
class CruiseInitialState:
    """
    Initial state vector for cruise phase from climb endpoint.
    
    State: X_0 = (h_0, M_0, m_0) at climb termination
    Cumulative quantities: Σm_fuel (climb), Σt (climb)
    """
    altitude_m: float                  # h_0 [m]: initial altitude
    mach: float                        # M_0 [-]: initial Mach number
    mass_kg: float                     # m_0 [kg]: aircraft mass at cruise start
    fuel_consumed_climb_kg: float      # Σm_fuel,climb [kg]: fuel consumed in climb
    climb_time_s: float                # Σt_climb [s]: time spent in climb
    
    def __post_init__(self):
        """Validate state: h ∈ [h_min, h_max], M ∈ [M_min, M_max], m > 0."""
        if not (MIN_CRUISE_ALT_M <= self.altitude_m <= MAX_CRUISE_ALT_M):
            raise ValueError(f"Altitude {self.altitude_m:.0f}m outside cruise range [{MIN_CRUISE_ALT_M}, {MAX_CRUISE_ALT_M}]m")
        if not (MIN_CRUISE_MACH <= self.mach <= MAX_CRUISE_MACH):
            raise ValueError(f"Mach {self.mach:.3f} outside cruise range [{MIN_CRUISE_MACH}, {MAX_CRUISE_MACH}]")
        if self.mass_kg <= 0:
            raise ValueError(f"Mass must be positive: {self.mass_kg:.1f} kg")

@dataclass
class CruiseResults:
    """
    Cruise trajectory from level-flight simulation or cruise-climb optimization.
    
    State variables: h(t) [m], M(t) [-], δ(t) [-], s(t) [km]
    Performance: Ps [m/s], ṁ [kg/s], T [N], D [N]
    Atmospheric: T [K], ρ [kg/m³], V [m/s]
    """
    # Configuration
    initial_state: CruiseInitialState      # X_0: initial state
    target_distance_km: float              # s_target [km]: target distance
    time_step_s: float                     # Δt [s]: simulation time step
    
    # State trajectory arrays
    time_s: np.ndarray                     # t [s]: time
    distance_km: np.ndarray                # s(t) [km]: ground distance
    mass_kg: np.ndarray                    # m(t) [kg]: aircraft mass
    fuel_consumed_kg: np.ndarray           # m_fuel(t) [kg]: cumulative fuel
    thrust_total_N: np.ndarray             # T(t) [N]: total thrust
    T_per_engine_N: np.ndarray             # T_eng(t) [N]: thrust per engine
    D_N: np.ndarray                        # D(t) [N]: drag
    mdot_kgps: np.ndarray                  # ṁ(t) [kg/s]: fuel flow rate
    Ps_mps: np.ndarray                     # Ps(t) [m/s]: specific excess power
    lever_position: np.ndarray             # δ(t) [-]: throttle lever [0,1]
    altitude_m: np.ndarray                 # h(t) [m]: altitude
    mach_number: np.ndarray                # M(t) [-]: Mach number
    
    # Atmospheric arrays
    temperature_K: np.ndarray              # T_atm(t) [K]: atmospheric temperature
    density_kgpm3: np.ndarray              # ρ(t) [kg/m³]: air density
    true_airspeed_mps: np.ndarray          # V(t) [m/s]: true airspeed
    
    # Summary statistics
    total_time_s: float                    # t_total [s]: total cruise time
    total_fuel_consumed_kg: float          # m_fuel,total [kg]: total fuel consumed
    final_mass_kg: float                   # m_f [kg]: final mass
    average_fuel_flow_kgps: float          # <ṁ> [kg/s]: mean fuel flow
    average_thrust_N: float                # <T> [N]: mean thrust
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Extract summary statistics for reporting."""
        return {
            'cruise_distance_km': self.target_distance_km,
            'cruise_time_hours': self.total_time_s / 3600.0,
            'cruise_fuel_kg': self.total_fuel_consumed_kg,
            'avg_fuel_flow_kg_h': self.average_fuel_flow_kgps * 3600.0,
            'avg_thrust_N': self.average_thrust_N,
            'initial_mass_kg': self.initial_state.mass_kg,
            'final_mass_kg': self.final_mass_kg,
            'cruise_mach': self.initial_state.mach,
            'cruise_altitude_m': self.initial_state.altitude_m,
        }

# ========================================================================
# SECTION 3: SYSTEM UTILITIES
# ========================================================================

class SystemUtilities:
    """Diagnostic and logging utilities."""
    
    @staticmethod
    def dbg(msg: str):
        """Debug output method (currently disabled)."""
        pass

def dbg(msg: str):
    """Module-level debug function for backward compatibility."""
    SystemUtilities.dbg(msg)

# Module constants
DEFAULT_TIME_STEP_S = CRUISE_TIME_STEP_S  # Δt [s]: default integration time step

# ========================================================================
# SECTION 4: TRAJECTORY CONSTRUCTION
# ========================================================================

def combine_cruise_segments(initial_cruise: CruiseResults,
                           cruise_climb: ClimbingCore.MinFuelSchedule,
                           continued_cruise: Optional[CruiseResults],
                           initial_state: CruiseInitialState,
                           target_distance_km: float) -> CruiseResults:
    """
    Concatenate cruise segments: initial + climb + continued.
    
    Segment structure:
        1. Initial cruise: level flight h=const, M=const
        2. Cruise climb: altitude increase Δh via DP optimization
        3. Continued cruise: level flight at h+Δh
    
    Concatenation: Align time and distance arrays with proper offsets.
    
    Parameters:
        initial_cruise: CruiseResults - initial level segment
        cruise_climb: MinFuelSchedule - climb via DP
        continued_cruise: CruiseResults - continued level segment (optional)
        initial_state: CruiseInitialState - reference initial state
        target_distance_km: s_target [km] - total cruise distance
        
    Returns:
        CruiseResults: combined trajectory with all segments
    """
    print("[CRUISE] Combining cruise segments")
    
    # Calculate time offsets for proper concatenation
    initial_cruise_time_offset = 0.0
    cruise_climb_time_offset = initial_cruise.total_time_s
    continued_cruise_time_offset = cruise_climb_time_offset + np.sum(cruise_climb.dt_s)
    
    # Calculate horizontal distance covered during cruise climb
    # Distance = integral of true airspeed over time
    cruise_climb_distance_km = 0.0
    cruise_climb_distance_array = []
    cruise_climb_cumulative_distance = initial_cruise.distance_km[-1]  # Start from end of initial cruise
    
    for i in range(len(cruise_climb.alt_m) - 1):
        # Get average conditions for this segment
        alt_avg = 0.5 * (cruise_climb.alt_m[i] + cruise_climb.alt_m[i + 1])
        mach_avg = 0.5 * (cruise_climb.mach[i] + cruise_climb.mach[i + 1])
        dt = cruise_climb.dt_s[i + 1] if i + 1 < len(cruise_climb.dt_s) else cruise_climb.dt_s[i]
        
        # Calculate true airspeed
        a = a_from_altitude(alt_avg)
        tas = mach_avg * a
        
        # Add distance for this segment
        segment_distance_km = (tas * dt) / 1000.0  # Convert m to km
        cruise_climb_distance_km += segment_distance_km
        cruise_climb_cumulative_distance += segment_distance_km
        cruise_climb_distance_array.append(cruise_climb_cumulative_distance)
    
    # Add final point
    cruise_climb_distance_array.append(cruise_climb_cumulative_distance)
    cruise_climb_distance_array = np.array(cruise_climb_distance_array)
    
    # Calculate distance offsets
    initial_cruise_distance = initial_cruise.distance_km[-1]
    continued_cruise_distance_offset = initial_cruise_distance + cruise_climb_distance_km
    
    # Convert cruise climb to compatible format
    # Cruise climb has different structure (MinFuelSchedule), need to convert to cruise-like arrays
    cruise_climb_time_array = np.cumsum(np.concatenate([[0.0], cruise_climb.dt_s[1:]]))
    
    # Calculate true airspeed for cruise climb (needed for distance calculation)
    cruise_climb_tas = []
    cruise_climb_temp = []
    cruise_climb_density = []
    for i, (alt, mach) in enumerate(zip(cruise_climb.alt_m, cruise_climb.mach)):
        temp_K, pressure_Pa, density_kgpm3 = isa_properties(alt)
        a = a_from_altitude(alt)
        tas = mach * a
        cruise_climb_tas.append(tas)
        cruise_climb_temp.append(temp_K)  # Temperature in Kelvin
        cruise_climb_density.append(density_kgpm3)
    
    cruise_climb_tas = np.array(cruise_climb_tas)
    cruise_climb_temp = np.array(cruise_climb_temp)
    cruise_climb_density = np.array(cruise_climb_density)
    
    # Concatenate time arrays with proper offsets
    if continued_cruise is not None:
        time_combined = np.concatenate([
            initial_cruise.time_s,
            cruise_climb_time_array + cruise_climb_time_offset,
            continued_cruise.time_s + continued_cruise_time_offset
        ])
    else:
        # No continued cruise - only initial + climb
        time_combined = np.concatenate([
            initial_cruise.time_s,
            cruise_climb_time_array + cruise_climb_time_offset
        ])
    
    # Concatenate distance arrays
    if continued_cruise is not None:
        distance_combined = np.concatenate([
            initial_cruise.distance_km,
            cruise_climb_distance_array,
            continued_cruise.distance_km + continued_cruise_distance_offset
        ])
    else:
        # No continued cruise - only initial + climb
        distance_combined = np.concatenate([
            initial_cruise.distance_km,
            cruise_climb_distance_array
        ])
    
    # Concatenate mass arrays
    if continued_cruise is not None:
        mass_combined = np.concatenate([
            initial_cruise.mass_kg,
            cruise_climb.mass_kg,
            continued_cruise.mass_kg
        ])
    else:
        mass_combined = np.concatenate([
            initial_cruise.mass_kg,
            cruise_climb.mass_kg
        ])
    
    # Concatenate fuel consumed arrays (cumulative)
    initial_fuel = initial_cruise.total_fuel_consumed_kg
    cruise_climb_fuel = cruise_climb.cumFuel_kg[-1]
    if continued_cruise is not None:
        fuel_consumed_combined = np.concatenate([
            initial_cruise.fuel_consumed_kg,
            initial_fuel + cruise_climb.cumFuel_kg,
            initial_fuel + cruise_climb_fuel + continued_cruise.fuel_consumed_kg
        ])
    else:
        fuel_consumed_combined = np.concatenate([
            initial_cruise.fuel_consumed_kg,
            initial_fuel + cruise_climb.cumFuel_kg
        ])
    
    # Concatenate thrust arrays
    if continued_cruise is not None:
        thrust_combined = np.concatenate([
            initial_cruise.thrust_total_N,
            cruise_climb.thrust_total_N,
            continued_cruise.thrust_total_N
        ])
        thrust_per_engine_combined = np.concatenate([
            initial_cruise.T_per_engine_N,
            cruise_climb.T_per_engine_N,
            continued_cruise.T_per_engine_N
        ])
    else:
        thrust_combined = np.concatenate([
            initial_cruise.thrust_total_N,
            cruise_climb.thrust_total_N
        ])
        thrust_per_engine_combined = np.concatenate([
            initial_cruise.T_per_engine_N,
            cruise_climb.T_per_engine_N
        ])
    
    # Concatenate drag arrays
    if continued_cruise is not None:
        drag_combined = np.concatenate([
            initial_cruise.D_N,
            cruise_climb.D_N,
            continued_cruise.D_N
        ])
    else:
        drag_combined = np.concatenate([
            initial_cruise.D_N,
            cruise_climb.D_N
        ])
    
    # Concatenate fuel flow arrays
    if continued_cruise is not None:
        fuel_flow_combined = np.concatenate([
            initial_cruise.mdot_kgps,
            cruise_climb.mdot_kgps,
            continued_cruise.mdot_kgps
        ])
    else:
        fuel_flow_combined = np.concatenate([
            initial_cruise.mdot_kgps,
            cruise_climb.mdot_kgps
        ])
    
    # Concatenate specific excess power arrays
    if continued_cruise is not None:
        ps_combined = np.concatenate([
            initial_cruise.Ps_mps,
            cruise_climb.Ps_mps,
            continued_cruise.Ps_mps
        ])
    else:
        ps_combined = np.concatenate([
            initial_cruise.Ps_mps,
            cruise_climb.Ps_mps
        ])
    
    # Concatenate lever arrays
    if continued_cruise is not None:
        lever_combined = np.concatenate([
            initial_cruise.lever_position,
            cruise_climb.lever,
            continued_cruise.lever_position
        ])
    else:
        lever_combined = np.concatenate([
            initial_cruise.lever_position,
            cruise_climb.lever
        ])
    
    # Concatenate altitude arrays
    if continued_cruise is not None:
        altitude_combined = np.concatenate([
            initial_cruise.altitude_m,
            cruise_climb.alt_m,
            continued_cruise.altitude_m
        ])
    else:
        altitude_combined = np.concatenate([
            initial_cruise.altitude_m,
            cruise_climb.alt_m
        ])
    
    # Concatenate Mach arrays
    if continued_cruise is not None:
        mach_combined = np.concatenate([
            initial_cruise.mach_number,
            cruise_climb.mach,
            continued_cruise.mach_number
        ])
    else:
        mach_combined = np.concatenate([
            initial_cruise.mach_number,
            cruise_climb.mach
        ])
    
    # Concatenate temperature arrays
    if continued_cruise is not None:
        temperature_combined = np.concatenate([
            initial_cruise.temperature_K,
            cruise_climb_temp,
            continued_cruise.temperature_K
        ])
    else:
        temperature_combined = np.concatenate([
            initial_cruise.temperature_K,
            cruise_climb_temp
        ])
    
    # Concatenate density arrays
    if continued_cruise is not None:
        density_combined = np.concatenate([
            initial_cruise.density_kgpm3,
            cruise_climb_density,
            continued_cruise.density_kgpm3
        ])
    else:
        density_combined = np.concatenate([
            initial_cruise.density_kgpm3,
            cruise_climb_density
        ])
    
    # Concatenate true airspeed arrays
    if continued_cruise is not None:
        tas_combined = np.concatenate([
            initial_cruise.true_airspeed_mps,
            cruise_climb_tas,
            continued_cruise.true_airspeed_mps
        ])
    else:
        tas_combined = np.concatenate([
            initial_cruise.true_airspeed_mps,
            cruise_climb_tas
        ])
    
    # Calculate summary statistics
    total_time = time_combined[-1]
    total_fuel = fuel_consumed_combined[-1]
    final_mass = mass_combined[-1]  # Renamed for physics accuracy
    avg_fuel_flow = np.mean(fuel_flow_combined[fuel_flow_combined > 0]) if np.any(fuel_flow_combined > 0) else 0.0
    avg_thrust = np.mean(thrust_combined[thrust_combined > 0]) if np.any(thrust_combined > 0) else 0.0
    
    print("[CRUISE] Combined segments summary:")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total fuel: {total_fuel:.1f} kg")
    print(f"  Final mass: {final_mass:.1f} kg")
    print(f"  Total distance: {distance_combined[-1]:.1f} km")
    
    return CruiseResults(
        initial_state=initial_state,
        target_distance_km=target_distance_km,
        time_step_s=initial_cruise.time_step_s,
        time_s=time_combined,
        distance_km=distance_combined,
        mass_kg=mass_combined,  # Renamed for physics accuracy
        fuel_consumed_kg=fuel_consumed_combined,
        thrust_total_N=thrust_combined,
        T_per_engine_N=thrust_per_engine_combined,
        D_N=drag_combined,
        mdot_kgps=fuel_flow_combined,
        Ps_mps=ps_combined,
        lever_position=lever_combined,
        altitude_m=altitude_combined,
        mach_number=mach_combined,
        temperature_K=temperature_combined,
        density_kgpm3=density_combined,
        true_airspeed_mps=tas_combined,
        total_time_s=total_time,
        total_fuel_consumed_kg=total_fuel,
        final_mass_kg=final_mass,  # Renamed for physics accuracy
        average_fuel_flow_kgps=avg_fuel_flow,
        average_thrust_N=avg_thrust
    )

def extract_cruise_initial_state(climb_result: ClimbingCore.MinFuelSchedule, 
                                initial_mass_kg: float) -> CruiseInitialState:
    """
    Extract initial state vector X_0 for cruise from climb endpoint.
    
    State extraction: X_0 = (h_f^climb, M_f^climb, m_f^climb)
    Cumulative accounting: Σm_fuel = m_climb, Σt = t_climb
    
    Parameters:
        climb_result: MinFuelSchedule - climb phase terminal state
        initial_mass_kg: m_0 [kg] - initial mass (pre-climb)
        
    Returns:
        CruiseInitialState: initial state for cruise phase
    """
    # Extract terminal climb state: X_f^climb = (h_f, M_f, m_f)
    final_altitude = float(climb_result.alt_m[-1])     # h_f [m]
    final_mach = float(climb_result.mach[-1])          # M_f [-]
    fuel_consumed_climb = float(climb_result.cumFuel_kg[-1]) if len(climb_result.cumFuel_kg) > 0 else 0.0
    climb_time = float(np.sum(climb_result.dt_s)) if len(climb_result.dt_s) > 0 else 0.0
    
    # Final mass from dynamic tracking: m_f = m_0 - Σm_fuel
    current_mass = float(climb_result.mass_kg[-1]) if len(climb_result.mass_kg) > 0 else (initial_mass_kg - fuel_consumed_climb)
    
    print("[CRUISE] Initial state extraction:")
    print(f"  h_0 = {final_altitude:.0f} m")
    print(f"  M_0 = {final_mach:.3f}")
    print(f"  m_0 = {current_mass:.1f} kg (Δm_climb = {fuel_consumed_climb:.1f} kg)")
    print(f"  t_climb = {climb_time:.0f} s ({climb_time/60:.1f} min)")
    
    return CruiseInitialState(
        altitude_m=final_altitude,
        mach=final_mach,
        mass_kg=current_mass,
        fuel_consumed_climb_kg=fuel_consumed_climb,
        climb_time_s=climb_time
    )

# ========================================================================
# SECTION 5: LEVEL CRUISE SIMULATION
# ========================================================================

def simulate_steady_cruise(initial_state: CruiseInitialState,
                          target_distance_km: float,
                          aero: PyAerodynamicsWrapper,
                          engine: EngineWrapper,
                          time_step_s: float = DEFAULT_TIME_STEP_S) -> CruiseResults:
    """
    Simulate steady-state level cruise: h=const, M=const.
    
    Mathematical formulation:
        Constraint: ḣ = 0 (level flight) → T = D at constant M, h
        Kinematic: ds/dt = V(M,h) → s(t) = ∫V dt
        Mass evolution: dm/dt = -ṁ = -TSFC·T
        
    Integration: Forward Euler with fixed time step Δt.
    
    Parameters:
        initial_state: CruiseInitialState - X_0 = (h_0, M_0, m_0)
        target_distance_km: s_target [km] - target ground distance
        aero: PyAerodynamicsWrapper - drag model D(M,h,m)
        engine: EngineWrapper - thrust model T(δ,M,h)
        time_step_s: Δt [s] - integration time step
        
    Returns:
        CruiseResults: trajectory X(t) from t=0 to s=s_target
    """
    print("[CRUISE] Starting steady-level cruise simulation")
    print(f"  Distance: {target_distance_km:.1f} km")
    print(f"  Time step: {time_step_s:.0f} s")
    print(f"  Altitude: {initial_state.altitude_m:.0f} m")
    print(f"  Mach: {initial_state.mach:.3f}")
    
    # Calculate true airspeed (constant for constant Mach and altitude)
    a = a_from_altitude(initial_state.altitude_m)
    true_airspeed_mps = initial_state.mach * a
    distance_per_step_km = (true_airspeed_mps * time_step_s) / 1000.0
    
    # Estimate number of steps needed
    n_steps = int(np.ceil(target_distance_km / distance_per_step_km))
    print(f"[CRUISE] Estimated simulation steps: {n_steps} (estimated duration: {n_steps * time_step_s / 3600:.2f} hours)")
    
    # Initialize trajectory arrays
    time_array = np.zeros(n_steps + 1)
    distance_array = np.zeros(n_steps + 1)
    mass_array = np.zeros(n_steps + 1)  # Renamed for physics accuracy
    fuel_consumed_array = np.zeros(n_steps + 1)
    thrust_array = np.zeros(n_steps + 1)
    thrust_per_engine_array = np.zeros(n_steps + 1)
    drag_array = np.zeros(n_steps + 1)
    fuel_flow_array = np.zeros(n_steps + 1)
    ps_array = np.zeros(n_steps + 1)
    lever_array = np.zeros(n_steps + 1)
    altitude_array = np.zeros(n_steps + 1)
    mach_array = np.zeros(n_steps + 1)
    temperature_array = np.zeros(n_steps + 1)
    density_array = np.zeros(n_steps + 1)
    tas_array = np.zeros(n_steps + 1)
    
    # Set initial conditions
    current_mass = initial_state.mass_kg  # Renamed for physics accuracy
    cumulative_fuel = 0.0
    cumulative_distance = 0.0
    
    # Get atmospheric properties (constant altitude)
    temp_K, pressure_Pa, density_kgpm3 = isa_properties(initial_state.altitude_m)
    sound_speed_mps = a_from_altitude(initial_state.altitude_m)
    
    print(f"[CRUISE] Atmospheric conditions at cruise altitude ({initial_state.altitude_m:.0f} m):")
    print(f"  Temperature: {temp_K:.1f} K ({temp_K-273.15:.1f}°C)")
    print(f"  Density: {density_kgpm3:.4f} kg/m³")
    print(f"  Speed of sound: {sound_speed_mps:.1f} m/s")
    print(f"  True airspeed: {true_airspeed_mps:.1f} m/s ({true_airspeed_mps * 1.944:.0f} kts)")
    
    # Simulation loop
    for step in range(n_steps + 1):
        # Update atmospheric properties (constant for level cruise, but store for each step)
        tas_array[step] = true_airspeed_mps
        temperature_array[step] = temp_K  # Temperature in Kelvin
        density_array[step] = density_kgpm3
        
        # Get drag from aerodynamics wrapper with current dynamic mass
        drag_N = aero.get_drag(initial_state.mach, initial_state.altitude_m, current_mass)
        
        # For steady cruise: thrust = drag (accounts for mass effects on drag)
        thrust_required_N = calculate_required_thrust_level_flight(drag_N)
        
        # Find required lever position for this thrust (dynamically updated each step)
        lever_required, _, _ = find_lever_for_thrust(engine, thrust_required_N, 
                                                     initial_state.mach, initial_state.altitude_m,
                                                     n_engines=N_ENGINES)
        
        # Get actual thrust at this lever position (engine performance at current conditions)
        thrust_per_engine_N = engine.thrust_with_lever(lever_required, initial_state.mach, 
                                                     initial_state.altitude_m)
        thrust_total_N = N_ENGINES * thrust_per_engine_N if thrust_per_engine_N else 0.0
        
        # Get TSFC at current engine operating point (TSFC depends on current lever and conditions)
        # TSFC should be called after setting the engine to current operating point
        tsfc = engine.tsfc_current()
        if tsfc is None or not np.isfinite(tsfc):
            tsfc = 0.0
        
        # Calculate fuel flow (dynamically updated with current thrust and TSFC)
        fuel_flow_kgps = thrust_total_N * tsfc if tsfc > 0 else 0.0
        
        # Calculate specific excess power with current mass (dynamically updated)
        ps = calculate_specific_excess_power(thrust_total_N, drag_N, current_mass, 
                                           true_airspeed_mps)
        
        # Store current state
        time_array[step] = step * time_step_s
        distance_array[step] = cumulative_distance
        mass_array[step] = current_mass  # Renamed for physics accuracy
        fuel_consumed_array[step] = cumulative_fuel
        thrust_array[step] = thrust_total_N
        thrust_per_engine_array[step] = thrust_per_engine_N if thrust_per_engine_N else 0.0
        drag_array[step] = drag_N
        fuel_flow_array[step] = fuel_flow_kgps
        ps_array[step] = ps
        lever_array[step] = lever_required
        altitude_array[step] = initial_state.altitude_m  # Constant altitude cruise
        mach_array[step] = initial_state.mach  # Constant Mach cruise
        
        # Break if target distance reached
        if cumulative_distance >= target_distance_km:
            # Truncate arrays to actual length
            actual_steps = step + 1
            time_array = time_array[:actual_steps]
            distance_array = distance_array[:actual_steps]
            mass_array = mass_array[:actual_steps]  # Renamed for physics accuracy
            fuel_consumed_array = fuel_consumed_array[:actual_steps]
            thrust_array = thrust_array[:actual_steps]
            thrust_per_engine_array = thrust_per_engine_array[:actual_steps]
            drag_array = drag_array[:actual_steps]
            fuel_flow_array = fuel_flow_array[:actual_steps]
            ps_array = ps_array[:actual_steps]
            lever_array = lever_array[:actual_steps]
            altitude_array = altitude_array[:actual_steps]
            mach_array = mach_array[:actual_steps]
            temperature_array = temperature_array[:actual_steps]
            density_array = density_array[:actual_steps]
            tas_array = tas_array[:actual_steps]
            break
        
        # Update for next step (if not last step)
        if step < n_steps:
            # Calculate fuel consumed in this step
            fuel_step = calculate_fuel_consumption_step(thrust_total_N, tsfc, time_step_s)
            cumulative_fuel += fuel_step
            
            # Update mass
            current_mass = update_weight_after_burn(current_mass, fuel_step)
            
            # Update distance
            cumulative_distance += distance_per_step_km
    
    # Calculate summary statistics
    total_time = time_array[-1]
    total_fuel = fuel_consumed_array[-1]
    final_mass = mass_array[-1]  # Renamed for physics accuracy
    avg_fuel_flow = np.mean(fuel_flow_array[fuel_flow_array > 0]) if np.any(fuel_flow_array > 0) else 0.0
    avg_thrust = np.mean(thrust_array[thrust_array > 0]) if np.any(thrust_array > 0) else 0.0
    
    print("[CRUISE] Cruise simulation completed")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total fuel: {total_fuel:.1f} kg")
    print(f"  Final mass: {final_mass:.1f} kg")
    print(f"  Average fuel flow: {avg_fuel_flow*3600:.1f} kg/h")
    print(f"  Average thrust: {avg_thrust:.0f} N")
    
    return CruiseResults(
        initial_state=initial_state,
        target_distance_km=target_distance_km,
        time_step_s=time_step_s,
        time_s=time_array,
        distance_km=distance_array,
        mass_kg=mass_array,  # Renamed for physics accuracy
        fuel_consumed_kg=fuel_consumed_array,
        thrust_total_N=thrust_array,
        T_per_engine_N=thrust_per_engine_array,
        D_N=drag_array,
        mdot_kgps=fuel_flow_array,
        Ps_mps=ps_array,
        lever_position=lever_array,
        altitude_m=altitude_array,
        mach_number=mach_array,
        temperature_K=temperature_array,
        density_kgpm3=density_array,
        true_airspeed_mps=tas_array,
        total_time_s=total_time,
        total_fuel_consumed_kg=total_fuel,
        final_mass_kg=final_mass,  # Renamed for physics accuracy
        average_fuel_flow_kgps=avg_fuel_flow,
        average_thrust_N=avg_thrust
    )

# ========================================================================
# SECTION 6: CRUISE CLIMB MANAGEMENT
# ========================================================================

class CruiseSegmentManager:
    """
    Cruise phase segmentation with optional cruise climb.
    
    Segment structure:
        1. Initial cruise: level flight h_0, M_0 for distance s_1
        2. Cruise climb (optional): h_0 → h_0+Δh via DP optimization
        3. Continued cruise: level flight h_0+Δh, M_0 for distance s_2
    
    Configuration:
        - enable_climb: Boolean flag for cruise climb feature
        - trigger_distance_fraction: s_1/s_total ∈ [0,1]
        - altitude_increment_m: Δh [m] for climb segment
        - mach_tolerance: |M_final - M_0| tolerance during climb
    """
    
    def __init__(self, enable_climb: bool = False,
                 trigger_distance_fraction: float = 0.30,
                 altitude_increment_m: float = 1000.0,
                 mach_tolerance: float = 0.015):
        """
        Initialize segment manager with cruise climb parameters.
        
        Parameters:
            enable_climb: Boolean - enable/disable cruise climb
            trigger_distance_fraction: ξ ∈ [0,1] - fraction of distance before climb
            altitude_increment_m: Δh [m] - altitude increment for climb
            mach_tolerance: tol_M [-] - Mach tolerance for climb optimization
        """
        self.enable_climb = enable_climb
        self.trigger_distance_fraction = trigger_distance_fraction
        self.altitude_increment_m = altitude_increment_m
        self.mach_tolerance = mach_tolerance
    
    def calculate_segment_distances(self, total_distance_km: float) -> Tuple[float, float]:
        """
        Compute distance allocation for cruise segmentation.
        
        Segmentation formula:
            s_1 = ξ · s_total  (initial cruise before climb)
            s_remaining = s_total - s_1  (to be covered by climb + continued cruise)
        
        Parameters:
            total_distance_km: s_total [km] - total cruise distance
            
        Returns:
            (s_1, s_remaining) [km]: initial and remaining distances
            If cruise climb disabled: (s_total, 0)
        """
        if not self.enable_climb:
            return total_distance_km, 0.0
        
        # Initial segment distance: s_1 = ξ · s_total
        initial_distance_km = total_distance_km * self.trigger_distance_fraction
        
        # Remaining distance: s_remaining = s_total - s_1
        # Note: Climb covers horizontal distance s_climb, reducing continued cruise
        remaining_distance_km = total_distance_km - initial_distance_km
        
        return initial_distance_km, remaining_distance_km
    
    def execute_cruise_climb(self, initial_state: CruiseInitialState,
                            aero: PyAerodynamicsWrapper,
                            engine: EngineWrapper,
                            mach_grid: np.ndarray,
                            altitude_sched: Optional[np.ndarray] = None) -> ClimbingCore.MinFuelSchedule:
        """
        Execute cruise climb via 3D dynamic programming optimization.
        
        Objective: Climb from h_0 to h_0+Δh while maintaining M≈M_0.
        Method: 3D Bellman recursion over (h,M,δ) with target_mach=M_0.
        
        Constraint: |M_final - M_0| < tolerance (maintain cruise Mach)
        
        Parameters:
            initial_state: CruiseInitialState - X_0 before climb
            aero: PyAerodynamicsWrapper - drag model
            engine: EngineWrapper - thrust model
            mach_grid: np.ndarray - M discretization for DP
            altitude_sched: np.ndarray - h discretization (optional, auto-generated)
            
        Returns:
            MinFuelSchedule: optimal cruise climb trajectory
        """
        print("\n[CRUISE] Starting cruise climb optimization")
        print(f"  Initial altitude: {initial_state.altitude_m:.0f} m")
        print(f"  Target altitude: {initial_state.altitude_m + self.altitude_increment_m:.0f} m")
        print(f"  Altitude increment: {self.altitude_increment_m:.0f} m")
        print(f"  Cruise Mach: {initial_state.mach:.3f}")
        
        # Calculate target altitude
        target_altitude_m = initial_state.altitude_m + self.altitude_increment_m
        
        # Ensure target altitude is within limits
        if target_altitude_m > MAX_CRUISE_ALT_M:
            print(f"[CRUISE] WARNING: Target altitude {target_altitude_m:.0f} m exceeds maximum {MAX_CRUISE_ALT_M:.0f} m")
            target_altitude_m = MAX_CRUISE_ALT_M
            self.altitude_increment_m = target_altitude_m - initial_state.altitude_m
        
        # Create altitude schedule for cruise climb
        # Use uniform step size consistent with climb phase
        uniform_step_size = self.altitude_increment_m / N_ALTITUDE_STEPS_CLIMB
        climb_h_sched = np.arange(initial_state.altitude_m,
                                 target_altitude_m + uniform_step_size,
                                 uniform_step_size)
        
        # Ensure we have at least 2 points
        if len(climb_h_sched) < 2:
            climb_h_sched = np.array([initial_state.altitude_m, target_altitude_m])
        
        print(f"[CRUISE] Cruise climb altitude schedule: {len(climb_h_sched)} points")
        print(f"  From {climb_h_sched[0]:.0f} m to {climb_h_sched[-1]:.0f} m")
        
        # Find feasible starting lever for cruise climb
        # First, create the lever grid that will be used by the DP solver
        lever_grid_dp = np.linspace(0.0, 1.0, N_LEVER_SAMPLES_CLIMB)
        
        # Calculate required thrust for level flight at starting conditions
        drag_N = aero.get_drag(initial_state.mach, initial_state.altitude_m, initial_state.mass_kg)
        required_thrust_N = drag_N  # For level flight, thrust = drag
        
        # Find lever position that provides required thrust (using the DP grid)
        start_lever, start_thrust, thrust_limited = find_lever_for_thrust(
            engine=engine,
            required_thrust_total=required_thrust_N,
            mach=initial_state.mach,
            altitude_m=initial_state.altitude_m,
            lever_grid=lever_grid_dp,  # Use the same grid as DP solver
            allow_refine=True
        )
        
        # If we found a lever, verify it's feasible by checking cost
        if start_lever is not None:
            # Find the closest lever on the grid
            start_lever_idx = np.argmin(np.abs(lever_grid_dp - start_lever))
            start_lever_on_grid = lever_grid_dp[start_lever_idx]
            
            # Verify feasibility by computing cost
            # Use climb_fraction=0.0 for start of cruise climb
            test_cost = ClimbingCore.compute_cost(
                aero, engine, 
                initial_state.altitude_m, 
                initial_state.mach, 
                start_lever_on_grid,
                initial_state.mass_kg,
                target_mach=initial_state.mach,
                prev_mach=None,
                climb_fraction=0.0
            )
            
            if np.isfinite(test_cost) and test_cost > 0:
                start_lever = start_lever_on_grid
                print(f"[CRUISE] Cruise climb starting lever: {start_lever:.3f} (on grid, cost={test_cost:.4f})")
            else:
                # Try to find a feasible lever on the grid by testing multiple levers
                print(f"[CRUISE] WARNING: Lever {start_lever_on_grid:.3f} not feasible (cost={test_cost}), searching grid")
                start_lever = None
                for test_lever in lever_grid_dp[::-1]:  # Try from high to low
                    test_cost = ClimbingCore.compute_cost(
                        aero, engine,
                        initial_state.altitude_m,
                        initial_state.mach,
                        test_lever,
                        initial_state.mass_kg,
                        target_mach=initial_state.mach,
                        prev_mach=None,
                        climb_fraction=0.0
                    )
                    if np.isfinite(test_cost) and test_cost > 0:
                        start_lever = test_lever
                        print(f"[CRUISE] Found feasible cruise climb lever: {start_lever:.3f} (cost={test_cost:.4f})")
                        break
        
        if start_lever is None:
            # Last resort: use a reasonable lever position
            print(f"[CRUISE] WARNING: Could not find feasible lever, using {CRUISE_CLIMB_FALLBACK_LEVER:.2f} as fallback")
            start_lever = CRUISE_CLIMB_FALLBACK_LEVER
        
        # Create initial state for cruise climb
        cruise_climb_initial_state = ClimbInitialState(
            altitude_m=initial_state.altitude_m,
            mach=initial_state.mach,
            mass_kg=initial_state.mass_kg,
            lever=start_lever
        )
        
        # Run DP optimization for cruise climb
        # Target Mach is the current cruise Mach (maintain same Mach)
        cruise_climb_result, cruise_climb_info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_dp(
            aero=aero,
            engine=engine,
            mach_grid=mach_grid,
            altitude_sched=climb_h_sched,
            initial_state=cruise_climb_initial_state,
            lever_samples=N_LEVER_SAMPLES_CLIMB,
            target_mach=initial_state.mach,  # Maintain cruise Mach
            target_mach_tolerance=self.mach_tolerance
        )
        
        print("[CRUISE] Cruise climb optimization completed")
        print(f"  Fuel consumed: {cruise_climb_result.cumFuel_kg[-1]:.2f} kg")
        print(f"  Time: {np.sum(cruise_climb_result.dt_s):.2f} s ({np.sum(cruise_climb_result.dt_s)/60:.2f} min)")
        print(f"  Final altitude: {cruise_climb_result.alt_m[-1]:.0f} m")
        print(f"  Final Mach: {cruise_climb_result.mach[-1]:.3f} (target: {initial_state.mach:.3f})")
        
        return cruise_climb_result

# ========================================================================
# SECTION 7: MAIN SIMULATION INTERFACE
# ========================================================================

def run_cruise_simulation(climb_result: ClimbingCore.MinFuelSchedule, 
                         initial_mass_kg: float,
                         target_distance_km: float,
                         aero: PyAerodynamicsWrapper,
                         engine: EngineWrapper,
                         time_step_s: float = DEFAULT_TIME_STEP_S,
                         create_plots: bool = True,
                         mach_grid: Optional[np.ndarray] = None) -> CruiseResults:
    """
    Primary interface for cruise phase simulation.
    
    Method: Time-marching integration for level cruise with optional cruise climb.
    
    Cruise modes:
        1. Standard: level flight h=const, M=const for distance s_target
        2. Segmented (if ENABLE_CRUISE_CLIMB):
           - Initial cruise: level flight for s_1 = ξ·s_target
           - Cruise climb: DP optimization h → h+Δh maintaining M≈const
           - Continued cruise: level flight at h+Δh for remaining distance
    
    Parameters:
        climb_result: MinFuelSchedule - climb phase terminal state
        initial_mass_kg: m_0 [kg] - initial mass (pre-climb)
        target_distance_km: s_target [km] - total cruise distance
        aero: PyAerodynamicsWrapper - drag model
        engine: EngineWrapper - thrust model
        time_step_s: Δt [s] - integration time step
        create_plots: Boolean - enable visualization
        mach_grid: np.ndarray - Mach grid for cruise climb DP (optional)
        
    Returns:
        CruiseResults: complete cruise trajectory X(t)
    """
    print(f"\n{'='*60}")
    print("CRUISE PHASE SIMULATION")
    print(f"{'='*60}")
    
    # Initialize cruise segment manager with configuration parameters
    segment_manager = CruiseSegmentManager(
        enable_climb=ENABLE_CRUISE_CLIMB,
        trigger_distance_fraction=CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION,
        altitude_increment_m=CRUISE_CLIMB_ALTITUDE_INCREMENT_M,
        mach_tolerance=TARGET_MACH_TOLERANCE
    )
    
    # Extract initial state from climb
    initial_state = extract_cruise_initial_state(climb_result, initial_mass_kg)
    
    # Check if cruise climb is enabled
    if not segment_manager.enable_climb:
        # Standard cruise simulation (no climb)
        print("[CRUISE] Cruise climb disabled, running standard cruise simulation")
        cruise_results = simulate_steady_cruise(
            initial_state=initial_state,
            target_distance_km=target_distance_km,
            aero=aero,
            engine=engine,
            time_step_s=time_step_s
        )
    else:
        # Segmented cruise with optional climb
        print("[CRUISE] Cruise climb enabled, running segmented cruise simulation")
        
        # Calculate initial cruise distance
        initial_dist, remaining_dist = segment_manager.calculate_segment_distances(target_distance_km)
        print(f"[CRUISE] Initial cruise segment: {initial_dist:.1f} km ({initial_dist/target_distance_km*100:.1f}% of total)")
        print(f"[CRUISE] Remaining distance: {remaining_dist:.1f} km (will be covered by climb and continued cruise)")
        
        # Run initial cruise segment
        print(f"\n[CRUISE] Running initial cruise segment ({initial_dist:.1f} km)")
        initial_cruise_results = simulate_steady_cruise(
            initial_state=initial_state,
            target_distance_km=initial_dist,
            aero=aero,
            engine=engine,
            time_step_s=time_step_s
        )
        
        # Extract state after initial cruise
        initial_cruise_final_state = CruiseInitialState(
            altitude_m=initial_cruise_results.altitude_m[-1],
            mach=initial_cruise_results.mach_number[-1],
            mass_kg=initial_cruise_results.mass_kg[-1],  # Renamed for physics accuracy
            fuel_consumed_climb_kg=initial_cruise_results.total_fuel_consumed_kg,
            climb_time_s=initial_cruise_results.total_time_s
        )
        
        # Execute cruise climb using DP optimization
        if mach_grid is None:
            # Create default Mach grid if not provided
            M_min = max(MIN_CRUISE_MACH, initial_cruise_final_state.mach - CRUISE_CLIMB_MACH_GRID_MARGIN)
            M_max = min(MAX_CRUISE_MACH, initial_cruise_final_state.mach + CRUISE_CLIMB_MACH_GRID_MARGIN)
            mach_grid = np.linspace(M_min, M_max, N_MACH_SAMPLES_CLIMB)
        
        cruise_climb_result = segment_manager.execute_cruise_climb(
            initial_state=initial_cruise_final_state,
            aero=aero,
            engine=engine,
            mach_grid=mach_grid,
            altitude_sched=None  # altitude_sched is created internally in execute_cruise_climb
        )
        
        # Calculate horizontal distance covered during cruise climb
        # Distance = integral of true airspeed over time
        cruise_climb_distance_km = 0.0
        for i in range(len(cruise_climb_result.alt_m) - 1):
            # Get average conditions for this segment
            alt_avg = 0.5 * (cruise_climb_result.alt_m[i] + cruise_climb_result.alt_m[i + 1])
            mach_avg = 0.5 * (cruise_climb_result.mach[i] + cruise_climb_result.mach[i + 1])
            dt = cruise_climb_result.dt_s[i + 1] if i + 1 < len(cruise_climb_result.dt_s) else cruise_climb_result.dt_s[i]
            
            # Calculate true airspeed
            a = a_from_altitude(alt_avg)
            tas = mach_avg * a
            
            # Add distance for this segment
            cruise_climb_distance_km += (tas * dt) / 1000.0  # Convert m to km
        
        print(f"[CRUISE] Cruise climb covered {cruise_climb_distance_km:.2f} km horizontally")
        
        # Calculate remaining distance for continued cruise
        # Total distance = initial + climb + continued
        # Therefore: continued = total - initial - climb
        continued_dist = target_distance_km - initial_dist - cruise_climb_distance_km
        
        if continued_dist < 0:
            print(f"[CRUISE] WARNING: Climb distance ({cruise_climb_distance_km:.2f} km) exceeds remaining distance ({remaining_dist:.2f} km)")
            print("[CRUISE] Setting continued cruise distance to 0 km")
            continued_dist = 0.0
        
        print(f"[CRUISE] Continued cruise segment: {continued_dist:.1f} km")
        print("[CRUISE] Total distance breakdown:")
        print(f"  Initial cruise: {initial_dist:.2f} km")
        print(f"  Cruise climb: {cruise_climb_distance_km:.2f} km")
        print(f"  Continued cruise: {continued_dist:.2f} km")
        print(f"  Total: {initial_dist + cruise_climb_distance_km + continued_dist:.2f} km (target: {target_distance_km:.2f} km)")
        
        # Extract state after cruise climb
        cruise_climb_final_state = CruiseInitialState(
            altitude_m=cruise_climb_result.alt_m[-1],
            mach=cruise_climb_result.mach[-1],
            mass_kg=cruise_climb_result.mass_kg[-1],  # Already using mass_kg (consistent)
            fuel_consumed_climb_kg=cruise_climb_result.cumFuel_kg[-1],
            climb_time_s=np.sum(cruise_climb_result.dt_s)
        )
        
        # Run continued cruise segment (if there's remaining distance)
        if continued_dist > MIN_CONTINUED_CRUISE_DISTANCE_KM:  # Only if more than minimum distance remaining
            print(f"\n[CRUISE] Running continued cruise segment ({continued_dist:.1f} km)")
            continued_cruise_results = simulate_steady_cruise(
                initial_state=cruise_climb_final_state,
                target_distance_km=continued_dist,
                aero=aero,
                engine=engine,
                time_step_s=time_step_s
            )
        else:
            # No continued cruise needed - create empty results
            print("\n[CRUISE] No continued cruise needed (climb covered all remaining distance)")
            continued_cruise_results = None
        
        # Combine all cruise segments into single result
        if continued_cruise_results is not None:
            cruise_results = combine_cruise_segments(
                initial_cruise=initial_cruise_results,
                cruise_climb=cruise_climb_result,
                continued_cruise=continued_cruise_results,
                initial_state=initial_state,
                target_distance_km=target_distance_km
            )
        else:
            # Only initial cruise + climb (no continued cruise)
            cruise_results = combine_cruise_segments(
                initial_cruise=initial_cruise_results,
                cruise_climb=cruise_climb_result,
                continued_cruise=None,
                initial_state=initial_state,
                target_distance_km=target_distance_km
            )
    
    # Skip individual cruise plots - only show combined mission plots
    if create_plots:
        print("[CRUISE] Skipping individual cruise plots (combined mission plots will be shown instead)")
    
    print(f"{'='*60}")
    print("CRUISE SIMULATION COMPLETED")
    print(f"{'='*60}\n")
    
    return cruise_results

# ========================================================================
# SECTION 8: MODULE EXPORTS
# ========================================================================

def run_simulation(climb_result: ClimbingCore.MinFuelSchedule, 
                  initial_mass_kg: float,
                  target_distance_km: float,
                  aero: PyAerodynamicsWrapper,
                  engine: EngineWrapper,
                  time_step_s: float = None,
                  create_plots: bool = True,
                  mach_grid: Optional[np.ndarray] = None) -> CruiseResults:
    """
    Generic interface for cruise simulation (phase-agnostic naming).
    
    Provides consistent interface with climb.run_optimization() and
    descent.run_optimization() for unified mission analysis workflow.
    
    Delegates to run_cruise_simulation() with proper time step handling.
    
    Parameters:
        climb_result: MinFuelSchedule - climb phase terminal state
        initial_mass_kg: m_0 [kg] - initial mass
        target_distance_km: s_target [km] - cruise distance
        aero: PyAerodynamicsWrapper - drag model
        engine: EngineWrapper - thrust model
        time_step_s: Δt [s] - time step (default DEFAULT_TIME_STEP_S)
        create_plots: Boolean - enable visualization
        mach_grid: np.ndarray - Mach grid for cruise climb (optional)
        
    Returns:
        CruiseResults: cruise trajectory
    """
    if time_step_s is None:
        time_step_s = DEFAULT_TIME_STEP_S
    
    return run_cruise_simulation(
        climb_result=climb_result,
        initial_mass_kg=initial_mass_kg,
        target_distance_km=target_distance_km,
        aero=aero,
        engine=engine,
        time_step_s=time_step_s,
        create_plots=create_plots,
        mach_grid=mach_grid
    )

# Phase-agnostic type alias for mission analysis
OptimalTrajectory = CruiseResults