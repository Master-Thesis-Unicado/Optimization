from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Import necessary components from existing modules
from atmosphere import Atmosphere
from aircraft_config import N_ENGINES
import climb
from climb import ClimbingCore
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper
from mission_config import (
    THRUST_CONVERGENCE_TOL_CRUISE,
    MAX_ITERATIONS_CRUISE,
    MIN_CRUISE_MACH,
    MAX_CRUISE_MACH,
    MIN_CRUISE_ALT_M,
    MAX_CRUISE_ALT_M,
    ENABLE_CRUISE_CLIMB,
    CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION,
    CRUISE_CLIMB_ALTITUDE_INCREMENT_M,
    CRUISE_CLIMB_MACH_TOLERANCE,
    N_MACH_SAMPLES_CLIMB,
    N_ALTITUDE_STEPS_CLIMB,
    N_LEVER_SAMPLES_CLIMB
)

# ========= CRUISE CONSTANTS AND SETTINGS =============================================

# Default cruise parameters (user adjustable)
DEFAULT_TIME_STEP_S = 60.0    # 1 minute time steps
# Note: Gravity constant is now obtained from aero.G_C (PyAerodynamicsWrapper)

# Import phase-specific parameters from centralized configuration
THRUST_CONVERGENCE_TOL = THRUST_CONVERGENCE_TOL_CRUISE  # Newton tolerance for thrust balance
MAX_ITERATIONS = MAX_ITERATIONS_CRUISE                   # Maximum iterations for convergence

# ========= DATA STRUCTURES =============================================

@dataclass
class CruiseInitialState:
    """Initial state for cruise phase extracted from climb optimization."""
    altitude_m: float
    mach: float
    weight_kg: float          # Updated weight after climb fuel consumption
    fuel_consumed_climb_kg: float
    climb_time_s: float
    
    def __post_init__(self):
        """Validate initial cruise state."""
        if not (MIN_CRUISE_ALT_M <= self.altitude_m <= MAX_CRUISE_ALT_M):
            raise ValueError(f"Cruise altitude {self.altitude_m:.0f}m outside safe range [{MIN_CRUISE_ALT_M}-{MAX_CRUISE_ALT_M}]m")
        if not (MIN_CRUISE_MACH <= self.mach <= MAX_CRUISE_MACH):
            raise ValueError(f"Cruise Mach {self.mach:.3f} outside safe range [{MIN_CRUISE_MACH}-{MAX_CRUISE_MACH}]")
        if self.weight_kg <= 0:
            raise ValueError(f"Aircraft weight must be positive, got {self.weight_kg:.1f} kg")

@dataclass
class CruiseResults:
    """Complete results from cruise simulation."""
    # Input parameters
    initial_state: CruiseInitialState
    target_distance_km: float
    time_step_s: float
    
    # Trajectory arrays
    time_s: np.ndarray
    distance_km: np.ndarray
    weight_kg: np.ndarray
    fuel_consumed_kg: np.ndarray
    thrust_total_N: np.ndarray
    drag_N: np.ndarray
    fuel_flow_kgps: np.ndarray
    specific_excess_power_mps: np.ndarray
    lever_position: np.ndarray
    altitude_m: np.ndarray
    mach_number: np.ndarray
    
    # Atmospheric arrays
    temperature_K: np.ndarray
    density_kgpm3: np.ndarray
    true_airspeed_mps: np.ndarray
    
    # Summary statistics
    total_time_s: float
    total_fuel_consumed_kg: float
    final_weight_kg: float
    average_fuel_flow_kgps: float
    average_thrust_N: float
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics as dictionary."""
        return {
            'cruise_distance_km': self.target_distance_km,
            'cruise_time_hours': self.total_time_s / 3600.0,
            'cruise_fuel_kg': self.total_fuel_consumed_kg,
            'avg_fuel_flow_kg_h': self.average_fuel_flow_kgps * 3600.0,
            'avg_thrust_N': self.average_thrust_N,
            'initial_weight_kg': self.initial_state.weight_kg,
            'final_weight_kg': self.final_weight_kg,
            'cruise_mach': self.initial_state.mach,
            'cruise_altitude_m': self.initial_state.altitude_m,
        }

# ========= ATMOSPHERIC AND FLIGHT CALCULATIONS =============================================

def calculate_atmospheric_properties(altitude_m: float) -> Tuple[float, float, float, float]:
    """
    Calculate atmospheric properties at given altitude using ISA model.
    
    Args:
        altitude_m: Altitude in meters
        
    Returns:
        Tuple of (temperature_K, pressure_Pa, density_kgpm3, speed_of_sound_mps)
    """
    atm = Atmosphere()
    
    # Convert to flight level for atmosphere calculation
    flight_level = altitude_m / 0.3048
    T, p, rho = atm.calculate_atmospheric_properties(flight_level)
    
    # Calculate speed of sound
    a = atm.get_speed_of_sound(altitude_m)
    
    return T, p, rho, a

def calculate_true_airspeed(mach: float, altitude_m: float) -> float:
    """Calculate true airspeed from Mach number and altitude."""
    _, _, _, a = calculate_atmospheric_properties(altitude_m)
    return mach * a

def calculate_required_thrust_cruise(drag_N: float, weight_kg: float, 
                                   altitude_m: float) -> float:
    """
    Calculate required thrust for steady level cruise.
    
    For level flight at constant speed: T_required = D
    
    Args:
        drag_N: Drag force in Newtons
        weight_kg: Aircraft weight in kg
        altitude_m: Cruise altitude in meters
        
    Returns:
        Required total thrust in Newtons
    """
    # For steady level cruise, thrust exactly balances drag
    return float(drag_N)

def calculate_specific_excess_power(thrust_total_N: float, drag_N: float, 
                                  weight_kg: float, true_airspeed_mps: float,
                                  aero: PyAerodynamicsWrapper = None) -> float:
    """
    Calculate specific excess power for cruise.
    
    Ps = (T_total - D) * V / W
    
    For ideal steady cruise, Ps should be approximately 0.
    
    Args:
        thrust_total_N: Total thrust in Newtons
        drag_N: Drag force in Newtons  
        weight_kg: Aircraft weight in kg
        true_airspeed_mps: True airspeed in m/s
        aero: Aerodynamics wrapper (for G_C access). If None, falls back to Atmosphere.G_C
        
    Returns:
        Specific excess power in m/s
    """
    if aero is not None:
        g = aero.G_C
    else:
        from atmosphere import Atmosphere
        g = Atmosphere.G_C
    weight_N = weight_kg * g
    return (thrust_total_N - drag_N) * true_airspeed_mps / weight_N

def calculate_fuel_consumption_step(thrust_total_N: float, tsfc_kg_per_N_s: float, 
                                   time_step_s: float) -> float:
    """
    Calculate fuel consumed in one time step.
    
    Args:
        thrust_total_N: Total thrust in Newtons
        tsfc_kg_per_N_s: Thrust-specific fuel consumption in kg/(N·s)
        time_step_s: Time step in seconds
        
    Returns:
        Fuel consumed in kg
    """
    fuel_flow_kgps = thrust_total_N * tsfc_kg_per_N_s
    return fuel_flow_kgps * time_step_s

def update_weight_after_fuel_consumption(current_weight_kg: float, 
                                       fuel_consumed_kg: float) -> float:
    """
    Update aircraft weight after fuel consumption.
    
    Args:
        current_weight_kg: Current aircraft weight in kg
        fuel_consumed_kg: Fuel consumed in kg
        
    Returns:
        Updated weight in kg
    """
    new_weight = current_weight_kg - fuel_consumed_kg
    
    # Ensure weight doesn't become unreasonably low
    if new_weight < 0.5 * current_weight_kg:
        raise ValueError(f"Weight reduction too large: {current_weight_kg:.1f} -> {new_weight:.1f} kg")
    
    return new_weight

def combine_cruise_segments(initial_cruise: CruiseResults,
                           cruise_climb: ClimbingCore.MinFuelSchedule,
                           continued_cruise: Optional[CruiseResults],
                           initial_state: CruiseInitialState,
                           target_distance_km: float) -> CruiseResults:
    """
    Combine multiple cruise segments into a single CruiseResults object.
    
    Args:
        initial_cruise: Results from initial cruise segment
        cruise_climb: Results from cruise climb DP optimization
        continued_cruise: Results from continued cruise segment (None if no continued cruise)
        initial_state: Initial cruise state
        target_distance_km: Total target cruise distance
        
    Returns:
        Combined CruiseResults with all segments concatenated
    """
    print(f"[CRUISE] Combining cruise segments...")
    
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
        _, _, _, a = calculate_atmospheric_properties(alt_avg)
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
        T, p, rho, a = calculate_atmospheric_properties(alt)
        tas = mach * a
        cruise_climb_tas.append(tas)
        cruise_climb_temp.append(T)
        cruise_climb_density.append(rho)
    
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
    
    # Concatenate weight arrays
    if continued_cruise is not None:
        weight_combined = np.concatenate([
            initial_cruise.weight_kg,
            cruise_climb.mass_kg,
            continued_cruise.weight_kg
        ])
    else:
        weight_combined = np.concatenate([
            initial_cruise.weight_kg,
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
            cruise_climb.T_total_N,
            continued_cruise.thrust_total_N
        ])
    else:
        thrust_combined = np.concatenate([
            initial_cruise.thrust_total_N,
            cruise_climb.T_total_N
        ])
    
    # Concatenate drag arrays
    if continued_cruise is not None:
        drag_combined = np.concatenate([
            initial_cruise.drag_N,
            cruise_climb.D_N,
            continued_cruise.drag_N
        ])
    else:
        drag_combined = np.concatenate([
            initial_cruise.drag_N,
            cruise_climb.D_N
        ])
    
    # Concatenate fuel flow arrays
    if continued_cruise is not None:
        fuel_flow_combined = np.concatenate([
            initial_cruise.fuel_flow_kgps,
            cruise_climb.mdot_kgps,
            continued_cruise.fuel_flow_kgps
        ])
    else:
        fuel_flow_combined = np.concatenate([
            initial_cruise.fuel_flow_kgps,
            cruise_climb.mdot_kgps
        ])
    
    # Concatenate specific excess power arrays
    if continued_cruise is not None:
        ps_combined = np.concatenate([
            initial_cruise.specific_excess_power_mps,
            cruise_climb.Ps_mps,
            continued_cruise.specific_excess_power_mps
        ])
    else:
        ps_combined = np.concatenate([
            initial_cruise.specific_excess_power_mps,
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
    final_weight = weight_combined[-1]
    avg_fuel_flow = np.mean(fuel_flow_combined[fuel_flow_combined > 0]) if np.any(fuel_flow_combined > 0) else 0.0
    avg_thrust = np.mean(thrust_combined[thrust_combined > 0]) if np.any(thrust_combined > 0) else 0.0
    
    print(f"[CRUISE] Combined segments:")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total fuel: {total_fuel:.1f} kg")
    print(f"  Final weight: {final_weight:.1f} kg")
    print(f"  Total distance: {distance_combined[-1]:.1f} km")
    
    return CruiseResults(
        initial_state=initial_state,
        target_distance_km=target_distance_km,
        time_step_s=initial_cruise.time_step_s,
        time_s=time_combined,
        distance_km=distance_combined,
        weight_kg=weight_combined,
        fuel_consumed_kg=fuel_consumed_combined,
        thrust_total_N=thrust_combined,
        drag_N=drag_combined,
        fuel_flow_kgps=fuel_flow_combined,
        specific_excess_power_mps=ps_combined,
        lever_position=lever_combined,
        altitude_m=altitude_combined,
        mach_number=mach_combined,
        temperature_K=temperature_combined,
        density_kgpm3=density_combined,
        true_airspeed_mps=tas_combined,
        total_time_s=total_time,
        total_fuel_consumed_kg=total_fuel,
        final_weight_kg=final_weight,
        average_fuel_flow_kgps=avg_fuel_flow,
        average_thrust_N=avg_thrust
    )

def extract_cruise_initial_state(climb_result: ClimbingCore.MinFuelSchedule, 
                                initial_mass_kg: float) -> CruiseInitialState:
    """
    Extract initial cruise state from climb optimization results.
    
    Args:
        climb_result: Results from 3D DP climb optimization
        initial_mass_kg: Initial aircraft mass before climb (used only for fuel consumed calculation)
        
    Returns:
        CruiseInitialState object with extracted parameters
    """
    # Get final state from climb
    final_altitude = float(climb_result.alt_m[-1])
    final_mach = float(climb_result.mach[-1])
    fuel_consumed_climb = float(climb_result.cumFuel_kg[-1]) if len(climb_result.cumFuel_kg) > 0 else 0.0
    climb_time = float(np.sum(climb_result.dt_s)) if len(climb_result.dt_s) > 0 else 0.0
    
    # Get actual weight at end of climb from dynamic weight tracking
    # This is more accurate than calculating by subtraction
    current_weight = float(climb_result.mass_kg[-1]) if len(climb_result.mass_kg) > 0 else (initial_mass_kg - fuel_consumed_climb)
    
    print(f"[CRUISE] Extracted initial state:")
    print(f"  Altitude: {final_altitude:.0f} m")
    print(f"  Mach: {final_mach:.3f}")
    print(f"  Weight: {current_weight:.1f} kg (consumed {fuel_consumed_climb:.1f} kg in climb)")
    print(f"  Initial weight: {initial_mass_kg:.1f} kg")
    print(f"  Climb time: {climb_time:.0f} s ({climb_time/60:.1f} min)")
    
    return CruiseInitialState(
        altitude_m=final_altitude,
        mach=final_mach,
        weight_kg=current_weight,
        fuel_consumed_climb_kg=fuel_consumed_climb,
        climb_time_s=climb_time
    )

def find_required_lever(engine: EngineWrapper, required_thrust_N: float, 
                       mach: float, altitude_m: float) -> float:
    """
    Find the lever position required to achieve target thrust.
    
    Args:
        engine: Engine wrapper
        required_thrust_N: Required total thrust in Newtons
        mach: Mach number
        altitude_m: Altitude in meters
        
    Returns:
        Required lever position (0.0 to 1.0)
    """
    required_per_engine = required_thrust_N / N_ENGINES
    
    # Check idle and max thrust bounds
    thrust_idle = engine.thrust_with_lever(0.0, mach, altitude_m)
    thrust_max = engine.thrust_with_lever(1.0, mach, altitude_m)
    
    if thrust_idle is None or thrust_max is None:
        return 0.5  # Default fallback
    
    if required_per_engine <= thrust_idle:
        return 0.0  # Idle sufficient
    elif required_per_engine >= thrust_max:
        return 1.0  # Maximum required
    
    # Binary search for required lever
    lever_low, lever_high = 0.0, 1.0
    
    for _ in range(MAX_ITERATIONS):
        lever_mid = (lever_low + lever_high) / 2.0
        thrust_mid = engine.thrust_with_lever(lever_mid, mach, altitude_m)
        
        if thrust_mid is None:
            break
        
        if abs(thrust_mid - required_per_engine) < THRUST_CONVERGENCE_TOL / N_ENGINES:
            return lever_mid
        
        if thrust_mid < required_per_engine:
            lever_low = lever_mid
        else:
            lever_high = lever_mid
    
    return (lever_low + lever_high) / 2.0

def simulate_steady_cruise(initial_state: CruiseInitialState,
                          target_distance_km: float,
                          aero: PyAerodynamicsWrapper,
                          engine: EngineWrapper,
                          time_step_s: float = DEFAULT_TIME_STEP_S) -> CruiseResults:
    """
    Simulate steady level cruise at constant Mach and altitude.
    
    Args:
        initial_state: Initial cruise state from climb
        target_distance_km: Cruise distance in kilometers
        aero: Aerodynamics wrapper
        engine: Engine wrapper
        time_step_s: Time step for simulation in seconds
        
    Returns:
        CruiseResults object with complete trajectory
    """
    print(f"[CRUISE] Starting cruise simulation:")
    print(f"  Distance: {target_distance_km:.1f} km")
    print(f"  Time step: {time_step_s:.0f} s")
    print(f"  Altitude: {initial_state.altitude_m:.0f} m")
    print(f"  Mach: {initial_state.mach:.3f}")
    
    # Calculate true airspeed (constant for constant Mach and altitude)
    true_airspeed_mps = calculate_true_airspeed(initial_state.mach, initial_state.altitude_m)
    distance_per_step_km = (true_airspeed_mps * time_step_s) / 1000.0
    
    # Estimate number of steps needed
    n_steps = int(np.ceil(target_distance_km / distance_per_step_km))
    print(f"[CRUISE] Estimated steps: {n_steps} ({n_steps * time_step_s / 3600:.2f} hours)")
    
    # Initialize trajectory arrays
    time_array = np.zeros(n_steps + 1)
    distance_array = np.zeros(n_steps + 1)
    weight_array = np.zeros(n_steps + 1)
    fuel_consumed_array = np.zeros(n_steps + 1)
    thrust_array = np.zeros(n_steps + 1)
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
    current_weight = initial_state.weight_kg
    cumulative_fuel = 0.0
    cumulative_distance = 0.0
    
    # Get atmospheric properties (constant altitude)
    T, p, rho, a = calculate_atmospheric_properties(initial_state.altitude_m)
    
    print(f"[CRUISE] Atmospheric conditions at {initial_state.altitude_m:.0f}m:")
    print(f"  Temperature: {T:.1f} K ({T-273.15:.1f}°C)")
    print(f"  Density: {rho:.4f} kg/m³")
    print(f"  Speed of sound: {a:.1f} m/s")
    print(f"  True airspeed: {true_airspeed_mps:.1f} m/s ({true_airspeed_mps * 1.944:.0f} kts)")
    
    # Simulation loop
    for step in range(n_steps + 1):
        # Update atmospheric properties (constant for level cruise, but store for each step)
        tas_array[step] = true_airspeed_mps
        temperature_array[step] = T
        density_array[step] = rho
        
        # Get drag from aerodynamics wrapper with current dynamic weight
        drag_N = aero.get_drag(initial_state.mach, initial_state.altitude_m, current_weight)
        
        # For steady cruise: thrust = drag (now properly accounts for weight effects on drag)
        thrust_required_N = calculate_required_thrust_cruise(drag_N, current_weight, 
                                                          initial_state.altitude_m)
        
        # Find required lever position for this thrust (dynamically updated each step)
        lever_required = find_required_lever(engine, thrust_required_N, 
                                           initial_state.mach, initial_state.altitude_m)
        
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
        
        # Calculate specific excess power with current weight (dynamically updated)
        ps = calculate_specific_excess_power(thrust_total_N, drag_N, current_weight, 
                                           true_airspeed_mps, aero=aero)
        
        # Store current state
        time_array[step] = step * time_step_s
        distance_array[step] = cumulative_distance
        weight_array[step] = current_weight
        fuel_consumed_array[step] = cumulative_fuel
        thrust_array[step] = thrust_total_N
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
            weight_array = weight_array[:actual_steps]
            fuel_consumed_array = fuel_consumed_array[:actual_steps]
            thrust_array = thrust_array[:actual_steps]
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
            
            # Update weight
            current_weight = update_weight_after_fuel_consumption(current_weight, fuel_step)
            
            # Update distance
            cumulative_distance += distance_per_step_km
    
    # Calculate summary statistics
    total_time = time_array[-1]
    total_fuel = fuel_consumed_array[-1]
    final_weight = weight_array[-1]
    avg_fuel_flow = np.mean(fuel_flow_array[fuel_flow_array > 0]) if np.any(fuel_flow_array > 0) else 0.0
    avg_thrust = np.mean(thrust_array[thrust_array > 0]) if np.any(thrust_array > 0) else 0.0
    
    print(f"[CRUISE] Simulation completed:")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total fuel: {total_fuel:.1f} kg")
    print(f"  Final weight: {final_weight:.1f} kg")
    print(f"  Average fuel flow: {avg_fuel_flow*3600:.1f} kg/h")
    print(f"  Average thrust: {avg_thrust:.0f} N")
    
    return CruiseResults(
        initial_state=initial_state,
        target_distance_km=target_distance_km,
        time_step_s=time_step_s,
        time_s=time_array,
        distance_km=distance_array,
        weight_kg=weight_array,
        fuel_consumed_kg=fuel_consumed_array,
        thrust_total_N=thrust_array,
        drag_N=drag_array,
        fuel_flow_kgps=fuel_flow_array,
        specific_excess_power_mps=ps_array,
        lever_position=lever_array,
        altitude_m=altitude_array,
        mach_number=mach_array,
        temperature_K=temperature_array,
        density_kgpm3=density_array,
        true_airspeed_mps=tas_array,
        total_time_s=total_time,
        total_fuel_consumed_kg=total_fuel,
        final_weight_kg=final_weight,
        average_fuel_flow_kgps=avg_fuel_flow,
        average_thrust_N=avg_thrust
    )

# ========= CRUISE SEGMENT MANAGER =============================================

class CruiseSegmentManager:
    """
    Manages cruise phase segments including optional cruise climb capability.
    
    This class handles the segmentation of cruise phase into:
    - Initial cruise segment (before optional climb)
    - Cruise climb segment (optional, using DP optimization)
    - Continued cruise segment (after optional climb)
    
    The cruise climb feature can be enabled/disabled via configuration flags
    and uses dynamic programming optimization similar to the initial climb phase.
    """
    
    def __init__(self, enable_climb: bool = False,
                 trigger_distance_fraction: float = 0.30,
                 altitude_increment_m: float = 1000.0,
                 mach_tolerance: float = 0.015):
        """
        Initialize cruise segment manager.
        
        Args:
            enable_climb: Enable/disable cruise climb feature
            trigger_distance_fraction: Fraction of cruise distance before climb (0.0-1.0)
            altitude_increment_m: Altitude increment for cruise climb [m]
            mach_tolerance: Mach tolerance for maintaining cruise Mach during climb
        """
        self.enable_climb = enable_climb
        self.trigger_distance_fraction = trigger_distance_fraction
        self.altitude_increment_m = altitude_increment_m
        self.mach_tolerance = mach_tolerance
    
    def calculate_segment_distances(self, total_distance_km: float) -> Tuple[float, float]:
        """
        Calculate distances for initial cruise segment.
        The continued cruise distance will be calculated after the climb to ensure total distance is completed.
        
        Args:
            total_distance_km: Total cruise distance [km]
            
        Returns:
            Tuple of (initial_distance_km, remaining_distance_km)
            If cruise climb is disabled, returns (total_distance_km, 0.0)
        """
        if not self.enable_climb:
            return total_distance_km, 0.0
        
        # Initial cruise segment: up to trigger point
        initial_distance_km = total_distance_km * self.trigger_distance_fraction
        
        # Remaining distance after initial segment
        # Note: Climb will cover some horizontal distance, which will be subtracted from remaining
        remaining_distance_km = total_distance_km - initial_distance_km
        
        return initial_distance_km, remaining_distance_km
    
    def execute_cruise_climb(self, initial_state: CruiseInitialState,
                            aero: PyAerodynamicsWrapper,
                            engine: EngineWrapper,
                            M_grid: np.ndarray,
                            H_sched: Optional[np.ndarray] = None) -> ClimbingCore.MinFuelSchedule:
        """
        Execute cruise climb using DP optimization.
        
        Args:
            initial_state: Current cruise state before climb
            aero: Aerodynamics wrapper
            engine: Engine wrapper
            M_grid: Mach number grid for DP optimization
            H_sched: Altitude schedule for DP optimization
            
        Returns:
            MinFuelSchedule: Results from cruise climb DP optimization
        """
        print(f"\n[CRUISE-CLIMB] Starting cruise climb optimization...")
        print(f"  Initial altitude: {initial_state.altitude_m:.0f} m")
        print(f"  Target altitude: {initial_state.altitude_m + self.altitude_increment_m:.0f} m")
        print(f"  Altitude increment: {self.altitude_increment_m:.0f} m")
        print(f"  Cruise Mach: {initial_state.mach:.3f}")
        
        # Calculate target altitude
        target_altitude_m = initial_state.altitude_m + self.altitude_increment_m
        
        # Ensure target altitude is within limits
        if target_altitude_m > MAX_CRUISE_ALT_M:
            print(f"[CRUISE-CLIMB] WARNING: Target altitude {target_altitude_m:.0f}m exceeds maximum {MAX_CRUISE_ALT_M:.0f}m")
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
        
        print(f"[CRUISE-CLIMB] Altitude schedule: {len(climb_h_sched)} points")
        print(f"  From {climb_h_sched[0]:.0f}m to {climb_h_sched[-1]:.0f}m")
        
        # Find feasible starting lever for cruise climb
        # First, create the lever grid that will be used by the DP solver
        lever_grid_dp = np.linspace(0.0, 1.0, N_LEVER_SAMPLES_CLIMB)
        
        # Calculate required thrust for level flight at starting conditions
        drag_N = aero.get_drag(initial_state.mach, initial_state.altitude_m, initial_state.weight_kg)
        required_thrust_N = drag_N  # For level flight, thrust = drag
        
        # Find lever position that provides required thrust (using the DP grid)
        from climb import find_lever_for_thrust
        start_lever, start_thrust, thrust_limited = find_lever_for_thrust(
            eng=engine,
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
            # Use a simple altitude fraction (0.0 for start of cruise climb)
            test_cost = ClimbingCore.compute_3d_cost(
                aero, engine, 
                initial_state.altitude_m, 
                initial_state.mach, 
                start_lever_on_grid,
                target_mach=initial_state.mach,
                prev_mach=None,
                altitude_fraction=0.0,
                mass_kg=initial_state.weight_kg
            )
            
            if np.isfinite(test_cost) and test_cost > 0:
                start_lever = start_lever_on_grid
                print(f"[CRUISE-CLIMB] Starting lever: {start_lever:.3f} (on grid, cost={test_cost:.4f})")
            else:
                # Try to find a feasible lever on the grid by testing multiple levers
                print(f"[CRUISE-CLIMB] WARNING: Lever {start_lever_on_grid:.3f} not feasible (cost={test_cost}), searching grid...")
                start_lever = None
                for test_lever in lever_grid_dp[::-1]:  # Try from high to low
                    test_cost = ClimbingCore.compute_3d_cost(
                        aero, engine,
                        initial_state.altitude_m,
                        initial_state.mach,
                        test_lever,
                        target_mach=initial_state.mach,
                        prev_mach=None,
                        altitude_fraction=0.0,
                        mass_kg=initial_state.weight_kg
                    )
                    if np.isfinite(test_cost) and test_cost > 0:
                        start_lever = test_lever
                        print(f"[CRUISE-CLIMB] Found feasible lever: {start_lever:.3f} (cost={test_cost:.4f})")
                        break
        
        if start_lever is None:
            # Last resort: use a reasonable lever position
            print(f"[CRUISE-CLIMB] WARNING: Could not find feasible lever, using 0.7 as fallback")
            start_lever = 0.7
        
        # Run DP optimization for cruise climb
        # Target Mach is the current cruise Mach (maintain same Mach)
        cruise_climb_result, cruise_climb_info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_fixed_mass(
            aero=aero,
            eng=engine,
            M_grid=M_grid,
            H_sched=climb_h_sched,
            lever_samples=N_LEVER_SAMPLES_CLIMB,
            target_mach=initial_state.mach,  # Maintain cruise Mach
            target_mach_tolerance=self.mach_tolerance,
            start_mach=initial_state.mach,  # Start at current cruise Mach
            start_lever=start_lever,  # Use feasible starting lever (on grid)
            mass_kg=initial_state.weight_kg  # Use current weight
        )
        
        print(f"[CRUISE-CLIMB] Cruise climb optimization complete:")
        print(f"  Fuel consumed: {cruise_climb_result.cumFuel_kg[-1]:.2f} kg")
        print(f"  Time: {np.sum(cruise_climb_result.dt_s):.2f} s ({np.sum(cruise_climb_result.dt_s)/60:.2f} min)")
        print(f"  Final altitude: {cruise_climb_result.alt_m[-1]:.0f} m")
        print(f"  Final Mach: {cruise_climb_result.mach[-1]:.3f} (target: {initial_state.mach:.3f})")
        
        return cruise_climb_result

# ========= MAIN INTERFACE =============================================

def run_cruise_simulation(climb_result: ClimbingCore.MinFuelSchedule, 
                         initial_mass_kg: float,
                         target_distance_km: float,
                         aero: PyAerodynamicsWrapper,
                         engine: EngineWrapper,
                         time_step_s: float = DEFAULT_TIME_STEP_S,
                         create_plots: bool = True,
                         M_grid: Optional[np.ndarray] = None) -> CruiseResults:
    """
    Main function to run complete cruise simulation with optional cruise climb.
    
    Args:
        climb_result: Results from 3D DP climb optimization
        initial_mass_kg: Initial aircraft mass before climb  
        target_distance_km: Cruise distance in kilometers
        aero: Aerodynamics wrapper
        engine: Engine wrapper
        time_step_s: Time step for simulation (default 60s)
        create_plots: Whether to create visualization plots
        M_grid: Mach number grid for cruise climb DP optimization (optional)
        
    Returns:
        Complete cruise simulation results (may include cruise climb if enabled)
    """
    print(f"\n{'='*60}")
    print("CRUISE PHASE SIMULATION")
    print(f"{'='*60}")
    
    # Initialize cruise segment manager with configuration parameters
    segment_manager = CruiseSegmentManager(
        enable_climb=ENABLE_CRUISE_CLIMB,
        trigger_distance_fraction=CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION,
        altitude_increment_m=CRUISE_CLIMB_ALTITUDE_INCREMENT_M,
        mach_tolerance=CRUISE_CLIMB_MACH_TOLERANCE
    )
    
    # Extract initial state from climb
    initial_state = extract_cruise_initial_state(climb_result, initial_mass_kg)
    
    # Check if cruise climb is enabled
    if not segment_manager.enable_climb:
        # Standard cruise simulation (no climb)
        print(f"[CRUISE] Cruise climb disabled - running standard cruise simulation")
        cruise_results = simulate_steady_cruise(
            initial_state=initial_state,
            target_distance_km=target_distance_km,
            aero=aero,
            engine=engine,
            time_step_s=time_step_s
        )
    else:
        # Segmented cruise with optional climb
        print(f"[CRUISE] Cruise climb enabled - running segmented cruise simulation")
        
        # Calculate initial cruise distance
        initial_dist, remaining_dist = segment_manager.calculate_segment_distances(target_distance_km)
        print(f"[CRUISE] Initial cruise segment: {initial_dist:.1f} km ({initial_dist/target_distance_km*100:.1f}% of total)")
        print(f"[CRUISE] Remaining distance: {remaining_dist:.1f} km (will be covered by climb + continued cruise)")
        
        # Run initial cruise segment
        print(f"\n[CRUISE] Running initial cruise segment ({initial_dist:.1f} km)...")
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
            weight_kg=initial_cruise_results.weight_kg[-1],
            fuel_consumed_climb_kg=initial_cruise_results.total_fuel_consumed_kg,
            climb_time_s=initial_cruise_results.total_time_s
        )
        
        # Execute cruise climb using DP optimization
        if M_grid is None:
            # Create default Mach grid if not provided
            M_min = max(MIN_CRUISE_MACH, initial_cruise_final_state.mach - 0.1)
            M_max = min(MAX_CRUISE_MACH, initial_cruise_final_state.mach + 0.1)
            M_grid = np.linspace(M_min, M_max, N_MACH_SAMPLES_CLIMB)
        
        cruise_climb_result = segment_manager.execute_cruise_climb(
            initial_state=initial_cruise_final_state,
            aero=aero,
            engine=engine,
            M_grid=M_grid,
            H_sched=None  # H_sched is created internally in execute_cruise_climb
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
            _, _, _, a = calculate_atmospheric_properties(alt_avg)
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
            print(f"[CRUISE] Setting continued cruise to 0 km")
            continued_dist = 0.0
        
        print(f"[CRUISE] Continued cruise segment: {continued_dist:.1f} km")
        print(f"[CRUISE] Total distance breakdown:")
        print(f"  Initial cruise: {initial_dist:.2f} km")
        print(f"  Cruise climb: {cruise_climb_distance_km:.2f} km")
        print(f"  Continued cruise: {continued_dist:.2f} km")
        print(f"  Total: {initial_dist + cruise_climb_distance_km + continued_dist:.2f} km (target: {target_distance_km:.2f} km)")
        
        # Extract state after cruise climb
        cruise_climb_final_state = CruiseInitialState(
            altitude_m=cruise_climb_result.alt_m[-1],
            mach=cruise_climb_result.mach[-1],
            weight_kg=cruise_climb_result.mass_kg[-1],
            fuel_consumed_climb_kg=cruise_climb_result.cumFuel_kg[-1],
            climb_time_s=np.sum(cruise_climb_result.dt_s)
        )
        
        # Run continued cruise segment (if there's remaining distance)
        if continued_dist > 0.01:  # Only if more than 10m remaining
            print(f"\n[CRUISE] Running continued cruise segment ({continued_dist:.1f} km)...")
            continued_cruise_results = simulate_steady_cruise(
                initial_state=cruise_climb_final_state,
                target_distance_km=continued_dist,
                aero=aero,
                engine=engine,
                time_step_s=time_step_s
            )
        else:
            # No continued cruise needed - create empty results
            print(f"\n[CRUISE] No continued cruise needed (climb covered all remaining distance)")
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
        print(f"[CRUISE] Skipping individual cruise plots (combined mission plots will be shown instead)...")
    
    print(f"{'='*60}")
    print("CRUISE SIMULATION COMPLETED")
    print(f"{'='*60}\n")
    
    return cruise_results