from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import time

# Import necessary components from existing modules
from atmosphere import Atmosphere
from aircraft_config import N_ENGINES, isa_properties, G_C
import climb
from climb import EngineWrapper, ClimbingCore
from pyaerodynamics_wrapper import PyAerodynamicsWrapper

# ========= CRUISE CONSTANTS AND SETTINGS =============================================

# Default cruise parameters (user adjustable)
DEFAULT_TIME_STEP_S = 60.0    # 1 minute time steps
DEFAULT_DISTANCE_KM = 1000.0  # Default cruise distance
GRAVITY_MS2 = G_C             # Standard gravity

# Convergence criteria for steady cruise calculations
THRUST_CONVERGENCE_TOL = 1.0   # Newton tolerance for thrust balance
MAX_ITERATIONS = 50            # Maximum iterations for convergence

# Safety and validation limits
MIN_CRUISE_MACH = 0.3         # Minimum safe cruise Mach
MAX_CRUISE_MACH = 0.9         # Maximum reasonable cruise Mach  
MIN_CRUISE_ALT_M = 1000.0     # Minimum cruise altitude
MAX_CRUISE_ALT_M = 15000.0    # Maximum cruise altitude

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

def calculate_weight_adjusted_drag(base_drag_N: float, current_weight_kg: float, 
                                 reference_weight_kg: float, mach: float, 
                                 altitude_m: float) -> float:
    """
    Calculate drag adjusted for current weight (induced drag variation).
    
    For constant Mach/altitude cruise, drag varies with weight due to:
    CL = Weight / (0.5 × ρ × V² × S)
    CDi = CL² / (π × AR × e)  (induced drag)
    CD_total = CD0 + CDi
    
    Args:
        base_drag_N: Base drag from aero tables at reference weight
        current_weight_kg: Current aircraft weight
        reference_weight_kg: Reference weight for base drag
        mach: Mach number
        altitude_m: Altitude in meters
        
    Returns:
        Weight-adjusted drag in Newtons
    """
    # Calculate weight ratio
    weight_ratio = current_weight_kg / reference_weight_kg
    
    # Get atmospheric properties
    _, _, rho, a = calculate_atmospheric_properties(altitude_m)
    true_airspeed = mach * a
    
    # Estimate induced drag scaling with weight²
    # This is a simplified model: CDi ∝ CL² ∝ Weight²
    # Assume induced drag is ~40% of total drag for typical cruise conditions
    induced_drag_fraction = 0.4
    parasitic_drag_fraction = 1.0 - induced_drag_fraction
    
    # Scale drag components
    parasitic_drag = base_drag_N * parasitic_drag_fraction  # Constant
    induced_drag_base = base_drag_N * induced_drag_fraction
    induced_drag_current = induced_drag_base * (weight_ratio ** 2)
    
    total_drag = parasitic_drag + induced_drag_current
    
    return float(total_drag)

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
                                  weight_kg: float, true_airspeed_mps: float) -> float:
    """
    Calculate specific excess power for cruise.
    
    Ps = (T_total - D) * V / W
    
    For ideal steady cruise, Ps should be approximately 0.
    
    Args:
        thrust_total_N: Total thrust in Newtons
        drag_N: Drag force in Newtons  
        weight_kg: Aircraft weight in kg
        true_airspeed_mps: True airspeed in m/s
        
    Returns:
        Specific excess power in m/s
    """
    weight_N = weight_kg * GRAVITY_MS2
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
        aero: Aerodynamics tables
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
        
        # Get base drag from aero tables and adjust for current weight
        base_drag_N = aero.get_drag(initial_state.mach, initial_state.altitude_m)
        
        # Calculate weight-adjusted drag (accounts for induced drag variation with weight)
        drag_N = calculate_weight_adjusted_drag(
            base_drag_N, current_weight, initial_state.weight_kg, 
            initial_state.mach, initial_state.altitude_m
        )
        
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
                                           true_airspeed_mps)
        
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

# ========= MAIN INTERFACE =============================================

def run_cruise_simulation(climb_result: ClimbingCore.MinFuelSchedule, 
                         initial_mass_kg: float,
                         target_distance_km: float,
                         aero: PyAerodynamicsWrapper,
                         engine: EngineWrapper,
                         time_step_s: float = DEFAULT_TIME_STEP_S,
                         create_plots: bool = True) -> CruiseResults:
    """
    Main function to run complete cruise simulation.
    
    Args:
        climb_result: Results from 3D DP climb optimization
        initial_mass_kg: Initial aircraft mass before climb  
        target_distance_km: Cruise distance in kilometers
        aero: Aerodynamics tables
        engine: Engine wrapper
        time_step_s: Time step for simulation (default 60s)
        create_plots: Whether to create visualization plots
        
    Returns:
        Complete cruise simulation results
    """
    print(f"\n{'='*60}")
    print("CRUISE PHASE SIMULATION")
    print(f"{'='*60}")
    
    # Extract initial state from climb
    initial_state = extract_cruise_initial_state(climb_result, initial_mass_kg)
    
    # Run cruise simulation
    cruise_results = simulate_steady_cruise(
        initial_state=initial_state,
        target_distance_km=target_distance_km,
        aero=aero,
        engine=engine,
        time_step_s=time_step_s
    )
    
    # Skip individual cruise plots - only show combined mission plots
    if create_plots:
        print(f"[CRUISE] Skipping individual cruise plots (combined mission plots will be shown instead)...")
    
    print(f"{'='*60}")
    print("CRUISE SIMULATION COMPLETED")
    print(f"{'='*60}\n")
    
    return cruise_results