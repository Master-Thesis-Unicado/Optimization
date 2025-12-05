# ========================================================================
# MISSION CONFIGURATION MODULE
# ========================================================================
"""
Mission-specific parameters for trajectory optimization and simulation.

Configuration domains organized by parameter type:
    1. Cache pre-computation: grid resolution for model caching
    2. Initial conditions: takeoff state variables
    3. Terminal constraints: target altitude/Mach for phase transitions
    4. Operational bounds: flight envelope limits
    5. Grid discretization: DP optimization grid dimensions
    6. Penalty systems: trajectory guidance and control penalties
    7. Numerical thresholds: integration and convergence tolerances
    8. Feature flags: enable/disable optional features
    9. Mission parameters: distance, time step, cruise climb settings
    10. Mission optimization: range/fuel iteration parameters
    11. Center of gravity: CG calculation settings

Parameters specify boundary conditions, discretization, and penalty coefficients
for dynamic programming optimization across all flight phases.
"""

from __future__ import annotations
from aircraft_config import W_FUEL_KG


# ========================================================================
# SECTION 0: CACHE PRE-COMPUTATION CONFIGURATION
# ========================================================================
"""
Cache pre-computation parameters define the grid resolution for engine and
aerodynamic model caching across all mission phases. These grids should encompass
the entire operational envelope to maximize cache hit rates during optimization.

Grid resolution trade-offs:
    - Higher resolution → better cache coverage → fewer runtime computations
    - Lower resolution → faster initialization → potential cache misses during optimization
    
Recommended: Set pre-computation grid to cover maximum bounds used across
climb, cruise, descent optimization and visualization.
"""

# ────────────────────────────────────────────────────────────────────────────
# Pre-computation Grid Dimensions
# ────────────────────────────────────────────────────────────────────────────
N_MACH_SAMPLES_PRECOMPUTE = 100                     # I_cache: Mach discretization for cache
N_ALTITUDE_SAMPLES_PRECOMPUTE = 100                 # K_cache: altitude samples for cache
N_LEVER_SAMPLES_PRECOMPUTE = 100                    # L_cache: throttle lever samples for cache

# ────────────────────────────────────────────────────────────────────────────
# Altitude Grid Configuration
# ────────────────────────────────────────────────────────────────────────────
ALTITUDE_MAX_PRECOMPUTE_M = 15000.0                 # h_max_cache [m]: maximum altitude (covers all phases)


# ========================================================================
# SECTION 1: INITIAL CONDITIONS
# ========================================================================
"""
Initial state variables for each flight phase.
Defines starting altitude, velocity, and throttle position.
"""

# ────────────────────────────────────────────────────────────────────────────
# Climb Phase: X_0 = (h_0, V_0, δ_0)
# ────────────────────────────────────────────────────────────────────────────
START_ALTITUDE_CLIMB_M = 10.0                       # h_0 [m]: takeoff altitude
START_VELOCITY_CLIMB_MS = 85.0                      # V_0 [m/s]: takeoff velocity
START_LEVER_CLIMB = 0.85                            # δ_0 [-]: initial throttle position


# ========================================================================
# SECTION 2: TERMINAL CONSTRAINTS
# ========================================================================
"""
Target state variables for phase transitions and mission completion.
Defines altitude and Mach number targets with associated tolerances.
"""

# ────────────────────────────────────────────────────────────────────────────
# Climb Phase: X_f = (h_target, M_target)
# ────────────────────────────────────────────────────────────────────────────
TARGET_ALT_CLIMB_M = 10060                          # h_target [m]: target altitude (FL330)
TARGET_MACH_CRUISE = 0.78                           # M_target [-]: target Mach (cruise entry)

# ────────────────────────────────────────────────────────────────────────────
# Descent Phase: X_f = (h_target, M_target)
# ────────────────────────────────────────────────────────────────────────────
TARGET_DESCENT_ALT_M = 300.0                        # h_target [m]: approach altitude (~1000 ft)
TARGET_DESCENT_MACH = 0.25                          # M_target [-]: approach Mach

# ────────────────────────────────────────────────────────────────────────────
# Terminal Mach Tolerance (All Phases)
# ────────────────────────────────────────────────────────────────────────────
TARGET_MACH_TOLERANCE = 0.010                       # tol_M [-]: terminal Mach tolerance


# ========================================================================
# SECTION 3: OPERATIONAL BOUNDS
# ========================================================================
"""
Flight envelope limits for each phase.
Defines minimum/maximum Mach numbers and altitudes.
"""

# ────────────────────────────────────────────────────────────────────────────
# Cruise Phase Bounds
# ────────────────────────────────────────────────────────────────────────────
MIN_CRUISE_MACH = 0.3                               # M_min [-]: minimum cruise Mach
MAX_CRUISE_MACH = 0.9                               # M_max [-]: maximum cruise Mach
MIN_CRUISE_ALT_M = 1000.0                           # h_min [m]: minimum cruise altitude
MAX_CRUISE_ALT_M = 15000.0                          # h_max [m]: maximum cruise altitude

# ────────────────────────────────────────────────────────────────────────────
# Descent Phase Bounds
# ────────────────────────────────────────────────────────────────────────────
MIN_DESCENT_MACH = 0.2                              # M_min [-]: minimum descent Mach
MAX_DESCENT_MACH = 0.85                             # M_max [-]: maximum descent Mach
STALL_SPEED_SAFETY_MARGIN = 1.15                    # Safety factor: M_min = M_stall × 1.15
ABSOLUTE_MIN_DESCENT_MACH = 0.15                    # M_abs_min [-]: absolute minimum (fallback)


# ========================================================================
# SECTION 4: DP GRID DISCRETIZATION
# ========================================================================
"""
Dynamic programming optimization grid dimensions for each phase.
Grid structure: (M_i, h_k, δ_j) where:
    - M_i: Mach number discretization (I points)
    - h_k: Altitude discretization (K steps)
    - δ_j: Throttle lever discretization (L samples)
"""

# ────────────────────────────────────────────────────────────────────────────
# Climb Phase Grid: (M_i, h_k, δ_j)
# ────────────────────────────────────────────────────────────────────────────
N_MACH_SAMPLES_CLIMB = 50                           # I: number of Mach points
N_ALTITUDE_STEPS_CLIMB = 50                         # K: number of altitude levels
N_LEVER_SAMPLES_CLIMB = 50                          # L: number of throttle positions

# ────────────────────────────────────────────────────────────────────────────
# Descent Phase Grid: (M_i, h_k, δ_j)
# ────────────────────────────────────────────────────────────────────────────
N_MACH_SAMPLES_DESCENT = 50                         # I: number of Mach points
N_ALTITUDE_STEPS_DESCENT = 50                       # K: number of altitude levels
N_LEVER_SAMPLES_DESCENT = 50                        # L: number of throttle positions


# ========================================================================
# SECTION 5: PENALTY SYSTEMS
# ========================================================================
"""
Trajectory guidance and control penalty coefficients for DP optimization.
Penalties guide the optimizer toward fuel-efficient trajectories while
maintaining safe operating conditions and target constraints.
"""

# ────────────────────────────────────────────────────────────────────────────
# 5.1: Climb Phase - Mach Trajectory Guidance
# ────────────────────────────────────────────────────────────────────────────
PENALTY_CLIMB_MACH_TRAJECTORY_GUIDANCE = True       # Enable Mach guidance
PENALTY_CLIMB_TARGET_MACH_TOLERANCE = 0.010         # tol_M [-]
PENALTY_CLIMB_MACH_PENALTY_BASE_WEIGHT = 0.15       # w_base [kg/Mach²]
PENALTY_CLIMB_MAX_REASONABLE_MACH_RATE = 0.1        # dM/dk_max [-]
PENALTY_CLIMB_TOTAL_STEPS_ESTIMATE = N_ALTITUDE_STEPS_CLIMB  # K_total
PENALTY_CLIMB_URGENCY_MULTIPLIER = 1.8              # α_urgency
PENALTY_CLIMB_GUIDANCE_PENALTY_WEIGHT = 0.3         # w_guidance

# ────────────────────────────────────────────────────────────────────────────
# 5.2: Climb Phase - Lever Penalty System
# ────────────────────────────────────────────────────────────────────────────
# Penalty function: P(δ) = w·[(δ-δ_MCT)^p + critical terms]
PENALTY_CLIMB_LEVER_PENALTY_GUIDANCE = True         # Enable lever penalties
PENALTY_CLIMB_LEVER_PENALTY_WEIGHT = 3.0            # w_lever
PENALTY_CLIMB_LEVER_PENALTY_THRESHOLD = 0.75        # δ_MCT (75%)
PENALTY_CLIMB_LEVER_PENALTY_EXPONENT = 3.0          # p
PENALTY_CLIMB_LEVER_PENALTY_CRITICAL_THRESHOLD = 0.90  # δ_crit (90%)
PENALTY_CLIMB_LEVER_PENALTY_CRITICAL_MULTIPLIER = 5.0  # α_crit
PENALTY_CLIMB_LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD = 0.95  # δ_ultra (95%)
PENALTY_CLIMB_LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER = 20.0  # α_ultra

# ────────────────────────────────────────────────────────────────────────────
# 5.3: Descent Phase - Mach Trajectory Guidance
# ────────────────────────────────────────────────────────────────────────────
PENALTY_DESCENT_MACH_TRAJECTORY_GUIDANCE = True     # Enable Mach guidance
PENALTY_DESCENT_TARGET_MACH_TOLERANCE = 0.030       # tol_M [-]
PENALTY_DESCENT_MACH_PENALTY_BASE_WEIGHT = 0.5      # w_base [kg/Mach²]
PENALTY_DESCENT_MAX_REASONABLE_MACH_RATE = 0.018    # dM/dk_max [-]
PENALTY_DESCENT_TOTAL_STEPS_ESTIMATE = N_ALTITUDE_STEPS_DESCENT  # K_total
PENALTY_DESCENT_URGENCY_MULTIPLIER = 2.5            # α_urgency
PENALTY_DESCENT_GUIDANCE_PENALTY_WEIGHT = 0.8       # w_guidance

# ────────────────────────────────────────────────────────────────────────────
# 5.4: Descent Phase - Lever Penalty System
# ────────────────────────────────────────────────────────────────────────────
# Penalty function: P(δ) = w·[(δ-δ_MCT)^p + critical terms]
PENALTY_DESCENT_LEVER_PENALTY_GUIDANCE = True       # Enable lever penalties
PENALTY_DESCENT_LEVER_PENALTY_WEIGHT = 3.0          # w_lever
PENALTY_DESCENT_LEVER_PENALTY_THRESHOLD = 0.85      # δ_MCT (85%)
PENALTY_DESCENT_LEVER_PENALTY_EXPONENT = 3.0        # p
PENALTY_DESCENT_LEVER_PENALTY_CRITICAL_THRESHOLD = 0.90  # δ_crit (90%)
PENALTY_DESCENT_LEVER_PENALTY_CRITICAL_MULTIPLIER = 5.0  # α_crit
PENALTY_DESCENT_LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD = 0.95  # δ_ultra (95%)
PENALTY_DESCENT_LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER = 20.0  # α_ultra

# ────────────────────────────────────────────────────────────────────────────
# 5.5: Mach Guidance Phase Thresholds (Shared)
# ────────────────────────────────────────────────────────────────────────────
MACH_GUIDANCE_FINAL_PHASE_START = 0.7               # ξ_final [-]: final phase start (70%)
MACH_GUIDANCE_FINAL_PHASE_RANGE = 0.3               # Δξ_final [-]: final phase range (30%)
MACH_GUIDANCE_TERMINAL_PHASE_START = 0.9            # ξ_terminal [-]: terminal phase start (90%)
MACH_GUIDANCE_TERMINAL_PHASE_RANGE = 0.1            # Δξ_terminal [-]: terminal phase range (10%)
MACH_GUIDANCE_TERMINAL_BOOST_MULTIPLIER = 2.0       # α_boost [-]: terminal boost multiplier


# ========================================================================
# SECTION 6: NUMERICAL THRESHOLDS & TOLERANCES
# ========================================================================
"""
Integration, convergence, and numerical stability parameters.
Defines minimum thresholds for trajectory post-processing and edge case handling.
"""

# ────────────────────────────────────────────────────────────────────────────
# 6.1: DP Post-Processing Thresholds (Shared)
# ────────────────────────────────────────────────────────────────────────────
DP_MIN_ALTITUDE_SEGMENT_M = 1.0                     # Δh_min [m]: minimum altitude for vertical segment
DP_MIN_VELOCITY_CHANGE_MPS = 0.1                    # ΔV_min [m/s]: minimum velocity change threshold
DP_MIN_TIME_STEP_S = 0.1                            # Δt_min [s]: minimum time step for integration
DP_MIN_SEGMENT_DISTANCE_M = 1.0                     # s_min [m]: fallback distance for minimal segments

# ────────────────────────────────────────────────────────────────────────────
# 6.2: Climb Phase - Numerical Thresholds
# ────────────────────────────────────────────────────────────────────────────
CLIMB_ALTITUDE_UNIFORMITY_TOLERANCE = 0.1           # ε_uniformity [-]: altitude spacing tolerance (10%)
CLIMB_THRUST_LIMITED_ATOL = 1e-3                    # ε_thrust [-]: thrust-limited detection tolerance
CLIMB_MIN_RESAMPLE_POINTS = 2                       # N_min [-]: minimum points for trajectory resampling
CLIMB_STALL_SAFETY_MARGIN = 1.0                     # γ_stall [-]: stall speed safety margin multiplier

# ────────────────────────────────────────────────────────────────────────────
# 6.3: Diagnostic and Reporting (Shared)
# ────────────────────────────────────────────────────────────────────────────
DP_PROGRESS_REPORT_INTERVAL = 10                    # N_report [-]: progress reporting frequency (steps)
DP_TRAJECTORY_DEBUG_LIMIT = 10                      # N_debug [-]: trajectory output point limit


# ========================================================================
# SECTION 7: FEATURE FLAGS & MODES
# ========================================================================
"""
Boolean flags to enable or disable optional features and comparison modes.
"""

# ────────────────────────────────────────────────────────────────────────────
# Climb Phase Features
# ────────────────────────────────────────────────────────────────────────────
ENABLE_STRATEGY_COMPARISON = False                  # Boolean: enable strategy comparison

# ────────────────────────────────────────────────────────────────────────────
# Cruise Phase Features
# ────────────────────────────────────────────────────────────────────────────
ENABLE_CRUISE_CLIMB = False                         # Boolean: enable cruise climb

# ────────────────────────────────────────────────────────────────────────────
# Export Features
# ────────────────────────────────────────────────────────────────────────────
ENABLE_EXCEL_EXPORT = False                         # Boolean: enable Excel export of mission data


# ========================================================================
# SECTION 8: MISSION PARAMETERS
# ========================================================================
"""
Phase-specific mission execution parameters.
Defines distances, time steps, and operational parameters for each flight phase.
"""

# ────────────────────────────────────────────────────────────────────────────
# 8.1: Climb Phase - Strategy Comparison Parameters
# ────────────────────────────────────────────────────────────────────────────
STRATEGY_DT_CLIMB_S = 0.1                           # Δt [s]: integration time step
E_DOT_CMD_CLIMB = 14                                # Ė_cmd [m/s]: specific energy rate command

# ────────────────────────────────────────────────────────────────────────────
# 8.2: Cruise Phase - Basic Mission Parameters
# ────────────────────────────────────────────────────────────────────────────
CRUISE_DISTANCE_KM = 3800                           # s_cruise [km]: cruise distance
CRUISE_TIME_STEP_S = 0.1                            # Δt [s]: integration time step

# ────────────────────────────────────────────────────────────────────────────
# 8.3: Cruise Phase - Cruise Climb Configuration
# ────────────────────────────────────────────────────────────────────────────
CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION = 0.50       # ξ_trigger [-]: fraction before climb
CRUISE_CLIMB_ALTITUDE_INCREMENT_M = 600             # Δh_cruise [m]: altitude step
CRUISE_CLIMB_FALLBACK_LEVER = 0.7                   # δ_fallback [-]: fallback throttle position
CRUISE_CLIMB_MACH_GRID_MARGIN = 0.1                 # ΔM_grid [-]: Mach grid margin for DP
MIN_CONTINUED_CRUISE_DISTANCE_KM = 0.01             # s_min [km]: minimum continued cruise distance


# ========================================================================
# SECTION 9: MISSION OPTIMIZATION
# ========================================================================
"""
Iterative optimization parameters for mission planning.
Includes range matching and fuel optimization convergence criteria.
"""

# ────────────────────────────────────────────────────────────────────────────
# 9.1: Range Optimization (Iterative Distance Matching)
# ────────────────────────────────────────────────────────────────────────────
TARGET_MISSION_RANGE_KM = 4537.4                    # s_target [km]: target total range
INITIAL_CRUISE_DISTANCE_KM = 4278.0                 # s_cruise,0 [km]: initial cruise estimate
RANGE_OPTIMIZATION_TOLERANCE_KM = 3.0               # ε_range [km]: convergence tolerance
MAX_RANGE_OPTIMIZATION_ITERATIONS = 15              # N_iter,max: iteration limit
RANGE_OPTIMIZATION_DAMPING_FACTOR = 0.75            # α_damp ∈ [0,1]: adjustment damping

# ────────────────────────────────────────────────────────────────────────────
# 9.2: Fuel Optimization (Minimum Fuel Computation)
# ────────────────────────────────────────────────────────────────────────────
FUEL_OPTIMIZATION_CONVERGENCE_TOLERANCE_KG = 5.0    # ε_fuel [kg]: convergence tolerance
FUEL_OPTIMIZATION_SAFETY_BUFFER_PERCENT = 0.05      # β_safety (5%): safety margin
FUEL_OPTIMIZATION_MAX_ITERATIONS = 25               # N_iter,max: iteration limit
FUEL_OPTIMIZATION_INITIAL_FUEL_LOW_KG = 1000.0      # m_fuel,low [kg]: lower bound
FUEL_OPTIMIZATION_INITIAL_FUEL_HIGH_KG = W_FUEL_KG + 5000.0  # m_fuel,high [kg]: upper bound


# ========================================================================
# SECTION 10: CENTER OF GRAVITY CONFIGURATION
# ========================================================================
"""
Center of gravity calculation and fuel distribution parameters.
Defines CG computation mode and fuel consumption sequence.
"""

# ────────────────────────────────────────────────────────────────────────────
# CG Calculation Mode and Default Values
# ────────────────────────────────────────────────────────────────────────────
USE_DYNAMIC_CG = True                               # Boolean: True = dynamic CG from fuel distribution
                                                    #          False = static CG_X_DEFAULT
CG_X_DEFAULT = 14.0                                 # x_CG,default [m]: default CG position

# ────────────────────────────────────────────────────────────────────────────
# Fuel Consumption Scenario
# ────────────────────────────────────────────────────────────────────────────
CG_CONSUMPTION_SCENARIO = "CENTER_FIRST"            # Fuel depletion sequence
                                                    # Options: "OUTER_FIRST", "CENTER_FIRST", "PROPORTIONAL"
