# =========================================================================
# MISSION CONFIGURATION MODULE
# =========================================================================
"""
Centralized mission configuration parameters.

This module consolidates all mission-specific parameters that were previously
scattered throughout the codebase into a single location for easy management
and modification.
"""

from __future__ import annotations

# ========= MISSION PARAMETERS (CONSOLIDATED FROM EXISTING CODE) =========

# Climb phase parameters (from climb.py and main.py)
TARGET_ALT_CLIMB_M = 8000    # Target climb altitude [m]
ALT_STEP_M = 200.0              # Altitude step size for plotting [m]
Y_AXIS_TOP_M = 14000.0          # Maximum altitude for plots [m]

# Starting conditions (from main.py) - Climb phase initialization
START_ALTITUDE_CLIMB_M = 10.0   # Takeoff altitude [m]
START_VELOCITY_CLIMB_MS = 85.0  # Takeoff velocity [m/s]
START_LEVER_CLIMB = 0.85        # Initial throttle lever position

# Climb optimization parameters (from main.py) - Consistent phase-specific naming
# ULTRA-HIGH RESOLUTION SETTINGS for maximum fuel efficiency optimization
N_MACH_SAMPLES_CLIMB =15     # Number of Mach samples for DP optimization (ultra-fine: 101 samples)
N_ALTITUDE_STEPS_CLIMB = 15      # Number of altitude steps for DP optimization (ultra-fine: 50 steps)
N_LEVER_SAMPLES_CLIMB = 15     # Number of lever samples for DP optimization (ultra-fine: 50 samples)
TARGET_MACH_CRUISE = 0.78        # Target Mach number at end of climb (cruise Mach)
TARGET_MACH_TOLERANCE_CLIMB = 0.015  # Tolerance for target Mach achievement in climb
STRATEGY_DT_CLIMB_S = 0.2       # Time step for climb strategy simulation [s]

# Cruise phase parameters (from main.py and cruise.py)
CRUISE_DISTANCE_KM = 4290.0     # Cruise distance [km]
CRUISE_TIME_STEP_S = 15.0       # Time step for cruise simulation [s] (ultra-fine: 15s for maximum accuracy)

# Range optimization parameters (for mission_range_optimizer.py)
TARGET_MISSION_RANGE_KM = 4537.0        # Target total mission range [km]
INITIAL_CRUISE_DISTANCE_KM = 4278.0     # Initial cruise distance estimate [km]
RANGE_OPTIMIZATION_TOLERANCE_KM = 10.0  # Convergence tolerance [km] (±10 km)
MAX_RANGE_OPTIMIZATION_ITERATIONS = 10  # Maximum optimization iterations
RANGE_OPTIMIZATION_DAMPING_FACTOR = 0.8 # Damping factor for cruise distance adjustment (0 < factor ≤ 1)

# Cruise constraints (from cruise.py)
MIN_CRUISE_MACH = 0.3           # Minimum safe cruise Mach
MAX_CRUISE_MACH = 0.9           # Maximum reasonable cruise Mach
MIN_CRUISE_ALT_M = 1000.0       # Minimum cruise altitude [m]
MAX_CRUISE_ALT_M = 15000.0      # Maximum cruise altitude [m]

# Cruise convergence (from cruise.py) - Phase-specific parameters
THRUST_CONVERGENCE_TOL_CRUISE = 1.0  # Thrust balance tolerance [N] for cruise equilibrium
MAX_ITERATIONS_CRUISE = 50           # Maximum convergence iterations for cruise thrust balance

# Cruise climb parameters (optional cruise altitude increase during cruise phase)
ENABLE_CRUISE_CLIMB = False          # Enable/disable cruise climb feature (True/False)
CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION = 0.30  # When to start cruise climb: fraction of TOTAL cruise distance before climb (0.30 = 30% of total distance)
CRUISE_CLIMB_ALTITUDE_INCREMENT_M = 3000.0     # Altitude increment for cruise climb [m]
CRUISE_CLIMB_MACH_TOLERANCE = 0.015             # Mach tolerance for maintaining cruise Mach during climb

# Descent phase parameters (from main.py and descent.py)
TARGET_DESCENT_ALT_M = 300.0    # Approach altitude [m] (~1000 ft)
TARGET_DESCENT_MACH = 0.25      # Approach Mach number

# Descent optimization parameters (from main.py) - Consistent naming with climb convention
# ULTRA-HIGH RESOLUTION SETTINGS for maximum fuel efficiency optimization
N_MACH_SAMPLES_DESCENT = 15     #  Number of Mach samples for DP optimization (ultra-fine: 101 samples)
N_ALTITUDE_STEPS_DESCENT = 15    #  Number of altitude steps for DP optimization (ultra-fine: 50 steps)
N_LEVER_SAMPLES_DESCENT = 15     #  Number of lever samples for DP optimization (ultra-fine: 50 samples)

# Descent constraints (from descent.py)
MIN_DESCENT_MACH = 0.2          # Minimum descent Mach
MAX_DESCENT_MACH = 0.85         # Maximum descent Mach

# Mission constraints (from various files)
DELTA_ISA_K = 0.0               # Temperature offset from ISA [K]
MAX_MISSION_TIME_HOURS = 12.0   # Maximum total mission time [hours]
CONTINGENCY_FUEL_PERCENT = 0.06 # Contingency fuel percentage
RESERVE_FUEL_PERCENT = 0.05     # Reserve fuel percentage

# Simulation settings (from various files)
DEBUG = True                    # Enable debug output
VERBOSE = True                  # Enable verbose logging
CREATE_PLOTS = True             # Generate visualization plots
SAVE_RESULTS = True             # Save results to files

# Output settings (from various files)
OUTPUT_DIRECTORY = "Images"     # Directory for output files
PLOT_FORMAT = "png"             # Plot file format
HTML_OUTPUT = True              # Generate HTML visualizations