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
TARGET_ALT_M = 10000.0          # Target climb altitude [m]
N_PLOT_STEPS = 50               # Number of altitude steps for plotting
ALT_STEP_M = 200.0              # Altitude step size [m]
Y_AXIS_TOP_M = 12000.0          # Maximum altitude for plots [m]
MACH_COLS = 41                  # Number of Mach grid points

# Starting conditions (from main.py)
START_ALTITUDE_M = 10.0         # Takeoff altitude [m]
START_VELOCITY_MS = 85.0        # Takeoff velocity [m/s]
START_LEVER = 0.85              # Initial throttle lever position

# Climb optimization parameters (from main.py)
LEVER_SAMPLES = 50              # Number of lever positions for optimization
TARGET_MACH = 0.7               # Target cruise Mach number
TARGET_MACH_TOLERANCE = 0.015   # Tolerance for target Mach achievement
STRATEGY_DT_S = 0.2             # Time step for strategy simulation [s]

# Cruise phase parameters (from main.py and cruise.py)
CRUISE_DISTANCE_KM = 1500.0     # Cruise distance [km]
CRUISE_TIME_STEP_S = 60.0       # Time step for cruise simulation [s]

# Cruise constraints (from cruise.py)
MIN_CRUISE_MACH = 0.3           # Minimum safe cruise Mach
MAX_CRUISE_MACH = 0.9           # Maximum reasonable cruise Mach
MIN_CRUISE_ALT_M = 1000.0       # Minimum cruise altitude [m]
MAX_CRUISE_ALT_M = 15000.0      # Maximum cruise altitude [m]

# Cruise convergence (from cruise.py)
THRUST_CONVERGENCE_TOL = 1.0    # Thrust balance tolerance [N]
MAX_ITERATIONS = 50             # Maximum convergence iterations

# Descent phase parameters (from main.py and descent.py)
TARGET_DESCENT_ALT_M = 300.0    # Approach altitude [m] (~1000 ft)
TARGET_DESCENT_MACH = 0.25      # Approach Mach number

# Descent optimization parameters (from main.py)
N_ALTITUDE_STEPS = 50           # Number of altitude steps for DP
N_MACH_SAMPLES = 41             # Number of Mach samples for DP
N_LEVER_SAMPLES = 11            # Number of lever samples for DP

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