# Climb Phase Documentation

> **Scope**: Complete documentation of the climb phase simulation system, including strategy-based climb trajectories, 3D dynamic programming optimization, penalty systems, and integration with mission analysis for optimal fuel consumption.

---

## Table of Contents

1. [System Overview and Objectives](#1-system-overview-and-objectives)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [System Architecture and Data Structures](#3-system-architecture-and-data-structures)
4. [Strategy System Implementation](#4-strategy-system-implementation)
5. [Dynamic Programming Optimization](#5-dynamic-programming-optimization)
6. [Penalty System and Guidance](#6-penalty-system-and-guidance)
7. [Code Execution Flow and Logic](#7-code-execution-flow-and-logic)
8. [Integration and Interface](#8-integration-and-interface)
9. [Validation and Quality Assurance](#9-validation-and-quality-assurance)

---

## 1) System Overview and Objectives

### 1.1 Purpose and Scope

The climb phase simulation implements a comprehensive climb trajectory analysis system that models aircraft performance during the climb segment of flight. The system provides multiple climb strategies, 3D dynamic programming optimization, and penalty-based guidance for realistic climb trajectory simulation.

### 1.2 System Objectives

**Primary Objectives:**
- **Fuel-Optimal Climb**: Find minimum fuel consumption climb trajectories
- **Strategy Analysis**: Compare different climb strategies and their performance
- **Realistic Trajectories**: Generate physically realizable climb paths
- **Mission Integration**: Provide seamless connection with cruise and descent phases

**Key Components:**
- Multiple climb strategies (linear, exponential, constant speed/Mach)
- 3D dynamic programming optimization (Mach × Altitude × Lever)
- Penalty-based guidance system for trajectory realism
- Engine performance integration with FADEC-like control
- Atmospheric property calculations using ISA model
- Integration with aerodynamics tables from Excel

### 1.3 System Flow Overview

The climb simulation follows a logical progression:

1. **Initialization**: Load aerodynamics data, engine model, and system parameters
2. **Strategy Simulation**: Execute various climb strategies for comparison
3. **3D Optimization**: Run dynamic programming optimization for fuel-optimal path
4. **Penalty Integration**: Apply guidance penalties for realistic trajectories
5. **Results Analysis**: Generate comprehensive performance data and visualizations
6. **Mission Integration**: Provide climb results for cruise phase initialization

---

## 2) Mathematical Foundation

### 2.1 Climb Flight Physics

**Theory**: Climb flight requires excess thrust beyond that needed for level flight. The specific excess power (Ps) determines the aircraft's climb capability and fuel consumption rate.

**Mathematical Formulation:**
```math
P_s = \frac{(T_{total} - D) \times V_{TAS}}{W}
```

```math
\dot{h} = P_s \times \sin(\gamma) \approx P_s \times \frac{\dot{h}}{V_{TAS}}
```

```math
J = \frac{\dot{m}_{fuel}}{P_s} = \frac{\dot{m}_{fuel} \times W}{(T_{total} - D) \times V_{TAS}} \quad [kg/m]
```

Where:
- `P_s` = Specific excess power (m/s)
- `T_{total}` = Total thrust (N)
- `D` = Total drag (N)
- `V_{TAS}` = True airspeed (m/s)
- `W` = Aircraft weight (N)
- `J` = Fuel cost density (kg/m)

**Weight Dependency**: The cost density `J` is directly proportional to weight `W` through the specific excess power calculation. As fuel is consumed during climb, the aircraft weight decreases, which affects the cost calculation for subsequent states. This creates a circular dependency: `next_cost` depends on `next_weight`, while `next_weight` depends on `fuel_burned`, which depends on `next_cost`.

### 2.2 Dynamic Programming Formulation

**Theory**: The climb optimization problem can be formulated as a discrete-time dynamic programming problem using Bellman's principle of optimality.

**Continuous Problem Formulation:**
```math
\min_{\ell(\cdot),\,M(\cdot)} \quad J_{tot} \,=\, \int_{h_0}^{h_f} \underbrace{\frac{\dot m(\ell,M,h)}{P_s(\ell,M,h)}}_{J(\ell,M,h)\;[\mathrm{kg/m}]}\, \mathrm{d}h
```

**Discrete State Space:**
```math
S = \{M_i\} \times \{h_j\} \times \{\ell_k\}
```

**Bellman Equation:**
```math
J^*(M_i, h_j, \ell_k) = \min_{\ell'} \left[ J(M_i, h_j, \ell_k, \ell') + \gamma J^*(M_{i'}, h_{j'}, \ell') \right]
```

### 2.3 Strategy Energy Allocation

**Theory**: Climb strategies allocate energy between climb rate and speed optimization using configurable energy fractions.

**Energy Allocation Model:**
```math
E_{total} = E_{climb} + E_{speed}
```

```math
\dot{h} = cw \times P_s
```

```math
\dot{V} = sw \times \frac{P_s \times W}{V \times m}
```

Where:
- `cw` = Climb weight fraction
- `sw` = Speed weight fraction
- `cw + sw = 1.0`

---

## 3) System Architecture and Data Structures

### 3.1 Core System Parameters

**Aircraft and Engine Parameters:**
```python
N_ENGINES = 2                    # Number of engines
INITIAL_MASS_KG = 65000.0        # Fixed aircraft mass (kg)
S_REF_M2 = 122.4                 # Reference wing area (m²)
```

**Engine Operational Limits:**
```python
M_MIN_DEFAULT = 0.0              # Default minimum operational Mach
M_MMO = 0.94                     # Maximum operating Mach
CL_MAX = None                    # Maximum lift coefficient
```

**Grid and Discretization Parameters:**
```python
TARGET_ALT_M = 10000.0           # Target altitude for 3D DP optimization (m)
ALT_STEP_M = 200.0               # Altitude step size (m)
MACH_COLS = 81                   # Number of Mach discretization points
N_PLOT_STEPS = 50                # Number of trajectory points for plotting
```

**Strategy Energy Management:**
```python
E_DOT_CMD = 14                   # Command energy rate (m/s)
```

### 3.2 Data Structures

#### 3.2.1 StrategyRun

**Purpose**: Complete results container for strategy-based climb simulation.

**Structure:**
```python
@dataclass
class StrategyRun:
    label: str                    # Strategy identifier
    h_m: np.ndarray              # Altitude trajectory
    V_ms: np.ndarray             # Velocity trajectory
    t_s: np.ndarray              # Time progression
    lever: np.ndarray            # Engine lever positions
    Ttot_N: np.ndarray           # Total thrust
    D_N: np.ndarray              # Total drag
    Ps_ms: np.ndarray            # Specific excess power
    mdot_kgps: np.ndarray        # Fuel flow rate
    dt_s: np.ndarray             # Time steps
    dFuel_kg: np.ndarray         # Fuel consumed per step
    cumFuel_kg: np.ndarray       # Cumulative fuel consumption
```

#### 3.2.2 MinFuelSchedule

**Purpose**: Results container for 3D dynamic programming optimization.

**Structure:**
```python
@dataclass
class MinFuelSchedule:
    alt_m: np.ndarray            # Optimal altitude trajectory
    mach: np.ndarray             # Optimal Mach trajectory
    lever: np.ndarray            # Optimal lever trajectory
    cumFuel_kg: np.ndarray       # Cumulative fuel consumption
    dt_s: np.ndarray             # Time steps
    total_time_s: float          # Total climb time
    total_fuel_kg: float         # Total fuel consumed
```

### 3.3 System Integration Components

**Aerodynamics System:**
```python
class AeroTables:
    def get_drag(self, mach: float, altitude_m: float) -> float
    def get_lift_coefficient(self, mach: float, altitude_m: float) -> float
```

**Engine System:**
```python
class EngineWrapper:
    def thrust_with_lever(self, lever: float, mach: float, altitude_m: float) -> float
    def tsfc_current(self) -> float
```

**Atmosphere System:**
```python
def isa_properties(h_m: float) -> Tuple[float, float, float]
def a_from_altitude(h_m: float) -> float
```

---

## 4) Strategy System Implementation

### 4.1 Strategy Types and Energy Allocation

**Linear Strategies:**
```python
    @staticmethod
def linear_climb(altitude, velocity, altitude_fraction):
    """Linear climb strategy with constant energy allocation."""
    climb_norm = altitude_fraction  # e.g., 0.10 = 10% climb
    speed_norm = 1.0 - climb_norm   # 90% speed
    return climb_norm, speed_norm
```

**Exponential Strategies:**
```python
@staticmethod
def increasing_climb(altitude, velocity, altitude_fraction):
    """Exponential climb strategy with increasing climb emphasis."""
        af_param, current_af = altitude_fraction
    min_climb = 0.1
    max_climb = 0.9
    exponent = 1.0 + 2.0 * af_param
    exp_factor = current_af ** exponent
    climb_norm = min_climb + (max_climb - min_climb) * exp_factor
    speed_norm = 1.0 - climb_norm
    return climb_norm, speed_norm
```

**Constant Strategies:**
```python
@staticmethod
def constant_mach(altitude, velocity, altitude_fraction):
    """Constant Mach strategy maintaining fixed Mach number."""
    return 1.0, 0.0  # 100% climb, 0% speed change
```

### 4.2 Strategy Simulation Engine

**Function**: `ClimbingCore.StrategyManager.simulate_strategy_path()`

**Simulation Flow:**
```python
@staticmethod
def simulate_strategy_path(*, label: str, aero: AeroTables, eng: EngineWrapper,
                          mass0_kg: float, h0_m: float, V0_ms: float,
                          target_alt_m: float, dt: float,
                          strategy_fn: Callable, altitude_fraction: Optional[float]) -> StrategyRun:
    
    # Initialize trajectory arrays
    h_hist, V_hist, t_hist = [float(h0_m)], [float(V0_ms)], [0.0]
    lever_hist, Ttot_hist, D_hist, Ps_hist = [], [], [], []
    mdot_hist, dt_hist, dFuel_hist, cumFuel_hist = [], [], [], [0.0]
    
    mass_kg = float(mass0_kg)  # Fixed mass for strategies
    
    # Main simulation loop
    while h_hist[-1] < target_alt_m:
        h = float(h_hist[-1])
        V = float(V_hist[-1])
        t = float(t_hist[-1])
        
        # Calculate current conditions
        a = a_from_altitude(h)
        M = V / max(a, 1e-9)
        Mq = float(np.clip(M, M_MIN_DEFAULT, 0.94))
        
        # Apply strategy energy allocation
        current_altitude_fraction = (h - h0_m) / (target_alt_m - h0_m)
        cw, sw = strategy_fn(h, V, altitude_fraction)
        
        # Calculate energy allocation
        climb_weight = cw
        speed_weight = sw
        
        # Continue simulation...
```

### 4.3 Strategy Set Configuration

**Strategy Set Builder:**
```python
@staticmethod
def build_strategy_set() -> List[Tuple[str, Callable, Optional[float]]]:
    """Build comprehensive set of climb strategies for comparison."""
    out = []
    for af in [0.10, 0.20, 0.30]:
        out.append((f"Linear AF={af:.2f}", 
                   ClimbingCore.StrategyManager.StrategyProfiles.FixedEnergy.Linear.profile, af))
    for af in [0.50]:
        out.append((f"Exp climb AF={af:.2f}", 
                   ClimbingCore.StrategyManager.StrategyProfiles.FixedEnergy.Exponential.increasing_climb, af))
        out.append((f"Exp speed AF={af:.2f}", 
                   ClimbingCore.StrategyManager.StrategyProfiles.FixedEnergy.Exponential.increasing_speed, af))
    out.append(("Constant speed", 
               ClimbingCore.StrategyManager.StrategyProfiles.ConstantRates.constant_speed, None))
    out.append(("Constant Mach",  
               ClimbingCore.StrategyManager.StrategyProfiles.ConstantRates.constant_mach(), None))
    return out
```

---

## 5) Dynamic Programming Optimization

### 5.1 3D State Space Definition

**State Variables:**
- **Mach Number**: `M ∈ [M_min, M_MMO]`
- **Altitude**: `h ∈ [h_0, h_target]`
- **Engine Lever**: `ℓ ∈ [0.0, 1.0]`

**Discretization:**
```python
# Mach grid
M_grid = np.linspace(M_MIN_DEFAULT, M_MMO, MACH_COLS)

# Altitude grid
H_sched = np.arange(h0_m, target_alt_m + ALT_STEP_M, ALT_STEP_M)

# Lever grid
lever_samples = 50
lever_grid = np.linspace(0.0, 1.0, lever_samples)
```

### 5.2 Bellman's Principle Implementation

**Function**: `ClimbingCore.DynamicProgrammingOptimizer.solve_3d_fixed_mass()`

**Algorithm:**
```python
@staticmethod
def solve_3d_fixed_mass(aero: AeroTables, eng: EngineWrapper,
                       M_grid: np.ndarray, H_sched: np.ndarray,
                       lever_samples: int = 10,
                       target_mach: float = None,
                       target_mach_tolerance: float = 0.02,
                       start_mach: float = None,
                       start_lever: float = None,
                       mass_kg: float = None) -> MinFuelSchedule:
    
    # Initialize 3D cost grid and weight matrix
    F = np.full((K, I, L), np.inf)
    weight_matrix = np.full((K, I, L), np.nan)  # Track weight at each state
    
    # Set terminal condition
    F[0, start_mach_idx, start_lever_idx] = 0.0  # Starting cost is 0
    weight_matrix[0, start_mach_idx, start_lever_idx] = initial_mass
    
    # Forward pass - Bellman's equation with weight-dependent costs
    for k in range(K - 1):  # For each altitude level
        current_alt = H_sched[k]
        next_alt = H_sched[k + 1]
        dh = next_alt - current_alt
        
        for each feasible state (k, i, j):
            current_weight = weight_matrix[k, i, j]
            
            # Calculate current cost with current weight
            current_cost = compute_3d_cost(..., mass_kg=current_weight)
            
            # Single recalculation approach (see Section 5.4)
            # 1. First pass: next_cost with current_weight
            # 2. Estimate fuel and next_weight
            # 3. Recalculate next_cost with estimated weight
            # 4. Final trapezoidal integration
            
            # Find optimal transition from previous altitude
            for each next state (k+1, i', j'):
                # Calculate transition cost using weight-dependent calculation
                step_cost = 0.5 * (current_cost + next_cost) * dh
                total_cost = F[k, i, j] + step_cost
                
                # Update if this path is better
                if total_cost < F[k+1, i', j']:
                    F[k+1, i', j'] = total_cost
                    weight_matrix[k+1, i', j'] = next_weight
    
    # Backtracking to find optimal path
    return backtrack_optimal_path(F, weight_matrix, M_grid, H_sched, lever_grid)
```

**Key Implementation Details:**
- **Weight Tracking**: The `weight_matrix` tracks aircraft weight at each DP state, accounting for fuel consumption during climb
- **Weight-Dependent Costs**: Cost calculations use the appropriate weight for each state (see Section 5.4)
- **Trapezoidal Integration**: Fuel consumption between states uses trapezoidal rule for improved accuracy

### 5.3 Cost Function Calculation

**Function**: `ClimbingCore.DynamicProgrammingOptimizer.compute_3d_cost()`

**Cost Calculation:**
```python
@staticmethod
def compute_3d_cost(aero: AeroTables, eng: EngineWrapper,
                   mach: float, altitude_m: float, lever: float,
                   mass_kg: float) -> float:
    
    # Calculate atmospheric properties
    T, p, rho = isa_properties(altitude_m)
    a = a_from_altitude(altitude_m)
    V_tas = mach * a
    
    # Calculate thrust and drag
    thrust_per_engine = eng.thrust_with_lever(lever, mach, altitude_m)
    thrust_total = N_ENGINES * thrust_per_engine if thrust_per_engine else 0.0
    drag = aero.get_drag(mach, altitude_m)
    
    # Calculate specific excess power (weight-dependent)
    weight = mass_kg * G_C
    Ps = (thrust_total - drag) * V_tas / weight if weight > 0 else 0.0
    
    # Calculate fuel flow
    tsfc = eng.tsfc_current()
    fuel_flow = thrust_total * tsfc if tsfc > 0 else 0.0
    
    # Calculate cost density J = fuel_flow / Ps = fuel_flow * W / ((T-D) * V)
    # Note: J is directly proportional to weight W
    if Ps > 0:
        J = fuel_flow / Ps  # kg/m
    else:
        J = np.inf  # Infeasible state
    
    return J
```

### 5.4 Weight-Dependent Cost Calculation with Single Recalculation

**Problem**: During dynamic programming state transitions, a circular dependency exists:
- `next_cost` depends on `next_weight` (through `Ps = (T-D)V/W`)
- `next_weight = current_weight - fuel_burned`
- `fuel_burned` depends on `next_cost` (via trapezoidal integration)

**Solution**: Single recalculation approach (Option 2) to capture first-order weight effects with minimal computational overhead.

**Algorithm:**
```python
# Step 1: Calculate current cost with current weight
current_cost = compute_3d_cost(..., mass_kg=current_weight)

# Step 2: First pass - calculate next_cost with current_weight
next_cost_initial = compute_3d_cost(..., mass_kg=current_weight)

# Step 3: Estimate fuel burned using trapezoidal integration
fuel_burned_initial = 0.5 * (current_cost + next_cost_initial) * dh
next_weight_estimate = current_weight - fuel_burned_initial

# Step 4: Recalculation - compute next_cost with estimated weight
# This accounts for weight-dependent effects: J ∝ W
next_cost_refined = compute_3d_cost(..., mass_kg=next_weight_estimate)

# Step 5: Final calculation with refined cost
step_cost = 0.5 * (current_cost + next_cost_refined) * dh
next_weight = current_weight - step_cost
```

**Mathematical Basis:**
- Cost density is linearly proportional to weight: `J = mdot × W / ((T-D) × V)`
- First-order weight change: `ΔJ/J ≈ ΔW/W` (typically 0.02-0.07% per 200m step)
- Single recalculation captures the dominant first-order effect
- Residual error is second-order: `~0.00005%` (negligible)

**Benefits:**
- **Accuracy**: Reduces error from ~0.07% to ~0.00005% per step
- **Efficiency**: One extra cost calculation per step (vs 5 in full iteration)
- **Robustness**: Fallback to initial calculation if refinement fails

---

## 6) Penalty System and Guidance

### 6.1 Mach Trajectory Guidance

**Purpose**: Guide optimization toward realistic Mach trajectories that follow typical climb profiles.

**Penalty Function:**
```python
def calculate_mach_penalty(mach: float, altitude_m: float, 
                          target_alt_m: float) -> float:
    """Calculate Mach trajectory guidance penalty."""
    
    altitude_fraction = altitude_m / target_alt_m
    
    # Typical climb Mach profile: start low, increase with altitude
    target_mach = 0.5 + 0.3 * altitude_fraction  # 0.5 to 0.8 Mach
    
    # Penalty for deviation from target
    mach_deviation = abs(mach - target_mach)
    penalty_weight = MACH_PENALTY_BASE_WEIGHT
    
    # Increase penalty weight with altitude (urgency)
    urgency_multiplier = 1.0 + URGENCY_MULTIPLIER * altitude_fraction
    penalty_weight *= urgency_multiplier
    
    return penalty_weight * (mach_deviation ** 2)
```

### 6.2 Lever Position Penalties

**Purpose**: Penalize excessive lever usage to maintain realistic engine operation.

**Penalty Function:**
```python
def calculate_lever_penalty(lever: float) -> float:
    """Calculate lever position penalty for excessive usage."""
    
    if lever <= LEVER_PENALTY_THRESHOLD:
        return 0.0  # No penalty below threshold
    
    # Cubic penalty growth above threshold
    excess = lever - LEVER_PENALTY_THRESHOLD
    penalty = LEVER_PENALTY_WEIGHT * (excess ** LEVER_PENALTY_EXPONENT)
    
    # Critical range penalties
    if lever >= LEVER_PENALTY_CRITICAL_THRESHOLD:
        penalty *= LEVER_PENALTY_CRITICAL_MULTIPLIER
    if lever >= LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD:
        penalty *= LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
    
    return penalty
```

### 6.3 Integrated Cost Function

**Total Cost Calculation:**
```python
def calculate_total_cost(aero: AeroTables, eng: EngineWrapper,
                        mach: float, altitude_m: float, lever: float,
                        mass_kg: float, target_alt_m: float) -> float:
    
    # Base fuel cost
    base_cost = compute_3d_cost(aero, eng, mach, altitude_m, lever, mass_kg)
    
    # Add penalties
    mach_penalty = calculate_mach_penalty(mach, altitude_m, target_alt_m)
    lever_penalty = calculate_lever_penalty(lever)
    
    total_cost = base_cost + mach_penalty + lever_penalty
    
    return total_cost
```

---

## 7) Code Execution Flow and Logic

### 7.1 System Entry Point and Initialization

**Main Entry Function**: `main()` in `main.py`

**Actual Execution Sequence:**
```python
def main():
    # 1. Load aerodynamics data from Excel
    print("[READ] Aerodynamics (Excel Sheet4) …")
    aero = AeroTables(AERO_XLSX, AERO_SHEET)
    
    # 2. Set up dense grids for contours
    M_min, M_max = float(aero.mach_grid[0]), float(aero.mach_grid[-1])
    M_dense = np.linspace(M_min, M_max, MACH_COLS)
    H_plot = np.arange(10.0, Y_AXIS_TOP_M + 0.5*ALT_STEP_M, ALT_STEP_M)
    
    # 3. Set effective minimum Mach
    climb.M_MIN_EFFECTIVE = max(climb.M_MIN_DEFAULT, float(aero.mach_grid[0]))
    
    # 4. Initialize engine wrapper
    print("[ENGINE] Loading engine stub …")
    eng = EngineWrapper(ENGINE_STUB_PATH)
    
    # 5. Performance optimization: Pre-compute grids
    print("[OPTIMIZATION] Pre-computing engine and drag grids...")
    lever_grid = np.linspace(0.0, 1.0, 21)
    eng.precompute_grid(M_dense, H_plot, lever_grid)
    aero.precompute_drag_grid(M_dense, H_plot)
    
    # 6. Compute background Ps grid
    print("[PS] Computing background Ps grid (max lever, ref mass) …")
    M_grid, H_plot, Ps_base = compute_sep_grid_maxlever(aero, eng, INITIAL_MASS_KG,
                                                        M_grid=M_dense, H_grid=H_plot)
```

### 7.2 Strategy Simulation Flow

**Strategy Execution:**
```python
# 2. Strategy Simulation
def run_strategy_simulation():
    strategies = ClimbingCore.StrategyManager.build_strategy_set()
    strategy_results = []
    
    DT_S = 0.2  # Time step for strategy simulation
    
    for name, fn, af in strategies:
        # Simulate each strategy
        sr = ClimbingCore.StrategyManager.simulate_strategy_path(
            label=name,
            aero=aero,
            eng=eng,
            mass0_kg=INITIAL_MASS_KG,
            h0_m=10.0,  # Start altitude
            V0_ms=85.0,  # Start velocity
            target_alt_m=TARGET_ALT_M,
            dt=DT_S,
            strategy_fn=fn,
            altitude_fraction=af
        )
        
        # Resample to common grid for comparison
        sr_resampled = ClimbingCore.StrategyManager.resample_strategy_run(sr, N_PLOT_STEPS)
        strategy_results.append(sr_resampled)
    
    return strategy_results
```

### 7.3 Dynamic Programming Flow

**3D Optimization:**
```python
# 3. 3D Dynamic Programming Optimization
def run_3d_optimization():
    # Set up grids
    M_min, M_max = float(aero.mach_grid[0]), float(aero.mach_grid[-1])
    M_grid = np.linspace(M_min, M_max, MACH_COLS)
    
    # Create uniform altitude steps
    uniform_step_size = TARGET_ALT_M / N_PLOT_STEPS  # 200m steps
    H_sched = np.arange(10.0, TARGET_ALT_M + uniform_step_size, uniform_step_size)
    
    # Calculate starting Mach from V0_ms=85.0 m/s at 10m altitude
    a = atmospheric_props.a_from_altitude(10.0)
    start_mach = 85.0 / a  # Starting Mach from V0_ms=85.0 m/s
    
    # Run 3D DP optimization
    dp_sched, dp_info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_fixed_mass(
        aero=aero,
        eng=eng,
        M_grid=M_grid,
        H_sched=H_sched,
        lever_samples=50,
        target_mach=0.7,
        target_mach_tolerance=0.015,
        start_mach=start_mach,
        start_lever=0.85
    )
    
    return dp_sched, dp_info
```

### 7.4 Visualization and Analysis Flow

**Results Processing:**
```python
# 4. Results Analysis and Visualization
def analyze_results(strategy_results, dp_result):
    # Convert DP result to StrategyRun format for plotting
    dp_strategy_run = convert_dp_to_strategy_run(dp_result, N_PLOT_STEPS)
    
    # Combine all results for comparison
    all_results = strategy_results + [dp_strategy_run]
    
    # Generate interactive plots
    plot_strategies_interactive(all_results, Ps_base, M_grid, H_plot)
    plot_J_3d_plotly(M_grid, H_sched, lever_grid, J_grid_3d, min_path=dp_result)
    create_strategy_comparison_plots(all_results)
    
    return all_results
```

### 7.5 Cruise Integration Flow

**Cruise Simulation Integration:**
```python
# 5. Cruise Simulation using climb results
def run_cruise_simulation_from_climb():
    # Use optimal climb result for cruise initialization
    cruise_results = run_cruise_simulation(
        climb_result=dp_sched,  # Use DP optimal result
        initial_mass_kg=INITIAL_MASS_KG,
        target_distance_km=1000.0,  # 1000 km cruise
        aero=aero,
        engine=eng,
        time_step_s=60.0,  # 1 minute steps
        create_plots=True
    )
    
    return cruise_results
```

### 7.6 Complete Execution Flow

```
CLIMB SIMULATION EXECUTION FLOW
===============================

┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM INITIALIZATION                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ main() entry point                                     │   │
│  │  ├── Load aerodynamics from Excel                      │   │
│  │  ├── Initialize engine wrapper                         │   │
│  │  ├── Apply parameters from Excel                       │   │
│  │  └── Set simulation parameters                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                STRATEGY SIMULATION                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FOR each strategy in strategy set:                     │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ simulate_strategy_path()                        │   │   │
│  │  │  ├── Initialize trajectory arrays               │   │   │
│  │  │  ├── WHILE altitude < target:                   │   │   │
│  │  │  │   ├── Calculate current conditions           │   │   │
│  │  │  │   ├── Apply strategy energy allocation       │   │   │
│  │  │  │   ├── Calculate thrust and drag              │   │   │
│  │  │  │   ├── Calculate specific excess power        │   │   │
│  │  │  │   ├── Update trajectory                      │   │   │
│  │  │  │   └── Advance time and altitude              │   │   │
│  │  │  └── Return StrategyRun object                  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│             3D DYNAMIC PROGRAMMING OPTIMIZATION                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ solve_3d_fixed_mass()                                  │   │
│  │  ├── Initialize 3D cost grid J_grid_3d                 │   │
│  │  ├── Set terminal conditions                           │   │
│  │  ├── FOR each altitude level (forward pass):           │   │
│  │  │   ├── FOR each Mach number:                         │   │
│  │  │   │   ├── FOR each lever position:                  │   │
│  │  │   │   │   ├── Calculate cost for this state         │   │
│  │  │   │   │   ├── Find optimal transition from previous │   │
│  │  │   │   │   └── Update cost grid                      │   │
│  │  │   │   └── Store minimum cost                        │   │
│  │  │   └── Complete altitude level                       │   │
│  │  └── Backtrack optimal path                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESULTS ANALYSIS AND VISUALIZATION                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ analyze_results()                                      │   │
│  │  ├── Resample all trajectories to common grid          │   │
│  │  ├── Convert DP result to StrategyRun format           │   │
│  │  ├── Generate strategy comparison plots                │   │
│  │  ├── Create 3D optimization visualization              │   │
│  │  └── Generate performance summary                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETURN RESULTS                               │
│              Complete climb simulation data                     │
└─────────────────────────────────────────────────────────────────┘
```

### 7.6 Function Call Hierarchy

```
main()
├── AeroTables(AERO_XLSX, AERO_SHEET)
│   ├── load_excel_data()
│   ├── apply_parameters_to_globals()
│   └── validate_data()
│
├── EngineWrapper(ENGINE_STUB_PATH)
│   ├── initialize_engine_model()
│   └── setup_caching()
│
├── run_strategy_simulation()
│   ├── StrategyManager.build_strategy_set()
│   └── FOR each strategy:
│       └── StrategyManager.simulate_strategy_path()
│           ├── initialize_trajectory_arrays()
│           ├── WHILE altitude < target:
│           │   ├── calculate_current_conditions()
│           │   ├── apply_strategy_energy_allocation()
│           │   ├── calculate_thrust_and_drag()
│           │   ├── calculate_specific_excess_power()
│           │   ├── update_trajectory()
│           │   └── advance_time_and_altitude()
│           └── return_StrategyRun()
│
├── run_3d_optimization()
│   └── DynamicProgrammingOptimizer.solve_3d_fixed_mass()
│       ├── initialize_3d_cost_grid()
│       ├── set_terminal_conditions()
│       ├── FOR each altitude level:
│       │   ├── FOR each Mach:
│       │   │   ├── FOR each lever:
│       │   │   │   ├── compute_3d_cost()
│       │   │   │   │   ├── calculate_atmospheric_properties()
│       │   │   │   │   ├── calculate_thrust_and_drag()
│       │   │   │   │   ├── calculate_specific_excess_power()
│       │   │   │   │   └── calculate_fuel_cost_density()
│       │   │   │   ├── find_optimal_transition()
│       │   │   │   └── update_cost_grid()
│       │   │   └── store_minimum_cost()
│       │   └── complete_altitude_level()
│       └── backtrack_optimal_path()
│
└── analyze_results()
    ├── resample_trajectories()
    ├── convert_dp_to_strategy_run()
    ├── plot_strategy_comparison()
    └── plot_3d_optimization()
```

---

## 8) Integration and Interface

### 8.1 Mission Analysis Integration

**Function**: `extract_climb_results_for_cruise()`

**Integration Process:**
```python
def extract_climb_results_for_cruise(dp_result: MinFuelSchedule) -> Dict[str, Any]:
    """Extract climb results for cruise phase initialization."""
    
    # Get final state from optimal climb
    final_altitude = float(dp_result.alt_m[-1])
    final_mach = float(dp_result.mach[-1])
    fuel_consumed_climb = float(dp_result.cumFuel_kg[-1])
    climb_time = float(dp_result.total_time_s)
    
    # Calculate current weight after climb
    current_weight = INITIAL_MASS_KG - fuel_consumed_climb
    
    return {
        'final_altitude_m': final_altitude,
        'final_mach': final_mach,
        'fuel_consumed_climb_kg': fuel_consumed_climb,
        'climb_time_s': climb_time,
        'current_weight_kg': current_weight
    }
```

### 8.2 Main Interface Functions

**Strategy Analysis Interface:**
```python
def run_climb_analysis(aero_xlsx: str, engine_stub_path: str,
                      initial_mass_kg: float = INITIAL_MASS_KG,
                      target_alt_m: float = TARGET_ALT_M,
                      create_plots: bool = True) -> Dict[str, Any]:
    """Main interface for complete climb analysis."""
    
    # Initialize systems
    aero = AeroTables(aero_xlsx, AERO_SHEET)
    eng = EngineWrapper(engine_stub_path)
    
    # Run strategy simulations
    strategy_results = run_strategy_simulation(aero, eng, initial_mass_kg, target_alt_m)
    
    # Run 3D optimization
    dp_result = run_3d_optimization(aero, eng, target_alt_m)
    
    # Analyze and visualize results
    if create_plots:
        analyze_results(strategy_results, dp_result)
    
    # Prepare results for mission integration
    cruise_inputs = extract_climb_results_for_cruise(dp_result)
    
    return {
        'strategy_results': strategy_results,
        'optimal_climb': dp_result,
        'cruise_inputs': cruise_inputs
    }
```

**Backward Compatibility Functions:**
```python
# Standalone functions for backward compatibility
def simulate_strategy_path(*, label: str, aero: AeroTables, eng: EngineWrapper,
                          mass0_kg: float, h0_m: float, V0_ms: float,
                          target_alt_m: float, dt: float,
                          strategy_fn: Callable, altitude_fraction: Optional[float]) -> StrategyRun:
    """Backward compatibility wrapper for ClimbingCore.simulate_strategy_path"""
    return ClimbingCore.StrategyManager.simulate_strategy_path(
        label=label, aero=aero, eng=eng, mass0_kg=mass0_kg, h0_m=h0_m, V0_ms=V0_ms,
        target_alt_m=target_alt_m, dt=dt, strategy_fn=strategy_fn, altitude_fraction=altitude_fraction
    )

def resample_strategy_run(sr: StrategyRun, n_samples: int) -> StrategyRun:
    """Backward compatibility wrapper for ClimbingCore.resample_strategy_run"""
    return ClimbingCore.StrategyManager.resample_strategy_run(sr, n_samples)
```

### 8.3 Performance Metrics

**Key Performance Indicators:**
- **Fuel Consumption**: Total fuel burned during climb
- **Climb Time**: Time required to reach target altitude
- **Average Climb Rate**: Mean vertical velocity
- **Energy Efficiency**: Fuel consumed per unit altitude gained
- **Engine Utilization**: Average lever position and thrust usage

---
