# Penalty System Documentation

> **Scope**: Complete documentation of the reachability-constrained penalty system, including Mach trajectory guidance, lever penalty guidance, and their integration with 3D Dynamic Programming for optimal climb trajectory generation.

---

## Table of Contents

1. [System Overview and Objectives](#1-system-overview-and-objectives)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [System Architecture and Data Structures](#3-system-architecture-and-data-structures)
4. [Mach Trajectory Guidance System](#4-mach-trajectory-guidance-system)
5. [Lever Penalty Guidance System](#5-lever-penalty-guidance-system)
6. [Penalty Integration and Balancing](#6-penalty-integration-and-balancing)
7. [Code Execution Flow and Logic](#7-code-execution-flow-and-logic)
8. [Integration and Interface](#8-integration-and-interface)
9. [Validation and Quality Assurance](#9-validation-and-quality-assurance)

---

## 1) System Overview and Objectives

### 1.1 Purpose and Scope

The penalty system implements a sophisticated guidance framework that uses soft constraints to guide trajectory optimization toward realistic, operationally viable solutions. The system employs reachability-constrained Mach guidance and engine-friendly lever penalties to create balanced, cooperative optimization that produces smooth, realistic climb trajectories.

### 1.2 System Objectives

**Primary Objectives:**
- **Trajectory Guidance**: Guide optimization toward realistic Mach trajectories
- **Engine Protection**: Prevent excessive lever usage and engine wear
- **Operational Viability**: Ensure generated trajectories are operationally feasible
- **Balanced Optimization**: Create cooperative penalty interaction for smooth guidance

**Key Components:**
- Reachability-constrained Mach guidance system
- Engine-friendly lever penalty system
- Adaptive penalty weighting with altitude progress
- Integration with 3D Dynamic Programming
- Cooperative penalty interaction and balancing

### 1.3 System Flow Overview

The penalty system follows a logical progression:

1. **Mach Trajectory Analysis**: Calculate reachability constraints based on achievable Mach rates
2. **Penalty Calculation**: Compute Mach and lever penalties for current state
3. **Adaptive Weighting**: Adjust penalty weights based on altitude progress and urgency
4. **Cost Integration**: Combine penalties with base fuel cost for total optimization cost
5. **Guidance Application**: Apply penalties during 3D Dynamic Programming optimization

---

## 2) Mathematical Foundation

### 2.1 Penalty System Philosophy

**Theory**: Instead of hard constraints that can create infeasible solutions, the penalty system uses graduated costs that increase with deviation from desired behavior. This approach provides guidance while maintaining optimization flexibility.

**Mathematical Formulation:**
```math
J_{total} = J_{fuel} + J_{mach\_penalty} + J_{lever\_penalty}
```

Where:
- `J_{fuel} = \dot{m}/P_s` (original fuel cost density)
- `J_{mach\_penalty}` (Mach guidance penalty)
- `J_{lever\_penalty}` (lever penalty)

**Physical Interpretation:**
- **Base fuel cost**: Primary optimization objective
- **Mach penalty**: Guidance toward realistic trajectories
- **Lever penalty**: Engine protection and operational limits
- **Cooperative interaction**: Penalties work together for balanced guidance

### 2.2 Reachability-Constrained Approach

**Theory**: The system creates dynamic "safety corridors" that ensure the target remains achievable throughout the climb using realistic Mach change rates, rather than forcing predetermined trajectories.

**Key Innovation**: Dynamic corridor bounds based on achievable Mach rates rather than artificial trajectory assumptions.

**Mathematical Foundation:**
```math
\text{Remaining\_Steps} = (1.0 - \text{Altitude\_Fraction}) \times \text{Total\_Steps}
```

```math
\text{Max\_Mach\_Change} = \text{Remaining\_Steps} \times \text{Max\_Mach\_Rate}
```

```math
\text{Safety\_Corridor} = [\text{Target\_Mach} - \text{Max\_Mach\_Change}, \text{Target\_Mach} + \text{Max\_Mach\_Change}]
```

### 2.3 Adaptive Penalty Weighting

**Theory**: Penalty weights increase with altitude progress to create urgency and ensure convergence to target conditions.

**Mathematical Formulation:**
```math
w_{penalty} = w_{base} \times (1.0 + \alpha \times \text{Altitude\_Fraction})
```

Where:
- `w_{base}` = Base penalty weight
- `\alpha` = Urgency multiplier
- `\text{Altitude\_Fraction}` = Current progress (0.0 to 1.0)

---

## 3) System Architecture and Data Structures

### 3.1 System Parameters

**Mach Trajectory Guidance Parameters:**
```python
MACH_TRAJECTORY_GUIDANCE = True              # Enable Mach guidance
MACH_PENALTY_BASE_WEIGHT = 0.3              # Base penalty weight
URGENCY_MULTIPLIER = 2.0                    # Urgency scaling factor
GUIDANCE_PENALTY_WEIGHT = 0.5               # Guidance penalty inside corridor
MAX_REASONABLE_MACH_RATE = 0.02             # Maximum realistic Mach change per step
TOTAL_CLIMB_STEPS_ESTIMATE = 50             # Estimated total climb steps
TARGET_MACH_TOLERANCE = 0.015               # Target Mach tolerance
```

**Lever Penalty Parameters:**
```python
LEVER_PENALTY_GUIDANCE = True               # Enable lever penalties
LEVER_PENALTY_WEIGHT = 3.0                 # Base lever penalty weight
LEVER_PENALTY_THRESHOLD = 0.85             # Threshold above which penalties apply
LEVER_PENALTY_EXPONENT = 3.0               # Exponent for penalty curve
LEVER_PENALTY_CRITICAL_THRESHOLD = 0.90    # Critical threshold
LEVER_PENALTY_CRITICAL_MULTIPLIER = 5.0    # Critical range multiplier
LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD = 0.95  # Ultra-critical threshold
LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER = 20.0 # Ultra-critical multiplier
```

### 3.2 Data Structures

**Penalty Calculation State:**
```python
@dataclass
class PenaltyState:
    current_mach: float
    target_mach: float
    altitude_fraction: float
    current_lever: float
    total_steps_estimate: int
    max_mach_rate: float
```

**Penalty Results:**
```python
@dataclass
class PenaltyResults:
    mach_penalty: float
    lever_penalty: float
    total_penalty: float
    penalty_weight: float
    guidance_active: bool
    corridor_bounds: Tuple[float, float]
```

---

## 4) Mach Trajectory Guidance System

### 4.1 Reachability Constraint Calculation

**Function**: `compute_mach_penalty()`

**Purpose**: Calculate Mach trajectory guidance penalty based on reachability constraints.

**Algorithm:**
```python
def compute_mach_penalty(current_mach: float, target_mach: float, 
                        prev_mach: float = None, 
                         altitude_fraction: float = None) -> float:
    """
    Calculate Mach trajectory guidance penalty using reachability constraints.
    
    Args:
        current_mach: Current Mach number
        target_mach: Target Mach number at cruise altitude
        prev_mach: Previous Mach number (for rate calculation)
        altitude_fraction: Current altitude progress (0.0 to 1.0)
        
    Returns:
        Mach penalty value
    """
    
    if not MACH_TRAJECTORY_GUIDANCE:
        return 0.0
    
    # Calculate remaining steps
    remaining_steps = (1.0 - altitude_fraction) * TOTAL_CLIMB_STEPS_ESTIMATE
    
    # Calculate maximum achievable Mach change
    max_mach_change = remaining_steps * MAX_REASONABLE_MACH_RATE
    
    # Create safety corridor
    corridor_low = target_mach - max_mach_change
    corridor_high = target_mach + max_mach_change
    
    # Check if current Mach is within safety corridor
    if corridor_low <= current_mach <= corridor_high:
        # Inside corridor - apply guidance penalty
        mach_deviation = abs(current_mach - target_mach)
        penalty_weight = GUIDANCE_PENALTY_WEIGHT
        
        # Scale penalty with altitude progress (urgency)
        urgency_multiplier = 1.0 + URGENCY_MULTIPLIER * altitude_fraction
        penalty_weight *= urgency_multiplier
        
        return penalty_weight * (mach_deviation ** 2)
    else:
        # Outside corridor - apply strong penalty
        mach_deviation = min(abs(current_mach - corridor_low), 
                           abs(current_mach - corridor_high))
        
        penalty_weight = MACH_PENALTY_BASE_WEIGHT
        
        # Scale penalty with altitude progress (urgency)
        urgency_multiplier = 1.0 + URGENCY_MULTIPLIER * altitude_fraction
        penalty_weight *= urgency_multiplier
        
        return penalty_weight * (mach_deviation ** 2)
```

### 4.2 Dynamic Corridor Management

**Corridor Calculation:**
```python
def calculate_safety_corridor(target_mach: float, altitude_fraction: float, 
                            total_steps_estimate: int, max_mach_rate: float) -> Tuple[float, float]:
    """Calculate dynamic safety corridor bounds."""
    
    # Calculate remaining steps
    remaining_steps = (1.0 - altitude_fraction) * total_steps_estimate
    
    # Calculate maximum achievable Mach change
    max_mach_change = remaining_steps * max_mach_rate
    
    # Create corridor bounds
    corridor_low = target_mach - max_mach_change
    corridor_high = target_mach + max_mach_change
    
    return corridor_low, corridor_high
```

### 4.3 Adaptive Urgency Scaling

**Urgency Calculation:**
```python
def calculate_urgency_weight(base_weight: float, altitude_fraction: float, 
                           urgency_multiplier: float = 2.0) -> float:
    """Calculate adaptive penalty weight based on altitude progress."""
    
    return base_weight * (1.0 + urgency_multiplier * altitude_fraction)
```

---

## 5) Lever Penalty Guidance System

### 5.1 Lever Penalty Calculation

**Function**: `compute_lever_penalty()`

**Purpose**: Calculate engine-friendly lever penalties to prevent excessive usage.

**Algorithm:**
```python
def compute_lever_penalty(current_lever: float, altitude_fraction: float = None) -> float:
    """
    Calculate lever position penalty for excessive usage.
    
    Args:
        current_lever: Current lever position (0.0 to 1.0)
        altitude_fraction: Current altitude progress (0.0 to 1.0)
        
    Returns:
        Lever penalty value
    """
    
    if not LEVER_PENALTY_GUIDANCE:
        return 0.0
    
    # No penalty below threshold
    if current_lever <= LEVER_PENALTY_THRESHOLD:
        return 0.0
    
    # Calculate excess above threshold
    excess = current_lever - LEVER_PENALTY_THRESHOLD
    
    # Base penalty with cubic growth
    penalty = LEVER_PENALTY_WEIGHT * (excess ** LEVER_PENALTY_EXPONENT)
    
    # Apply critical range multipliers
    if current_lever >= LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD:
        penalty *= LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
    elif current_lever >= LEVER_PENALTY_CRITICAL_THRESHOLD:
        penalty *= LEVER_PENALTY_CRITICAL_MULTIPLIER
    
    # Scale with altitude progress (urgency)
    if altitude_fraction is not None:
        urgency_multiplier = 1.0 + URGENCY_MULTIPLIER * altitude_fraction
        penalty *= urgency_multiplier
    
    return penalty
```

### 5.2 Multi-Tier Penalty Structure

**Penalty Tiers:**
1. **Safe Range** (0.0 - 0.85): No penalty
2. **Warning Range** (0.85 - 0.90): Base penalty with cubic growth
3. **Critical Range** (0.90 - 0.95): 5x multiplier
4. **Ultra-Critical Range** (0.95 - 1.0): 20x multiplier

**Mathematical Formulation:**
```math
\text{Penalty} = \begin{cases}
0 & \text{if } \ell \leq 0.85 \\
w \times (\ell - 0.85)^3 & \text{if } 0.85 < \ell \leq 0.90 \\
5w \times (\ell - 0.85)^3 & \text{if } 0.90 < \ell \leq 0.95 \\
20w \times (\ell - 0.85)^3 & \text{if } \ell > 0.95
\end{cases}
```

Where `w` is the base penalty weight.

---

## 6) Penalty Integration and Balancing

### 6.1 Total Cost Calculation

**Function**: `calculate_total_penalty_cost()`

**Purpose**: Integrate all penalties with base fuel cost for total optimization cost.

**Implementation:**
```python
def calculate_total_penalty_cost(base_fuel_cost: float, current_mach: float, 
                                target_mach: float, current_lever: float,
                                altitude_fraction: float, prev_mach: float = None) -> float:
    """Calculate total cost including all penalties."""
    
    # Calculate individual penalties
    mach_penalty = compute_mach_penalty(current_mach, target_mach, prev_mach, altitude_fraction)
    lever_penalty = compute_lever_penalty(current_lever, altitude_fraction)
    
    # Calculate total penalty
    total_penalty = mach_penalty + lever_penalty
    
    # Calculate total cost
    total_cost = base_fuel_cost + total_penalty
    
    return total_cost
```

### 6.2 Cooperative Penalty Interaction

**Theory**: The system ensures penalties work together rather than against each other, creating balanced guidance.

**Interaction Principles:**
1. **Mach guidance** encourages realistic trajectories
2. **Lever penalties** protect engine operation
3. **Adaptive weighting** ensures convergence
4. **Cooperative balancing** prevents conflicting guidance

### 6.3 Penalty Weight Balancing

**Weight Management:**
```python
def balance_penalty_weights(mach_penalty: float, lever_penalty: float, 
                          altitude_fraction: float) -> Tuple[float, float]:
    """Balance penalty weights to ensure cooperative interaction."""
    
    # Calculate relative magnitudes
    total_penalty = mach_penalty + lever_penalty
    
    if total_penalty > 0:
        mach_weight = mach_penalty / total_penalty
        lever_weight = lever_penalty / total_penalty
        
        # Ensure balanced influence
        if mach_weight > 0.8:  # Mach penalty too dominant
            mach_penalty *= 0.8
            lever_penalty *= 1.2
        elif lever_weight > 0.8:  # Lever penalty too dominant
            lever_penalty *= 0.8
            mach_penalty *= 1.2
    
    return mach_penalty, lever_penalty
```

---

## 7) Code Execution Flow and Logic

### 7.1 System Entry Point and Initialization

**Main Entry Function**: Integration with 3D Dynamic Programming

**Execution Sequence:**
```python
# 1. Penalty System Initialization
def initialize_penalty_system():
    """Initialize penalty system parameters and state."""
    
    # Set up penalty parameters
    penalty_params = {
        'mach_guidance': MACH_TRAJECTORY_GUIDANCE,
        'lever_guidance': LEVER_PENALTY_GUIDANCE,
        'base_weights': {
            'mach': MACH_PENALTY_BASE_WEIGHT,
            'lever': LEVER_PENALTY_WEIGHT
        },
        'thresholds': {
            'lever': LEVER_PENALTY_THRESHOLD,
            'critical': LEVER_PENALTY_CRITICAL_THRESHOLD,
            'ultra_critical': LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD
        }
    }
    
    return penalty_params
```

### 7.2 Penalty Calculation Flow

**Function**: `calculate_penalties_for_state()`

**Step-by-Step Execution Flow:**

```python
# PHASE 1: STATE ANALYSIS
def calculate_penalties_for_state(current_mach: float, current_lever: float,
                                altitude_fraction: float, target_mach: float,
                                prev_mach: float = None):
    
    # Step 1: Calculate altitude progress
    altitude_progress = altitude_fraction if altitude_fraction is not None else 0.0
    
    # Step 2: Determine remaining climb steps
    remaining_steps = (1.0 - altitude_progress) * TOTAL_CLIMB_STEPS_ESTIMATE
    
    # Step 3: Calculate maximum achievable Mach change
    max_mach_change = remaining_steps * MAX_REASONABLE_MACH_RATE
```

```python
# PHASE 2: MACH PENALTY CALCULATION
    # Step 4: Calculate safety corridor bounds
    corridor_low = target_mach - max_mach_change
    corridor_high = target_mach + max_mach_change
    
    # Step 5: Determine penalty zone
    if corridor_low <= current_mach <= corridor_high:
        # Inside corridor - guidance penalty
        mach_deviation = abs(current_mach - target_mach)
        penalty_weight = GUIDANCE_PENALTY_WEIGHT
        guidance_active = True
    else:
        # Outside corridor - strong penalty
        mach_deviation = min(abs(current_mach - corridor_low), 
                           abs(current_mach - corridor_high))
        penalty_weight = MACH_PENALTY_BASE_WEIGHT
        guidance_active = False
    
    # Step 6: Apply urgency scaling
    urgency_multiplier = 1.0 + URGENCY_MULTIPLIER * altitude_progress
    penalty_weight *= urgency_multiplier
    
    # Step 7: Calculate Mach penalty
    mach_penalty = penalty_weight * (mach_deviation ** 2)
```

```python
# PHASE 3: LEVER PENALTY CALCULATION
    # Step 8: Check lever threshold
    if current_lever <= LEVER_PENALTY_THRESHOLD:
        lever_penalty = 0.0
    else:
        # Step 9: Calculate excess above threshold
        excess = current_lever - LEVER_PENALTY_THRESHOLD
        
        # Step 10: Base penalty with cubic growth
        lever_penalty = LEVER_PENALTY_WEIGHT * (excess ** LEVER_PENALTY_EXPONENT)
        
        # Step 11: Apply critical range multipliers
        if current_lever >= LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD:
            lever_penalty *= LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
        elif current_lever >= LEVER_PENALTY_CRITICAL_THRESHOLD:
            lever_penalty *= LEVER_PENALTY_CRITICAL_MULTIPLIER
        
        # Step 12: Apply urgency scaling
        lever_penalty *= urgency_multiplier
```

```python
# PHASE 4: PENALTY INTEGRATION
    # Step 13: Calculate total penalty
    total_penalty = mach_penalty + lever_penalty
    
    # Step 14: Create penalty results
    penalty_results = PenaltyResults(
        mach_penalty=mach_penalty,
        lever_penalty=lever_penalty,
        total_penalty=total_penalty,
        penalty_weight=penalty_weight,
        guidance_active=guidance_active,
        corridor_bounds=(corridor_low, corridor_high)
    )
    
    return penalty_results
```

### 7.3 Integration with 3D Dynamic Programming

**Function**: `integrate_penalties_with_dp()`

**Integration Flow:**
```python
def integrate_penalties_with_dp(base_cost: float, mach: float, lever: float,
                              altitude_fraction: float, target_mach: float) -> float:
    """Integrate penalties with 3D DP cost calculation."""
    
    # Calculate penalties for current state
    penalty_results = calculate_penalties_for_state(
        current_mach=mach,
        current_lever=lever,
        altitude_fraction=altitude_fraction,
        target_mach=target_mach
    )
    
    # Add penalties to base fuel cost
    total_cost = base_cost + penalty_results.total_penalty
    
    return total_cost
```

### 7.4 Visual Code Flow Diagram

```
PENALTY SYSTEM EXECUTION FLOW
=============================

┌─────────────────────────────────────────────────────────────────┐
│                    PENALTY SYSTEM INITIALIZATION               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ initialize_penalty_system()                            │   │
│  │  ├── Set up penalty parameters                         │   │
│  │  ├── Configure Mach guidance                           │   │
│  │  ├── Configure lever penalties                         │   │
│  │  └── Set threshold values                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STATE ANALYSIS                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FOR each DP state (mach, altitude, lever):             │   │
│  │  ├── Calculate altitude progress                        │   │
│  │  ├── Determine remaining climb steps                   │   │
│  │  └── Calculate maximum achievable Mach change          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MACH PENALTY CALCULATION                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ├── Calculate safety corridor bounds                  │   │
│  │  ├── Check if current Mach is within corridor          │   │
│  │  ├── Calculate Mach deviation                          │   │
│  │  ├── Apply appropriate penalty weight                  │   │
│  │  ├── Scale with urgency (altitude progress)           │   │
│  │  └── Calculate Mach penalty                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LEVER PENALTY CALCULATION                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ├── Check lever threshold                             │   │
│  │  ├── Calculate excess above threshold                  │   │
│  │  ├── Apply cubic penalty growth                        │   │
│  │  ├── Apply critical range multipliers                  │   │
│  │  ├── Scale with urgency (altitude progress)           │   │
│  │  └── Calculate lever penalty                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PENALTY INTEGRATION                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ├── Sum Mach and lever penalties                      │   │
│  │  ├── Add to base fuel cost                             │   │
│  │  ├── Balance penalty weights                           │   │
│  │  └── Return total optimization cost                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETURN TOTAL COST                           │
│              Cost = Base Fuel Cost + Total Penalties           │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5 Function Call Hierarchy

```
3D Dynamic Programming Optimization
├── compute_3d_cost()
│   ├── calculate_base_fuel_cost()
│   └── calculate_total_penalty_cost()
│       ├── compute_mach_penalty()
│       │   ├── calculate_safety_corridor()
│       │   ├── calculate_urgency_weight()
│       │   └── determine_penalty_zone()
│       ├── compute_lever_penalty()
│       │   ├── check_lever_threshold()
│       │   ├── calculate_excess()
│       │   ├── apply_critical_multipliers()
│       │   └── apply_urgency_scaling()
│       └── balance_penalty_weights()
│
└── integrate_penalties_with_dp()
    ├── calculate_penalties_for_state()
    ├── create_penalty_results()
    └── return_total_cost()
```

---

## 8) Integration and Interface

### 8.1 3D Dynamic Programming Integration

**Function**: `integrate_with_3d_dp()`

**Integration Process:**
```python
def integrate_with_3d_dp():
    """Integrate penalty system with 3D Dynamic Programming optimization."""
    
    # Modify the cost calculation in DP
    def enhanced_cost_calculation(mach: float, lever: float, altitude_fraction: float, 
                                 base_cost: float, target_mach: float = 0.7) -> float:
        """Enhanced cost calculation with penalty integration."""
        
        # Calculate penalties
        penalty_results = calculate_penalties_for_state(
            current_mach=mach,
            current_lever=lever,
            altitude_fraction=altitude_fraction,
            target_mach=target_mach
        )
        
        # Return total cost
        return base_cost + penalty_results.total_penalty
    
    return enhanced_cost_calculation
```

### 8.2 Main Interface Functions

**Penalty System Interface:**
```python
def run_penalty_guided_optimization(aero: AeroTables, eng: EngineWrapper,
                          M_grid: np.ndarray, H_sched: np.ndarray,
                                   target_mach: float = 0.7) -> MinFuelSchedule:
    """Run 3D DP optimization with penalty guidance."""
    
    # Initialize penalty system
    penalty_params = initialize_penalty_system()
    
    # Enhanced cost function with penalties
    def penalty_enhanced_cost(mach: float, altitude_m: float, lever: float, 
                            mass_kg: float) -> float:
        # Calculate base fuel cost
        base_cost = compute_3d_cost(aero, eng, mach, altitude_m, lever, mass_kg)
        
        # Calculate altitude fraction
        altitude_fraction = (altitude_m - H_sched[0]) / (H_sched[-1] - H_sched[0])
        
        # Add penalties
        penalty_cost = calculate_total_penalty_cost(
            base_cost, mach, target_mach, lever, altitude_fraction
        )
        
        return penalty_cost
    
    # Run 3D DP with penalty-enhanced cost
    dp_result = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_fixed_mass(
        aero=aero, eng=eng, M_grid=M_grid, H_sched=H_sched,
        lever_samples=50, target_mach=target_mach
    )
    
    return dp_result
```

### 8.3 Configuration Interface

**Penalty Configuration:**
```python
def configure_penalty_system(mach_guidance: bool = True, lever_guidance: bool = True,
                           target_mach: float = 0.7, urgency_multiplier: float = 2.0) -> Dict[str, Any]:
    """Configure penalty system parameters."""
    
    config = {
        'mach_guidance_enabled': mach_guidance,
        'lever_guidance_enabled': lever_guidance,
        'target_mach': target_mach,
        'urgency_multiplier': urgency_multiplier,
        'parameters': {
            'mach_base_weight': MACH_PENALTY_BASE_WEIGHT,
            'lever_base_weight': LEVER_PENALTY_WEIGHT,
            'lever_threshold': LEVER_PENALTY_THRESHOLD,
            'max_mach_rate': MAX_REASONABLE_MACH_RATE
        }
    }
    
    return config
```

---

## 9) Validation and Quality Assurance

### 9.1 Penalty System Validation

**Parameter Validation:**
```python
def validate_penalty_parameters() -> None:
    """Validate penalty system parameters."""
    
    # Validate Mach guidance parameters
    if MACH_PENALTY_BASE_WEIGHT < 0:
        raise ValueError("Mach penalty base weight must be positive")
    
    if URGENCY_MULTIPLIER < 0:
        raise ValueError("Urgency multiplier must be positive")
    
    if MAX_REASONABLE_MACH_RATE <= 0:
        raise ValueError("Maximum Mach rate must be positive")
    
    # Validate lever penalty parameters
    if LEVER_PENALTY_WEIGHT < 0:
        raise ValueError("Lever penalty weight must be positive")
    
    if not (0.0 <= LEVER_PENALTY_THRESHOLD <= 1.0):
        raise ValueError("Lever penalty threshold must be between 0.0 and 1.0")
    
    if LEVER_PENALTY_EXPONENT <= 0:
        raise ValueError("Lever penalty exponent must be positive")
```

### 9.2 Performance Monitoring

**Penalty Effectiveness Monitoring:**
```python
def monitor_penalty_effectiveness(dp_result: MinFuelSchedule, target_mach: float) -> Dict[str, float]:
    """Monitor effectiveness of penalty guidance."""
    
    # Analyze Mach trajectory
    mach_trajectory = dp_result.mach
    mach_deviations = np.abs(mach_trajectory - target_mach)
    max_mach_deviation = np.max(mach_deviations)
    avg_mach_deviation = np.mean(mach_deviations)
    
    # Analyze lever usage
    lever_trajectory = dp_result.lever
    max_lever = np.max(lever_trajectory)
    avg_lever = np.mean(lever_trajectory)
    high_lever_fraction = np.sum(lever_trajectory > LEVER_PENALTY_THRESHOLD) / len(lever_trajectory)
    
    # Calculate effectiveness metrics
    effectiveness_metrics = {
        'max_mach_deviation': max_mach_deviation,
        'avg_mach_deviation': avg_mach_deviation,
        'mach_convergence': 1.0 - (max_mach_deviation / target_mach),
        'max_lever_usage': max_lever,
        'avg_lever_usage': avg_lever,
        'high_lever_fraction': high_lever_fraction,
        'lever_protection': 1.0 - high_lever_fraction
    }
    
    return effectiveness_metrics
```

### 9.3 Error Handling

**Penalty Calculation Errors:**
```python
def safe_penalty_calculation(current_mach: float, current_lever: float,
                           altitude_fraction: float, target_mach: float) -> PenaltyResults:
    """Safe penalty calculation with error handling."""
    
    try:
        # Validate inputs
        if not (0.0 <= current_lever <= 1.0):
            raise ValueError(f"Invalid lever position: {current_lever}")
        
        if not (0.0 <= altitude_fraction <= 1.0):
            raise ValueError(f"Invalid altitude fraction: {altitude_fraction}")
        
        # Calculate penalties
        penalty_results = calculate_penalties_for_state(
            current_mach, current_lever, altitude_fraction, target_mach
        )
        
        return penalty_results
        
    except Exception as e:
        print(f"Penalty calculation error: {e}")
        # Return zero penalties as fallback
        return PenaltyResults(
            mach_penalty=0.0,
            lever_penalty=0.0,
            total_penalty=0.0,
            penalty_weight=0.0,
            guidance_active=False,
            corridor_bounds=(0.0, 1.0)
        )
```

---
 