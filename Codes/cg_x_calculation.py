# ========================================================================
# CENTER OF GRAVITY CALCULATION MODULE
# ========================================================================
"""
Total aircraft longitudinal CG position computation during fuel consumption.

Mathematical formulation:
    x_CG,total = (m_ZFW · x_CG,ZFW + Σ(m_i · x_i)) / (m_ZFW + Σm_i)
    where:
        m_ZFW = W_OE + W_PL [kg] - Zero Fuel Weight (empty aircraft + payload)
        x_CG,ZFW [m] - CG position at zero fuel weight
        m_i = fuel mass in tank i [kg]
        x_i = tank CG position [m]

Fuel distribution models:
    - OUTER_FIRST: Outer → Inner → Center depletion sequence
    - CENTER_FIRST: Center → Inner → Outer depletion sequence
    - PROPORTIONAL: Uniform depletion across all tanks

CG shift affects trim analysis and aerodynamic performance.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Literal, List

# Aircraft mass components and fuel tank configuration
from aircraft_config import (
    INITIAL_MASS_KG, W_FUEL_KG, W_OE_KG, W_PL_KG,
    KEROSENE_DENSITY_KGPM3, TANK_CG_POSITIONS, TANK_NAMES,
    ZERO_FUEL_CG_X,
    FUEL_LEVEL_PRINT_ENABLED, FUEL_LEVEL_PRINT_SAMPLE_RATE
)

# Default CG and consumption scenario
from mission_config import CG_X_DEFAULT, CG_CONSUMPTION_SCENARIO

# ========================================================================
# SECTION 1: TYPE DEFINITIONS AND CONSTANTS
# ========================================================================

# Consumption scenario enumeration
ConsumptionScenario = Literal["OUTER_FIRST", "CENTER_FIRST", "PROPORTIONAL"]

# Tank capacity parameters
TOTAL_CAPACITY_KG = W_FUEL_KG           # m_fuel,total [kg]: total fuel capacity
TANK_CAPACITY_KG = TOTAL_CAPACITY_KG / 5.0  # m_tank [kg]: capacity per tank (uniform)

# ========================================================================
# SECTION 2: FUEL DISTRIBUTION COMPUTATION
# ========================================================================

class FuelDistributionCalculator:
    """
    Deterministic fuel distribution across tanks via consumption scenarios.
    
    Mathematical basis: Given m_fuel,remaining, compute m_i for tanks i=0..4
    subject to Σm_i = m_fuel,remaining and scenario-specific sequencing rules.
    
    Computation: Pure function mapping m_fuel → {m_0, m_1, m_2, m_3, m_4}.
    Includes caching for performance optimization.
    """
    
    def __init__(self, scenario: ConsumptionScenario = "OUTER_FIRST"):
        """
        Initialize distribution calculator with consumption scenario.
        
        Scenarios:
            OUTER_FIRST: Sequence [Outer(1,3)] → [Inner(0,2)] → [Center(4)]
            CENTER_FIRST: Sequence [Center(4)] → [Inner(0,2)] → [Outer(1,3)]
            PROPORTIONAL: m_i = m_fuel,remaining / 5 ∀i
        
        Parameters:
            scenario: ConsumptionScenario - depletion order
        """
        self.scenario = scenario
        self._distribution_cache: Dict[float, Dict[int, float]] = {}
    
    def calculate_distribution(self, fuel_remaining: float) -> Dict[int, float]:
        """
        Compute fuel mass distribution {m_i} for given total fuel.
        
        Algorithm: Apply scenario-specific depletion logic to determine
        which tanks contain fuel and their respective masses.
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass per tank
        """
        # Clamp to physical bounds: m_fuel ∈ [0, m_fuel,max]
        fuel_remaining = max(0.0, min(fuel_remaining, W_FUEL_KG))
        
        # Cache lookup for performance
        cache_key = round(fuel_remaining, 6)
        if cache_key in self._distribution_cache:
            return self._distribution_cache[cache_key].copy()
        
        # Route to scenario-specific distribution function
        if self.scenario == "OUTER_FIRST":
            distribution = self._calculate_outer_inner_center(fuel_remaining)
        elif self.scenario == "CENTER_FIRST":
            distribution = self._calculate_inner_outer_center(fuel_remaining)
        elif self.scenario == "PROPORTIONAL":
            distribution = self._calculate_even(fuel_remaining)
        else:
            distribution = self._calculate_even(fuel_remaining)  # Default
        
        # Store in cache
        self._distribution_cache[cache_key] = distribution.copy()
        
        return distribution
    
    def _calculate_outer_inner_center(self, fuel_remaining: float) -> Dict[int, float]:
        """
        OUTER_FIRST depletion: Sequential emptying [Outer] → [Inner] → [Center].
        
        Depletion sequence:
            Phase 1 (100%→60%): m_consumed < m_outer,total
                → Deplete tanks {1,3} uniformly, {0,2,4} remain full
            Phase 2 (60%→20%): m_outer,total ≤ m_consumed < m_outer,total + m_inner,total
                → Tanks {1,3} empty, deplete {0,2} uniformly, {4} remains full
            Phase 3 (20%→0%): m_consumed ≥ m_outer,total + m_inner,total
                → Tanks {0,1,2,3} empty, deplete {4}
        
        Tank grouping:
            Outer: {1,3}, Inner: {0,2}, Center: {4}
            Capacity: m_tank = m_fuel,total / 5
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass distribution
        """
        tanks = {i: 0.0 for i in range(5)}
        
        # Fuel consumed: Δm_fuel = m_fuel,total - m_fuel,remaining
        fuel_consumed = W_FUEL_KG - fuel_remaining
        
        # Tank group capacities
        capacity_per_tank = TANK_CAPACITY_KG       # m_tank [kg]
        outer_capacity = capacity_per_tank * 2     # m_outer = 2·m_tank (tanks 1,3)
        inner_capacity = capacity_per_tank * 2     # m_inner = 2·m_tank (tanks 0,2)
        center_capacity = capacity_per_tank        # m_center = m_tank (tank 4)
        
        # Phase determination via consumed fuel thresholds
        if fuel_consumed < outer_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 1: Outer depletion (100% → 60%)
            # ────────────────────────────────────────────────────────────────
            outer_consumed = fuel_consumed
            outer_remaining = outer_capacity - outer_consumed
            
            # Distribution: Outer depleting, Inner and Center full
            tanks[1] = outer_remaining / 2.0  # m_1 = m_outer,rem / 2
            tanks[3] = outer_remaining / 2.0  # m_3 = m_outer,rem / 2
            tanks[0] = capacity_per_tank      # m_0 = m_tank (full)
            tanks[2] = capacity_per_tank      # m_2 = m_tank (full)
            tanks[4] = capacity_per_tank      # m_4 = m_tank (full)
        
        elif fuel_consumed < outer_capacity + inner_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 2: Inner depletion (60% → 20%)
            # ────────────────────────────────────────────────────────────────
            inner_consumed = fuel_consumed - outer_capacity
            inner_remaining = inner_capacity - inner_consumed
            
            # Distribution: Outer empty, Inner depleting, Center full
            tanks[1] = 0.0  # m_1 = 0 (empty)
            tanks[3] = 0.0  # m_3 = 0 (empty)
            tanks[0] = inner_remaining / 2.0  # m_0 = m_inner,rem / 2
            tanks[2] = inner_remaining / 2.0  # m_2 = m_inner,rem / 2
            tanks[4] = capacity_per_tank      # m_4 = m_tank (full)
        
        else:
            # ────────────────────────────────────────────────────────────────
            # Phase 3: Center depletion (20% → 0%)
            # ────────────────────────────────────────────────────────────────
            center_consumed = fuel_consumed - outer_capacity - inner_capacity
            center_remaining = center_capacity - center_consumed
            
            # Distribution: Outer and Inner empty, Center depleting
            tanks[1] = 0.0  # m_1 = 0 (empty)
            tanks[3] = 0.0  # m_3 = 0 (empty)
            tanks[0] = 0.0  # m_0 = 0 (empty)
            tanks[2] = 0.0  # m_2 = 0 (empty)
            tanks[4] = max(0.0, center_remaining)  # m_4 = m_center,rem
        
        return tanks
    
    def _calculate_inner_outer_center(self, fuel_remaining: float) -> Dict[int, float]:
        """
        CENTER_FIRST depletion: Sequential emptying [Center] → [Inner] → [Outer].
        
        Depletion sequence:
            Phase 1 (100%→80%): m_consumed < m_center,total
                → Deplete tank {4}, {0,1,2,3} remain full
            Phase 2 (80%→40%): m_center,total ≤ m_consumed < m_center,total + m_inner,total
                → Tank {4} empty, deplete {0,2} uniformly, {1,3} remain full
            Phase 3 (40%→0%): m_consumed ≥ m_center,total + m_inner,total
                → Tanks {0,2,4} empty, deplete {1,3} uniformly
        
        Tank grouping:
            Center: {4}, Inner: {0,2}, Outer: {1,3}
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass distribution
        """
        tanks = {i: 0.0 for i in range(5)}
        
        # Fuel consumed: Δm_fuel = m_fuel,total - m_fuel,remaining
        fuel_consumed = W_FUEL_KG - fuel_remaining
        
        # Tank group capacities
        capacity_per_tank = TANK_CAPACITY_KG       # m_tank [kg]
        center_capacity = capacity_per_tank        # m_center = m_tank (tank 4)
        inner_capacity = capacity_per_tank * 2     # m_inner = 2·m_tank (tanks 0,2)
        outer_capacity = capacity_per_tank * 2     # m_outer = 2·m_tank (tanks 1,3)
        
        # Phase determination via consumed fuel thresholds
        if fuel_consumed < center_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 1: Center depletion (100% → 80%)
            # ────────────────────────────────────────────────────────────────
            center_consumed = fuel_consumed
            center_remaining = center_capacity - center_consumed
            
            # Distribution: Center depleting, Inner and Outer full
            tanks[4] = center_remaining       # m_4 = m_center,rem
            tanks[0] = capacity_per_tank      # m_0 = m_tank (full)
            tanks[2] = capacity_per_tank      # m_2 = m_tank (full)
            tanks[1] = capacity_per_tank      # m_1 = m_tank (full)
            tanks[3] = capacity_per_tank      # m_3 = m_tank (full)
        
        elif fuel_consumed < center_capacity + inner_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 2: Inner depletion (80% → 40%)
            # ────────────────────────────────────────────────────────────────
            inner_consumed = fuel_consumed - center_capacity
            inner_remaining = inner_capacity - inner_consumed
            
            # Distribution: Center empty, Inner depleting, Outer full
            tanks[4] = 0.0  # m_4 = 0 (empty)
            tanks[0] = inner_remaining / 2.0  # m_0 = m_inner,rem / 2
            tanks[2] = inner_remaining / 2.0  # m_2 = m_inner,rem / 2
            tanks[1] = capacity_per_tank      # m_1 = m_tank (full)
            tanks[3] = capacity_per_tank      # m_3 = m_tank (full)
        
        else:
            # ────────────────────────────────────────────────────────────────
            # Phase 3: Outer depletion (40% → 0%)
            # ────────────────────────────────────────────────────────────────
            outer_consumed = fuel_consumed - center_capacity - inner_capacity
            outer_remaining = outer_capacity - outer_consumed
            
            # Distribution: Center and Inner empty, Outer depleting
            tanks[4] = 0.0  # m_4 = 0 (empty)
            tanks[0] = 0.0  # m_0 = 0 (empty)
            tanks[2] = 0.0  # m_2 = 0 (empty)
            tanks[1] = max(0.0, outer_remaining / 2.0)  # m_1 = m_outer,rem / 2
            tanks[3] = max(0.0, outer_remaining / 2.0)  # m_3 = m_outer,rem / 2
        
        return tanks
    
    def _calculate_even(self, fuel_remaining: float) -> Dict[int, float]:
        """
        PROPORTIONAL depletion: Uniform distribution across all tanks.
        
        Depletion: All tanks deplete at equal rate.
        Distribution: m_i = m_fuel,remaining / N_tanks ∀i
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass distribution
        """
        tanks = {i: 0.0 for i in range(5)}
        
        # Uniform distribution: m_i = m_fuel / 5
        fuel_per_tank = fuel_remaining / 5.0
        
        for tank_id in range(5):
            tanks[tank_id] = fuel_per_tank
        
        return tanks
    
    def calculate_cg_x(self, fuel_remaining: float) -> float:
        """
        Compute total aircraft longitudinal CG position including empty aircraft and fuel.
        
        Formula: x_CG,total = (m_ZFW · x_CG,ZFW + Σ(m_i · x_i)) / (m_ZFW + Σm_i)
        where:
            m_ZFW = W_OE + W_PL [kg] - Zero Fuel Weight (empty aircraft + payload)
            x_CG,ZFW = ZERO_FUEL_CG_X [m] - CG position at zero fuel weight
            m_i = fuel mass in tank i [kg]
            x_i = tank CG position [m]
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            x_CG [m]: total aircraft longitudinal CG position
        """
        # Zero Fuel Weight (OEW + Payload, no fuel)
        m_zfw = W_OE_KG + W_PL_KG
        x_cg_zfw = ZERO_FUEL_CG_X
        
        # Obtain fuel distribution {m_i}
        tanks = self.calculate_distribution(fuel_remaining)
        
        # Fuel contribution: Σ(m_i·x_i)
        total_fuel_mass = 0.0
        fuel_cg_weighted_sum = 0.0
        
        for tank_id, fuel_mass_kg in tanks.items():
            if fuel_mass_kg > 0:
                cg_position = TANK_CG_POSITIONS[tank_id]  # x_i [m]
                contribution = fuel_mass_kg * cg_position  # m_i·x_i
                fuel_cg_weighted_sum += contribution
                total_fuel_mass += fuel_mass_kg
        
        # Total aircraft CG: x_CG = (m_ZFW·x_CG,ZFW + Σ(m_i·x_i)) / (m_ZFW + Σm_i)
        total_mass = m_zfw + total_fuel_mass
        total_cg_weighted_sum = (m_zfw * x_cg_zfw) + fuel_cg_weighted_sum
        
        if total_mass > 0:
            cg_x = total_cg_weighted_sum / total_mass
        else:
            # Fallback: use default CG (should never happen)
            cg_x = CG_X_DEFAULT
        
        return cg_x
    
    def get_fuel_consumed(self, fuel_remaining: float) -> float:
        """
        Compute fuel consumed: Δm_fuel = m_fuel,total - m_fuel,remaining.
        
        Parameters:
            fuel_remaining: m_fuel,remaining [kg]
        
        Returns:
            Δm_fuel [kg]: fuel consumed
        """
        return W_FUEL_KG - fuel_remaining
    
    def clear_cache(self):
        """Clear distribution cache for memory management."""
        self._distribution_cache.clear()

# ========================================================================
# SECTION 3: HISTORY TRACKING
# ========================================================================

class FuelHistoryTracker:
    """
    Time-series storage for fuel state evolution during mission.
    
    Purpose: Record trajectory X(t) = (x_CG(t), m_fuel(t), m_i(t)) for
    post-processing and visualization.
    
    Stateless calculator is separated from history tracking to maintain
    pure function semantics for CG computation.
    """
    
    def __init__(self, scenario: ConsumptionScenario):
        """
        Initialize history arrays.
        
        Parameters:
            scenario: ConsumptionScenario - depletion order identifier
        """
        self.scenario = scenario
        
        # Time-series arrays
        self.cg_history: List[float] = []                              # x_CG(t) [m]
        self.fuel_consumed_history: List[float] = []                   # Δm_fuel(t) [kg]
        self.fuel_remaining_history: List[float] = []                  # m_fuel(t) [kg]
        self.weight_history: List[float] = []                          # m_total(t) [kg]
        self.tank_fuel_history: Dict[int, List[float]] = {i: [] for i in range(5)}  # m_i(t) [kg]
        
        # State tracking for monotonicity enforcement
        self._last_recorded_fuel: Optional[float] = None
        self._step_counter = 0
    
    def record_state(self, fuel_remaining: float, cg_x: float, tanks: Dict[int, float], 
                     weight_kg: Optional[float] = None):
        """
        Append current state to history with monotonicity enforcement.
        
        Monotonicity constraint: m_fuel(t+Δt) ≤ m_fuel(t)
        Enforces physical fuel consumption: dm_fuel/dt ≤ 0
        
        Parameters:
            fuel_remaining: m_fuel [kg] - current fuel mass
            cg_x: x_CG [m] - longitudinal CG position
            tanks: {tank_id: m_i [kg]} - tank fuel distribution
            weight_kg: m_total [kg] - total aircraft mass (optional)
        """
        # Duplicate rejection: skip if state unchanged (|Δm| < ε)
        if self._last_recorded_fuel is not None:
            fuel_change = abs(fuel_remaining - self._last_recorded_fuel)
            if fuel_change < 1e-6:
                return
            
            # Monotonicity enforcement: reject if m_fuel increases
            if fuel_remaining > self._last_recorded_fuel + 1e-6:
                return
        
        # Compute fuel consumed: Δm_fuel = m_fuel,total - m_fuel,remaining
        fuel_consumed = W_FUEL_KG - fuel_remaining
        
        # Append to time series
        self.cg_history.append(cg_x)
        self.fuel_consumed_history.append(fuel_consumed)
        self.fuel_remaining_history.append(fuel_remaining)
        
        if weight_kg is not None:
            self.weight_history.append(weight_kg)
        
        for tank_id in range(5):
            tank_fuel = tanks.get(tank_id, 0.0)
            self.tank_fuel_history[tank_id].append(max(0.0, tank_fuel))
        
        self._last_recorded_fuel = fuel_remaining
        self._step_counter += 1
        
        # Periodic diagnostic output
        if FUEL_LEVEL_PRINT_ENABLED and self._step_counter % FUEL_LEVEL_PRINT_SAMPLE_RATE == 0:
            self._print_fuel_levels(tanks, fuel_consumed, fuel_remaining)
    
    def _print_fuel_levels(self, tanks: Dict[int, float], fuel_consumed: float, fuel_remaining: float):
        """
        Diagnostic output for fuel distribution state.
        
        Parameters:
            tanks: {tank_id: m_i [kg]} - current tank masses
            fuel_consumed: Δm_fuel [kg] - fuel consumed
            fuel_remaining: m_fuel [kg] - fuel remaining
        """
        outer_tanks = [1, 3]
        inner_tanks = [0, 2]
        
        # Tank group totals
        outer_total = sum(tanks.get(i, 0.0) for i in outer_tanks)    # m_outer [kg]
        inner_total = sum(tanks.get(i, 0.0) for i in inner_tanks)    # m_inner [kg]
        center_total = tanks.get(4, 0.0)                              # m_center [kg]
        
        # Table header (periodic)
        if self._step_counter % (FUEL_LEVEL_PRINT_SAMPLE_RATE * 5) == 0:
            print("\n" + "="*100)
            print(f"{'Step':<8} {'Outer Total':<15} {'Inner Total':<15} {'Center':<15} "
                  f"{'Fuel Remaining':<18} {'Fuel Consumed':<15}")
            print("="*100)
        
        # Current state
        print(f"{self._step_counter:<8} {outer_total:<15.3f} {inner_total:<15.3f} "
              f"{center_total:<15.3f} {fuel_remaining:<18.3f} {fuel_consumed:<15.3f}")
        
        # Detailed breakdown (periodic)
        if self._step_counter % (FUEL_LEVEL_PRINT_SAMPLE_RATE * 10) == 0:
            print(f"  Detail: m_1={tanks.get(1, 0.0):.3f} kg, m_3={tanks.get(3, 0.0):.3f} kg, "
                  f"m_0={tanks.get(0, 0.0):.3f} kg, m_2={tanks.get(2, 0.0):.3f} kg, "
                  f"m_4={tanks.get(4, 0.0):.3f} kg")
            print()
    
    def reset(self):
        """Clear all history arrays and reset counters."""
        self.cg_history.clear()
        self.fuel_consumed_history.clear()
        self.fuel_remaining_history.clear()
        self.weight_history.clear()
        for i in range(5):
            self.tank_fuel_history[i].clear()
        self._last_recorded_fuel = None
        self._step_counter = 0

# ========================================================================
# SECTION 4: INTEGRATED FUEL SYSTEM
# ========================================================================

class FuelSystem:
    """
    Unified interface: CG computation + history tracking.
    
    Components:
        - FuelDistributionCalculator: Pure function x_CG = f(m_fuel)
        - FuelHistoryTracker: Time-series storage for visualization
    
    Primary API: calculate_cg_x(m_total, record_history) → x_CG
    """
    
    def __init__(self, scenario: ConsumptionScenario = "OUTER_FIRST"):
        """
        Initialize fuel system with consumption scenario.
        
        Parameters:
            scenario: ConsumptionScenario - depletion sequence
        """
        self.scenario = scenario
        self.calculator = FuelDistributionCalculator(scenario)
        self.history_tracker = FuelHistoryTracker(scenario)
        self.initial_fuel_kg = W_FUEL_KG  # m_fuel,0 [kg]
    
    def calculate_cg_x(self, current_weight_kg: float, record_history: bool = False) -> float:
        """
        Primary API: Compute longitudinal CG from total aircraft mass.
        
        Algorithm:
            1. Extract fuel mass: m_fuel = m_total - m_OE - m_PL
            2. Compute distribution: {m_i} = f_scenario(m_fuel)
            3. Calculate CG: x_CG = Σ(m_i·x_i) / Σm_i
            4. Optionally record in history
        
        History recording: Set record_history=False during DP optimization
        to avoid recording all evaluated candidate points. Set True only for
        optimal trajectory points in final mission analysis.
        
        Parameters:
            current_weight_kg: m_total [kg] - total aircraft mass
            record_history: Boolean - append to time series (default False)
        
        Returns:
            x_CG [m]: longitudinal CG position
        """
        # Extract fuel mass: m_fuel = m_total - m_OE - m_PL
        fuel_remaining = current_weight_kg - W_OE_KG - W_PL_KG
        
        # Clamp to physical bounds: m_fuel ∈ [0, m_fuel,max]
        fuel_remaining = max(0.0, min(fuel_remaining, W_FUEL_KG))
        
        # Compute CG position
        cg_x = self.calculator.calculate_cg_x(fuel_remaining)
        
        # Optional history recording
        if record_history:
            tanks = self.calculator.calculate_distribution(fuel_remaining)
            self.history_tracker.record_state(
                fuel_remaining=fuel_remaining,
                cg_x=cg_x,
                tanks=tanks,
                weight_kg=current_weight_kg
            )
            
        
        return cg_x
    
    def get_tank_distribution(self, fuel_remaining: float) -> Dict[int, float]:
        """
        Query fuel distribution for given fuel level.
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass distribution
        """
        return self.calculator.calculate_distribution(fuel_remaining)
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Extract current fuel state from history endpoint.
        
        Returns:
            Dictionary with current state variables and tank distribution
        """
        if len(self.history_tracker.fuel_remaining_history) == 0:
            # Initial state (no consumption history): m_fuel = m_fuel,0
            tanks = self.calculator.calculate_distribution(self.initial_fuel_kg)
            cg_initial = self.calculator.calculate_cg_x(self.initial_fuel_kg)
            
            return {
                'initialized': True,
                'scenario': self.scenario,
                'has_history': False,
                'initial_fuel_kg': self.initial_fuel_kg,
                'current_fuel_kg': self.initial_fuel_kg,
                'fuel_consumed_kg': 0.0,
                'tank_fuel_kg': {TANK_NAMES[i]: tanks[i] for i in range(5)},
                'cg_x_m': cg_initial,
                'history_length': 0,
                'message': 'Initial state (no consumption)'
            }
        
        # Current state from history endpoint: t = t_final
        latest_fuel_remaining = self.history_tracker.fuel_remaining_history[-1]  # m_fuel(t_final)
        latest_cg_x = self.history_tracker.cg_history[-1]                        # x_CG(t_final)
        latest_fuel_consumed = self.history_tracker.fuel_consumed_history[-1]    # Δm_fuel(t_final)
        
        # Current distribution
        tanks = self.calculator.calculate_distribution(latest_fuel_remaining)
        
        return {
            'initialized': True,
            'scenario': self.scenario,
            'has_history': True,
            'initial_fuel_kg': self.initial_fuel_kg,
            'current_fuel_kg': latest_fuel_remaining,
            'fuel_consumed_kg': latest_fuel_consumed,
            'tank_fuel_kg': {TANK_NAMES[i]: tanks[i] for i in range(5)},
            'cg_x_m': latest_cg_x,
            'history_length': len(self.history_tracker.cg_history)
        }
    
    def reset(self):
        """Reset system: clear history and distribution cache."""
        self.history_tracker.reset()
        self.calculator.clear_cache()
    
    @property
    def tank_fuel_kg(self) -> Dict[int, float]:
        """Query current tank distribution from history endpoint."""
        if len(self.history_tracker.fuel_remaining_history) == 0:
            return {i: TANK_CAPACITY_KG for i in range(5)}
        latest_fuel_remaining = self.history_tracker.fuel_remaining_history[-1]
        return self.calculator.calculate_distribution(latest_fuel_remaining)
    
    @property
    def cg_history(self) -> List[float]:
        """Access x_CG(t) time series."""
        return self.history_tracker.cg_history
    
    @property
    def fuel_consumed_history(self) -> List[float]:
        """Access Δm_fuel(t) time series."""
        return self.history_tracker.fuel_consumed_history
    
    @property
    def fuel_remaining_history(self) -> List[float]:
        """Access m_fuel,remaining(t) time series."""
        return self.history_tracker.fuel_remaining_history
    
    @property
    def weight_history(self) -> List[float]:
        """Access m_total(t) time series."""
        return self.history_tracker.weight_history
    
    @property
    def tank_fuel_history(self) -> Dict[int, List[float]]:
        """Access m_i(t) time series for all tanks."""
        return self.history_tracker.tank_fuel_history

# ========================================================================
# SECTION 5: MODULE-LEVEL API
# ========================================================================

# Global fuel system instance (singleton pattern)
_fuel_system: Optional[FuelSystem] = None

def _get_fuel_system(scenario: Optional[ConsumptionScenario] = None) -> FuelSystem:
    """
    Access or instantiate global fuel system singleton.
    
    Scenario management: Reads CG_CONSUMPTION_SCENARIO from mission_config
    to ensure consistency across all mission phases.
    
    Validation: If scenario parameter provided, must match configured value.
    
    Parameters:
        scenario: ConsumptionScenario - depletion order (optional)
                  If None, uses CG_CONSUMPTION_SCENARIO from mission_config
    
    Returns:
        FuelSystem: global singleton instance
    """
    global _fuel_system
    
    # Configuration source: mission_config.CG_CONSUMPTION_SCENARIO
    config_scenario = CG_CONSUMPTION_SCENARIO
    
    # Scenario validation: ensure consistency with configuration
    if scenario is not None and scenario != config_scenario:
        print(f"[CG_WARNING] Scenario mismatch: config='{config_scenario}', provided='{scenario}'")
        print(f"[CG_WARNING] Using config scenario: '{config_scenario}'")
        scenario = config_scenario
    else:
        scenario = config_scenario
    
    # Singleton instantiation
    if _fuel_system is None:
        print(f"[CG_SYSTEM] Initializing fuel system: scenario={scenario}, m_fuel,0={W_FUEL_KG:.3f} kg")
        _fuel_system = FuelSystem(scenario=scenario)
    else:
        # Validate scenario consistency
        if _fuel_system.scenario != scenario:
            print(f"[CG_WARNING] Scenario change detected: '{_fuel_system.scenario}' → '{scenario}'")
            print(f"[CG_WARNING] Reinitializing fuel system")
            _fuel_system = FuelSystem(scenario=scenario)
    
    return _fuel_system

# ========================================================================
# SECTION 6: PUBLIC API
# ========================================================================

def record_mission_history(climb_result=None, cruise_result=None, descent_result=None,
                          scenario: Optional[ConsumptionScenario] = None) -> None:
    """
    Populate history from mission trajectory m_total(t) arrays.
    
    Purpose: Extract optimal trajectory points from climb/cruise/descent phases
    and record in FuelSystem history for post-mission CG analysis.
    
    Data sources:
        - climb_result.mass_kg: m(t) during climb phase
        - cruise_result.mass_kg: m(t) during cruise phase
        - descent_result.mass_kg: m(t) during descent phase
    
    Sampling: Cruise history sampled to ~1000 points to limit memory usage.
    
    Parameters:
        climb_result: Object with mass_kg array (optional)
        cruise_result: Object with mass_kg array (optional)
        descent_result: Object with mass_kg array (optional)
        scenario: ConsumptionScenario (optional, uses config default)
    """
    fuel_system = _get_fuel_system(scenario)
    
    # ────────────────────────────────────────────────────────────────────
    # Climb Phase Recording
    # ────────────────────────────────────────────────────────────────────
    if climb_result is not None and hasattr(climb_result, 'mass_kg'):
        mass_array = np.array(climb_result.mass_kg)
        if len(mass_array) > 0:
            print(f"[CG_HISTORY] Climb: {len(mass_array)} points")
            for weight in mass_array:
                if np.isfinite(weight) and weight > 0:
                    fuel_system.calculate_cg_x(weight, record_history=True)
    
    # ────────────────────────────────────────────────────────────────────
    # Cruise Phase Recording (with sampling)
    # ────────────────────────────────────────────────────────────────────
    if cruise_result is not None and hasattr(cruise_result, 'mass_kg'):
        mass_array = np.array(cruise_result.mass_kg)
        if len(mass_array) > 0:
            # Sampling: reduce to ~1000 points to limit memory usage
            sample_rate = max(1, len(mass_array) // 1000)
            print(f"[CG_HISTORY] Cruise: {len(mass_array)} points (sampling every {sample_rate})")
            for i, mass in enumerate(mass_array):
                if i % sample_rate == 0 and np.isfinite(mass) and mass > 0:
                    fuel_system.calculate_cg_x(mass, record_history=True)
    
    # ────────────────────────────────────────────────────────────────────
    # Descent Phase Recording
    # ────────────────────────────────────────────────────────────────────
    if descent_result is not None and hasattr(descent_result, 'mass_kg'):
        mass_array = np.array(descent_result.mass_kg)
        if len(mass_array) > 0:
            print(f"[CG_HISTORY] Descent: {len(mass_array)} points")
            for mass in mass_array:
                if np.isfinite(mass) and mass > 0:
                    fuel_system.calculate_cg_x(mass, record_history=True)
    
    total_points = len(fuel_system.history_tracker.cg_history)
    print(f"[CG_HISTORY] Total recorded: {total_points} points")

def get_fuel_tank_status() -> Dict[str, Any]:
    """
    Query current fuel system state.
    
    Returns:
        Dictionary: {scenario, m_fuel,current, Δm_fuel, x_CG, {m_i}, history_length}
    """
    fuel_system = _get_fuel_system()
    return fuel_system.get_current_status()