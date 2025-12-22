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
    - INNER_FIRST: Inner → Outer → Center depletion sequence
    - PROPORTIONAL: Proportional depletion based on tank volumes

Tank capacities are calculated from physical volumes V_i [L] and fuel density ρ [kg/m³]:
    m_max,i = (V_i / 1000) × ρ

Capacity validation:
    Binary classification:
    - Low: W_FUEL_KG < theoretical capacity → use W_FUEL_KG as effective capacity
    - High: W_FUEL_KG >= theoretical capacity → cap to theoretical maximum
    If W_FUEL_KG > theoretical capacity, a warning is issued to indicate the mismatch.

CG shift affects trim analysis and aerodynamic performance.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Literal, List

# Aircraft mass components and fuel tank configuration
from aircraft_config import (
    INITIAL_MASS_KG, W_FUEL_KG, W_OE_KG, W_PL_KG,
    KEROSENE_DENSITY_KGPM3, TANK_VOLUMES_L, TANK_CG_POSITIONS, TANK_NAMES,
    ZERO_FUEL_CG_X,
    FUEL_LEVEL_PRINT_ENABLED, FUEL_LEVEL_PRINT_SAMPLE_RATE
)

# Default CG and consumption scenario
from mission_config import CG_CONSUMPTION_SCENARIO

# ========================================================================
# SECTION 1: TYPE DEFINITIONS AND CONSTANTS
# ========================================================================

# Consumption scenario enumeration
ConsumptionScenario = Literal["OUTER_FIRST", "CENTER_FIRST", "PROPORTIONAL", "INNER_FIRST"]

# Calculate individual tank mass capacities from volumes
# Conversion: V_i [L] → V_i [m³] = V_i / 1000, then m_max,i = V_i [m³] × ρ [kg/m³]
_L_TO_M3 = 1.0 / 1000.0  # Conversion factor: liters to cubic meters
_theoretical_max_mass_per_tank = {
    tank_id: (volume_L * _L_TO_M3) * KEROSENE_DENSITY_KGPM3
    for tank_id, volume_L in TANK_VOLUMES_L.items()
}
_theoretical_total_capacity_kg = sum(_theoretical_max_mass_per_tank.values())

# Validation: Binary classification of W_FUEL_KG vs theoretical tank capacity
# Low: W_FUEL_KG < theoretical_total_capacity_kg → use W_FUEL_KG
# High: W_FUEL_KG >= theoretical_total_capacity_kg → cap to theoretical maximum
if W_FUEL_KG >= _theoretical_total_capacity_kg:
    # Case: W_FUEL_KG >= theoretical capacity (high) - cap to physical maximum
    if W_FUEL_KG > _theoretical_total_capacity_kg + 1e-6:
        print(f"[CG_WARNING] W_FUEL_KG ({W_FUEL_KG:.2f} kg) exceeds theoretical tank capacity ({_theoretical_total_capacity_kg:.2f} kg)")
        print(f"[CG_WARNING] Capping effective fuel capacity to theoretical maximum: {_theoretical_total_capacity_kg:.2f} kg")
        print(f"[CG_WARNING] Excess fuel ({W_FUEL_KG - _theoretical_total_capacity_kg:.2f} kg) cannot be physically stored")
    _effective_fuel_capacity_kg = _theoretical_total_capacity_kg
else:
    # Case: W_FUEL_KG < theoretical capacity (low) - use W_FUEL_KG
    _effective_fuel_capacity_kg = W_FUEL_KG

# Scaling factor to match effective fuel capacity (accounts for not filling tanks to 100%)
# If theoretical capacity exceeds effective capacity, scale proportionally
if _theoretical_total_capacity_kg > _effective_fuel_capacity_kg + 1e-6:
    _capacity_scale_factor = _effective_fuel_capacity_kg / _theoretical_total_capacity_kg
else:
    _capacity_scale_factor = 1.0

# Per-tank mass capacities [kg]: m_max,i = theoretical_max,i × scale_factor
TANK_CAPACITY_KG = {
    tank_id: mass_kg * _capacity_scale_factor
    for tank_id, mass_kg in _theoretical_max_mass_per_tank.items()
}

# Effective total capacity (may be capped if W_FUEL_KG exceeded theoretical limit)
TOTAL_CAPACITY_KG = _effective_fuel_capacity_kg  # m_fuel,total [kg]: effective fuel capacity

# Legacy uniform capacity for backward compatibility (deprecated, use TANK_CAPACITY_KG[ID] instead)
TANK_CAPACITY_KG_UNIFORM = TOTAL_CAPACITY_KG / 5.0

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
    
    def __init__(
        self,
        scenario: ConsumptionScenario = "OUTER_FIRST",
        total_capacity_kg: float = TOTAL_CAPACITY_KG
    ):
        """
        Initialize distribution calculator with consumption scenario.
        
        Scenarios:
            OUTER_FIRST: Sequence [Outer(1,3)] → [Inner(0,2)] → [Center(4)]
            CENTER_FIRST: Sequence [Center(4)] → [Inner(0,2)] → [Outer(1,3)]
            INNER_FIRST: Sequence [Inner(0,2)] → [Outer(1,3)] → [Center(4)]
            PROPORTIONAL: m_i = m_fuel,remaining / 5 ∀i
        
        Parameters:
            scenario: ConsumptionScenario - depletion order
        """
        self.scenario = scenario
        self.total_capacity_kg = total_capacity_kg
        # Scale tank capacities to match the requested total capacity
        capacity_scale = total_capacity_kg / TOTAL_CAPACITY_KG if TOTAL_CAPACITY_KG > 0 else 1.0
        self.tank_capacity_kg = {
            tank_id: mass_kg * capacity_scale for tank_id, mass_kg in TANK_CAPACITY_KG.items()
        }
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
        fuel_remaining = max(0.0, min(fuel_remaining, self.total_capacity_kg))
        
        # Cache lookup for performance
        cache_key = round(fuel_remaining, 6)
        if cache_key in self._distribution_cache:
            return self._distribution_cache[cache_key].copy()
        
        # Route to scenario-specific distribution function
        if self.scenario == "OUTER_FIRST":
            distribution = self._calculate_outer_inner_center(fuel_remaining)
        elif self.scenario == "CENTER_FIRST":
            distribution = self._calculate_inner_outer_center(fuel_remaining)
        elif self.scenario == "INNER_FIRST":
            distribution = self._calculate_inner_first(fuel_remaining)
        elif self.scenario == "PROPORTIONAL":
            distribution = self._calculate_even(fuel_remaining)
        else:
            distribution = self._calculate_even(fuel_remaining)  # Default
        
        # Validation: Ensure fuel conservation (Σm_i = m_fuel,remaining)
        total_distributed = sum(distribution.values())
        fuel_conservation_error = abs(total_distributed - fuel_remaining)
        if fuel_conservation_error > 1e-4:  # Allow small floating-point errors
            print(f"[CG_ERROR] Fuel conservation violation in {self.scenario}:")
            print(f"  Expected: {fuel_remaining:.6f} kg")
            print(f"  Actual: {total_distributed:.6f} kg")
            print(f"  Error: {fuel_conservation_error:.6f} kg")
            # Normalize distribution to ensure exact conservation
            if total_distributed > 0:
                scale_factor = fuel_remaining / total_distributed
                distribution = {tank_id: mass * scale_factor for tank_id, mass in distribution.items()}
            else:
                distribution = {tank_id: 0.0 for tank_id in range(5)}
        
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
            Capacity: m_max,i calculated from tank volumes V_i [L] and fuel density ρ [kg/m³]
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass distribution
        """
        tanks = {i: 0.0 for i in range(5)}
        
        # Fuel consumed: Δm_fuel = m_fuel,total - m_fuel,remaining
        fuel_consumed = self.total_capacity_kg - fuel_remaining
        
        # Tank group capacities (volume-based, per-tank)
        outer_capacity = self.tank_capacity_kg[1] + self.tank_capacity_kg[3]  # m_outer = m_1 + m_3
        inner_capacity = self.tank_capacity_kg[0] + self.tank_capacity_kg[2]  # m_inner = m_0 + m_2
        center_capacity = self.tank_capacity_kg[4]                             # m_center = m_4
        
        # Proportional distribution factors for tanks within groups
        outer_tank_1_ratio = self.tank_capacity_kg[1] / outer_capacity if outer_capacity > 0 else 0.5
        inner_tank_0_ratio = self.tank_capacity_kg[0] / inner_capacity if inner_capacity > 0 else 0.5
        
        # Overlap threshold: start next phase when current phase has 20 kg remaining
        # This ensures continuous fuel consumption without gaps
        overlap_kg = 20.0  # [kg] - overlap amount for smooth transition
        
        # Phase determination with overlap transitions
        if fuel_consumed < outer_capacity - overlap_kg:
            # ────────────────────────────────────────────────────────────────
            # Phase 1: Outer depletion only
            # ────────────────────────────────────────────────────────────────
            outer_remaining = max(0.0, outer_capacity - fuel_consumed)
            
            # Distribution: Outer depleting proportionally, Inner and Center full
            tanks[1] = outer_remaining * outer_tank_1_ratio     # m_1 proportional to capacity
            tanks[3] = outer_remaining * (1.0 - outer_tank_1_ratio)  # m_3 remaining
            tanks[0] = self.tank_capacity_kg[0]                      # m_0 = m_max,0 (full)
            tanks[2] = self.tank_capacity_kg[2]                      # m_2 = m_max,2 (full)
            tanks[4] = self.tank_capacity_kg[4]                      # m_4 = m_max,4 (full)
        
        elif fuel_consumed < outer_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 1-2 Transition: Overlap period (both Outer and Inner depleting)
            # ────────────────────────────────────────────────────────────────
            # Outer tanks: remaining overlap_kg being depleted
            outer_remaining = max(0.0, outer_capacity - fuel_consumed)
            # Inner tanks: start depleting during overlap
            inner_consumed_in_overlap = fuel_consumed - (outer_capacity - overlap_kg)
            inner_remaining = max(0.0, inner_capacity - inner_consumed_in_overlap)
            
            # Distribution: Both Outer and Inner depleting, Center full
            tanks[1] = outer_remaining * outer_tank_1_ratio
            tanks[3] = outer_remaining * (1.0 - outer_tank_1_ratio)
            tanks[0] = inner_remaining * inner_tank_0_ratio
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)
            tanks[4] = self.tank_capacity_kg[4]                      # m_4 = m_max,4 (full)
        
        elif fuel_consumed < outer_capacity + inner_capacity - overlap_kg:
            # ────────────────────────────────────────────────────────────────
            # Phase 2: Inner depletion only
            # ────────────────────────────────────────────────────────────────
            inner_consumed = max(0.0, fuel_consumed - outer_capacity)
            inner_remaining = max(0.0, inner_capacity - inner_consumed)
            
            # Distribution: Outer empty, Inner depleting proportionally, Center full
            tanks[1] = 0.0                                       # m_1 = 0 (empty)
            tanks[3] = 0.0                                       # m_3 = 0 (empty)
            tanks[0] = inner_remaining * inner_tank_0_ratio     # m_0 proportional to capacity
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)  # m_2 remaining
            tanks[4] = self.tank_capacity_kg[4]                      # m_4 = m_max,4 (full)
        
        elif fuel_consumed < outer_capacity + inner_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 2-3 Transition: Overlap period (both Inner and Center depleting)
            # ────────────────────────────────────────────────────────────────
            # Inner tanks: remaining overlap_kg being depleted
            inner_remaining = max(0.0, inner_capacity - (fuel_consumed - outer_capacity))
            # Center tank: start depleting during overlap
            center_consumed_in_overlap = fuel_consumed - (outer_capacity + inner_capacity - overlap_kg)
            center_remaining = max(0.0, center_capacity - center_consumed_in_overlap)
            
            # Distribution: Outer empty, Inner and Center depleting
            tanks[1] = 0.0                                       # m_1 = 0 (empty)
            tanks[3] = 0.0                                       # m_3 = 0 (empty)
            tanks[0] = inner_remaining * inner_tank_0_ratio
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)
            tanks[4] = center_remaining                           # m_4 depleting
        
        else:
            # ────────────────────────────────────────────────────────────────
            # Phase 3: Center depletion only
            # ────────────────────────────────────────────────────────────────
            center_consumed = max(0.0, fuel_consumed - outer_capacity - inner_capacity)
            center_remaining = max(0.0, center_capacity - center_consumed)
            
            # Distribution: Outer and Inner empty, Center depleting
            tanks[1] = 0.0                                       # m_1 = 0 (empty)
            tanks[3] = 0.0                                       # m_3 = 0 (empty)
            tanks[0] = 0.0                                       # m_0 = 0 (empty)
            tanks[2] = 0.0                                       # m_2 = 0 (empty)
            tanks[4] = max(0.0, center_remaining)                # m_4 = m_center,rem
        
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
        fuel_consumed = self.total_capacity_kg - fuel_remaining
        
        # Tank group capacities (volume-based, per-tank)
        center_capacity = self.tank_capacity_kg[4]                        # m_center = m_4
        inner_capacity = self.tank_capacity_kg[0] + self.tank_capacity_kg[2]  # m_inner = m_0 + m_2
        outer_capacity = self.tank_capacity_kg[1] + self.tank_capacity_kg[3]  # m_outer = m_1 + m_3
        
        # Proportional distribution factors for tanks within groups
        inner_tank_0_ratio = self.tank_capacity_kg[0] / inner_capacity if inner_capacity > 0 else 0.5
        outer_tank_1_ratio = self.tank_capacity_kg[1] / outer_capacity if outer_capacity > 0 else 0.5
        
        # Overlap threshold: start next phase when current phase has 20 kg remaining
        # This ensures continuous fuel consumption without gaps
        overlap_kg = 20.0  # [kg] - overlap amount for smooth transition
        
        # Phase determination with overlap transitions
        if fuel_consumed < center_capacity - overlap_kg:
            # ────────────────────────────────────────────────────────────────
            # Phase 1: Center depletion only
            # ────────────────────────────────────────────────────────────────
            center_remaining = max(0.0, center_capacity - fuel_consumed)
            
            # Distribution: Center depleting, Inner and Outer full
            tanks[4] = center_remaining                            # m_4 = m_center,rem
            tanks[0] = self.tank_capacity_kg[0]                         # m_0 = m_max,0 (full)
            tanks[2] = self.tank_capacity_kg[2]                         # m_2 = m_max,2 (full)
            tanks[1] = self.tank_capacity_kg[1]                         # m_1 = m_max,1 (full)
            tanks[3] = self.tank_capacity_kg[3]                         # m_3 = m_max,3 (full)
        
        elif fuel_consumed < center_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 1-2 Transition: Overlap period (both Center and Inner depleting)
            # ────────────────────────────────────────────────────────────────
            # Center tank: remaining overlap_kg being depleted
            center_remaining = max(0.0, center_capacity - fuel_consumed)
            # Inner tanks: start depleting during overlap
            inner_consumed_in_overlap = fuel_consumed - (center_capacity - overlap_kg)
            inner_remaining = max(0.0, inner_capacity - inner_consumed_in_overlap)
            
            # Distribution: Both Center and Inner depleting, Outer full
            tanks[4] = center_remaining
            tanks[0] = inner_remaining * inner_tank_0_ratio
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)
            tanks[1] = self.tank_capacity_kg[1]                         # m_1 = m_max,1 (full)
            tanks[3] = self.tank_capacity_kg[3]                         # m_3 = m_max,3 (full)
        
        elif fuel_consumed < center_capacity + inner_capacity - overlap_kg:
            # ────────────────────────────────────────────────────────────────
            # Phase 2: Inner depletion only
            # ────────────────────────────────────────────────────────────────
            inner_consumed = max(0.0, fuel_consumed - center_capacity)
            inner_remaining = max(0.0, inner_capacity - inner_consumed)
            
            # Distribution: Center empty, Inner depleting proportionally, Outer full
            tanks[4] = 0.0                                         # m_4 = 0 (empty)
            tanks[0] = inner_remaining * inner_tank_0_ratio        # m_0 proportional to capacity
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)  # m_2 remaining
            tanks[1] = self.tank_capacity_kg[1]                         # m_1 = m_max,1 (full)
            tanks[3] = self.tank_capacity_kg[3]                         # m_3 = m_max,3 (full)
        
        elif fuel_consumed < center_capacity + inner_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 2-3 Transition: Overlap period (both Inner and Outer depleting)
            # ────────────────────────────────────────────────────────────────
            # Inner tanks: remaining overlap_kg being depleted
            inner_remaining = max(0.0, inner_capacity - (fuel_consumed - center_capacity))
            # Outer tanks: start depleting during overlap
            outer_consumed_in_overlap = fuel_consumed - (center_capacity + inner_capacity - overlap_kg)
            outer_remaining = max(0.0, outer_capacity - outer_consumed_in_overlap)
            
            # Distribution: Center empty, Inner and Outer depleting
            tanks[4] = 0.0                                         # m_4 = 0 (empty)
            tanks[0] = inner_remaining * inner_tank_0_ratio
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)
            tanks[1] = outer_remaining * outer_tank_1_ratio
            tanks[3] = outer_remaining * (1.0 - outer_tank_1_ratio)
        
        else:
            # ────────────────────────────────────────────────────────────────
            # Phase 3: Outer depletion only
            # ────────────────────────────────────────────────────────────────
            outer_consumed = max(0.0, fuel_consumed - center_capacity - inner_capacity)
            outer_remaining = max(0.0, outer_capacity - outer_consumed)
            
            # Distribution: Center and Inner empty, Outer depleting proportionally
            tanks[4] = 0.0                                         # m_4 = 0 (empty)
            tanks[0] = 0.0                                         # m_0 = 0 (empty)
            tanks[2] = 0.0                                         # m_2 = 0 (empty)
            tanks[1] = max(0.0, outer_remaining * outer_tank_1_ratio)  # m_1 proportional to capacity
            tanks[3] = max(0.0, outer_remaining * (1.0 - outer_tank_1_ratio))  # m_3 remaining
        
        return tanks
    
    def _calculate_inner_first(self, fuel_remaining: float) -> Dict[int, float]:
        """
        INNER_FIRST depletion: Sequential emptying [Inner] → [Outer] → [Center].
        
        Depletion sequence:
            Phase 1 (100%→60%): m_consumed < m_inner,total
                → Deplete tanks {0,2} uniformly, {1,3,4} remain full
            Phase 2 (60%→20%): m_inner,total ≤ m_consumed < m_inner,total + m_outer,total
                → Tanks {0,2} empty, deplete {1,3} uniformly, {4} remains full
            Phase 3 (20%→0%): m_consumed ≥ m_inner,total + m_outer,total
                → Tanks {0,1,2,3} empty, deplete {4}
        
        Tank grouping:
            Inner: {0,2}, Outer: {1,3}, Center: {4}
            Capacity: m_max,i calculated from tank volumes V_i [L] and fuel density ρ [kg/m³]
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass distribution
        """
        tanks = {i: 0.0 for i in range(5)}
        
        # Fuel consumed: Δm_fuel = m_fuel,total - m_fuel,remaining
        fuel_consumed = self.total_capacity_kg - fuel_remaining
        
        # Tank group capacities (volume-based, per-tank)
        inner_capacity = self.tank_capacity_kg[0] + self.tank_capacity_kg[2]  # m_inner = m_0 + m_2
        outer_capacity = self.tank_capacity_kg[1] + self.tank_capacity_kg[3]  # m_outer = m_1 + m_3
        center_capacity = self.tank_capacity_kg[4]                             # m_center = m_4
        
        # Proportional distribution factors for tanks within groups
        inner_tank_0_ratio = self.tank_capacity_kg[0] / inner_capacity if inner_capacity > 0 else 0.5
        outer_tank_1_ratio = self.tank_capacity_kg[1] / outer_capacity if outer_capacity > 0 else 0.5
        
        # Overlap threshold: start next phase when current phase has 20 kg remaining
        # This ensures continuous fuel consumption without gaps
        overlap_kg = 20.0  # [kg] - overlap amount for smooth transition
        
        # Phase determination with overlap transitions
        if fuel_consumed < inner_capacity - overlap_kg:
            # ────────────────────────────────────────────────────────────────
            # Phase 1: Inner depletion only
            # ────────────────────────────────────────────────────────────────
            inner_remaining = max(0.0, inner_capacity - fuel_consumed)
            
            # Distribution: Inner depleting proportionally, Outer and Center full
            tanks[0] = inner_remaining * inner_tank_0_ratio     # m_0 proportional to capacity
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)  # m_2 remaining
            tanks[1] = self.tank_capacity_kg[1]                      # m_1 = m_max,1 (full)
            tanks[3] = self.tank_capacity_kg[3]                      # m_3 = m_max,3 (full)
            tanks[4] = self.tank_capacity_kg[4]                      # m_4 = m_max,4 (full)
        
        elif fuel_consumed < inner_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 1-2 Transition: Overlap period (both Inner and Outer depleting)
            # ────────────────────────────────────────────────────────────────
            # Inner tanks: remaining overlap_kg being depleted
            inner_remaining = max(0.0, inner_capacity - fuel_consumed)
            # Outer tanks: start depleting during overlap
            outer_consumed_in_overlap = fuel_consumed - (inner_capacity - overlap_kg)
            outer_remaining = max(0.0, outer_capacity - outer_consumed_in_overlap)
            
            # Distribution: Both Inner and Outer depleting, Center full
            tanks[0] = inner_remaining * inner_tank_0_ratio
            tanks[2] = inner_remaining * (1.0 - inner_tank_0_ratio)
            tanks[1] = outer_remaining * outer_tank_1_ratio
            tanks[3] = outer_remaining * (1.0 - outer_tank_1_ratio)
            tanks[4] = self.tank_capacity_kg[4]                      # m_4 = m_max,4 (full)
        
        elif fuel_consumed < inner_capacity + outer_capacity - overlap_kg:
            # ────────────────────────────────────────────────────────────────
            # Phase 2: Outer depletion only
            # ────────────────────────────────────────────────────────────────
            outer_consumed = max(0.0, fuel_consumed - inner_capacity)
            outer_remaining = max(0.0, outer_capacity - outer_consumed)
            
            # Distribution: Inner empty, Outer depleting proportionally, Center full
            tanks[0] = 0.0                                       # m_0 = 0 (empty)
            tanks[2] = 0.0                                       # m_2 = 0 (empty)
            tanks[1] = outer_remaining * outer_tank_1_ratio     # m_1 proportional to capacity
            tanks[3] = outer_remaining * (1.0 - outer_tank_1_ratio)  # m_3 remaining
            tanks[4] = self.tank_capacity_kg[4]                      # m_4 = m_max,4 (full)
        
        elif fuel_consumed < inner_capacity + outer_capacity:
            # ────────────────────────────────────────────────────────────────
            # Phase 2-3 Transition: Overlap period (both Outer and Center depleting)
            # ────────────────────────────────────────────────────────────────
            # Outer tanks: remaining overlap_kg being depleted
            outer_remaining = max(0.0, outer_capacity - (fuel_consumed - inner_capacity))
            # Center tank: start depleting during overlap
            center_consumed_in_overlap = fuel_consumed - (inner_capacity + outer_capacity - overlap_kg)
            center_remaining = max(0.0, center_capacity - center_consumed_in_overlap)
            
            # Distribution: Inner empty, Outer and Center depleting
            tanks[0] = 0.0                                       # m_0 = 0 (empty)
            tanks[2] = 0.0                                       # m_2 = 0 (empty)
            tanks[1] = outer_remaining * outer_tank_1_ratio
            tanks[3] = outer_remaining * (1.0 - outer_tank_1_ratio)
            tanks[4] = center_remaining                           # m_4 depleting
        
        else:
            # ────────────────────────────────────────────────────────────────
            # Phase 3: Center depletion only
            # ────────────────────────────────────────────────────────────────
            center_consumed = max(0.0, fuel_consumed - inner_capacity - outer_capacity)
            center_remaining = max(0.0, center_capacity - center_consumed)
            
            # Distribution: Inner and Outer empty, Center depleting
            tanks[0] = 0.0                                       # m_0 = 0 (empty)
            tanks[2] = 0.0                                       # m_2 = 0 (empty)
            tanks[1] = 0.0                                       # m_1 = 0 (empty)
            tanks[3] = 0.0                                       # m_3 = 0 (empty)
            tanks[4] = max(0.0, center_remaining)                # m_4 = m_center,rem
        
        return tanks
    
    def _calculate_even(self, fuel_remaining: float) -> Dict[int, float]:
        """
        PROPORTIONAL depletion: Distribution proportional to tank capacities.
        
        Depletion: All tanks deplete proportionally to their volume-based capacities.
        Distribution: m_i = m_fuel,remaining × (m_max,i / Σm_max,j)
        where m_max,i is the mass capacity of tank i.
        
        Parameters:
            fuel_remaining: m_fuel [kg] - total fuel remaining
        
        Returns:
            {tank_id: m_i [kg]} - fuel mass distribution
        """
        tanks = {i: 0.0 for i in range(5)}
        
        # Total capacity: Σm_max,i
        total_capacity = sum(self.tank_capacity_kg.values())
        
        if total_capacity > 0:
            # Proportional distribution: m_i = m_fuel × (m_max,i / Σm_max)
            for tank_id in range(5):
                capacity_ratio = self.tank_capacity_kg[tank_id] / total_capacity
                tanks[tank_id] = fuel_remaining * capacity_ratio
        else:
            # Fallback: uniform distribution (should not occur)
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
            # Fallback: use zero-fuel CG (should never happen)
            cg_x = ZERO_FUEL_CG_X
        
        return cg_x
    
    def get_fuel_consumed(self, fuel_remaining: float) -> float:
        """
        Compute fuel consumed: Δm_fuel = m_fuel,total - m_fuel,remaining.
        
        Parameters:
            fuel_remaining: m_fuel,remaining [kg]
        
        Returns:
            Δm_fuel [kg]: fuel consumed
        """
        return self.total_capacity_kg - fuel_remaining
    
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
    
    def __init__(self, scenario: ConsumptionScenario, total_capacity_kg: float):
        """
        Initialize history arrays.
        
        Parameters:
            scenario: ConsumptionScenario - depletion order identifier
        """
        self.scenario = scenario
        self.total_capacity_kg = total_capacity_kg
        
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
        fuel_consumed = self.total_capacity_kg - fuel_remaining
        
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
    
    def __init__(self, scenario: ConsumptionScenario = "OUTER_FIRST", initial_fuel_kg: float = TOTAL_CAPACITY_KG):
        """
        Initialize fuel system with consumption scenario.
        
        Parameters:
            scenario: ConsumptionScenario - depletion sequence
        """
        self.scenario = scenario
        self.initial_fuel_kg = initial_fuel_kg  # m_fuel,0 [kg]: effective initial fuel capacity
        self.calculator = FuelDistributionCalculator(scenario, total_capacity_kg=self.initial_fuel_kg)
        self.history_tracker = FuelHistoryTracker(scenario, total_capacity_kg=self.initial_fuel_kg)
    
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
        fuel_remaining = max(0.0, min(fuel_remaining, self.initial_fuel_kg))
        
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
            # Initial state: all tanks at full capacity
            return self.calculator.tank_capacity_kg.copy()
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

def _get_fuel_system(
    scenario: Optional[ConsumptionScenario] = None,
    initial_fuel_kg: Optional[float] = None
) -> FuelSystem:
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
    
    # Determine effective initial fuel
    effective_initial_fuel = initial_fuel_kg if initial_fuel_kg is not None else TOTAL_CAPACITY_KG
    
    # Singleton instantiation
    if _fuel_system is None:
        print(f"[CG_SYSTEM] Initializing fuel system: scenario={scenario}, m_fuel,0={effective_initial_fuel:.3f} kg")
        if effective_initial_fuel < W_FUEL_KG - 1e-6:
            print(f"[CG_SYSTEM] Note: Effective capacity ({effective_initial_fuel:.3f} kg) < W_FUEL_KG ({W_FUEL_KG:.3f} kg) due to tank volume limits")
        _fuel_system = FuelSystem(scenario=scenario, initial_fuel_kg=effective_initial_fuel)
    else:
        # Validate scenario consistency
        if _fuel_system.scenario != scenario:
            print(f"[CG_WARNING] Scenario change detected: '{_fuel_system.scenario}' → '{scenario}'")
            print(f"[CG_WARNING] Reinitializing fuel system")
            _fuel_system = FuelSystem(scenario=scenario, initial_fuel_kg=effective_initial_fuel)
        else:
            # If an override fuel is provided and differs, reinitialize
            if abs(_fuel_system.initial_fuel_kg - effective_initial_fuel) > 1e-6:
                print(f"[CG_SYSTEM] Reinitializing fuel system with updated initial fuel: {effective_initial_fuel:.3f} kg")
                _fuel_system = FuelSystem(scenario=scenario, initial_fuel_kg=effective_initial_fuel)
    
    return _fuel_system

# ========================================================================
# SECTION 6: PUBLIC API
# ========================================================================

def record_mission_history(climb_result=None, cruise_result=None, descent_result=None,
                          scenario: Optional[ConsumptionScenario] = None,
                          initial_fuel_kg: Optional[float] = None) -> None:
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
    fuel_system = _get_fuel_system(scenario, initial_fuel_kg=initial_fuel_kg)
    
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


def initialize_fuel_system(initial_fuel_kg: Optional[float] = None,
                           scenario: Optional[ConsumptionScenario] = None) -> None:
    """
    Initialize or reinitialize the global fuel system with an explicit fuel load.
    
    This helper lets callers (e.g., per-iteration optimizers) set the initial fuel
    used by CG calculations without altering the default mission configuration.
    """
    _get_fuel_system(scenario=scenario, initial_fuel_kg=initial_fuel_kg)