# ========================================================================
# PYAERODYNAMICS INTEGRATION MODULE
# ========================================================================
"""
Aerodynamic force and coefficient computation via pyaerodynamics library.

Computational interface:
    - Trim analysis: α_wing(M,h,m), η_HT(M,h,m) via linearized equations
    - Force computation: D = CD·q·S, L = CL·q·S where q = 0.5·ρ·V²
    - Coefficients: CD(M,h,m), CL(M,h,m), L/D via trim solution
    
Cache system:
    Key: (M, h, m) rounded to precision (3, 1, 1) decimal places
    Hit rate tracking for performance analysis
    
Mathematical context:
    Trim condition: ΣM = 0, ΣF_z = 0 about CG
    Lift equation: L = W → CL = W/(q·S)
    Drag force: D = CD·q·S [N]
    Dynamic pressure: q = 0.5·ρ·V² where V = M·a(h)
    
Compatibility: AeroTables interface maintained for legacy code.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# ────────────────────────────────────────────────────────────────────
# Path Resolution and Module Loading
# ────────────────────────────────────────────────────────────────────
_current_file = Path(__file__).resolve()
_root_dir = _current_file.parent.parent  # Codes/ → root
_pyaerodynamics_release_path = _root_dir / 'Aero' / 'pyaerodynamics' / 'pyaerodynamics' / 'Release'
_pyaerodynamics_package_path = _root_dir / 'Aero' / 'pyaerodynamics'

# Add to sys.path for import resolution
if str(_pyaerodynamics_release_path) not in sys.path:
    sys.path.insert(0, str(_pyaerodynamics_release_path))
if str(_pyaerodynamics_package_path) not in sys.path:
    sys.path.insert(0, str(_pyaerodynamics_package_path))

# Direct .pyd loading for compiled module
import importlib.util
_pyd_file = _pyaerodynamics_release_path / 'pyaerodynamics.cp311-win_amd64.pyd'
if _pyd_file.exists():
    spec = importlib.util.spec_from_file_location("pyaerodynamics", _pyd_file)
    _pyaerodynamics_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_pyaerodynamics_core)
    Aircraft = _pyaerodynamics_core.Aircraft
    Flight_Condition = _pyaerodynamics_core.Flight_Condition
else:
    from pyaerodynamics import Aircraft, Flight_Condition

# Atmospheric model and aircraft parameters
from atmosphere import Atmosphere
from aircraft_config import S_REF_M2, M_MMO, DEFAULT_COG_LOCATION, CL_MAX

# CG calculation configuration and dynamic computation
from mission_config import USE_DYNAMIC_CG, CG_CONSUMPTION_SCENARIO

# Gravity constant: g = 9.80665 m/s²
G_C = Atmosphere.G_C

# ========================================================================
# SECTION 1: FLIGHT STATE DEFINITION
# ========================================================================

@dataclass
class FlightState:
    """
    Flight condition state vector for trim analysis.
    
    State: X = (h, M, m, x_CG)
    
    Fields:
        altitude_m: h [m] - altitude
        mach: M [-] - Mach number
        weight_kg: m [kg] - aircraft mass
        cog_location: x_CG [m, m, m] - center of gravity coordinates
    """
    altitude_m: float
    mach: float
    weight_kg: float
    cog_location: list = None
    
    def __post_init__(self):
        if self.cog_location is None:
            self.cog_location = DEFAULT_COG_LOCATION

# ────────────────────────────────────────────────────────────────────
# CG Location Computation
# ────────────────────────────────────────────────────────────────────

def _compute_cog_location(weight_kg: float) -> list:
    """
    Compute CG location based on USE_DYNAMIC_CG flag.
    
    Algorithm:
        - If USE_DYNAMIC_CG = True: Compute x_CG from fuel distribution
          via cg_x_calculation module, return [x_CG, 0.0, 0.0]
        - If USE_DYNAMIC_CG = False: Return DEFAULT_COG_LOCATION
    
    Parameters:
        weight_kg: m [kg] - total aircraft mass
    
    Returns:
        [x, y, z] [m]: CG coordinates in body frame
    """
    if USE_DYNAMIC_CG:
        # Dynamic CG computation from fuel distribution
        from cg_x_calculation import _get_fuel_system
        fuel_system = _get_fuel_system()
        cg_x = fuel_system.calculate_cg_x(weight_kg, record_history=False)
        return [cg_x, 0.0, 0.0]
    else:
        # Static default CG location
        return DEFAULT_COG_LOCATION

# ========================================================================
# SECTION 2: AERODYNAMICS MODEL WRAPPER
# ========================================================================

class PyAerodynamicsWrapper:
    """
    Aerodynamics model interface with trim analysis and caching.
    
    Mathematical operations:
        - Trim solution: (α_wing, η_HT) satisfying ΣM = 0, L = W
        - Coefficients: CD(M,h,m), CL(M,h,m), L/D from trim state
        - Forces: D = CD·q·S, L = CL·q·S [N]
        - Dynamic pressure: q = 0.5·ρ·V² where V = M·a(h), ρ = ρ(h)
    
    Cache:
        Key: (M, h, m) with precision (3, 1, 1) decimals
        Purpose: Avoid redundant trim computations
        Metrics: Hit rate, cache size tracked
    
    Compatibility: AeroTables interface (M_grid, H_grid, cl_max, G_C).
    
    Source: pyaerodynamics library via Aircraft and Flight_Condition classes.
    """
    
    def __init__(self, xml_path: str = "Aero/aero_data/polar.xml", high_accuracy: bool = False):
        """
        Initialize aerodynamics model from XML configuration.
        
        Parameters:
            xml_path: str - path to aircraft polar XML (relative to root)
            high_accuracy: bool - use 4-decimal Mach precision (default: 3)
        """
        # Path resolution
        xml_path_obj = Path(xml_path)
        if not xml_path_obj.is_absolute():
            self.xml_path = str(_root_dir / xml_path)
        else:
            self.xml_path = xml_path
        
        # Cache storage
        self.aircraft = None
        self._drag_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Cache precision configuration
        self._high_accuracy = high_accuracy
        if high_accuracy:
            self._precision = {'mach': 4, 'altitude': 1, 'weight': 1}  # 4 decimal Mach
            print(f"[AERO] High-accuracy mode enabled")
        else:
            self._precision = {'mach': 3, 'altitude': 1, 'weight': 1}  # 3 decimal Mach
        
        # Initialize aircraft model
        self._initialize_aircraft()
        
        # Compatibility grids
        self._setup_default_grids()
        
        # Aircraft parameters
        self.params = {
            'S_REF_M2': S_REF_M2,
            'M_MMO': M_MMO,
            'CL_MAX': CL_MAX,
        }
    
    def _initialize_aircraft(self):
        """Load aircraft configuration from XML."""
        try:
            self.aircraft = Aircraft(self.xml_path)
            print(f"[PYAERO] Aircraft loaded from {self.xml_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load aircraft from {self.xml_path}: {e}")
    
    def _setup_default_grids(self):
        """
        Initialize default grids for AeroTables compatibility.
        
        Grids:
            M ∈ [0.2, 0.9]: 15 points (ΔM ≈ 0.047)
            h ∈ [0, 12000]: 16 points (Δh = 800 m)
        """
        self._M = np.linspace(0.2, 0.9, 15)
        self._H = np.linspace(0, 12000, 16)
    
    # ────────────────────────────────────────────────────────────────────
    # AeroTables Compatibility Properties
    # ────────────────────────────────────────────────────────────────────
    
    @property
    def mach_grid(self) -> np.ndarray:
        """M_grid: Mach sampling points for compatibility."""
        return self._M
    
    @property
    def alt_grid_m(self) -> np.ndarray:
        """H_grid [m]: altitude sampling points for compatibility."""
        return self._H
    
    @property
    def cl_max(self) -> float:
        """CL_max [-]: maximum lift coefficient."""
        return self.params['CL_MAX']
    
    @property
    def G_C(self) -> float:
        """g [m/s²]: standard gravity constant."""
        return Atmosphere.G_C
    
    # ────────────────────────────────────────────────────────────────────
    # Trim Analysis Primitives
    # ────────────────────────────────────────────────────────────────────
    
    def _create_flight_condition(self, altitude_m: float, mach: float) -> Flight_Condition:
        """Instantiate Flight_Condition(h, M) object."""
        return Flight_Condition(altitude_m, mach)
    
    def _perform_trim_analysis(self, flight_state: FlightState) -> Dict[str, Any]:
        """
        Solve trim equations for equilibrium state.
        
        Trim conditions:
            ΣF_z = 0: L = W
            ΣM_y = 0: Moment balance about CG
        
        Solution: (α_wing, η_HT) via linearized equations
        
        Output:
            - Control angles: α_wing [rad], η_HT [rad]
            - Coefficients: CD, CL, L/D
            - Conditions: Flight_Condition object
            - Forces: W [N]
        
        Parameters:
            flight_state: FlightState - (h, M, m, x_CG)
            
        Returns:
            dict: {wing_aoa, ht_incidence, cd, cl, ld, conditions, weight_force}
        """
        # Flight condition: (h, M)
        conditions = self._create_flight_condition(flight_state.altitude_m, flight_state.mach)
        
        # Aircraft configuration: main wing + horizontal stabilizer, linearized
        self.aircraft.change_settings("main_wing", "horizontal_stabiliser", "linear")
        
        # Gravitational force: W = m·g [N]
        weight_force = flight_state.weight_kg * G_C
        
        # Solve trim: (α_wing, η_HT) satisfying ΣM = 0, L = W
        self.aircraft.linearized_trim(conditions, weight_force, flight_state.cog_location)
        
        # Extract control angles
        wing_aoa = self.aircraft.settings.reference_wing_angle         # α_wing [rad]
        ht_incidence = self.aircraft.settings.adjustable_surface_angle # η_HT [rad]
        
        # Compute coefficients at trim state
        cd = self.aircraft.get_CD(conditions, weight_force)            # CD(M,h,m,α,η)
        cl = self.aircraft.get_CL(conditions, weight_force)            # CL(M,h,m,α,η)
        ld = self.aircraft.get_CL_CD(conditions, weight_force, flight_state.cog_location)  # L/D
        
        return {
            'wing_aoa': wing_aoa,
            'ht_incidence': ht_incidence,
            'cd': cd,
            'cl': cl,
            'ld': ld,
            'conditions': conditions,
            'weight_force': weight_force
        }
    
    # ────────────────────────────────────────────────────────────────────
    # Drag Force Computation with Caching
    # ────────────────────────────────────────────────────────────────────
    
    def get_drag(self, M: float, h_m: float, weight_kg: float) -> float:
        """
        Compute drag force: D(M, h, m) = CD·q·S [N].
        
        Algorithm:
            1. Check cache: key = (M, h, m) rounded to precision
            2. If miss: Solve trim → CD(M,h,m,α,η)
            3. Compute: q = 0.5·ρ(h)·V² where V = M·a(h)
            4. Force: D = CD·q·S
            5. Store in cache
        
        Parameters:
            M: Mach number [-]
            h_m: altitude [m]
            weight_kg: mass [kg]
            
        Returns:
            D [N]: drag force (0.0 if computation fails)
        """
        # Cache lookup with rounded key
        cache_key = (
            round(M, self._precision['mach']),
            round(h_m, self._precision['altitude']),
            round(weight_kg, self._precision['weight'])
        )
        if cache_key in self._drag_cache:
            self._cache_hits += 1
            return self._drag_cache[cache_key]
        
        try:
            # Flight state: X = (h, M, m, x_CG)
            # CG computed dynamically if USE_DYNAMIC_CG=True, else uses default
            flight_state = FlightState(
                altitude_m=h_m,
                mach=M,
                weight_kg=weight_kg,
                cog_location=_compute_cog_location(weight_kg)
            )
            
            # Trim solution
            trim_results = self._perform_trim_analysis(flight_state)
            cd = trim_results['cd']  # CD(M,h,m,α,η)
            
            # Atmospheric properties: ρ(h), a(h)
            atm = Atmosphere()
            flight_level = h_m / 0.3048  # Convert m → ft
            T, p, rho = atm.calculate_atmospheric_properties(flight_level)
            a = atm.get_speed_of_sound(h_m)
            V = M * a  # V = M·a(h) [m/s]
            
            # Dynamic pressure: q = 0.5·ρ·V² [Pa]
            q = 0.5 * rho * V**2
            
            # Drag force: D = CD·q·S [N]
            drag_force = cd * q * S_REF_M2
            
            # Store in cache
            self._drag_cache[cache_key] = drag_force
            self._cache_misses += 1
            
            return drag_force
            
        except Exception as e:
            print(f"[PYAERO] Error calculating drag: {e}")
            return 0.0
    
    def get_comprehensive_aerodynamics(self, M: float, h_m: float, weight_kg: float) -> Dict[str, Any]:
        """
        Compute complete aerodynamic state at (M, h, m).
        
        Output package:
            - State: M, h, m
            - Trim angles: α_wing [rad], η_HT [rad]
            - Coefficients: CD, CL, L/D
            - Forces: D [N], L [N]
            - Atmospheric: ρ [kg/m³], T [K], p [Pa], a [m/s]
            - Kinematics: V = M·a [m/s], q = 0.5·ρ·V² [Pa]
        
        Algorithm:
            1. Solve trim: (α_wing, η_HT) for ΣM = 0, L = W
            2. Compute: CD, CL, L/D at trim state
            3. Atmospheric: ρ(h), T(h), p(h), a(h)
            4. Forces: D = CD·q·S, L = CL·q·S
        
        Parameters:
            M: Mach number [-]
            h_m: altitude [m]
            weight_kg: mass [kg]
            
        Returns:
            dict: complete aerodynamic state (empty dict if error)
        """
        try:
            # Flight state: X = (h, M, m, x_CG)
            # CG computed dynamically if USE_DYNAMIC_CG=True, else uses default
            flight_state = FlightState(
                altitude_m=h_m,
                mach=M,
                weight_kg=weight_kg,
                cog_location=_compute_cog_location(weight_kg)
            )
            
            # Trim solution
            trim_results = self._perform_trim_analysis(flight_state)
            
            # Atmospheric properties: ρ(h), T(h), p(h), a(h)
            atm = Atmosphere()
            flight_level = h_m / 0.3048  # m → ft
            T, p, rho = atm.calculate_atmospheric_properties(flight_level)
            a = atm.get_speed_of_sound(h_m)
            V = M * a  # V = M·a(h) [m/s]
            
            # Dynamic pressure and forces
            q = 0.5 * rho * V**2                          # q = 0.5·ρ·V² [Pa]
            drag_force = trim_results['cd'] * q * S_REF_M2  # D = CD·q·S [N]
            lift_force = trim_results['cl'] * q * S_REF_M2  # L = CL·q·S [N]
            
            return {
                'mach': M,
                'altitude_m': h_m,
                'weight_kg': weight_kg,
                'cd': trim_results['cd'],
                'cl': trim_results['cl'],
                'ld': trim_results['ld'],
                'wing_aoa': trim_results['wing_aoa'],
                'ht_incidence': trim_results['ht_incidence'],
                'drag_force_N': drag_force,
                'lift_force_N': lift_force,
                'dynamic_pressure_Pa': q,
                'density_kgpm3': rho,
                'speed_of_sound_mps': a,
                'true_airspeed_mps': V,
                'temperature_K': T,
                'pressure_Pa': p
            }
            
        except Exception as e:
            print(f"[PYAERO] Error calculating comprehensive aerodynamics: {e}")
            return {}
    
    # ────────────────────────────────────────────────────────────────────
    # Grid Pre-computation and Cache Management
    # ────────────────────────────────────────────────────────────────────
    
    def precompute_drag_grid(self, mach_grid: np.ndarray, H_grid: np.ndarray, weight_kg: float):
        """
        Pre-populate cache with D(M, h, m) over grid.
        
        Grid iteration: For h in H_grid, for M in M_grid, compute D(M,h,m).
        Purpose: Amortize trim computation cost for repeated queries.
        
        Parameters:
            mach_grid: M_grid - Mach sampling points
            H_grid: h_grid [m] - altitude sampling points
            weight_kg: m [kg] - aircraft mass (constant for grid)
        """
        print(f"[AERO] Pre-computing drag grid: {len(mach_grid)}×{len(H_grid)} points")
        
        total_points = len(mach_grid) * len(H_grid)
        computed = 0
        
        for h in H_grid:
            for m in mach_grid:
                self.get_drag(m, h, weight_kg)  # Populates cache
                computed += 1
                
                if computed % 100 == 0:
                    progress = computed / total_points * 100
                    print(f"[AERO] Pre-computation progress: {progress:.1f}% ({computed}/{total_points})")
        
        print("[AERO] Drag grid pre-computation completed")
        stats = self.get_cache_stats()
        print(f"[AERO] Cache statistics: Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.1%}")
    
    def get_cache_stats(self) -> dict:
        """
        Extract cache performance metrics.
        
        Returns:
            dict: {hits, misses, hit_rate, cache_size}
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._drag_cache)
        }
