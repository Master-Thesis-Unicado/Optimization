"""
PyAerodynamics Wrapper for Mission Analysis

This module provides a wrapper class that integrates the pyaerodynamics library
with the existing mission analysis system, replacing Excel-based aerodynamic data
with real-time calculations from the pyaerodynamics library.

The wrapper maintains compatibility with the existing AeroTables interface while
providing dynamic aerodynamic calculations based on current flight conditions.

The module uses pathlib to dynamically locate the pyaerodynamics library relative
to this file's location, ensuring robust operation regardless of the current
working directory or import location.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Add the correct path to the pyaerodynamics module relative to this file
_current_file = Path(__file__).resolve()
# Since this file is in Codes/, go up one level to reach root, then into Aero
_root_dir = _current_file.parent.parent  # Go from Codes/ to root
_pyaerodynamics_release_path = _root_dir / 'Aero' / 'pyaerodynamics' / 'pyaerodynamics' / 'Release'
_pyaerodynamics_package_path = _root_dir / 'Aero' / 'pyaerodynamics'

# Add Release folder to path first (for DLL dependencies and .pyd module)
if str(_pyaerodynamics_release_path) not in sys.path:
    sys.path.insert(0, str(_pyaerodynamics_release_path))
# Add package parent directory to path (for package imports)
if str(_pyaerodynamics_package_path) not in sys.path:
    sys.path.insert(0, str(_pyaerodynamics_package_path))

# Load the .pyd module directly from Release directory
import importlib.util
_pyd_file = _pyaerodynamics_release_path / 'pyaerodynamics.cp311-win_amd64.pyd'
if _pyd_file.exists():
    # Load the compiled .pyd file directly as a module
    # The module name must match the PyInit function in the compiled module
    spec = importlib.util.spec_from_file_location("pyaerodynamics", _pyd_file)
    _pyaerodynamics_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_pyaerodynamics_core)
    Aircraft = _pyaerodynamics_core.Aircraft
    Flight_Condition = _pyaerodynamics_core.Flight_Condition
else:
    # Fallback: try regular import (in case module is installed)
    from pyaerodynamics import Aircraft, Flight_Condition

# Import atmospheric functions
from atmosphere import Atmosphere
from aircraft_config import S_REF_M2, M_MMO

# Get gravity constant from Atmosphere class (standard gravity)
G_C = Atmosphere.G_C  # Standard gravity [m/s²] = 9.80665

@dataclass
class FlightState:
    """Current flight state for aerodynamic calculations."""
    altitude_m: float
    mach: float
    weight_kg: float
    cog_location: list = None
    
    def __post_init__(self):
        if self.cog_location is None:
            self.cog_location = [14.0, 0.0, 0.0]  # Default CoG location

class PyAerodynamicsWrapper:
    """
    Wrapper class that replaces AeroTables with pyaerodynamics library integration.
    
    This class provides the same interface as AeroTables but uses the pyaerodynamics
    library for real-time aerodynamic calculations instead of Excel data lookup.
    
    Key Features:
    - Dynamic aerodynamic calculations based on current flight conditions
    - Maintains compatibility with existing AeroTables interface
    - Real-time trim analysis and coefficient calculations
    - Caching system for performance optimization
    - Weight-dependent drag calculations
    """
    
    def __init__(self, xml_path: str = "Aero/aero_data/polar.xml"):
        """
        Initialize the pyaerodynamics wrapper.
        
        Args:
            xml_path: Path to the aircraft configuration XML file
        """
        self.xml_path = xml_path
        self.aircraft = None
        self._drag_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize aircraft from XML
        self._initialize_aircraft()
        
        # Set up default grids for compatibility
        self._setup_default_grids()
        
        # Aircraft parameters (extracted from pyaerodynamics or defaults)
        # Set CL_MAX to a reasonable default (typical for commercial aircraft)
        cl_max_value = 1.4  # Typical CL_MAX for commercial aircraft
        
        self.params = {
            'S_REF_M2': S_REF_M2,
            'M_MMO': M_MMO,
            'CL_MAX': cl_max_value,
        }
        
        # Also set the global CL_MAX for compatibility
        import aircraft_config
        aircraft_config.CL_MAX = cl_max_value
    
    def _initialize_aircraft(self):
        """Initialize the aircraft object from XML configuration."""
        try:
            self.aircraft = Aircraft(self.xml_path)
            print(f"[PYAERO] Aircraft loaded from {self.xml_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load aircraft from {self.xml_path}: {e}")
    
    def _setup_default_grids(self):
        """Set up default Mach and altitude grids for compatibility."""
        # Mach grid (typical range for mission analysis)
        self._M = np.linspace(0.2, 0.9, 15)
        
        # Altitude grid (typical range for mission analysis)
        self._H = np.linspace(0, 12000, 16)
    
    @property
    def mach_grid(self) -> np.ndarray:
        """Get Mach number grid for compatibility."""
        return self._M
    
    @property
    def alt_grid_m(self) -> np.ndarray:
        """Get altitude grid in meters for compatibility."""
        return self._H
    
    @property
    def cl_max(self) -> float:
        """Get maximum lift coefficient for compatibility."""
        return self.params['CL_MAX']
    
    @property
    def G_C(self) -> float:
        """Get standard gravity constant [m/s²] from Atmosphere class."""
        return Atmosphere.G_C
    
    def _create_flight_condition(self, altitude_m: float, mach: float) -> Flight_Condition:
        """Create a Flight_Condition object for the given altitude and Mach."""
        return Flight_Condition(altitude_m, mach)
    
    def _perform_trim_analysis(self, flight_state: FlightState) -> Dict[str, Any]:
        """
        Perform trim analysis for the given flight state.
        
        Args:
            flight_state: Current flight state
            
        Returns:
            Dictionary containing trim results and aerodynamic coefficients
        """
        # Create flight condition
        conditions = self._create_flight_condition(flight_state.altitude_m, flight_state.mach)
        
        # Configure aircraft settings
        self.aircraft.change_settings("main_wing", "horizontal_stabiliser", "linear")
        
        # Calculate weight force (N)
        weight_force = flight_state.weight_kg * G_C
        
        # Perform linearized trim
        self.aircraft.linearized_trim(conditions, weight_force, flight_state.cog_location)
        
        # Extract results
        wing_aoa = self.aircraft.settings.reference_wing_angle
        ht_incidence = self.aircraft.settings.adjustable_surface_angle
        
        # Calculate aerodynamic coefficients
        cd = self.aircraft.get_CD(conditions, weight_force)
        cl = self.aircraft.get_CL(conditions, weight_force)
        ld = self.aircraft.get_CL_CD(conditions, weight_force, flight_state.cog_location)
        
        return {
            'wing_aoa': wing_aoa,
            'ht_incidence': ht_incidence,
            'cd': cd,
            'cl': cl,
            'ld': ld,
            'conditions': conditions,
            'weight_force': weight_force
        }
    
    def get_drag(self, M: float, h_m: float, weight_kg: float) -> float:
        """
        Get drag force in Newtons for given Mach, altitude, and weight.
        
        Args:
            M: Mach number
            h_m: Altitude in meters
            weight_kg: Aircraft weight in kg
            
        Returns:
            Drag force in Newtons
        """
        # Check cache first
        cache_key = (round(M, 3), round(h_m, 1), round(weight_kg, 1))
        if cache_key in self._drag_cache:
            self._cache_hits += 1
            return self._drag_cache[cache_key]
        
        try:
            # Create flight state
            flight_state = FlightState(
                altitude_m=h_m,
                mach=M,
                weight_kg=weight_kg
            )
            
            # Perform trim analysis
            trim_results = self._perform_trim_analysis(flight_state)
            
            # Calculate drag force from coefficient
            # D = CD * 0.5 * rho * V^2 * S_ref
            conditions = trim_results['conditions']
            cd = trim_results['cd']
            
            # Get atmospheric properties
            atm = Atmosphere()
            flight_level = h_m / 0.3048
            T, p, rho = atm.calculate_atmospheric_properties(flight_level)
            a = atm.get_speed_of_sound(h_m)
            V = M * a
            
            # Calculate dynamic pressure
            q = 0.5 * rho * V**2
            
            # Calculate drag force
            drag_force = cd * q * S_REF_M2
            
            # Cache result
            self._drag_cache[cache_key] = drag_force
            self._cache_misses += 1
            
            return drag_force
            
        except Exception as e:
            print(f"[PYAERO] Error calculating drag: {e}")
            return 0.0
    
    def get_cl(self, M: float, h_m: float, weight_kg: float) -> float:
        """
        Get lift coefficient for given Mach, altitude, and weight.
        
        Args:
            M: Mach number
            h_m: Altitude in meters
            weight_kg: Aircraft weight in kg
            
        Returns:
            Lift coefficient
        """
        try:
            # Create flight state
            flight_state = FlightState(
                altitude_m=h_m,
                mach=M,
                weight_kg=weight_kg
            )
            
            # Perform trim analysis
            trim_results = self._perform_trim_analysis(flight_state)
            
            return trim_results['cl']
            
        except Exception as e:
            print(f"[PYAERO] Error calculating CL: {e}")
            return 0.0
    
    def get_lift_drag_ratio(self, M: float, h_m: float, weight_kg: float) -> float:
        """
        Get lift-to-drag ratio for given Mach, altitude, and weight.
        
        Args:
            M: Mach number
            h_m: Altitude in meters
            weight_kg: Aircraft weight in kg
            
        Returns:
            Lift-to-drag ratio
        """
        try:
            # Create flight state
            flight_state = FlightState(
                altitude_m=h_m,
                mach=M,
                weight_kg=weight_kg
            )
            
            # Perform trim analysis
            trim_results = self._perform_trim_analysis(flight_state)
            
            return trim_results['ld']
            
        except Exception as e:
            print(f"[PYAERO] Error calculating L/D: {e}")
            return 0.0
    
    def get_drag_coefficient(self, M: float, h_m: float, weight_kg: float) -> float:
        """
        Get drag coefficient for given Mach, altitude, and weight.
        
        Args:
            M: Mach number
            h_m: Altitude in meters
            weight_kg: Aircraft weight in kg
            
        Returns:
            Drag coefficient
        """
        try:
            # Create flight state
            flight_state = FlightState(
                altitude_m=h_m,
                mach=M,
                weight_kg=weight_kg
            )
            
            # Perform trim analysis
            trim_results = self._perform_trim_analysis(flight_state)
            
            return trim_results['cd']
            
        except Exception as e:
            print(f"[PYAERO] Error calculating CD: {e}")
            return 0.0
    
    def get_comprehensive_aerodynamics(self, M: float, h_m: float, weight_kg: float) -> Dict[str, Any]:
        """
        Get comprehensive aerodynamic data for given flight conditions.
        
        Args:
            M: Mach number
            h_m: Altitude in meters
            weight_kg: Aircraft weight in kg
            
        Returns:
            Dictionary containing all aerodynamic data
        """
        try:
            # Create flight state
            flight_state = FlightState(
                altitude_m=h_m,
                mach=M,
                weight_kg=weight_kg
            )
            
            # Perform trim analysis
            trim_results = self._perform_trim_analysis(flight_state)
            
            # Get atmospheric properties
            atm = Atmosphere()
            flight_level = h_m / 0.3048
            T, p, rho = atm.calculate_atmospheric_properties(flight_level)
            a = atm.get_speed_of_sound(h_m)
            V = M * a
            
            # Calculate forces
            q = 0.5 * rho * V**2
            drag_force = trim_results['cd'] * q * S_REF_M2
            lift_force = trim_results['cl'] * q * S_REF_M2
            
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
    
    def precompute_drag_grid(self, M_grid: np.ndarray, H_grid: np.ndarray, weight_kg: float):
        """
        Pre-compute drag values for a grid of Mach and altitude values.
        
        Args:
            M_grid: Array of Mach numbers
            H_grid: Array of altitudes in meters
            weight_kg: Aircraft weight in kg
        """
        print(f"[PYAERO] Pre-computing drag grid: {len(M_grid)}×{len(H_grid)} points")
        
        total_points = len(M_grid) * len(H_grid)
        computed = 0
        
        for h in H_grid:
            for m in M_grid:
                # This will populate the cache
                self.get_drag(m, h, weight_kg)
                computed += 1
                
                if computed % 100 == 0:
                    progress = computed / total_points * 100
                    print(f"[PYAERO] Progress: {progress:.1f}% ({computed}/{total_points})")
        
        print(f"[PYAERO] Pre-computation completed")
        stats = self.get_cache_stats()
        print(f"[PYAERO] Cache stats: Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.1%}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._drag_cache)
        }
    
    def clear_cache(self):
        """Clear the drag calculation cache."""
        self._drag_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        print("[PYAERO] Cache cleared")
