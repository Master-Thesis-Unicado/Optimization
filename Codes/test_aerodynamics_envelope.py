"""
Aerodynamics Envelope Analysis Tool

This module provides independent testing and analysis of the PyAerodynamicsWrapper
working envelope, including operational limits across Mach number, altitude, and
weight ranges.

The analysis systematically tests aerodynamic calculations across parameter ranges
to identify operational limits and working envelope boundaries.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import aerodynamics wrapper
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from aircraft_config import INITIAL_MASS_KG


class AerodynamicsEnvelopeAnalyzer:
    """
    Analyzer for determining aerodynamics wrapper operational envelope and limits.
    
    This class performs systematic testing of aerodynamic calculations across
    parameter ranges to identify:
    - Working envelope boundaries
    - Maximum operational Mach number
    - Maximum operational altitude
    - Weight-dependent envelope variations
    """
    
    def __init__(self, xml_path: str = None):
        """
        Initialize the analyzer with aerodynamics wrapper.
        
        Args:
            xml_path: Path to aircraft XML configuration. If None, uses default.
        """
        if xml_path is None:
            xml_path = "Aero/aero_data/polar.xml"
        
        print(f"[AERO-ENVELOPE] Initializing aerodynamics wrapper with XML: {xml_path}")
        try:
            self.aero = PyAerodynamicsWrapper(xml_path)
            self.results = {}
        except Exception as e:
            print(f"[AERO-ENVELOPE] ERROR: Failed to initialize aerodynamics wrapper: {e}")
            raise
    
    def test_working_envelope(self,
                              mach_range: np.ndarray = None,
                              altitude_range: np.ndarray = None,
                              weight_range: np.ndarray = None,
                              verbose: bool = True) -> dict:
        """
        Test aerodynamics working envelope across specified parameter ranges.
        
        Args:
            mach_range: Array of Mach numbers to test
            altitude_range: Array of altitudes to test [m]
            weight_range: Array of weights to test [kg]
            verbose: Print progress information
            
        Returns:
            Dictionary containing test results with success/failure status
            for each parameter combination
        """
        if mach_range is None:
            mach_range = np.linspace(0.1, 0.94, 20)
        if altitude_range is None:
            altitude_range = np.linspace(0, 15000, 16)
        if weight_range is None:
            # Test at different weight fractions
            weight_range = np.array([
                INITIAL_MASS_KG * 0.5,  # Light weight
                INITIAL_MASS_KG * 0.75,  # Medium weight
                INITIAL_MASS_KG,  # Initial weight
                INITIAL_MASS_KG * 1.1  # Heavy weight
            ])
        
        if verbose:
            total_tests = len(mach_range) * len(altitude_range) * len(weight_range)
            print(f"[AERO-ENVELOPE] Testing {total_tests} parameter combinations...")
            print(f"  Mach range: {mach_range[0]:.3f} to {mach_range[-1]:.3f} ({len(mach_range)} points)")
            print(f"  Altitude range: {altitude_range[0]:.0f} to {altitude_range[-1]:.0f} m ({len(altitude_range)} points)")
            print(f"  Weight range: {weight_range[0]:.0f} to {weight_range[-1]:.0f} kg ({len(weight_range)} points)")
        
        results = {
            'mach_range': mach_range,
            'altitude_range': altitude_range,
            'weight_range': weight_range,
            'drag_data': np.zeros((len(mach_range), len(altitude_range), len(weight_range))),
            'cl_data': np.zeros((len(mach_range), len(altitude_range), len(weight_range))),
            'ld_data': np.zeros((len(mach_range), len(altitude_range), len(weight_range))),
            'valid_mask': np.zeros((len(mach_range), len(altitude_range), len(weight_range)), dtype=bool),
            'error_mask': np.zeros((len(mach_range), len(altitude_range), len(weight_range)), dtype=bool)
        }
        
        test_count = 0
        valid_count = 0
        error_count = 0
        
        for i, mach in enumerate(mach_range):
            for j, altitude in enumerate(altitude_range):
                for k, weight in enumerate(weight_range):
                    test_count += 1
                    
                    try:
                        drag = self.aero.get_drag(mach, altitude, weight)
                        cl = self.aero.get_cl(mach, altitude, weight)
                        ld = self.aero.get_lift_drag_ratio(mach, altitude, weight)
                        
                        if (drag is not None and np.isfinite(drag) and drag >= 0 and
                            cl is not None and np.isfinite(cl) and
                            ld is not None and np.isfinite(ld) and ld > 0):
                            results['drag_data'][i, j, k] = drag
                            results['cl_data'][i, j, k] = cl
                            results['ld_data'][i, j, k] = ld
                            results['valid_mask'][i, j, k] = True
                            valid_count += 1
                        else:
                            results['drag_data'][i, j, k] = np.nan
                            results['cl_data'][i, j, k] = np.nan
                            results['ld_data'][i, j, k] = np.nan
                            results['valid_mask'][i, j, k] = False
                            results['error_mask'][i, j, k] = True
                            error_count += 1
                    except Exception as e:
                        results['drag_data'][i, j, k] = np.nan
                        results['cl_data'][i, j, k] = np.nan
                        results['ld_data'][i, j, k] = np.nan
                        results['valid_mask'][i, j, k] = False
                        results['error_mask'][i, j, k] = True
                        error_count += 1
                    
                    if verbose and test_count % 100 == 0:
                        progress = test_count / total_tests * 100
                        print(f"  Progress: {progress:.1f}% ({test_count} tests, {valid_count} valid, {error_count} errors)")
        
        if verbose:
            print(f"[AERO-ENVELOPE] Testing complete: {valid_count}/{test_count} valid combinations ({100*valid_count/test_count:.1f}%)")
            print(f"[AERO-ENVELOPE] Errors encountered: {error_count}/{test_count} ({100*error_count/test_count:.1f}%)")
        
        self.results['envelope'] = results
        return results
    
    def find_maximum_mach(self,
                         altitude: float = 0.0,
                         weight_kg: float = None,
                         mach_start: float = 0.1,
                         mach_end: float = 0.94,
                         tolerance: float = 0.001,
                         verbose: bool = True) -> dict:
        """
        Find maximum operational Mach number at specified conditions.
        
        Uses binary search to efficiently find the maximum Mach where
        aerodynamics calculations produce valid results.
        
        Args:
            altitude: Altitude [m]
            weight_kg: Aircraft weight [kg]. If None, uses INITIAL_MASS_KG.
            mach_start: Starting Mach number for search
            mach_end: Ending Mach number for search
            tolerance: Tolerance for binary search convergence
            verbose: Print search progress
            
        Returns:
            Dictionary containing maximum Mach and test details
        """
        if weight_kg is None:
            weight_kg = INITIAL_MASS_KG
        
        if verbose:
            print(f"[MAX-MACH] Finding maximum Mach at altitude={altitude:.0f} m, weight={weight_kg:.0f} kg")
        
        # Binary search for maximum valid Mach
        mach_low = mach_start
        mach_high = mach_end
        max_valid_mach = None
        
        # First, verify that mach_start works
        try:
            test_drag = self.aero.get_drag(mach_start, altitude, weight_kg)
            if test_drag is None or not np.isfinite(test_drag) or test_drag < 0:
                if verbose:
                    print(f"[MAX-MACH] WARNING: Starting Mach {mach_start:.3f} produces invalid drag")
                return {'max_mach': None, 'drag_at_max': None, 'valid': False}
        except Exception:
            if verbose:
                print(f"[MAX-MACH] WARNING: Starting Mach {mach_start:.3f} produces error")
            return {'max_mach': None, 'drag_at_max': None, 'valid': False}
        
        # Binary search
        while mach_high - mach_low > tolerance:
            mach_mid = (mach_low + mach_high) / 2.0
            try:
                drag = self.aero.get_drag(mach_mid, altitude, weight_kg)
                if drag is not None and np.isfinite(drag) and drag >= 0:
                    max_valid_mach = mach_mid
                    mach_low = mach_mid
                else:
                    mach_high = mach_mid
            except Exception:
                mach_high = mach_mid
        
        if max_valid_mach is None:
            max_valid_mach = mach_low
        
        # Get final drag value
        try:
            final_drag = self.aero.get_drag(max_valid_mach, altitude, weight_kg)
            final_cl = self.aero.get_cl(max_valid_mach, altitude, weight_kg)
            final_ld = self.aero.get_lift_drag_ratio(max_valid_mach, altitude, weight_kg)
        except Exception:
            final_drag = None
            final_cl = None
            final_ld = None
        
        result = {
            'max_mach': max_valid_mach,
            'drag_at_max': final_drag,
            'cl_at_max': final_cl,
            'ld_at_max': final_ld,
            'altitude': altitude,
            'weight': weight_kg,
            'valid': True
        }
        
        if verbose:
            print(f"[MAX-MACH] Maximum Mach: {max_valid_mach:.4f}")
            if final_drag is not None:
                print(f"[MAX-MACH] Drag at max Mach: {final_drag:.1f} N")
            if final_ld is not None:
                print(f"[MAX-MACH] L/D at max Mach: {final_ld:.2f}")
        
        self.results['max_mach'] = result
        return result
    
    def find_maximum_altitude(self,
                             mach: float = 0.78,
                             weight_kg: float = None,
                             altitude_start: float = 0.0,
                             altitude_end: float = 20000.0,
                             tolerance: float = 10.0,
                             verbose: bool = True) -> dict:
        """
        Find maximum operational altitude at specified conditions.
        
        Maximum altitude is defined as the altitude where aerodynamics calculations
        still produce valid results. Uses binary search for efficient determination.
        
        Args:
            mach: Mach number
            weight_kg: Aircraft weight [kg]. If None, uses INITIAL_MASS_KG.
            altitude_start: Starting altitude for search [m]
            altitude_end: Maximum altitude to test [m]
            tolerance: Tolerance for binary search convergence [m]
            verbose: Print search progress
            
        Returns:
            Dictionary containing maximum altitude and test details
        """
        if weight_kg is None:
            weight_kg = INITIAL_MASS_KG
        
        if verbose:
            print(f"[MAX-ALTITUDE] Finding maximum altitude at Mach={mach:.3f}, weight={weight_kg:.0f} kg")
        
        # Binary search for maximum valid altitude
        alt_low = altitude_start
        alt_high = altitude_end
        max_valid_altitude = None
        
        # First, verify that altitude_start works
        try:
            test_drag = self.aero.get_drag(mach, altitude_start, weight_kg)
            if test_drag is None or not np.isfinite(test_drag) or test_drag < 0:
                if verbose:
                    print(f"[MAX-ALTITUDE] WARNING: Starting altitude {altitude_start:.0f} m produces invalid drag")
                return {'max_altitude': None, 'drag_at_max': None, 'valid': False}
        except Exception:
            if verbose:
                print(f"[MAX-ALTITUDE] WARNING: Starting altitude {altitude_start:.0f} m produces error")
            return {'max_altitude': None, 'drag_at_max': None, 'valid': False}
        
        # Binary search
        while alt_high - alt_low > tolerance:
            alt_mid = (alt_low + alt_high) / 2.0
            try:
                drag = self.aero.get_drag(mach, alt_mid, weight_kg)
                if drag is not None and np.isfinite(drag) and drag >= 0:
                    max_valid_altitude = alt_mid
                    alt_low = alt_mid
                else:
                    alt_high = alt_mid
            except Exception:
                alt_high = alt_mid
        
        if max_valid_altitude is None:
            max_valid_altitude = alt_low
        
        # Get final drag value
        try:
            final_drag = self.aero.get_drag(mach, max_valid_altitude, weight_kg)
            final_cl = self.aero.get_cl(mach, max_valid_altitude, weight_kg)
            final_ld = self.aero.get_lift_drag_ratio(mach, max_valid_altitude, weight_kg)
        except Exception:
            final_drag = None
            final_cl = None
            final_ld = None
        
        result = {
            'max_altitude': max_valid_altitude,
            'drag_at_max': final_drag,
            'cl_at_max': final_cl,
            'ld_at_max': final_ld,
            'mach': mach,
            'weight': weight_kg,
            'valid': True
        }
        
        if verbose:
            print(f"[MAX-ALTITUDE] Maximum altitude: {max_valid_altitude:.1f} m ({max_valid_altitude/1000:.2f} km)")
            if final_drag is not None:
                print(f"[MAX-ALTITUDE] Drag at max altitude: {final_drag:.1f} N")
            if final_ld is not None:
                print(f"[MAX-ALTITUDE] L/D at max altitude: {final_ld:.2f}")
        
        self.results['max_altitude'] = result
        return result
    
    def find_minimum_mach(self,
                         altitude: float = 0.0,
                         weight_kg: float = None,
                         mach_start: float = 0.1,
                         mach_end: float = 0.5,
                         tolerance: float = 0.001,
                         verbose: bool = True) -> dict:
        """
        Find minimum operational Mach number at specified conditions.
        
        Minimum Mach is typically limited by stall speed. Uses binary search
        to find the minimum Mach where valid calculations are possible.
        
        Args:
            altitude: Altitude [m]
            weight_kg: Aircraft weight [kg]. If None, uses INITIAL_MASS_KG.
            mach_start: Starting Mach number for search (lower bound)
            mach_end: Ending Mach number for search (upper bound)
            tolerance: Tolerance for binary search convergence
            verbose: Print search progress
            
        Returns:
            Dictionary containing minimum Mach and test details
        """
        if weight_kg is None:
            weight_kg = INITIAL_MASS_KG
        
        if verbose:
            print(f"[MIN-MACH] Finding minimum Mach at altitude={altitude:.0f} m, weight={weight_kg:.0f} kg")
        
        # Binary search for minimum valid Mach
        mach_low = mach_start
        mach_high = mach_end
        min_valid_mach = None
        
        # First, verify that mach_end works
        try:
            test_drag = self.aero.get_drag(mach_end, altitude, weight_kg)
            if test_drag is None or not np.isfinite(test_drag) or test_drag < 0:
                if verbose:
                    print(f"[MIN-MACH] WARNING: Upper bound Mach {mach_end:.3f} produces invalid drag")
                return {'min_mach': None, 'drag_at_min': None, 'valid': False}
        except Exception:
            if verbose:
                print(f"[MIN-MACH] WARNING: Upper bound Mach {mach_end:.3f} produces error")
            return {'min_mach': None, 'drag_at_min': None, 'valid': False}
        
        # Binary search
        while mach_high - mach_low > tolerance:
            mach_mid = (mach_low + mach_high) / 2.0
            try:
                drag = self.aero.get_drag(mach_mid, altitude, weight_kg)
                if drag is not None and np.isfinite(drag) and drag >= 0:
                    min_valid_mach = mach_mid
                    mach_high = mach_mid
                else:
                    mach_low = mach_mid
            except Exception:
                mach_low = mach_mid
        
        if min_valid_mach is None:
            min_valid_mach = mach_high
        
        # Get final drag value
        try:
            final_drag = self.aero.get_drag(min_valid_mach, altitude, weight_kg)
            final_cl = self.aero.get_cl(min_valid_mach, altitude, weight_kg)
            final_ld = self.aero.get_lift_drag_ratio(min_valid_mach, altitude, weight_kg)
        except Exception:
            final_drag = None
            final_cl = None
            final_ld = None
        
        result = {
            'min_mach': min_valid_mach,
            'drag_at_min': final_drag,
            'cl_at_min': final_cl,
            'ld_at_min': final_ld,
            'altitude': altitude,
            'weight': weight_kg,
            'valid': True
        }
        
        if verbose:
            print(f"[MIN-MACH] Minimum Mach: {min_valid_mach:.4f}")
            if final_drag is not None:
                print(f"[MIN-MACH] Drag at min Mach: {final_drag:.1f} N")
            if final_cl is not None:
                print(f"[MIN-MACH] CL at min Mach: {final_cl:.3f}")
        
        self.results['min_mach'] = result
        return result
    
    def analyze_envelope_at_multiple_conditions(self,
                                               mach_values: list = None,
                                               weight_values: list = None,
                                               verbose: bool = True) -> dict:
        """
        Analyze maximum altitude at multiple Mach and weight combinations.
        
        Args:
            mach_values: List of Mach numbers to test
            weight_values: List of weights to test [kg]
            verbose: Print progress information
            
        Returns:
            Dictionary containing maximum altitude results for each condition
        """
        if mach_values is None:
            mach_values = [0.3, 0.5, 0.7, 0.78, 0.85, 0.9]
        if weight_values is None:
            weight_values = [
                INITIAL_MASS_KG * 0.5,
                INITIAL_MASS_KG * 0.75,
                INITIAL_MASS_KG,
                INITIAL_MASS_KG * 1.1
            ]
        
        if verbose:
            print(f"[MULTI-ALTITUDE] Analyzing maximum altitude at {len(mach_values)}×{len(weight_values)} conditions")
        
        results = {}
        for mach in mach_values:
            for weight in weight_values:
                key = f"mach_{mach:.3f}_weight_{weight:.0f}"
                results[key] = self.find_maximum_altitude(mach=mach, weight_kg=weight, verbose=False)
        
        if verbose:
            print("\n[MULTI-ALTITUDE] Results summary:")
            print("Mach  | Weight [kg] | Max Altitude [m] | Drag [N] | L/D")
            print("-" * 70)
            for key, result in results.items():
                if result['valid']:
                    mach = result['mach']
                    weight = result['weight']
                    altitude = result['max_altitude']
                    drag = result['drag_at_max']
                    ld = result['ld_at_max']
                    print(f"{mach:5.3f} | {weight:11.0f} | {altitude:15.1f} | {drag:8.1f} | {ld:5.2f}" if drag is not None else f"{mach:5.3f} | {weight:11.0f} | {altitude:15.1f} | {'N/A':8s} | {'N/A':5s}")
                else:
                    # Extract mach and weight from key
                    parts = key.split('_')
                    mach = float(parts[1])
                    weight = float(parts[3])
                    print(f"{mach:5.3f} | {weight:11.0f} | {'INVALID':15s} | {'N/A':8s} | {'N/A':5s}")
        
        self.results['multi_altitude'] = results
        return results
    
    def plot_envelope(self, weight_idx: int = None, save_path: str = None):
        """
        Plot working envelope visualization.
        
        Args:
            weight_idx: Index of weight to plot (if None, plots multiple)
            save_path: Path to save figure (if None, displays interactively)
        """
        if 'envelope' not in self.results:
            print("[PLOT] No envelope data available. Run test_working_envelope() first.")
            return
        
        envelope_data = self.results['envelope']
        mach_range = envelope_data['mach_range']
        altitude_range = envelope_data['altitude_range']
        weight_range = envelope_data['weight_range']
        valid_mask = envelope_data['valid_mask']
        drag_data = envelope_data['drag_data']
        
        if weight_idx is None:
            # Plot multiple weights
            n_weights = min(3, len(weight_range))
            weight_indices = np.linspace(0, len(weight_range)-1, n_weights, dtype=int)
            fig, axes = plt.subplots(1, n_weights, figsize=(6*n_weights, 5))
            if n_weights == 1:
                axes = [axes]
        else:
            weight_indices = [weight_idx]
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes = [axes]
        
        for idx, (ax, weight_i) in enumerate(zip(axes, weight_indices)):
            weight = weight_range[weight_i]
            
            # Create 2D plot: Mach vs Altitude
            M_grid, H_grid = np.meshgrid(mach_range, altitude_range)
            valid_2d = valid_mask[:, :, weight_i].T
            drag_2d = drag_data[:, :, weight_i].T
            
            # Plot valid region
            ax.contourf(M_grid, H_grid, valid_2d.astype(float), levels=[0.5, 1.5],
                       colors=['lightcoral', 'lightgreen'], alpha=0.6)
            
            # Plot drag contours
            valid_drag = np.ma.masked_where(~valid_2d, drag_2d)
            if np.any(valid_2d):
                contour = ax.contour(M_grid, H_grid, valid_drag, levels=10, colors='black', alpha=0.3, linewidths=0.5)
                ax.clabel(contour, inline=True, fontsize=8)
            
            ax.set_xlabel('Mach Number')
            ax.set_ylabel('Altitude [m]')
            ax.set_title(f'Aerodynamics Envelope (Weight = {weight:.0f} kg)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[PLOT] Figure saved to {save_path}")
        else:
            plt.show()
    
    def list_available_parameters(self, 
                                  mach: float = 0.78,
                                  altitude_m: float = 10500.0,
                                  weight_kg: float = None,
                                  verbose: bool = True) -> dict:
        """
        List all available parameters that can be extracted from the aerodynamics library.
        
        Args:
            mach: Mach number for test calculation
            altitude_m: Altitude in meters for test calculation
            weight_kg: Aircraft weight in kg. If None, uses INITIAL_MASS_KG.
            verbose: Print detailed parameter information
            
        Returns:
            Dictionary containing all available parameters
        """
        if weight_kg is None:
            weight_kg = INITIAL_MASS_KG
        
        if verbose:
            print(f"\n[PARAMETERS] Listing available parameters at:")
            print(f"  Mach: {mach:.3f}")
            print(f"  Altitude: {altitude_m:.0f} m")
            print(f"  Weight: {weight_kg:.0f} kg")
            print("-" * 60)
        
        parameters = {}
        
        # Test basic methods
        try:
            drag = self.aero.get_drag(mach, altitude_m, weight_kg)
            parameters['drag_force_N'] = drag
            if verbose:
                print(f"✓ Drag Force: {drag:.1f} N")
        except Exception as e:
            parameters['drag_force_N'] = None
            if verbose:
                print(f"✗ Drag Force: Error - {e}")
        
        try:
            cl = self.aero.get_cl(mach, altitude_m, weight_kg)
            parameters['lift_coefficient_CL'] = cl
            if verbose:
                print(f"✓ Lift Coefficient (CL): {cl:.4f}")
        except Exception as e:
            parameters['lift_coefficient_CL'] = None
            if verbose:
                print(f"✗ Lift Coefficient (CL): Error - {e}")
        
        try:
            ld = self.aero.get_lift_drag_ratio(mach, altitude_m, weight_kg)
            parameters['lift_drag_ratio_LD'] = ld
            if verbose:
                print(f"✓ Lift-to-Drag Ratio (L/D): {ld:.2f}")
        except Exception as e:
            parameters['lift_drag_ratio_LD'] = None
            if verbose:
                print(f"✗ Lift-to-Drag Ratio (L/D): Error - {e}")
        
        try:
            cd = self.aero.get_drag_coefficient(mach, altitude_m, weight_kg)
            parameters['drag_coefficient_CD'] = cd
            if verbose:
                print(f"✓ Drag Coefficient (CD): {cd:.4f}")
        except Exception as e:
            parameters['drag_coefficient_CD'] = None
            if verbose:
                print(f"✗ Drag Coefficient (CD): Error - {e}")
        
        # Test comprehensive aerodynamics method
        try:
            comprehensive = self.aero.get_comprehensive_aerodynamics(mach, altitude_m, weight_kg)
            if comprehensive:
                parameters['comprehensive_data'] = comprehensive
                if verbose:
                    print(f"\n✓ Comprehensive Aerodynamics Data:")
                    for key, value in comprehensive.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.4g}")
                        else:
                            print(f"    {key}: {value}")
            else:
                parameters['comprehensive_data'] = None
                if verbose:
                    print(f"✗ Comprehensive Aerodynamics: No data returned")
        except Exception as e:
            parameters['comprehensive_data'] = None
            if verbose:
                print(f"✗ Comprehensive Aerodynamics: Error - {e}")
        
        # Get wrapper properties
        try:
            parameters['mach_grid'] = self.aero.mach_grid.tolist()
            if verbose:
                print(f"\n✓ Mach Grid: {len(parameters['mach_grid'])} points from {parameters['mach_grid'][0]:.3f} to {parameters['mach_grid'][-1]:.3f}")
        except Exception as e:
            parameters['mach_grid'] = None
            if verbose:
                print(f"✗ Mach Grid: Error - {e}")
        
        try:
            parameters['altitude_grid_m'] = self.aero.alt_grid_m.tolist()
            if verbose:
                print(f"✓ Altitude Grid: {len(parameters['altitude_grid_m'])} points from {parameters['altitude_grid_m'][0]:.0f} to {parameters['altitude_grid_m'][-1]:.0f} m")
        except Exception as e:
            parameters['altitude_grid_m'] = None
            if verbose:
                print(f"✗ Altitude Grid: Error - {e}")
        
        try:
            parameters['cl_max'] = self.aero.cl_max
            if verbose:
                print(f"✓ Maximum Lift Coefficient (CL_MAX): {parameters['cl_max']:.2f}")
        except Exception as e:
            parameters['cl_max'] = None
            if verbose:
                print(f"✗ Maximum Lift Coefficient (CL_MAX): Error - {e}")
        
        try:
            parameters['gravity_constant_ms2'] = self.aero.G_C
            if verbose:
                print(f"✓ Gravity Constant (G_C): {parameters['gravity_constant_ms2']:.5f} m/s²")
        except Exception as e:
            parameters['gravity_constant_ms2'] = None
            if verbose:
                print(f"✗ Gravity Constant (G_C): Error - {e}")
        
        # Get wrapper parameters
        try:
            params = self.aero.params
            parameters['wrapper_params'] = params
            if verbose:
                print(f"\n✓ Wrapper Parameters:")
                for key, value in params.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4g}")
                    else:
                        print(f"    {key}: {value}")
        except Exception as e:
            parameters['wrapper_params'] = None
            if verbose:
                print(f"✗ Wrapper Parameters: Error - {e}")
        
        # Try to access internal aircraft object properties (if accessible)
        if verbose:
            print(f"\n[PARAMETERS] Attempting to access internal aircraft properties...")
        
        try:
            if hasattr(self.aero, 'aircraft') and self.aero.aircraft is not None:
                aircraft = self.aero.aircraft
                if verbose:
                    print(f"✓ Aircraft object available")
                
                # Try to get aircraft settings
                if hasattr(aircraft, 'settings'):
                    settings = aircraft.settings
                    if verbose:
                        print(f"✓ Aircraft settings available")
                    
                    # Try to get reference wing angle
                    if hasattr(settings, 'reference_wing_angle'):
                        parameters['reference_wing_angle_rad'] = settings.reference_wing_angle
                        if verbose:
                            print(f"    Reference Wing Angle: {settings.reference_wing_angle:.4f} rad ({np.degrees(settings.reference_wing_angle):.2f} deg)")
                    
                    # Try to get adjustable surface angle
                    if hasattr(settings, 'adjustable_surface_angle'):
                        parameters['adjustable_surface_angle_rad'] = settings.adjustable_surface_angle
                        if verbose:
                            print(f"    Adjustable Surface Angle: {settings.adjustable_surface_angle:.4f} rad ({np.degrees(settings.adjustable_surface_angle):.2f} deg)")
        except Exception as e:
            if verbose:
                print(f"✗ Internal aircraft properties: Error - {e}")
        
        # Extract atmospheric data from comprehensive results
        if 'comprehensive_data' in parameters and parameters['comprehensive_data']:
            comp_data = parameters['comprehensive_data']
            atmospheric_params = {
                'temperature_K': comp_data.get('temperature_K'),
                'pressure_Pa': comp_data.get('pressure_Pa'),
                'density_kgpm3': comp_data.get('density_kgpm3'),
                'speed_of_sound_mps': comp_data.get('speed_of_sound_mps'),
                'true_airspeed_mps': comp_data.get('true_airspeed_mps'),
                'dynamic_pressure_Pa': comp_data.get('dynamic_pressure_Pa')
            }
            parameters['atmospheric_data'] = atmospheric_params
            
            if verbose:
                print(f"\n[ATMOSPHERIC DATA] Available atmospheric parameters:")
                for key, value in atmospheric_params.items():
                    if value is not None:
                        print(f"  ✓ {key:25s}: {value:12.4g}")
                    else:
                        print(f"  ✗ {key:25s}: Not available")
        
        if verbose:
            print("-" * 60)
            print(f"[PARAMETERS] Total parameters extracted: {len([k for k, v in parameters.items() if v is not None])}")
        
        self.results['available_parameters'] = parameters
        return parameters
    
    def print_summary(self):
        """Print summary of all analysis results."""
        print("\n" + "="*60)
        print("AERODYNAMICS ENVELOPE ANALYSIS SUMMARY")
        print("="*60)
        
        if 'max_mach' in self.results:
            result = self.results['max_mach']
            if result['valid']:
                print(f"\nMaximum Mach Number:")
                print(f"  Value: {result['max_mach']:.4f}")
                print(f"  Conditions: Altitude={result['altitude']:.0f} m, Weight={result['weight']:.0f} kg")
                if result['drag_at_max'] is not None:
                    print(f"  Drag at max: {result['drag_at_max']:.1f} N")
                if result['ld_at_max'] is not None:
                    print(f"  L/D at max: {result['ld_at_max']:.2f}")
        
        if 'min_mach' in self.results:
            result = self.results['min_mach']
            if result['valid']:
                print(f"\nMinimum Mach Number:")
                print(f"  Value: {result['min_mach']:.4f}")
                print(f"  Conditions: Altitude={result['altitude']:.0f} m, Weight={result['weight']:.0f} kg")
                if result['drag_at_min'] is not None:
                    print(f"  Drag at min: {result['drag_at_min']:.1f} N")
                if result['cl_at_min'] is not None:
                    print(f"  CL at min: {result['cl_at_min']:.3f}")
        
        if 'max_altitude' in self.results:
            result = self.results['max_altitude']
            if result['valid']:
                print(f"\nMaximum Operational Altitude:")
                print(f"  Altitude: {result['max_altitude']:.1f} m ({result['max_altitude']/1000:.2f} km)")
                print(f"  Conditions: Mach={result['mach']:.3f}, Weight={result['weight']:.0f} kg")
                if result['drag_at_max'] is not None:
                    print(f"  Drag at max: {result['drag_at_max']:.1f} N")
                if result['ld_at_max'] is not None:
                    print(f"  L/D at max: {result['ld_at_max']:.2f}")
        
        if 'envelope' in self.results:
            envelope = self.results['envelope']
            total_tests = np.prod(envelope['valid_mask'].shape)
            valid_tests = np.sum(envelope['valid_mask'])
            error_tests = np.sum(envelope['error_mask'])
            print(f"\nWorking Envelope Statistics:")
            print(f"  Valid combinations: {valid_tests}/{total_tests} ({100*valid_tests/total_tests:.1f}%)")
            print(f"  Error combinations: {error_tests}/{total_tests} ({100*error_tests/total_tests:.1f}%)")
            print(f"  Mach range: {envelope['mach_range'][0]:.3f} to {envelope['mach_range'][-1]:.3f}")
            print(f"  Altitude range: {envelope['altitude_range'][0]:.0f} to {envelope['altitude_range'][-1]:.0f} m")
            print(f"  Weight range: {envelope['weight_range'][0]:.0f} to {envelope['weight_range'][-1]:.0f} kg")
        
        cache_stats = self.aero.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"  Hits: {cache_stats['hits']}")
        print(f"  Misses: {cache_stats['misses']}")
        print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
        print(f"  Cache size: {cache_stats['cache_size']}")
        
        print("="*60 + "\n")


def main():
    """Main function to run aerodynamics envelope analysis."""
    print("="*60)
    print("AERODYNAMICS ENVELOPE ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = AerodynamicsEnvelopeAnalyzer()
    
    # Test 0: List all available parameters
    print("\n[TEST 0] Available Parameters from Aerodynamics Library")
    print("="*60)
    available_params = analyzer.list_available_parameters(
        mach=0.78,
        altitude_m=10500.0,
        weight_kg=INITIAL_MASS_KG,
        verbose=True
    )
    
    # Print comprehensive data summary if available
    if 'comprehensive_data' in available_params and available_params['comprehensive_data']:
        print("\n[PARAMETERS] Comprehensive Aerodynamics Data Summary:")
        comp_data = available_params['comprehensive_data']
        print(f"  Available keys: {list(comp_data.keys())}")
        print(f"  Total parameters: {len(comp_data)}")
        
        # Show example data extraction
        print("\n[EXAMPLE] Example data extraction at Mach=0.78, Altitude=10500m:")
        print("-" * 60)
        for key, value in comp_data.items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:12.4g}")
            else:
                print(f"  {key:25s}: {value}")
    
    # Demonstrate data extraction at multiple conditions
    print("\n[TEST 0.5] Data Extraction Examples at Different Conditions")
    print("="*60)
    test_conditions = [
        (0.3, 0.0, INITIAL_MASS_KG, "Low Mach, Sea Level"),
        (0.78, 10500.0, INITIAL_MASS_KG, "Cruise Conditions"),
        (0.85, 12000.0, INITIAL_MASS_KG * 0.9, "High Mach, High Altitude, Light Weight"),
    ]
    
    for mach, alt, weight, desc in test_conditions:
        print(f"\n[{desc}]")
        print(f"  Mach: {mach:.3f}, Altitude: {alt:.0f} m, Weight: {weight:.0f} kg")
        try:
            comp_data = analyzer.aero.get_comprehensive_aerodynamics(mach, alt, weight)
            if comp_data:
                print(f"  ✓ Successfully extracted {len(comp_data)} parameters:")
                print(f"\n  Aerodynamic Coefficients:")
                print(f"    - CD: {comp_data.get('cd', 'N/A'):.4f}")
                print(f"    - CL: {comp_data.get('cl', 'N/A'):.4f}")
                print(f"    - L/D: {comp_data.get('ld', 'N/A'):.2f}")
                print(f"\n  Forces:")
                print(f"    - Drag Force: {comp_data.get('drag_force_N', 'N/A'):.1f} N")
                print(f"    - Lift Force: {comp_data.get('lift_force_N', 'N/A'):.1f} N")
                print(f"\n  Trim Angles:")
                print(f"    - Wing AoA: {comp_data.get('wing_aoa', 'N/A'):.4f} rad ({np.degrees(comp_data.get('wing_aoa', 0)):.2f} deg)")
                print(f"    - HT Incidence: {comp_data.get('ht_incidence', 'N/A'):.4f} rad ({np.degrees(comp_data.get('ht_incidence', 0)):.2f} deg)")
                print(f"\n  Atmospheric Properties:")
                print(f"    - Temperature: {comp_data.get('temperature_K', 'N/A'):.2f} K ({comp_data.get('temperature_K', 0) - 273.15:.2f} °C)")
                print(f"    - Pressure: {comp_data.get('pressure_Pa', 'N/A'):.1f} Pa ({comp_data.get('pressure_Pa', 0)/100:.1f} hPa)")
                print(f"    - Density: {comp_data.get('density_kgpm3', 'N/A'):.4f} kg/m³")
                print(f"    - Speed of Sound: {comp_data.get('speed_of_sound_mps', 'N/A'):.2f} m/s")
                print(f"    - True Airspeed: {comp_data.get('true_airspeed_mps', 'N/A'):.2f} m/s")
                print(f"    - Dynamic Pressure: {comp_data.get('dynamic_pressure_Pa', 'N/A'):.1f} Pa")
            else:
                print(f"  ✗ No data returned")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Test 1: Find maximum Mach at sea level, initial weight
    print("\n[TEST 1] Maximum Mach Number Analysis")
    print("-" * 60)
    max_mach_result = analyzer.find_maximum_mach(altitude=0.0, weight_kg=INITIAL_MASS_KG, verbose=True)
    
    # Test 2: Find maximum Mach at cruise altitude
    print("\n[TEST 2] Maximum Mach at Cruise Altitude")
    print("-" * 60)
    max_mach_cruise = analyzer.find_maximum_mach(altitude=10500.0, weight_kg=INITIAL_MASS_KG, verbose=True)
    
    # Test 3: Find minimum Mach at sea level
    print("\n[TEST 3] Minimum Mach Number Analysis")
    print("-" * 60)
    min_mach_result = analyzer.find_minimum_mach(altitude=0.0, weight_kg=INITIAL_MASS_KG, verbose=True)
    
    # Test 4: Find maximum altitude at cruise Mach
    print("\n[TEST 4] Maximum Altitude Analysis")
    print("-" * 60)
    max_altitude_result = analyzer.find_maximum_altitude(mach=0.78, weight_kg=INITIAL_MASS_KG, verbose=True)
    
    # Test 5: Find maximum altitude at different Mach numbers
    print("\n[TEST 5] Maximum Altitude at Multiple Mach Numbers")
    print("-" * 60)
    analyzer.find_maximum_altitude(mach=0.5, weight_kg=INITIAL_MASS_KG, verbose=True)
    analyzer.find_maximum_altitude(mach=0.7, weight_kg=INITIAL_MASS_KG, verbose=True)
    analyzer.find_maximum_altitude(mach=0.85, weight_kg=INITIAL_MASS_KG, verbose=True)
    analyzer.find_maximum_altitude(mach=0.9, weight_kg=INITIAL_MASS_KG, verbose=True)
    
    # Test 6: Multi-condition altitude analysis
    print("\n[TEST 6] Multi-Condition Altitude Analysis")
    print("-" * 60)
    analyzer.analyze_envelope_at_multiple_conditions(
        mach_values=[0.3, 0.5, 0.7, 0.78, 0.85, 0.9],
        weight_values=[INITIAL_MASS_KG * 0.5, INITIAL_MASS_KG * 0.75, INITIAL_MASS_KG, INITIAL_MASS_KG * 1.1],
        verbose=True
    )
    
    # Test 7: Working envelope test (coarse grid for speed)
    print("\n[TEST 7] Working Envelope Grid Test")
    print("-" * 60)
    mach_test = np.linspace(0.2, 0.94, 15)
    alt_test = np.linspace(0, 15000, 16)
    weight_test = np.array([
        INITIAL_MASS_KG * 0.5,
        INITIAL_MASS_KG * 0.75,
        INITIAL_MASS_KG,
        INITIAL_MASS_KG * 1.1
    ])
    analyzer.test_working_envelope(
        mach_range=mach_test,
        altitude_range=alt_test,
        weight_range=weight_test,
        verbose=True
    )
    
    # Plot results
    print("\n[PLOT] Generating envelope visualization...")
    analyzer.plot_envelope(weight_idx=None, save_path="aerodynamics_envelope.png")
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()

