"""
Engine Envelope Analysis Tool

This module provides independent testing and analysis of the EngineWrapper
working envelope, including maximum Mach number and maximum service ceiling
altitude determination.

The analysis systematically tests engine performance across parameter ranges
to identify operational limits and working envelope boundaries.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import engine wrapper
from pyengine_wrapper import EngineWrapper
from aircraft_config import ENGINE_STUB_PATH


class EngineEnvelopeAnalyzer:
    """
    Analyzer for determining engine operational envelope and limits.
    
    This class performs systematic testing of engine performance across
    parameter ranges to identify:
    - Working envelope boundaries
    - Maximum operational Mach number
    - Maximum service ceiling altitude
    """
    
    def __init__(self, stub_path: str = None):
        """
        Initialize the analyzer with engine wrapper.
        
        Args:
            stub_path: Path to engine stub. If None, uses default from aircraft_config.
        """
        if stub_path is None:
            stub_path = ENGINE_STUB_PATH
        
        print(f"[ENVELOPE] Initializing engine wrapper with stub: {stub_path}")
        self.engine = EngineWrapper(stub_path)
        self.results = {}
        
    def test_working_envelope(self, 
                              lever_range: np.ndarray = None,
                              mach_range: np.ndarray = None,
                              altitude_range: np.ndarray = None,
                              verbose: bool = True) -> dict:
        """
        Test engine working envelope across specified parameter ranges.
        
        Args:
            lever_range: Array of lever positions to test (0.0 to 1.0)
            mach_range: Array of Mach numbers to test
            altitude_range: Array of altitudes to test [m]
            verbose: Print progress information
            
        Returns:
            Dictionary containing test results with success/failure status
            for each parameter combination
        """
        if lever_range is None:
            lever_range = np.linspace(0.1, 1.0, 10)
        if mach_range is None:
            mach_range = np.linspace(0.1, 0.94, 20)
        if altitude_range is None:
            altitude_range = np.linspace(0, 15000, 16)
        
        if verbose:
            total_tests = len(lever_range) * len(mach_range) * len(altitude_range)
            print(f"[ENVELOPE] Testing {total_tests} parameter combinations...")
            print(f"  Lever range: {lever_range[0]:.2f} to {lever_range[-1]:.2f} ({len(lever_range)} points)")
            print(f"  Mach range: {mach_range[0]:.3f} to {mach_range[-1]:.3f} ({len(mach_range)} points)")
            print(f"  Altitude range: {altitude_range[0]:.0f} to {altitude_range[-1]:.0f} m ({len(altitude_range)} points)")
        
        results = {
            'lever_range': lever_range,
            'mach_range': mach_range,
            'altitude_range': altitude_range,
            'thrust_data': np.zeros((len(lever_range), len(mach_range), len(altitude_range))),
            'valid_mask': np.zeros((len(lever_range), len(mach_range), len(altitude_range)), dtype=bool),
            'tsfc_data': np.zeros((len(lever_range), len(mach_range), len(altitude_range)))
        }
        
        test_count = 0
        valid_count = 0
        
        for i, lever in enumerate(lever_range):
            for j, mach in enumerate(mach_range):
                for k, altitude in enumerate(altitude_range):
                    test_count += 1
                    
                    thrust = self.engine.thrust_with_lever(lever, mach, altitude)
                    tsfc = self.engine.tsfc_current()
                    
                    if thrust is not None and np.isfinite(thrust) and thrust > 0:
                        results['thrust_data'][i, j, k] = thrust
                        results['valid_mask'][i, j, k] = True
                        valid_count += 1
                        if tsfc is not None and np.isfinite(tsfc):
                            results['tsfc_data'][i, j, k] = tsfc
                    else:
                        results['thrust_data'][i, j, k] = np.nan
                        results['valid_mask'][i, j, k] = False
                        results['tsfc_data'][i, j, k] = np.nan
                    
                    if verbose and test_count % 100 == 0:
                        progress = test_count / (len(lever_range) * len(mach_range) * len(altitude_range)) * 100
                        print(f"  Progress: {progress:.1f}% ({test_count} tests, {valid_count} valid)")
        
        if verbose:
            print(f"[ENVELOPE] Testing complete: {valid_count}/{test_count} valid combinations ({100*valid_count/test_count:.1f}%)")
        
        self.results['envelope'] = results
        return results
    
    def find_maximum_mach(self,
                         lever: float = 1.0,
                         altitude: float = 0.0,
                         mach_start: float = 0.1,
                         mach_end: float = 0.94,
                         tolerance: float = 0.001,
                         verbose: bool = True) -> dict:
        """
        Find maximum operational Mach number at specified conditions.
        
        Uses binary search to efficiently find the maximum Mach where
        engine produces valid thrust.
        
        Args:
            lever: Lever position (0.0 to 1.0)
            altitude: Altitude [m]
            mach_start: Starting Mach number for search
            mach_end: Ending Mach number for search
            tolerance: Tolerance for binary search convergence
            verbose: Print search progress
            
        Returns:
            Dictionary containing maximum Mach and test details
        """
        if verbose:
            print(f"[MAX-MACH] Finding maximum Mach at lever={lever:.2f}, altitude={altitude:.0f} m")
        
        # Binary search for maximum valid Mach
        mach_low = mach_start
        mach_high = mach_end
        max_valid_mach = None
        
        # First, verify that mach_start works
        test_thrust = self.engine.thrust_with_lever(lever, mach_start, altitude)
        if test_thrust is None or not np.isfinite(test_thrust) or test_thrust <= 0:
            if verbose:
                print(f"[MAX-MACH] WARNING: Starting Mach {mach_start:.3f} produces invalid thrust")
            return {'max_mach': None, 'thrust_at_max': None, 'valid': False}
        
        # Binary search
        while mach_high - mach_low > tolerance:
            mach_mid = (mach_low + mach_high) / 2.0
            thrust = self.engine.thrust_with_lever(lever, mach_mid, altitude)
            
            if thrust is not None and np.isfinite(thrust) and thrust > 0:
                max_valid_mach = mach_mid
                mach_low = mach_mid
            else:
                mach_high = mach_mid
        
        if max_valid_mach is None:
            max_valid_mach = mach_low
        
        # Get final thrust value
        final_thrust = self.engine.thrust_with_lever(lever, max_valid_mach, altitude)
        
        result = {
            'max_mach': max_valid_mach,
            'thrust_at_max': final_thrust,
            'lever': lever,
            'altitude': altitude,
            'valid': True
        }
        
        if verbose:
            print(f"[MAX-MACH] Maximum Mach: {max_valid_mach:.4f}")
            print(f"[MAX-MACH] Thrust at max Mach: {final_thrust:.1f} N")
        
        self.results['max_mach'] = result
        return result
    
    def find_maximum_ceiling(self,
                            lever: float = 1.0,
                            mach: float = 0.78,
                            altitude_start: float = 0.0,
                            altitude_end: float = 20000.0,
                            tolerance: float = 10.0,
                            verbose: bool = True) -> dict:
        """
        Find maximum service ceiling altitude at specified conditions.
        
        Service ceiling is defined as the altitude where engine can still
        produce positive thrust. Uses binary search for efficient determination.
        
        Args:
            lever: Lever position (0.0 to 1.0)
            mach: Mach number
            altitude_start: Starting altitude for search [m]
            altitude_end: Maximum altitude to test [m]
            tolerance: Tolerance for binary search convergence [m]
            verbose: Print search progress
            
        Returns:
            Dictionary containing maximum ceiling altitude and test details
        """
        if verbose:
            print(f"[MAX-CEILING] Finding maximum ceiling at lever={lever:.2f}, Mach={mach:.3f}")
        
        # Binary search for maximum valid altitude
        alt_low = altitude_start
        alt_high = altitude_end
        max_valid_altitude = None
        
        # First, verify that altitude_start works
        test_thrust = self.engine.thrust_with_lever(lever, mach, altitude_start)
        if test_thrust is None or not np.isfinite(test_thrust) or test_thrust <= 0:
            if verbose:
                print(f"[MAX-CEILING] WARNING: Starting altitude {altitude_start:.0f} m produces invalid thrust")
            return {'max_ceiling': None, 'thrust_at_max': None, 'valid': False}
        
        # Binary search
        while alt_high - alt_low > tolerance:
            alt_mid = (alt_low + alt_high) / 2.0
            thrust = self.engine.thrust_with_lever(lever, mach, alt_mid)
            
            if thrust is not None and np.isfinite(thrust) and thrust > 0:
                max_valid_altitude = alt_mid
                alt_low = alt_mid
            else:
                alt_high = alt_mid
        
        if max_valid_altitude is None:
            max_valid_altitude = alt_low
        
        # Get final thrust value
        final_thrust = self.engine.thrust_with_lever(lever, mach, max_valid_altitude)
        
        result = {
            'max_ceiling': max_valid_altitude,
            'thrust_at_max': final_thrust,
            'lever': lever,
            'mach': mach,
            'valid': True
        }
        
        if verbose:
            print(f"[MAX-CEILING] Maximum service ceiling: {max_valid_altitude:.1f} m ({max_valid_altitude/1000:.2f} km)")
            print(f"[MAX-CEILING] Thrust at ceiling: {final_thrust:.1f} N")
        
        self.results['max_ceiling'] = result
        return result
    
    def analyze_envelope_at_multiple_conditions(self,
                                               lever_values: list = None,
                                               mach_values: list = None,
                                               verbose: bool = True) -> dict:
        """
        Analyze maximum ceiling at multiple lever and Mach combinations.
        
        Args:
            lever_values: List of lever positions to test
            mach_values: List of Mach numbers to test
            verbose: Print progress information
            
        Returns:
            Dictionary containing maximum ceiling results for each condition
        """
        if lever_values is None:
            lever_values = [0.5, 0.75, 1.0]
        if mach_values is None:
            mach_values = [0.5, 0.7, 0.78, 0.85, 0.9]
        
        if verbose:
            print(f"[MULTI-CEILING] Analyzing ceiling at {len(lever_values)}×{len(mach_values)} conditions")
        
        results = {}
        for lever in lever_values:
            for mach in mach_values:
                key = f"lever_{lever:.2f}_mach_{mach:.3f}"
                results[key] = self.find_maximum_ceiling(lever=lever, mach=mach, verbose=False)
        
        if verbose:
            print("\n[MULTI-CEILING] Results summary:")
            print("Lever | Mach  | Max Ceiling [m] | Thrust [N]")
            print("-" * 50)
            for key, result in results.items():
                if result['valid']:
                    lever = result['lever']
                    mach = result['mach']
                    ceiling = result['max_ceiling']
                    thrust = result['thrust_at_max']
                    print(f"{lever:5.2f} | {mach:5.3f} | {ceiling:15.1f} | {thrust:10.1f}")
                else:
                    lever = result.get('lever', 0)
                    mach = result.get('mach', 0)
                    print(f"{lever:5.2f} | {mach:5.3f} | {'INVALID':15s} | {'N/A':10s}")
        
        self.results['multi_ceiling'] = results
        return results
    
    def plot_envelope(self, lever_idx: int = None, save_path: str = None):
        """
        Plot working envelope visualization.
        
        Args:
            lever_idx: Index of lever position to plot (if None, plots multiple)
            save_path: Path to save figure (if None, displays interactively)
        """
        if 'envelope' not in self.results:
            print("[PLOT] No envelope data available. Run test_working_envelope() first.")
            return
        
        envelope_data = self.results['envelope']
        lever_range = envelope_data['lever_range']
        mach_range = envelope_data['mach_range']
        altitude_range = envelope_data['altitude_range']
        valid_mask = envelope_data['valid_mask']
        thrust_data = envelope_data['thrust_data']
        
        if lever_idx is None:
            # Plot multiple lever positions
            n_levers = min(3, len(lever_range))
            lever_indices = np.linspace(0, len(lever_range)-1, n_levers, dtype=int)
            fig, axes = plt.subplots(1, n_levers, figsize=(6*n_levers, 5))
            if n_levers == 1:
                axes = [axes]
        else:
            lever_indices = [lever_idx]
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes = [axes]
        
        for idx, (ax, lever_i) in enumerate(zip(axes, lever_indices)):
            lever = lever_range[lever_i]
            
            # Create 2D plot: Mach vs Altitude
            M_grid, H_grid = np.meshgrid(mach_range, altitude_range)
            valid_2d = valid_mask[lever_i, :, :].T
            thrust_2d = thrust_data[lever_i, :, :].T
            
            # Plot valid region
            ax.contourf(M_grid, H_grid, valid_2d.astype(float), levels=[0.5, 1.5], 
                       colors=['lightcoral', 'lightgreen'], alpha=0.6)
            
            # Plot thrust contours
            valid_thrust = np.ma.masked_where(~valid_2d, thrust_2d)
            if np.any(valid_2d):
                contour = ax.contour(M_grid, H_grid, valid_thrust, levels=10, colors='black', alpha=0.3, linewidths=0.5)
                ax.clabel(contour, inline=True, fontsize=8)
            
            ax.set_xlabel('Mach Number')
            ax.set_ylabel('Altitude [m]')
            ax.set_title(f'Engine Envelope (Lever = {lever:.2f})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[PLOT] Figure saved to {save_path}")
        else:
            plt.show()
    
    def print_summary(self):
        """Print summary of all analysis results."""
        print("\n" + "="*60)
        print("ENGINE ENVELOPE ANALYSIS SUMMARY")
        print("="*60)
        
        if 'max_mach' in self.results:
            result = self.results['max_mach']
            if result['valid']:
                print(f"\nMaximum Mach Number:")
                print(f"  Value: {result['max_mach']:.4f}")
                print(f"  Conditions: Lever={result['lever']:.2f}, Altitude={result['altitude']:.0f} m")
                print(f"  Thrust at max: {result['thrust_at_max']:.1f} N")
        
        if 'max_ceiling' in self.results:
            result = self.results['max_ceiling']
            if result['valid']:
                print(f"\nMaximum Service Ceiling:")
                print(f"  Altitude: {result['max_ceiling']:.1f} m ({result['max_ceiling']/1000:.2f} km)")
                print(f"  Conditions: Lever={result['lever']:.2f}, Mach={result['mach']:.3f}")
                print(f"  Thrust at ceiling: {result['thrust_at_max']:.1f} N")
        
        if 'envelope' in self.results:
            envelope = self.results['envelope']
            total_tests = np.prod(envelope['valid_mask'].shape)
            valid_tests = np.sum(envelope['valid_mask'])
            print(f"\nWorking Envelope Statistics:")
            print(f"  Valid combinations: {valid_tests}/{total_tests} ({100*valid_tests/total_tests:.1f}%)")
            print(f"  Lever range: {envelope['lever_range'][0]:.2f} to {envelope['lever_range'][-1]:.2f}")
            print(f"  Mach range: {envelope['mach_range'][0]:.3f} to {envelope['mach_range'][-1]:.3f}")
            print(f"  Altitude range: {envelope['altitude_range'][0]:.0f} to {envelope['altitude_range'][-1]:.0f} m")
        
        cache_stats = self.engine.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"  Hits: {cache_stats['hits']}")
        print(f"  Misses: {cache_stats['misses']}")
        print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
        print(f"  Cache size: {cache_stats['cache_size']}")
        
        print("="*60 + "\n")


def main():
    """Main function to run engine envelope analysis."""
    print("="*60)
    print("ENGINE ENVELOPE ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = EngineEnvelopeAnalyzer()
    
    # Test 1: Find maximum Mach at sea level, full throttle
    print("\n[TEST 1] Maximum Mach Number Analysis")
    print("-" * 60)
    max_mach_result = analyzer.find_maximum_mach(lever=1.0, altitude=0.0, verbose=True)
    
    # Test 2: Find maximum Mach at cruise altitude
    print("\n[TEST 2] Maximum Mach at Cruise Altitude")
    print("-" * 60)
    max_mach_cruise = analyzer.find_maximum_mach(lever=1.0, altitude=10500.0, verbose=True)
    
    # Test 3: Find maximum service ceiling at cruise Mach
    print("\n[TEST 3] Maximum Service Ceiling Analysis")
    print("-" * 60)
    max_ceiling_result = analyzer.find_maximum_ceiling(lever=1.0, mach=0.78, verbose=True)
    
    # Test 4: Find maximum service ceiling at different Mach numbers
    print("\n[TEST 4] Maximum Ceiling at Multiple Mach Numbers")
    print("-" * 60)
    analyzer.find_maximum_ceiling(lever=1.0, mach=0.5, verbose=True)
    analyzer.find_maximum_ceiling(lever=1.0, mach=0.7, verbose=True)
    analyzer.find_maximum_ceiling(lever=1.0, mach=0.85, verbose=True)
    analyzer.find_maximum_ceiling(lever=1.0, mach=0.9, verbose=True)
    
    # Test 5: Multi-condition ceiling analysis
    print("\n[TEST 5] Multi-Condition Ceiling Analysis")
    print("-" * 60)
    analyzer.analyze_envelope_at_multiple_conditions(
        lever_values=[0.5, 0.75, 1.0],
        mach_values=[0.5, 0.7, 0.78, 0.85, 0.9],
        verbose=True
    )
    
    # Test 6: Working envelope test (coarse grid for speed)
    print("\n[TEST 6] Working Envelope Grid Test")
    print("-" * 60)
    lever_test = np.linspace(0.3, 1.0, 8)
    mach_test = np.linspace(0.2, 0.94, 15)
    alt_test = np.linspace(0, 15000, 16)
    analyzer.test_working_envelope(
        lever_range=lever_test,
        mach_range=mach_test,
        altitude_range=alt_test,
        verbose=True
    )
    
    # Plot results
    print("\n[PLOT] Generating envelope visualization...")
    analyzer.plot_envelope(lever_idx=None, save_path="engine_envelope.png")
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()

