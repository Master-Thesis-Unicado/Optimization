"""
PyEngine Wrapper for Mission Analysis

This module provides a wrapper class that integrates the pyengine library
with the existing mission analysis system, implementing computational caching
and simplified interface for engine thrust calculations.

The wrapper maintains a caching mechanism for performance optimization and
provides methods for thrust and TSFC calculations employed throughout the
mission analysis phases (climb, cruise, descent).
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
import time
import pyengine as engine

# Import aircraft configuration parameters
from aircraft_config import (
    ENGINE_ALT_CLIP,
    DEBUG
)


class EngineWrapper:
    """
    Computational wrapper for pyengine.Engine providing aircraft propulsion calculations.
    
    This class implements a caching mechanism and simplified interface for engine thrust
    calculations employed in mission analysis. The wrapper performs unit conversions
    (meters to Newtons) and implements computational caching to minimize redundant
    engine computations.
    
    Computational Features:
    - Caches thrust calculations indexed by (lever position, Mach number, altitude)
    - Pre-computes engine performance values across parameter grids for optimization
    - Manages computational errors and constrains Mach numbers to operational limits (≤0.94)
    - Provides computational performance statistics for analysis
    
    Implementation:
        engine = EngineWrapper("path/to/engine/stub")
        thrust = engine.thrust_with_lever(lever=0.8, M=0.8, h_m=10000)  # 10km altitude
        engine.precompute_grid(M_grid, H_grid, lever_grid)  # Pre-compute for optimization
    """
    def __init__(self, stub_path: str):
        self._eng = engine.Engine(str(Path(stub_path)), 0.5)
        # Initialize computational caching system
        self._thrust_cache = {}
        self._tsfc_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _to_engine_alt(self, h_m: float) -> float:
        val = float(h_m)
        if ENGINE_ALT_CLIP is not None:
            return float(np.clip(val, 0.0, ENGINE_ALT_CLIP))
        return val

    def thrust_with_lever(self, lever: float, M: float, h_m: float) -> float | None:
        """Return per-engine thrust [N] for lever in [0,1], Mach (clipped to <=0.94), altitude [m]."""
        # Check computational cache for existing results
        cache_key = (round(lever, 3), round(M, 3), round(h_m, 1))
        if cache_key in self._thrust_cache:
            self._cache_hits += 1
            return self._thrust_cache[cache_key]
        
        Mq = float(np.clip(M, 0.0, 0.94))  # avoid M >= 0.94
        alt_in_m = self._to_engine_alt(h_m)
        try:
            Tv = self._eng.get_thrust_with_lever_position(float(lever), Mq, float(alt_in_m))
            if Tv is None or not np.isfinite(Tv) or Tv < 0:
                result = None
            else:
                result = float(Tv)
            
            # Store result in computational cache
            self._thrust_cache[cache_key] = result
            self._cache_misses += 1
            return result
        except Exception:
            self._thrust_cache[cache_key] = None
            self._cache_misses += 1
            return None

    def tsfc_current(self) -> float | None:
        """Return TSFC as provided by engine (assumed kg/(N*s) by downstream logic)."""
        try:
            tsfc = self._eng.get_tsfc()
            if tsfc is None or not np.isfinite(tsfc):
                if DEBUG: print("[TSFC] get_tsfc() returned None/NaN.")
                return None
            return float(tsfc)
        except Exception as e:
            if DEBUG: print(f"[TSFC][ERR] {e}")
            return None
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._thrust_cache)
        }
    
    def precompute_grid(self, M_grid: np.ndarray, H_grid: np.ndarray, lever_grid: np.ndarray):
        """Pre-compute engine values for entire grid."""
        print(f"[ENGINE-CACHE] Pre-computing engine grid: {len(M_grid)}×{len(H_grid)}×{len(lever_grid)} = {len(M_grid)*len(H_grid)*len(lever_grid)} points")
        start_time = time.time()
        
        total_points = len(M_grid) * len(H_grid) * len(lever_grid)
        computed = 0
        
        for h in H_grid:
            for m in M_grid:
                for l in lever_grid:
                    cache_key = (round(l, 3), round(m, 3), round(h, 1))
                    if cache_key not in self._thrust_cache:
                        self.thrust_with_lever(l, m, h)
                        computed += 1
                        
                        if computed % 1000 == 0:
                            progress = computed / total_points * 100
                            print(f"[ENGINE-CACHE] Progress: {progress:.1f}% ({computed}/{total_points})")
        
        elapsed = time.time() - start_time
        print(f"[ENGINE-CACHE] Pre-computation completed in {elapsed:.2f}s")
        print(f"[ENGINE-CACHE] Cache stats: {self.get_cache_stats()}")

