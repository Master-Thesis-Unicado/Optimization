# ========================================================================
# PYENGINE WRAPPER MODULE
# ========================================================================
"""
Propulsion model interface via pyengine library integration.

Mathematical interface:
    T(δ, M, h): Thrust as function of lever position, Mach, altitude
    TSFC(δ, M, h): Thrust-specific fuel consumption

Features:
    - Computational caching: Memoization of T(δ,M,h) evaluations
    - Grid pre-computation: Batch evaluation for performance
    - Altitude clipping: Optional h ≤ h_max constraint

Performance: Cache reduces redundant engine model calls during DP optimization.
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
import time

# ========================================================================
# SECTION 1: MODULE INITIALIZATION
# ========================================================================

# Python path configuration for pyengine library
_current_file = Path(__file__).resolve()
_root_dir = _current_file.parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

import pyengine as engine

# Aircraft configuration: altitude limits
from aircraft_config import ENGINE_ALT_CLIP

# Engine model path
ENGINE_STUB_PATH = r"D:/Icloud/iCloudDrive/Master Thesis/Mission Analysis Code/lls/stubs/engines/PW1127G-JM"

# ========================================================================
# SECTION 2: ENGINE MODEL WRAPPER
# ========================================================================

class EngineWrapper:
    """
    Propulsion model interface with computational caching.
    
    Mathematical model: T = T(δ, M, h), TSFC = TSFC(δ, M, h)
    where δ ∈ [0,1] (throttle lever), M (Mach), h [m] (altitude).
    
    Cache structure: Dictionary indexed by (δ_rounded, M_rounded, h_rounded)
    Precision: Configurable rounding for accuracy vs. memory tradeoff.
    
    Performance: Cache reduces O(10^6) engine calls to O(10^3) during DP optimization.
    """
    
    def __init__(self, stub_path: str = None, high_accuracy: bool = False):
        """
        Initialize engine model with cache configuration.
        
        Parameters:
            stub_path: str - path to engine stub file (default: ENGINE_STUB_PATH)
            high_accuracy: bool - enable high-precision cache (4 decimals vs 3)
        """
        # Path resolution
        if stub_path is None:
            stub_path = ENGINE_STUB_PATH
        
        stub_path_obj = Path(stub_path)
        if not stub_path_obj.is_absolute():
            resolved_stub_path = _root_dir / stub_path
        else:
            resolved_stub_path = stub_path_obj
        
        # Initialize pyengine model
        self._eng = engine.Engine(str(resolved_stub_path), 0.9945)
        
        # Cache precision configuration
        self._high_accuracy = high_accuracy
        if high_accuracy:
            self._precision = {'lever': 4, 'mach': 4, 'altitude': 1}  # 4 decimals
            print(f"[ENGINE] High-accuracy cache: δ±0.0001, M±0.0001, h±1m")
        else:
            self._precision = {'lever': 3, 'mach': 3, 'altitude': 1}  # 3 decimals
        
        # Cache storage: {(δ,M,h): value}
        self._thrust_cache = {}    # {key: T [N]}
        self._tsfc_cache = {}      # {key: TSFC [kg/(N·s)]}
        self._last_call_key = None
        
        # Cache performance metrics
        self._cache_hits = 0
        self._cache_misses = 0

    def _to_engine_alt(self, h_m: float) -> float:
        """
        Apply altitude clipping if configured.
        
        Constraint: h ≤ h_clip (if ENGINE_ALT_CLIP is set)
        
        Parameters:
            h_m: h [m] - altitude
        
        Returns:
            h_clipped [m]: clipped altitude
        """
        val = float(h_m)
        if ENGINE_ALT_CLIP is not None:
            return float(np.clip(val, 0.0, ENGINE_ALT_CLIP))
        return val

    def thrust_with_lever(self, lever: float, M: float, h_m: float) -> float | None:
        """
        Compute per-engine thrust: T = T(δ, M, h).
        
        Model: pyengine library evaluation with caching.
        Cache lookup: O(1) dictionary access by (δ,M,h) key.
        
        Parameters:
            lever: δ [-] - throttle lever position ∈ [0,1]
            M: M [-] - Mach number
            h_m: h [m] - altitude
        
        Returns:
            T [N]: per-engine thrust, or None if infeasible
        """
        # Cache key generation: round to configured precision
        cache_key = (
            round(lever, self._precision['lever']),
            round(M, self._precision['mach']),
            round(h_m, self._precision['altitude'])
        )
        
        # Cache lookup: O(1) dictionary access
        if cache_key in self._thrust_cache:
            self._cache_hits += 1
            self._last_call_key = cache_key
            return self._thrust_cache[cache_key]
        
        # Cache miss: evaluate engine model
        from aircraft_config import M_MMO
        Mq = float(np.clip(M, 0.0, M_MMO))  # M ∈ [0, M_MMO]
        alt_in_m = self._to_engine_alt(h_m)  # Apply altitude clipping
        
        try:
            # Engine model evaluation: T = T(δ, M, h)
            Tv = self._eng.get_thrust_with_lever_position(float(lever), Mq, float(alt_in_m))
            
            # Validation: T > 0, finite
            if Tv is None or not np.isfinite(Tv) or Tv < 0:
                result = None
            else:
                result = float(Tv)
            
            # TSFC retrieval: TSFC = TSFC(δ, M, h)
            tsfc = self._eng.get_tsfc()
            if tsfc is not None and np.isfinite(tsfc):
                self._tsfc_cache[cache_key] = float(tsfc)
            else:
                self._tsfc_cache[cache_key] = None
            
            # Store in cache
            self._thrust_cache[cache_key] = result
            self._last_call_key = cache_key
            self._cache_misses += 1
            return result
            
        except Exception:
            # Failure: cache as None
            self._thrust_cache[cache_key] = None
            self._tsfc_cache[cache_key] = None
            self._last_call_key = cache_key
            self._cache_misses += 1
            return None

    def tsfc_current(self) -> float | None:
        """
        Query TSFC from last thrust evaluation.
        
        Mechanism: TSFC cached alongside thrust during thrust_with_lever() call.
        Retrieval: Access via last_call_key for state consistency.
        
        Returns:
            TSFC [kg/(N·s)]: thrust-specific fuel consumption, or None if unavailable
        """
        # Cache retrieval for last evaluation
        if hasattr(self, '_last_call_key') and self._last_call_key in self._tsfc_cache:
            return self._tsfc_cache[self._last_call_key]
        
        # Fallback: direct engine query
        try:
            tsfc = self._eng.get_tsfc()
            if tsfc is None or not np.isfinite(tsfc):
                return None
            return float(tsfc)
        except Exception:
            return None
    
    def get_cache_stats(self) -> dict:
        """
        Query cache performance metrics.
        
        Returns:
            dict: {hits, misses, hit_rate, cache_size}
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._thrust_cache)
        }
    
    def precompute_grid(self, mach_grid: np.ndarray, H_grid: np.ndarray, lever_grid: np.ndarray):
        """
        Pre-compute engine performance over 3D grid.
        
        Purpose: Populate cache before optimization to eliminate runtime overhead.
        Algorithm: Nested loop over (h,M,δ) with cache_key lookup.
        
        Parameters:
            mach_grid: np.ndarray - M_i discretization
            H_grid: np.ndarray - h_k discretization  
            lever_grid: np.ndarray - δ_j discretization
        """
        total_points = len(mach_grid) * len(H_grid) * len(lever_grid)
        print(f"[ENGINE] Pre-computing grid: {len(mach_grid)}×{len(H_grid)}×{len(lever_grid)} = {total_points} points")
        
        start_time = time.time()
        computed = 0
        
        # Triple loop over (h,M,δ) space
        for h in H_grid:
            for m in mach_grid:
                for l in lever_grid:
                    # Check if already cached
                    cache_key = (
                        round(l, self._precision['lever']),
                        round(m, self._precision['mach']),
                        round(h, self._precision['altitude'])
                    )
                    if cache_key not in self._thrust_cache:
                        self.thrust_with_lever(l, m, h)  # Populates cache
                        computed += 1
                        
                        # Progress reporting
                        if computed % 1000 == 0:
                            progress = computed / total_points * 100
                            print(f"[ENGINE] Progress: {progress:.1f}% ({computed}/{total_points})")
        
        elapsed = time.time() - start_time
        print(f"[ENGINE] Grid pre-computation complete: {elapsed:.2f} s")
        print(f"[ENGINE] Cache stats: {self.get_cache_stats()}")

