# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, NamedTuple, Callable, List, Dict, Any
import time
from atmosphere import Atmosphere
import pyengine as engine

# Import aircraft configuration from centralized module
from aircraft_config import (
    SystemConfiguration, AtmosphericProperties,
    ENGINE_STUB_PATH,
    N_ENGINES, INITIAL_MASS_KG, S_REF_M2,
    ENGINE_ALT_CLIP, M_MIN_DEFAULT, M_MIN_EFFECTIVE, M_MMO,
    G_C, DEBUG, AUTO_APPLY_PARAMS_FROM_EXCEL,
    isa_properties, a_from_altitude, _atmospheric_properties
)

# Import pyaerodynamics wrapper
from pyaerodynamics_wrapper import PyAerodynamicsWrapper

# =========  3 - ENGINE WRAPPER ========================
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
        self._eng = engine.Engine(str(Path(stub_path)))
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

# =========  4 - AERODYNAMICS SYSTEM =================
# Note: AeroTables class replaced with PyAerodynamicsWrapper from pyaerodynamics_wrapper.py
# The PyAerodynamicsWrapper provides the same interface as AeroTables but uses pyaerodynamics library

# =========  5 - CLIMBING CORE SYSTEM =====================
class ClimbingCore:
    """
    Comprehensive aircraft climb optimization and simulation framework for mission analysis.
    
    This class implements a complete computational framework for aircraft climb performance
    analysis through three integrated subsystems: strategy simulation, dynamic programming
    optimization, and penalty-based guidance. The system serves as the primary interface
    for climb trajectory analysis and optimization.
    
    System Components:
    - StrategyManager: Implements various climb strategies (linear, exponential, constant speed/Mach)
      with energy allocation between climb rate and speed optimization
    - DynamicProgrammingOptimizer: Computes optimal climb trajectories through 3D state space (altitude, Mach, lever)
      to minimize fuel consumption while satisfying aircraft constraints
    - PenaltySystem: Provides Mach trajectory guidance and lever position penalties to direct optimization
      toward physically realizable flight paths and avoid infeasible solutions
    - EnergyCalculator: Handles energy allocation and thrust calculations for all strategies
    
    Computational Features:
    - Multiple climb strategies with configurable energy allocation between climb and speed
    - Fuel-optimal climb path computation using dynamic programming in 3D state space
    - Penalty-based guidance to ensure physically realizable Mach trajectories and lever schedules
    - Strategy simulation and comparison capabilities for performance analysis
    - Integration with aerodynamic tables and engine models for accurate calculations
    
    Implementation:
        # Strategy simulation
        strategies = ClimbingCore.StrategyManager.build_strategy_set()
        results = []
        for label, strategy_func, af in strategies:
            result = ClimbingCore.StrategyManager.simulate_strategy_path(label=label, aero=aero, eng=engine, ...)
            results.append(result)
        
        # Optimal climb calculation
        optimal_path = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_fixed_mass(aero=aero, eng=engine, ...)
        
        # Cost evaluation
        cost = ClimbingCore.DynamicProgrammingOptimizer.compute_3d_cost(aero=aero, eng=engine, altitude=10000, mach=0.8, lever=0.7, ...)
    """
    
    # ========= STRATEGY MANAGEMENT SYSTEM =========
    class StrategyManager:
        """Manages climb strategy simulation and energy allocation."""
        
        class StrategyProfiles:
            """Return raw weights (cw, sw). Normalized to w_c + w_s = 1 in the integrator."""
            
            class FixedEnergy:
                class Linear:
                    @staticmethod
                    def profile(altitude, velocity, altitude_fraction):
                        """Linear climb strategy: cw increases linearly with altitude fraction."""
                        # altitude_fraction is the AF parameter (e.g., 0.10 for "Linear AF=0.10")
                        # This represents the constant energy split throughout the climb
                        af = float(np.clip(altitude_fraction, 0.0, 1.0))
                        cw = af  # Constant climb weight
                        sw = 1.0 - af  # Constant speed weight
                        return cw, sw

                class Exponential:
                    @staticmethod
                    def increasing_climb(altitude, velocity, altitude_fraction):
                        """Exponential climb strategy: cw increases exponentially with altitude fraction."""
                        # altitude_fraction is a tuple: (AF_parameter, current_altitude_fraction)
                        if isinstance(altitude_fraction, tuple):
                            af_param, current_af = altitude_fraction
                            af_param = float(np.clip(af_param, 0.0, 1.0))
                            current_af = float(np.clip(current_af, 0.0, 1.0))
                        else:
                            # Fallback for backward compatibility
                            af_param = 0.5
                            current_af = float(np.clip(altitude_fraction, 0.0, 1.0))
                        
                        min_climb = 0.1  # 10%
                        max_climb = 0.9  # 90%
                    
                        exponent = 1.0 + 2.0 * af_param   
                        
                        exp_factor = current_af ** exponent
                        
                        # Map exponential to 0.1-0.9 range
                        climb_norm = min_climb + (max_climb - min_climb) * exp_factor
                        speed_norm = 1.0 - climb_norm
                        
                        # Return normalized values (sum to 1.0)
                        cw = climb_norm
                        sw = speed_norm
                        return cw, sw

                    @staticmethod
                    def increasing_speed(altitude, velocity, altitude_fraction):
                        """Exponential speed strategy: sw increases exponentially with altitude fraction."""
                        if isinstance(altitude_fraction, tuple):
                            af_param, current_af = altitude_fraction
                            af_param = float(np.clip(af_param, 0.0, 1.0))
                            current_af = float(np.clip(current_af, 0.0, 1.0))
                        else:
                            af_param = 0.5
                            current_af = float(np.clip(altitude_fraction, 0.0, 1.0))
                        

                        min_speed = 0.1  # 10% speed at start
                        max_speed = 0.9  # 90% speed at end

                        exponent = 1.0 + 2.0 * af_param 
                        
                        # Exponential curve: current_af^exponent
                        exp_factor = current_af ** exponent
                        
                        # Map exponential to 0.1-0.9 range
                        speed_norm = min_speed + (max_speed - min_speed) * exp_factor
                        climb_norm = 1.0 - speed_norm
                        
                        # Return normalized values (sum to 1.0)
                        cw = climb_norm
                        sw = speed_norm
                        return cw, sw

            class ConstantRates:
                @staticmethod
                def constant_speed(altitude, velocity, altitude_fraction=None):
                    """Constant speed strategy: maintain constant speed, all energy to climb."""
                    return 1.0, 0.0

                @staticmethod
                def constant_mach():
                    """Constant Mach strategy: maintain constant Mach, all energy to climb."""
                    def _const_mach(altitude, velocity, altitude_fraction=None):
                        return 1.0, 0.0
                    _const_mach._const_mach = True
                    return _const_mach
        
        # ========= STRATEGY WEIGHT PROCESSING AND ENERGY ALLOCATION =========
        @staticmethod
        def process_strategy_weights(strategy_fn: Callable, h: float, V: float, 
                                    strategy_altitude_fraction, label: str, 
                                    h_hist: list, current_altitude_fraction: float) -> tuple[float, float, float, float]:
            """
            Process strategy weights and convert to normalized energy allocation weights.
            
            Args:
                strategy_fn: Strategy function that returns (cw, sw) weights
                h: Current altitude [m]
                V: Current velocity [m/s]
                strategy_altitude_fraction: Altitude fraction parameter for strategy
                label: Strategy label for debugging
                h_hist: History of altitudes for debugging
                current_altitude_fraction: Current altitude fraction for debugging
                
            Returns:
                tuple: (w_c, w_s, cw, sw) - normalized weights and raw weights
            """
            cw, sw = strategy_fn(h, V, strategy_altitude_fraction)
            
            # Validate strategy weights
            if not np.isfinite(cw) or not np.isfinite(sw):
                dbg(f"[WARN] Strategy '{label}' returned non-finite weights: cw={cw}, sw={sw} at h={h:.1f}m")
                cw, sw = 1.0, 0.0  # Default to climb-only
            
            if cw < 0 or sw < 0:
                dbg(f"[WARN] Strategy '{label}' returned negative weights: cw={cw}, sw={sw} at h={h:.1f}m")
                cw, sw = max(0, cw), max(0, sw)  # Clamp to non-negative
            
            # Normalize weights to ensure they sum to 1
            s = max(cw + sw, 1e-12)
            w_c, w_s = cw / s, sw / s
            
            # Debug output for strategy behavior analysis (every 10% altitude progress)
            # Calculate altitude progress percentage  
            altitude_progress = current_altitude_fraction
            
            # Print at 0%, 10%, 20%, ..., 90%, 100% progress
            progress_percent = int(altitude_progress * 10) * 10  # Round to nearest 10%
            should_print = (
                len(h_hist) <= 3 or  # Always print first 3 steps
                (altitude_progress >= 0.1 and progress_percent != getattr(ClimbingCore.StrategyManager.process_strategy_weights, '_last_progress', -1))  # Every 10%
            )
            
            if should_print:
                # Track last progress to prevent duplicate output
                ClimbingCore.StrategyManager.process_strategy_weights._last_progress = progress_percent
                if isinstance(strategy_altitude_fraction, tuple):
                    af_used_str = f"({strategy_altitude_fraction[0]:.3f}, {strategy_altitude_fraction[1]:.3f})"
                elif strategy_altitude_fraction is not None:
                    af_used_str = f"{strategy_altitude_fraction:.3f}"
                else:
                    af_used_str = "None"
                progress_pct = altitude_progress * 100
                dbg(f"[STRAT] {label} at h={h:.0f}m ({progress_pct:.1f}% progress): cw={cw:.3f}, sw={sw:.3f} -> w_c={w_c:.3f}, w_s={w_s:.3f}, af_used={af_used_str}, af_current={current_altitude_fraction:.3f}")
            
            return w_c, w_s, cw, sw
        
        @staticmethod
        def build_strategy_set() -> List[tuple[str, Callable, Optional[float]]]:
            """Build the complete set of climb strategies for comparison."""
            afs = [0.10, 0.30, 0.50, 0.70, 0.90]
            out = []
            for af in afs:
                out.append((f"Linear AF={af:.2f}", ClimbingCore.StrategyManager.StrategyProfiles.FixedEnergy.Linear.profile, af))
            for af in afs:
                out.append((f"Exp climb AF={af:.2f}", ClimbingCore.StrategyManager.StrategyProfiles.FixedEnergy.Exponential.increasing_climb, af))
            for af in afs:
                out.append((f"Exp speed AF={af:.2f}", ClimbingCore.StrategyManager.StrategyProfiles.FixedEnergy.Exponential.increasing_speed, af))
            out.append(("Constant speed", ClimbingCore.StrategyManager.StrategyProfiles.ConstantRates.constant_speed, None))
            out.append(("Constant Mach",  ClimbingCore.StrategyManager.StrategyProfiles.ConstantRates.constant_mach(), None))
            return out
        
        @staticmethod
        def simulate_strategy_path(*, label: str, aero: PyAerodynamicsWrapper, eng: EngineWrapper,
                                  mass0_kg: float, h0_m: float, V0_ms: float,
                                  target_alt_m: float, dt: float,
                                  strategy_fn: Callable[[float,float,Optional[float]], tuple],
                                  altitude_fraction: Optional[float]) -> 'ClimbingCore.EnergyCalculator.StrategyRun':
            """Integrate a strategy path with fixed mass; returns a StrategyRun with diagnostics."""
            g0 = G_C

            # Adjust initial velocity for constant speed and constant Mach strategies
            if "Constant speed" in label or "Constant Mach" in label:
                # For constant strategies, start at 0.5 Mach instead of using V0_ms
                a = a_from_altitude(h0_m)
                V0_ms = 0.5 * a  # Set initial velocity to achieve 0.5 Mach
                initial_mach = V0_ms / a
                dbg(f"[STRAT] {label}: Using initial velocity {V0_ms:.1f} m/s for {initial_mach:.3f} Mach at {h0_m:.0f}m (a={a:.1f} m/s)")

            # histories
            h_hist, V_hist, t_hist = [float(h0_m)], [float(V0_ms)], [0.0]
            lever_hist, Ttot_hist, D_hist, Ps_hist = [], [], [], []
            mdot_hist, dt_hist, dFuel_hist, cumFuel_hist, limited_hist = [], [], [], [0.0], []

            mass_kg = float(mass0_kg)  # FIXED for strategies

            while h_hist[-1] < target_alt_m:
                h = float(h_hist[-1]); V = float(V_hist[-1]); t = float(t_hist[-1])
                a = a_from_altitude(h); M = V / max(a, 1e-9)
                Mq = float(np.clip(M, M_MIN_DEFAULT, 0.94))

                # Calculate current altitude fraction for strategy functions
                current_altitude_fraction = (h - h0_m) / (target_alt_m - h0_m) if target_alt_m > h0_m else 0.0
                current_altitude_fraction = np.clip(current_altitude_fraction, 0.0, 1.0)
                
                # Determine which altitude fraction to use for each strategy type:
                # - Linear strategies: Use AF parameter as constant (e.g., AF=0.10 means always 10% climb)
                # - Exponential strategies: Use AF parameter as scaling factor, current altitude fraction for exponential behavior
                # - Constant strategies: Don't use altitude fraction (pass None)
                if "Linear" in label:
                    strategy_altitude_fraction = altitude_fraction  # Use the AF parameter (e.g., 0.10)
                elif "Exp" in label:
                    # For Exponential strategies, we need to pass both AF parameter and current altitude fraction
                    # We'll create a tuple: (AF_parameter, current_altitude_fraction)
                    strategy_altitude_fraction = (altitude_fraction, current_altitude_fraction)
                else:  # Constant strategies
                    strategy_altitude_fraction = None  # Don't use altitude fraction
                
                # Process strategy weights using centralized method
                w_c, w_s, cw, sw = ClimbingCore.StrategyManager.process_strategy_weights(
                    strategy_fn, h, V, strategy_altitude_fraction, label, h_hist, current_altitude_fraction
                )
                
                # Compute energy allocation using centralized method
                dh_dt, dv_dt = ClimbingCore.EnergyCalculator.compute_energy_allocation(strategy_fn, w_c, w_s, h, V, a)
                
                D = float(aero.get_drag(Mq, h))
                W = mass_kg * g0
                
                # Compute required thrust using centralized method
                F_required_total = ClimbingCore.EnergyCalculator.compute_required_thrust(mass_kg, dh_dt, dv_dt, V, D)

                lv, T_per, thrust_limited = find_lever_for_thrust(eng, F_required_total, Mq, h, lever_grid=None, allow_refine=True)
                if (lv is None) or (T_per is None):
                    dbg(f"[WARN] No valid lever at h={h:.1f} m, M={Mq:.3f}. Ending strategy '{label}'.")
                    # Ensure we have at least one valid point before breaking
                    if len(h_hist) < 2:
                        dbg(f"[ERROR] Strategy '{label}' failed at first step. Cannot create valid trajectory.")
                        return ClimbingCore.EnergyCalculator.StrategyRun(
                            label=label, alt_m=np.array([h0_m]), mach=np.array([V0_ms/max(a_from_altitude(h0_m), 1e-6)]),
                            time_s=np.array([0.0]), lever=np.array([np.nan]), T_total_N=np.array([np.nan]),
                            D_N=np.array([np.nan]), Ps_mps=np.array([np.nan]), mdot_kgps=np.array([np.nan]),
                            dt_s=np.array([0.0]), dFuel_kg=np.array([0.0]), cumFuel_kg=np.array([0.0]),
                            thrust_limited=np.array([False]), fuel_total_kg=0.0
                        )
                    break

                # Align TSFC at chosen lever
                _ = eng.thrust_with_lever(lv, Mq, h)
                tsfc = eng.tsfc_current()
                if (tsfc is None) or (not np.isfinite(tsfc)) or (tsfc < 0):
                    tsfc = np.nan
                mdot_total = (tsfc * max(T_per, 0.0)) * SystemConfiguration.N_ENGINES if np.isfinite(tsfc) else np.nan

                Ps = ((T_per*SystemConfiguration.N_ENGINES - D) * max(V,1e-9)) / max(W,1e-9)

                dt_use = float(dt)
                if h + dh_dt * dt_use > target_alt_m and dh_dt > 0:
                    dt_use = (target_alt_m - h) / dh_dt

                h_new = h + dh_dt * dt_use
                V_new = max(V + dv_dt * dt_use, 1.0)
                t_new = t + dt_use

                burned = (mdot_total * dt_use) if np.isfinite(mdot_total) else 0.0  # diagnostic only

                lever_hist.append(lv)
                Ttot_hist.append(T_per * SystemConfiguration.N_ENGINES)
                D_hist.append(D)
                Ps_hist.append(Ps)
                mdot_hist.append(mdot_total if np.isfinite(mdot_total) else np.nan)
                dt_hist.append(dt_use)
                dFuel_hist.append(burned)
                cumFuel_hist.append(cumFuel_hist[-1] + burned)
                limited_hist.append(bool(thrust_limited))

                h_hist.append(h_new); V_hist.append(V_new); t_hist.append(t_new)
                if h_new >= target_alt_m:
                    break

            alt = np.asarray(h_hist, float)
            V   = np.asarray(V_hist, float)
            time= np.asarray(t_hist, float)
            # Calculate Mach with safety check for division by zero
            def safe_mach_calc(hh):
                a = a_from_altitude(float(hh))
                return max(a, 1e-6)  # Prevent division by zero
            
            mach = V / np.vectorize(safe_mach_calc)(alt)
            lever = np.asarray(lever_hist + [lever_hist[-1] if lever_hist else np.nan], float)
            Ttot  = np.asarray(Ttot_hist + [Ttot_hist[-1] if Ttot_hist else np.nan], float)
            Darr  = np.asarray(D_hist + [D_hist[-1] if D_hist else np.nan], float)
            Psarr = np.asarray(Ps_hist + [Ps_hist[-1] if Ps_hist else np.nan], float)
            mdot  = np.asarray(mdot_hist + [mdot_hist[-1] if mdot_hist else np.nan], float)
            dtarr = np.asarray([0.0] + dt_hist, float)
            dFuel = np.asarray([0.0] + dFuel_hist, float)
            cumF  = np.asarray(cumFuel_hist, float)
            limited = np.asarray(limited_hist + [limited_hist[-1] if limited_hist else False], bool)
            fuel_total = float(cumF[-1])

            return ClimbingCore.EnergyCalculator.StrategyRun(
                label=label,
                alt_m=alt,
                mach=mach,
                time_s=time,
                lever=lever,
                T_total_N=Ttot,
                D_N=Darr,
                Ps_mps=Psarr,
                mdot_kgps=mdot,
                dt_s=dtarr,
                dFuel_kg=dFuel,
                cumFuel_kg=cumF,
                thrust_limited=limited,
                fuel_total_kg=fuel_total
            )
        
        @staticmethod
        def resample_strategy_run(sr: 'ClimbingCore.EnergyCalculator.StrategyRun', n_samples: int) -> 'ClimbingCore.EnergyCalculator.StrategyRun':
            """Resample a StrategyRun onto a uniform-in-altitude grid of length n_samples."""
            n = int(max(2, n_samples))

            # Find the maximum length among all arrays to preserve all data
            array_lengths = [
                len(sr.alt_m),
                len(sr.time_s),
                len(sr.mach),
                len(sr.lever),
                len(sr.T_total_N),
                len(sr.D_N),
                len(sr.Ps_mps),
                len(sr.mdot_kgps),
                len(sr.cumFuel_kg),
                len(sr.thrust_limited)
            ]
            max_length = max(array_lengths)

            # If arrays have different lengths, pad shorter ones to match the longest
            arrays_to_check = [
                ('alt_m', sr.alt_m),
                ('time_s', sr.time_s),
                ('mach', sr.mach),
                ('lever', sr.lever),
                ('T_total_N', sr.T_total_N),
                ('D_N', sr.D_N),
                ('Ps_mps', sr.Ps_mps),
                ('mdot_kgps', sr.mdot_kgps),
                ('cumFuel_kg', sr.cumFuel_kg),
                ('thrust_limited', sr.thrust_limited)
            ]

            for name, arr in arrays_to_check:
                if len(arr) != max_length:
                    print(f"[WARNING] Array {name} has length {len(arr)} but max length is {max_length}. Padding to match.")
                    if len(arr) < max_length:
                        # Pad with last value to match max length
                        if name == 'thrust_limited':
                            arr = np.concatenate([arr, np.full(max_length - len(arr), arr[-1] if len(arr) > 0 else False)])
                        else:
                            arr = np.concatenate([arr, np.full(max_length - len(arr), arr[-1] if len(arr) > 0 else 0.0)])

                    # Update the array in the StrategyRun object
                    setattr(sr, name, arr)

            # Use the updated max_length as the base length
            base_length = max_length

            # Use uniform step size consistent with 3D DP approach
            uniform_step_size = (sr.alt_m[-1] - sr.alt_m[0]) / n
            alt_new = np.arange(sr.alt_m[0], sr.alt_m[-1] + uniform_step_size, uniform_step_size)

            def safe_interp(y):
                """Safe interpolation with length validation."""
                if len(y) != len(sr.alt_m):
                    print(f"[ERROR] Array length mismatch: {len(y)} vs {len(sr.alt_m)}")
                    return np.full_like(alt_new, y[-1] if len(y) > 0 else 0.0)
                return np.interp(alt_new, sr.alt_m, y)

            time_new   = safe_interp(sr.time_s)
            mach_new   = safe_interp(sr.mach)
            lever_new  = safe_interp(sr.lever)
            Ttot_new   = safe_interp(sr.T_total_N)
            D_new      = safe_interp(sr.D_N)
            Ps_new     = safe_interp(sr.Ps_mps)
            mdot_new   = safe_interp(sr.mdot_kgps)
            cumF_new   = safe_interp(sr.cumFuel_kg)
            limited_f  = safe_interp(sr.thrust_limited.astype(float))
            limited_new= (limited_f >= 0.5)

            dt_new    = np.diff(time_new, prepend=time_new[0])
            dFuel_new = np.diff(cumF_new,  prepend=cumF_new[0])
            fuel_tot  = float(cumF_new[-1] - cumF_new[0])

            return ClimbingCore.EnergyCalculator.StrategyRun(
                label=sr.label,
                alt_m=alt_new,
                mach=mach_new,
                time_s=time_new,
                lever=lever_new,
                T_total_N=Ttot_new,
                D_N=D_new,
                Ps_mps=Ps_new,
                mdot_kgps=mdot_new,
                dt_s=dt_new,
                dFuel_kg=dFuel_new,
                cumFuel_kg=cumF_new,
                thrust_limited=limited_new,
                fuel_total_kg=fuel_tot
            )
    
    # ========= ENERGY CALCULATION SYSTEM =========
    class EnergyCalculator:
        """Handles energy allocation and thrust calculations for all strategies."""
        
        @staticmethod
        def compute_energy_allocation(strategy_fn: Callable, w_c: float, w_s: float, 
                                     h: float, V: float, a: float) -> tuple[float, float]:
            """
            Compute climb rate and acceleration rate based on energy allocation weights.
            
            Args:
                strategy_fn: Strategy function (used to check for const_mach attribute)
                w_c: Normalized climb weight (0-1)
                w_s: Normalized speed weight (0-1)
                h: Current altitude [m]
                V: Current velocity [m/s]
                a: Current speed of sound [m/s]
                
            Returns:
                tuple: (dh_dt, dv_dt) - climb rate [m/s] and acceleration rate [m/s²]
            """
            if getattr(strategy_fn, "_const_mach", False):
                eps = TARGET_ALT_M / N_PLOT_STEPS  # Uniform step size based on N_PLOT_STEPS
                a1 = _atmospheric_properties.a_from_altitude(h - eps/2)
                a2 = _atmospheric_properties.a_from_altitude(h + eps/2)
                dadh = (a2 - a1) / eps  # 
                
                # Climb rate
                dh_dt = w_c * StrategyConfig.E_DOT_CMD
                
                # Velocity change to maintain constant Mach: dV/dt = M * da/dh * dh/dt
                # where M = V/a is the current Mach number
                current_mach = V / max(a, 1e-9)
                dv_dt = current_mach * dadh * dh_dt
            else:
                # Energy allocation computation for climb and speed components
                # Energy rate balance: Ė = mg(dh/dt) + mV(dV/dt) = w_c*E_DOT + w_s*E_DOT
                # Where E_DOT_CMD is specific energy rate [J/kg/s] = [m²/s³] = [m/s]·[m/s²]
                dh_dt = w_c * StrategyConfig.E_DOT_CMD  # [m/s] climb rate from potential energy
                dv_dt = (w_s * StrategyConfig.E_DOT_CMD * G_C) / max(V, 1e-9)  # [m/s²] acceleration from kinetic energy
            
            return dh_dt, dv_dt
        
        @staticmethod
        def compute_required_thrust(mass_kg: float, dh_dt: float, dv_dt: float, 
                                  V: float, D: float) -> float:
            """
            Compute required total thrust based on energy allocation and drag.
            
            Args:
                mass_kg: Aircraft mass [kg]
                dh_dt: Climb rate [m/s]
                dv_dt: Acceleration rate [m/s²]
                V: Current velocity [m/s]
                D: Current drag [N]
                
            Returns:
                float: Required total thrust [N]
            """
            # Power balance: T*V - D*V = mg*dh/dt + mV*dV/dt (energy rate equation)
            # Solving for required thrust: T = D + (mg*dh/dt + mV*dV/dt)/V
            F_required_total = D + (mass_kg * G_C * dh_dt + mass_kg * V * dv_dt) / max(V, 1e-9)
            return F_required_total
        
        @dataclass
        class StrategyRun:
            """Container for strategy simulation results."""
            label: str
            alt_m: np.ndarray
            mach: np.ndarray
            time_s: np.ndarray
            lever: np.ndarray
            T_total_N: np.ndarray
            D_N: np.ndarray
            Ps_mps: np.ndarray
            mdot_kgps: np.ndarray
            dt_s: np.ndarray
            dFuel_kg: np.ndarray
            cumFuel_kg: np.ndarray
            thrust_limited: np.ndarray
            fuel_total_kg: float
    
    # ========= DYNAMIC PROGRAMMING OPTIMIZATION SYSTEM =========
    class DynamicProgrammingOptimizer:
        """Handles 3D dynamic programming optimization for minimum fuel climb paths."""
        
        @staticmethod
        def solve_3d_fixed_mass(aero: PyAerodynamicsWrapper, eng: EngineWrapper,
                                M_grid: np.ndarray, H_sched: np.ndarray,
                                lever_samples: int = 10,
                                target_mach: float = None,
                                target_mach_tolerance: float = 0.02,
                                start_mach: float = None,
                                start_lever: float = None):
            """
            True 3D Dynamic Programming solver for minimum fuel climb optimization.
            
            This implementation considers all three variables (altitude, Mach, lever) simultaneously
            and allows transitions to 25 neighboring points in the 3D space at each step (5x5 grid).
            
            Args:
                aero: Aerodynamics tables
                eng: Engine wrapper
                M_grid: Mach number grid
                H_sched: Altitude schedule
                lever_samples: Number of lever positions to consider
                target_mach: Target Mach number at final altitude
                target_mach_tolerance: Tolerance for target Mach constraint
            
            Returns:
                MinFuelSchedule: Optimal climb schedule
                dict: Additional information (costs, path, etc.)
            """
            K, I = len(H_sched), len(M_grid)
            L = lever_samples
            
            # Create lever grid
            lever_grid = np.linspace(0.0, 1.0, L)
            
            # Initialize 3D cost matrix, weight matrix, and predecessor array
            F = np.full((K, I, L), np.inf)
            weight_matrix = np.full((K, I, L), np.nan)  # Track weight at each state
            prv = np.full((K, I, L, 3), -1, dtype=int)  # [alt_idx, mach_idx, lever_idx]
        
            # Set starting Mach and lever
            if start_mach is not None:
                # Find the closest Mach in the grid to start_mach
                start_mach_idx = np.argmin(np.abs(M_grid - start_mach))
                actual_start_mach = M_grid[start_mach_idx]
                dbg(f"[3D-DP] Forcing start Mach to {actual_start_mach:.3f} (closest to {start_mach:.3f})")
            else:
                start_mach_idx = 0  # Start at minimum Mach
            
            if start_lever is not None:
                # Find the closest lever in the grid to start_lever
                start_lever_idx = np.argmin(np.abs(lever_grid - start_lever))
                actual_start_lever = lever_grid[start_lever_idx]
                dbg(f"[3D-DP] Forcing start lever to {actual_start_lever:.3f} (closest to {start_lever:.3f})")
            else:
                start_lever_idx = 0  # Start at minimum lever
            
            # Verify the starting point is feasible
            if (M_grid[start_mach_idx] >= M_MIN_EFFECTIVE and 
                M_grid[start_mach_idx] <= M_MMO and
                lever_grid[start_lever_idx] >= 0.0 and 
                lever_grid[start_lever_idx] <= 1.0):
                
                # Calculate altitude fraction for starting point (10m altitude)
                altitude_fraction = H_sched[0] / TARGET_ALT_M if TARGET_ALT_M > 0 else 0.0
                
                cost = ClimbingCore.compute_3d_cost(aero, eng, H_sched[0], M_grid[start_mach_idx], lever_grid[start_lever_idx],
                                                  target_mach=target_mach, prev_mach=None, altitude_fraction=altitude_fraction, mass_kg=INITIAL_MASS_KG)
                if np.isfinite(cost) and cost > 0:
                    F[0, start_mach_idx, start_lever_idx] = 0.0  # Starting cost is 0
                    weight_matrix[0, start_mach_idx, start_lever_idx] = INITIAL_MASS_KG  # Starting weight
                    dbg(f"[3D-DP] Starting point verified: h={H_sched[0]:.0f}m, M={M_grid[start_mach_idx]:.3f}, lever={lever_grid[start_lever_idx]:.3f}, weight={INITIAL_MASS_KG:.0f}kg")
                else:
                    raise RuntimeError(f"[3D-DP] Starting point not feasible: h={H_sched[0]:.0f}m, M={M_grid[start_mach_idx]:.3f}, lever={lever_grid[start_lever_idx]:.3f}")
            else:
                raise RuntimeError(f"[3D-DP] Starting point out of bounds: h={H_sched[0]:.0f}m, M={M_grid[start_mach_idx]:.3f}, lever={lever_grid[start_lever_idx]:.3f}")
            
            if not np.isfinite(F[0, start_mach_idx, start_lever_idx]):
                raise RuntimeError("[3D-DP] No feasible starting point found")
            
            dbg(f"[3D-DP] Starting at h={H_sched[0]:.0f}m, M={M_grid[start_mach_idx]:.3f}, lever={lever_grid[start_lever_idx]:.3f}")
        
            # Forward pass - 3D Dynamic Programming
            for k in range(K - 1):  # For each altitude level
                current_alt = H_sched[k]
                next_alt = H_sched[k + 1]
                dh = next_alt - current_alt
                
                dbg(f"[3D-DP] Processing altitude {current_alt:.0f}m -> {next_alt:.0f}m")
                
                # Find all feasible current states
                feasible_states = np.where(np.isfinite(F[k]))
                feasible_count = 0
            
                for state_idx in range(len(feasible_states[0])):
                    i = feasible_states[0][state_idx]  # Mach index
                    j = feasible_states[1][state_idx]  # Lever index
                    
                    if not np.isfinite(F[k, i, j]):
                        continue
                    
                    # Get current weight at this state
                    current_weight = weight_matrix[k, i, j]
                    if not np.isfinite(current_weight) or current_weight <= 0:
                        continue
                        
                    current_mach = M_grid[i]
                    current_lever = lever_grid[j]
                    
                    # Consider all 25 possible next moves (5x5 grid in Mach-Lever space)
                    # BUT ONLY to the next altitude level (k+1) to prevent altitude jumps
                    for di in [-2, -1, 0, 1, 2]:  # Mach change (broader range)
                        for dj in [-2, -1, 0, 1, 2]:  # Lever change (broader range)
                            next_mach_idx = i + di
                            next_lever_idx = j + dj
                            
                            # Check bounds - ONLY allow transition to next altitude level (k+1)
                            if (0 <= next_mach_idx < I and 
                                0 <= next_lever_idx < L and
                                k + 1 < K):  # This ensures we only go to k+1
                                
                                next_mach = M_grid[next_mach_idx]
                                next_lever = lever_grid[next_lever_idx]
                                
                                # Check feasibility constraints
                                if (next_mach >= M_MIN_EFFECTIVE and 
                                    next_mach <= M_MMO and
                                    next_lever >= 0.0 and 
                                    next_lever <= 1.0):
                                    
                                    # Compute costs for both states with penalties using dynamic weight
                                    # Calculate altitude fraction for adaptive penalties
                                    current_alt_fraction = current_alt / GridConfig.TARGET_ALT_M if GridConfig.TARGET_ALT_M > 0 else 0.0
                                    next_alt_fraction = next_alt / GridConfig.TARGET_ALT_M if GridConfig.TARGET_ALT_M > 0 else 0.0
                                    
                                    # Get previous Mach for smoothness penalty
                                    prev_mach = M_grid[i] if k > 0 else None
                                    
                                    # Use current weight for cost calculations (fuel burn will update weight)
                                    current_cost = ClimbingCore.compute_3d_cost(aero, eng, current_alt, current_mach, current_lever,
                                                                             target_mach=target_mach, prev_mach=prev_mach, 
                                                                             altitude_fraction=current_alt_fraction, mass_kg=current_weight)
                                    next_cost = ClimbingCore.compute_3d_cost(aero, eng, next_alt, next_mach, next_lever,
                                                                          target_mach=target_mach, prev_mach=current_mach, 
                                                                          altitude_fraction=next_alt_fraction, mass_kg=current_weight)
                                    
                                    if (np.isfinite(current_cost) and np.isfinite(next_cost) and
                                        current_cost > 0 and next_cost > 0):
                                        
                                        # Trapezoidal integration for fuel cost
                                        step_cost = 0.5 * (current_cost + next_cost) * dh
                                        total_cost = F[k, i, j] + step_cost
                                        
                                        # Calculate fuel burned during this step
                                        fuel_burned = step_cost
                                        next_weight = current_weight - fuel_burned
                                        
                                        # Ensure weight doesn't go negative
                                        if next_weight <= 0:
                                            continue
                                        
                                        # Update if this path is better
                                        if total_cost < F[k + 1, next_mach_idx, next_lever_idx]:
                                            F[k + 1, next_mach_idx, next_lever_idx] = total_cost
                                            weight_matrix[k + 1, next_mach_idx, next_lever_idx] = next_weight
                                            prv[k + 1, next_mach_idx, next_lever_idx] = [k, i, j]
                                            feasible_count += 1
            
                dbg(f"[3D-DP] Found {feasible_count} feasible transitions at altitude {current_alt:.0f}m")
        
            # Apply terminal Mach constraint
            if target_mach is not None:
                dbg(f"[3D-DP] Applying target Mach constraint: {target_mach:.3f} ± {target_mach_tolerance:.3f}")
                
                # Find valid final states
                valid_final = np.abs(M_grid - target_mach) < target_mach_tolerance
                
                if not valid_final.any():
                    dbg(f"[3D-DP] Warning: No Mach values within tolerance. Using closest Mach.")
                    closest_idx = np.argmin(np.abs(M_grid - target_mach))
                    valid_final = np.zeros_like(valid_final, dtype=bool)
                    valid_final[closest_idx] = True
                
                # Mask invalid final states
                for i in range(I):
                    if not valid_final[i]:
                        F[-1, i, :] = np.inf
            
            # Check if any path reached the final altitude
            if not np.isfinite(F[-1]).any():
                raise RuntimeError("[3D-DP] No feasible path reached the final altitude")
            
            # Find optimal final state
            final_flat_idx = np.nanargmin(F[-1])
            final_mach_idx, final_lever_idx = np.unravel_index(final_flat_idx, F[-1].shape)
            final_alt_idx = K - 1  # Final altitude index
            
            dbg(f"[3D-DP] Optimal final state: h={H_sched[final_alt_idx]:.0f}m, "
                f"M={M_grid[final_mach_idx]:.3f}, lever={lever_grid[final_lever_idx]:.3f}")
            dbg(f"[3D-DP] Total fuel cost: {F[final_alt_idx, final_mach_idx, final_lever_idx]:.1f} kg")
            
            # Backtrack to find optimal path
            path_alt = []
            path_mach = []
            path_lever = []
            path_costs = []
            path_weights = []
            
            current_state = [final_alt_idx, final_mach_idx, final_lever_idx]
            
            while current_state[0] >= 0:  # Backtrack to start
                alt_idx, mach_idx, lever_idx = current_state
                
                path_alt.append(H_sched[alt_idx])
                path_mach.append(M_grid[mach_idx])
                path_lever.append(lever_grid[lever_idx])
                path_costs.append(F[alt_idx, mach_idx, lever_idx])
                path_weights.append(weight_matrix[alt_idx, mach_idx, lever_idx])
                
                # Debug: Check for altitude jumps
                if len(path_alt) > 1:
                    alt_diff = path_alt[-1] - path_alt[-2]
                    if abs(alt_diff) > (H_sched[1] - H_sched[0]) * 1.5:  # More than 1.5x the altitude step
                        dbg(f"[3D-DP] WARNING: Potential altitude jump detected: {path_alt[-2]:.0f}m -> {path_alt[-1]:.0f}m (diff: {alt_diff:.0f}m)")
                
                # Move to predecessor
                if alt_idx > 0:
                    current_state = prv[alt_idx, mach_idx, lever_idx].tolist()
                else:
                    break
            
            # Reverse to get correct order (start to finish)
            path_alt = path_alt[::-1]
            path_mach = path_mach[::-1]
            path_lever = path_lever[::-1]
            path_costs = path_costs[::-1]
            path_weights = path_weights[::-1]
            
            # Debug: Show altitude progression with weight
            dbg(f"[3D-DP] Path altitude progression:")
            for i in range(min(10, len(path_alt))):  # Show first 10 points
                dbg(f"  Step {i}: {path_alt[i]:.0f}m, M={path_mach[i]:.3f}, lever={path_lever[i]:.3f}, weight={path_weights[i]:.0f}kg")
            if len(path_alt) > 10:
                dbg(f"  ... (showing first 10 of {len(path_alt)} points)")
            
            # Check for altitude jumps in the final path
            for i in range(1, len(path_alt)):
                alt_diff = path_alt[i] - path_alt[i-1]
                expected_diff = H_sched[1] - H_sched[0]  # Expected altitude step
                if abs(alt_diff - expected_diff) > expected_diff * 0.1:  # More than 10% deviation
                    dbg(f"[3D-DP] WARNING: Non-uniform altitude step at point {i}: {path_alt[i-1]:.0f}m -> {path_alt[i]:.0f}m (diff: {alt_diff:.0f}m, expected: {expected_diff:.0f}m)")
            
            # Compute additional trajectory data
            alt_array = np.array(path_alt)
            mach_array = np.array(path_mach)
            lever_array = np.array(path_lever)
            weight_array = np.array(path_weights)
            
            # Calculate time and fuel increments
            n_segments = len(alt_array) - 1  # Number of segments between points
            dt_array = np.zeros(n_segments)  # Time for each segment
            dF_array = np.zeros(n_segments)  # Fuel for each segment
            
            for i in range(n_segments):  # Loop through all segments: 0 to 48 for 50 points
                h_curr, h_next = alt_array[i], alt_array[i + 1]
                M_curr, M_next = mach_array[i], mach_array[i + 1]
                lever_curr, lever_next = lever_array[i], lever_array[i + 1]
                weight_curr, weight_next = weight_array[i], weight_array[i + 1]
                
                # Average values for this segment
                h_avg = 0.5 * (h_curr + h_next)
                M_avg = 0.5 * (M_curr + M_next)
                lever_avg = 0.5 * (lever_curr + lever_next)
                weight_avg = 0.5 * (weight_curr + weight_next)  # Use average weight for segment
                
                # Compute segment properties with dynamic weight
                a = a_from_altitude(h_avg)
                V = M_avg * a
                D = aero.get_drag(M_avg, h_avg)
                T_per = eng.thrust_with_lever(lever_avg, M_avg, h_avg)
                T_tot = T_per * SystemConfiguration.N_ENGINES
                Ps = ((T_tot - D) * V) / (weight_avg * G_C)  # Use dynamic weight
                
                if Ps > 0:
                    # Handle both vertical and horizontal moves
                    if abs(h_next - h_curr) > 1.0:  # Vertical move (altitude change)
                        dt_array[i] = (h_next - h_curr) / Ps
                        dbg(f"[3D-DP] Vertical move {i}: h={h_curr:.0f}->{h_next:.0f}m, dt={dt_array[i]:.3f}s, weight={weight_avg:.0f}kg")
                    else:  # Horizontal move (same altitude, different Mach/lever)
                        # Calculate time based on velocity change
                        V_curr = M_curr * a
                        V_next = M_next * a
                        if abs(V_next - V_curr) > 0.1:  # Significant velocity change
                            # Use acceleration rate: dt = dV / a_accel
                            # Where a_accel = (T_tot - D) / mass
                            a_accel = (T_tot - D) / weight_avg  # Use dynamic weight
                            if a_accel > 0:
                                dt_array[i] = abs(V_next - V_curr) / a_accel
                                dbg(f"[3D-DP] Horizontal move {i}: M={M_curr:.3f}->{M_next:.3f}, V={V_curr:.1f}->{V_next:.1f}m/s, dt={dt_array[i]:.3f}s")
                            else:
                                dt_array[i] = 0.1  # Small time step for horizontal moves
                                dbg(f"[3D-DP] Horizontal move {i}: M={M_curr:.3f}->{M_next:.3f}, dt={dt_array[i]:.3f}s (small step)")
                        else:
                            dt_array[i] = 0.1  # Small time step for horizontal moves
                            dbg(f"[3D-DP] Horizontal move {i}: M={M_curr:.3f}->{M_next:.3f}, dt={dt_array[i]:.3f}s (minimal change)")
                    
                    dF_array[i] = path_costs[i + 1] - path_costs[i]
                    
                    # Debug the final segment (step 49->50 issue)
                    if i >= n_segments - 2:  # Last two segments
                        dbg(f"[3D-DP] Segment {i} (step {i}->{i+1}): h={h_curr:.0f}->{h_next:.0f}m, dt={dt_array[i]:.3f}s")
                else:
                    dt_array[i] = 0.0
                    dF_array[i] = 0.0
                    dbg(f"[3D-DP] Invalid segment {i}: Ps={Ps:.3f}, dt=0.0s")
            
            # Time array construction with proper temporal progression
            n_points = len(alt_array)
            
            # Create proper time array starting from t=0
            # dt_array now has exactly n_segments elements for segments between points
            # We need n_points elements for the time array: [0, t1, t2, t3, ...]
            
            if len(dt_array) > 0:
                # Create time array: start at 0, then cumulate the dt values
                # dt_array[i] is the time to go from point i to point i+1
                time_array = np.zeros(n_points)
                
                # Fill in times: time[i+1] = time[i] + dt[i]
                for i in range(len(dt_array)):
                    if i + 1 < n_points:
                        time_array[i + 1] = time_array[i] + dt_array[i]
                
                dbg(f"[3D-DP] Time array constructed: start=0, end={time_array[-1]:.3f}s")
                dbg(f"[3D-DP] Sample time progression: {time_array[:5]} (first 5 points)")
                dbg(f"[3D-DP] Final time progression: {time_array[-3:]} (last 3 points: steps 47,48,49)")
                dbg(f"[3D-DP] Final dt values: {dt_array[-3:]} (last 3 segments: 46->47, 47->48, 48->49)")
                
                # Create dt_array_full for output (should match time differences)
                dt_array_full = np.zeros(n_points)
                dt_array_full[1:] = np.diff(time_array)  # dt[i] = time[i] - time[i-1]
                
            else:
                # Fallback if no segments calculated
                dbg(f"[3D-DP] WARNING: No time segments calculated, using uniform time steps")
                time_array = np.linspace(0, n_points * 1.0, n_points)  # 1 second per point
                dt_array_full = np.ones(n_points)
                dt_array_full[0] = 0.0  # First point has dt=0
            
            # Create fuel arrays - ensure all points have fuel values
            if len(dF_array) > 0:
                # Use the last dF value for the final point, or zero
                final_dF = dF_array[-1] if dF_array[-1] > 0 else 0.0
                dF_array_full = np.concatenate([dF_array, [final_dF]])
            else:
                # Fallback if no segments
                dF_array_full = np.array([0.0])
            
            # Ensure dF_array_full has exactly n_points elements
            if len(dF_array_full) > n_points:
                dF_array_full = dF_array_full[:n_points]
            elif len(dF_array_full) < n_points:
                # Pad with the last value
                last_dF = dF_array_full[-1] if len(dF_array_full) > 0 else 0.0
                dF_array_full = np.concatenate([dF_array_full, np.full(n_points - len(dF_array_full), last_dF)])
            
            # Create fuel array with exactly n_points elements
            fuel_array = np.cumsum(dF_array_full)
            
            # All arrays should now have the same length (n_points)
            dbg(f"[3D-DP] Final array lengths - alt: {len(alt_array)}, time: {len(time_array)}, fuel: {len(fuel_array)}")
            
            # Calculate gradients safely
            if len(time_array) > 1 and len(alt_array) > 1:
                try:
                    mdot_kgps = np.gradient(fuel_array, time_array)
                    Ps_mps = np.gradient(alt_array, time_array)
                except ValueError as e:
                    dbg(f"[3D-DP] WARNING: Gradient calculation failed: {e}")
                    mdot_kgps = np.zeros_like(alt_array)
                    Ps_mps = np.zeros_like(alt_array)
            else:
                mdot_kgps = np.zeros_like(alt_array)
                Ps_mps = np.zeros_like(alt_array)
            
            # Create the result schedule
            schedule = MinFuelSchedule(
                alt_m=alt_array,
                mach=mach_array,
                fuel_est_kg=float(path_costs[-1]),
                J_kg_per_m=np.array(path_costs),
                mdot_kgps=mdot_kgps,
                Ps_mps=Ps_mps,
                T_total_N=np.array([eng.thrust_with_lever(lever, mach, alt) * SystemConfiguration.N_ENGINES 
                                   for alt, mach, lever in zip(alt_array, mach_array, lever_array)]),
                D_N=np.array([aero.get_drag(mach, alt) for alt, mach in zip(alt_array, mach_array)]),
                lever=lever_array,
                T_per_engine_N=np.array([eng.thrust_with_lever(lever, mach, alt) 
                                        for alt, mach, lever in zip(alt_array, mach_array, lever_array)]),
                mass_kg=weight_array,  # Dynamic weight accounting for fuel burn
                thrust_limited=np.isclose(lever_array, 1.0, atol=1e-3),
                dt_s=dt_array_full,
                dFuel_kg=dF_array_full,
                cumFuel_kg=fuel_array
            )
            
            # Additional information
            info = {
                'total_fuel_kg': float(path_costs[-1]),
                'total_time_s': float(time_array[-1]),
                'final_mach': float(mach_array[-1]),
                'final_altitude': float(alt_array[-1]),
                'path_length': len(alt_array),
                'cost_matrix_3d': F,
                'predecessor_matrix': prv
            }
            
            return schedule, info
    
    @staticmethod
    def compute_3d_cost(aero: PyAerodynamicsWrapper, eng: EngineWrapper, 
                       altitude: float, mach: float, lever: float,
                       target_mach: float = None, prev_mach: float = None,
                       altitude_fraction: float = None, mass_kg: float = None) -> float:
        """
        Compute fuel cost density J = mdot/Ps + penalties for a given 3D state.
        
        Args:
            aero: Aerodynamics tables
            eng: Engine wrapper
            altitude: Altitude in meters
            mach: Mach number
            lever: Throttle lever position (0-1)
            target_mach: Target Mach number for penalty calculation
            prev_mach: Previous Mach number for smoothness penalty
            altitude_fraction: Fraction of altitude progress for adaptive penalties
            mass_kg: Aircraft mass in kg (defaults to INITIAL_MASS_KG for backward compatibility)
        
        Returns:
            float: Fuel cost density in kg/m + penalties, or inf if infeasible
        """
        try:
            # Use provided mass or default to initial mass for backward compatibility
            if mass_kg is None:
                mass_kg = INITIAL_MASS_KG
            
            # Get atmospheric properties
            a = a_from_altitude(altitude)
            V = mach * a
            
            # Get thrust
            T_per = eng.thrust_with_lever(lever, mach, altitude)
            if T_per is None or not np.isfinite(T_per) or T_per <= 0:
                return np.inf
                
            T_tot = T_per * SystemConfiguration.N_ENGINES
            
            # Get drag
            D = aero.get_drag(mach, altitude)
            if not np.isfinite(D) or D < 0:
                return np.inf
            
            # Calculate specific excess power using dynamic mass
            W = mass_kg * G_C
            Ps = ((T_tot - D) * V) / W
            
            if not np.isfinite(Ps) or Ps <= 0:
                return np.inf
            
            # Get fuel flow
            eng.thrust_with_lever(lever, mach, altitude)  # Align TSFC
            tsfc = eng.tsfc_current()
            if tsfc is None or not np.isfinite(tsfc) or tsfc < 0:
                return np.inf
            
            mdot = tsfc * T_per * SystemConfiguration.N_ENGINES
            
            # Base fuel cost density
            J = mdot / Ps
            
            # Add Mach penalty if target Mach is provided
            if target_mach is not None and ClimbingCore.PenaltySystem.MACH_TRAJECTORY_GUIDANCE:
                # Use simplified progressive target approach
                mach_penalty = ClimbingCore.PenaltySystem.compute_mach_penalty(mach, target_mach, prev_mach, altitude_fraction)
                J += mach_penalty
            
            # Add lever penalty if enabled
            if ClimbingCore.PenaltySystem.LEVER_PENALTY_GUIDANCE:
                lever_penalty = ClimbingCore.PenaltySystem.compute_lever_penalty(lever, altitude_fraction)
                J += lever_penalty
            
            return J
            
        except Exception:
            return np.inf
    
# Note: simulate_strategy_path and resample_strategy_run moved to StrategyManager class
    
    # ========= PENALTY SYSTEM (Nested Class) =========
    class PenaltySystem:
        """System for computing various penalties in the optimization process."""
        
        # Feature flags
        MACH_TRAJECTORY_GUIDANCE = True  # Enable reachability-constrained Mach guidance
        LEVER_PENALTY_GUIDANCE = True  # Enable penalties for high lever positions
        
        # Mach targeting constants
        TARGET_MACH_TOLERANCE = 0.015  # Tolerance for target Mach constraint in DP
        
        # Mach trajectory guidance constants
        MACH_PENALTY_BASE_WEIGHT = 0.3  # Base penalty weight (kg per Mach² deviation)
        MAX_REASONABLE_MACH_RATE = 0.02  # Max reasonable Mach change per optimization step
        TOTAL_CLIMB_STEPS_ESTIMATE = 50  # Matches N_PLOT_STEPS - actual DP grid steps
        URGENCY_MULTIPLIER = 2.0  # How much urgency scales with altitude progress - reduced from 8.0
        GUIDANCE_PENALTY_WEIGHT = 0.5  # Strong guidance penalty when inside reachable corridor (increased from 0.1)
        
        # Lever penalty guidance constants
        LEVER_PENALTY_WEIGHT = 3.0  # Base weight for lever penalty (kg per lever unit above threshold)
        LEVER_PENALTY_THRESHOLD = 0.85  # Lever threshold above which penalties apply (85% = realistic climb limit)
        LEVER_PENALTY_EXPONENT = 3.0  # Exponent for penalty curve (higher = more aggressive)
        LEVER_PENALTY_CRITICAL_THRESHOLD = 0.90  # Critical threshold for very high penalties (90%+)
        LEVER_PENALTY_CRITICAL_MULTIPLIER = 5.0  # Extra penalty multiplier for critical range
        LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD = 0.95  # Ultra-critical threshold for maximum penalties (95%+)
        LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER = 20.0  # Ultra-critical penalty multiplier (emergency thrust only)
        
        @staticmethod
        def compute_mach_penalty(current_mach: float, target_mach: float, prev_mach: float = None, 
                                 altitude_fraction: float = None) -> float:
            """
            Compute penalty using reachability-constrained approach.
            
            Creates a dynamic safety corridor that ensures target remains achievable
            with realistic Mach change rates. Penalties only apply outside the corridor.
            
            Args:
                current_mach: Current Mach number
                target_mach: Final target Mach number
                prev_mach: Previous Mach number (unused - kept for API compatibility)
                altitude_fraction: Fraction of altitude progress (0.0 = start, 1.0 = target)
            
            Returns:
                penalty: Penalty value in kg per meter
            """
            if altitude_fraction is None:
                altitude_fraction = 0.0
            
            # Calculate remaining altitude fraction and steps
            remaining_fraction = 1.0 - altitude_fraction
            estimated_steps_remaining = remaining_fraction * ClimbingCore.PenaltySystem.TOTAL_CLIMB_STEPS_ESTIMATE
            
            # Calculate maximum achievable Mach change with reasonable rates
            max_achievable_change = ClimbingCore.PenaltySystem.MAX_REASONABLE_MACH_RATE * estimated_steps_remaining
            
            # Define reachability corridor bounds
            min_reachable_mach = target_mach - max_achievable_change
            max_reachable_mach = target_mach + max_achievable_change
            
            # Calculate urgency factor (increases as we approach target altitude)
            urgency = (1.0 - remaining_fraction) * ClimbingCore.PenaltySystem.URGENCY_MULTIPLIER
            
            # Apply penalties based on position relative to corridor
            if current_mach < min_reachable_mach:
                # Below corridor - risk of not reaching target
                deviation = min_reachable_mach - current_mach
                penalty = urgency * ClimbingCore.PenaltySystem.MACH_PENALTY_BASE_WEIGHT * (deviation ** 2)
                
            elif current_mach > max_reachable_mach:
                # Above corridor - risk of overshooting target
                deviation = current_mach - max_reachable_mach  
                penalty = urgency * ClimbingCore.PenaltySystem.MACH_PENALTY_BASE_WEIGHT * (deviation ** 2)
                
            else:
                # Within corridor - apply progressive guidance toward target
                if altitude_fraction > 0.7:
                    # Strong final phase guidance (70-100% altitude)
                    final_phase_strength = (altitude_fraction - 0.7) / 0.3  # 0 to 1 scaling
                    mach_deviation = current_mach - target_mach
                    
                    # Extra penalty boost for final 10% of climb
                    if altitude_fraction > 0.9:
                        final_boost = ((altitude_fraction - 0.9) / 0.1) * 2.0  # 0 to 2x multiplier
                        final_phase_strength *= (1.0 + final_boost)
                        
                    penalty = final_phase_strength * ClimbingCore.PenaltySystem.GUIDANCE_PENALTY_WEIGHT * (mach_deviation ** 2)
                else:
                    penalty = 0.0  # No penalty in early climb phase
            
            return penalty
        
        @staticmethod
        def compute_lever_penalty(current_lever: float, altitude_fraction: float = None) -> float:
            """
            Compute penalty for high lever positions to encourage realistic engine usage.
            
            Engine limits are altitude-independent - high thrust settings cause the same
            thermal and mechanical stress regardless of altitude.
            
            Real-world considerations:
            - 85% lever = Maximum Continuous Thrust (MCT) - unlimited duration
            - 90%+ lever = Takeoff/Go-around thrust - limited duration, high wear
            - 95%+ lever = Maximum Takeoff Thrust - emergency use only, severe penalties
            
            Args:
                current_lever: Current lever position (0.0 to 1.0)
                altitude_fraction: Unused parameter (kept for backward compatibility)
            
            Returns:
                penalty: Penalty value in kg (altitude-independent)
            """
            penalty = 0.0
            
            # Only apply penalty if lever exceeds MCT threshold (90%)
            if current_lever > ClimbingCore.PenaltySystem.LEVER_PENALTY_THRESHOLD:
                # Calculate excess lever above MCT threshold
                excess_lever = current_lever - ClimbingCore.PenaltySystem.LEVER_PENALTY_THRESHOLD
                
                # Base penalty using exponential curve for realistic behavior
                lever_penalty = excess_lever ** ClimbingCore.PenaltySystem.LEVER_PENALTY_EXPONENT
                
                # Apply critical penalty for very high lever positions (90%+)
                if current_lever > ClimbingCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD:
                    critical_excess = current_lever - ClimbingCore.PenaltySystem.LEVER_PENALTY_CRITICAL_THRESHOLD
                    critical_penalty = critical_excess ** (ClimbingCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 1.0)
                    lever_penalty += critical_penalty * ClimbingCore.PenaltySystem.LEVER_PENALTY_CRITICAL_MULTIPLIER
                
                # Apply ultra-critical penalty for maximum thrust positions (95%+)
                if current_lever > ClimbingCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD:
                    ultra_critical_excess = current_lever - ClimbingCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_THRESHOLD
                    ultra_critical_penalty = ultra_critical_excess ** (ClimbingCore.PenaltySystem.LEVER_PENALTY_EXPONENT + 2.0)
                    lever_penalty += ultra_critical_penalty * ClimbingCore.PenaltySystem.LEVER_PENALTY_ULTRA_CRITICAL_MULTIPLIER
                
                # Use constant penalty weight - engine limits are altitude-independent
                penalty_weight = ClimbingCore.PenaltySystem.LEVER_PENALTY_WEIGHT
                
                penalty = penalty_weight * lever_penalty
            
            return penalty
    
    # ========= ENGINE ENVELOPE SYSTEM =========
    @staticmethod
    def compute_full_engine_envelope(aero: PyAerodynamicsWrapper, eng: EngineWrapper, M_grid: np.ndarray, 
                                   H_sched: np.ndarray, lever_samples: int = 50):
        """
        Compute the full engine envelope showing all possible J points (fuel cost density)
        that the engine can achieve across the entire Mach-Altitude-Lever space.
        
        This method provides comprehensive engine performance analysis for visualization
        and optimization purposes.
        
        Args:
            aero: Aerodynamics tables
            eng: Engine wrapper
            M_grid: Mach number grid
            H_sched: Altitude schedule
            lever_samples: Number of lever positions to sample
            
        Returns:
            tuple: (J_envelope_transposed, lever_grid) - Engine envelope data and lever grid
        """
        print("[ENVELOPE] Computing full engine envelope...")
        
        # Create lever grid
        lever_grid = np.linspace(0.0, 1.0, lever_samples)
        
        # Initialize 3D cost matrix
        K, I, L = len(H_sched), len(M_grid), len(lever_grid)
        J_envelope = np.full((K, I, L), np.nan)
        
        # Compute J = mdot/Ps for all feasible points
        feasible_count = 0
        total_points = K * I * L
        
        for k, h in enumerate(H_sched):
            for i, M in enumerate(M_grid):
                for j, lever in enumerate(lever_grid):
                    # Check basic constraints
                    if (M >= M_MIN_EFFECTIVE and M <= M_MMO and 
                        0.0 <= lever <= 1.0):
                        
                        # Compute cost using the same function as DP
                        cost = ClimbingCore.compute_3d_cost(aero, eng, h, M, lever)
                        
                        if np.isfinite(cost) and cost > 0:
                            J_envelope[k, i, j] = cost
                            feasible_count += 1
        
        print(f"[ENVELOPE] Computed {feasible_count}/{total_points} feasible points ({100*feasible_count/total_points:.1f}%)")
        
        # Transpose to (mach, altitude, lever) for visualization
        J_envelope_transposed = np.transpose(J_envelope, (1, 0, 2))
        
        return J_envelope_transposed, lever_grid
    
    # ========= FLIGHT ENVELOPE ANALYSIS =========
    @staticmethod
    def check_envelope_exceedance(strategy, aero):
        """
        Check if a strategy exceeds the flight envelope (MMO or CLmax).
        
        This method analyzes a climbing strategy to determine if it violates
        aircraft performance limits during the climb phase.
        
        Args:
            strategy: Climbing strategy object with mach and alt_m arrays
            aero: Aerodynamics tables for performance calculations
            
        Returns:
            str: Status message indicating envelope compliance
        """
        from atmosphere import Atmosphere
        
        atmospheric_props = Atmosphere()
        
        exceeds_mmo = False
        exceeds_clmax = False
        
        for i, (mach, alt) in enumerate(zip(strategy.mach, strategy.alt_m)):
            # Check MMO exceedance
            if mach > M_MMO:
                exceeds_mmo = True
                break
                
            # Check CLmax exceedance (stall)
            T, _, rho = atmospheric_props.calculate_atmospheric_properties_meters(float(alt))
            a = atmospheric_props.get_speed_of_sound(float(alt))
            V = mach * a
            W = INITIAL_MASS_KG * G_C
            q_req = W / (S_REF_M2 * aero.cl_max)
            
            if rho > 0:
                V_stall = np.sqrt(2 * q_req / rho)
                M_stall = V_stall / a
                if mach < M_stall:
                    exceeds_clmax = True
                    break
        
        if exceeds_mmo:
            return "Exceeds MMO"
        elif exceeds_clmax:
            return "Exceeds CLmax"
        else:
            return "Within Envelope"
    
    
    
# Create global climbing core instance
_climbing_core = ClimbingCore()

# =========  6 - SYSTEM UTILITIES =====================
class SystemUtilities:
    """Centralized utility functions for the entire system."""
    
    @staticmethod
    def dbg(msg: str):
        """Debug logging function."""
        if SystemConfiguration.DEBUG:
            print(msg)
    
    @staticmethod
    def apply_params_to_globals(params: Dict[str, Any]) -> None:
        """Apply recognized scalar parameters from Excel to module-level defaults."""
        g = globals()
        for key, val in params.items():
            if key in g:
                old = g[key]
                try:
                    g[key] = type(old)(val) if not isinstance(old, (int, float)) else (int(val) if isinstance(old, int) else float(val))
                except Exception:
                    g[key] = val
                SystemUtilities.dbg(f"[PARAM] {key}: {old} -> {g[key]}")
    
    @staticmethod
    def nanfill2d(A: np.ndarray) -> np.ndarray:
        """In-place-like forward/backward fill of NaNs first across rows, then across columns."""
        # Forward fill across rows
        for i in range(A.shape[0]):
            for j in range(1, A.shape[1]):
                if np.isnan(A[i, j]) and not np.isnan(A[i, j-1]):
                    A[i, j] = A[i, j-1]
        
        # Backward fill across rows
        for i in range(A.shape[0]):
            for j in range(A.shape[1]-2, -1, -1):
                if np.isnan(A[i, j]) and not np.isnan(A[i, j+1]):
                    A[i, j] = A[i, j+1]
        
        # Forward fill across columns
        for j in range(A.shape[1]):
            for i in range(1, A.shape[0]):
                if np.isnan(A[i, j]) and not np.isnan(A[i-1, j]):
                    A[i, j] = A[i-1, j]
        
        # Backward fill across columns
        for j in range(A.shape[1]):
            for i in range(A.shape[0]-2, -1, -1):
                if np.isnan(A[i, j]) and not np.isnan(A[i+1, j]):
                    A[i, j] = A[i+1, j]
        
        return A

# Create global system utilities instance
_system_utilities = SystemUtilities()

# Backward compatibility functions
def dbg(msg: str):
    """Debug logging function."""
    SystemUtilities.dbg(msg)

def _apply_params_to_globals(params: Dict[str, Any]) -> None:
    """Apply recognized scalar parameters from Excel to module-level defaults."""
    SystemUtilities.apply_params_to_globals(params)

def _nanfill2d(A: np.ndarray) -> np.ndarray:
    """In-place-like forward/backward fill of NaNs first across rows, then across columns."""
    return SystemUtilities.nanfill2d(A)

# =========  7 - GRID AND PLOTTING ========================
class GridAndPlotting:
    """Handles computational grid generation and data preparation for visualization."""
    
    @staticmethod
    def compute_sep_grid_maxlever(aero: PyAerodynamicsWrapper, engine: EngineWrapper, ref_mass_kg: float,
                                  M_grid: np.ndarray | None = None,
                                  H_grid: np.ndarray | None = None):
        """Compute specific excess power Ps = ((T-D)V)/W at maximum lever for visualization backgrounds."""
        if M_grid is None: M_grid = aero.mach_grid
        if H_grid is None: H_grid = aero.alt_grid_m
        Ps = np.full((len(H_grid), len(M_grid)), np.nan)
        W = ref_mass_kg * G_C
        for k, h in enumerate(H_grid):
            a = a_from_altitude(float(h))
            for i, M in enumerate(M_grid):
                V = max(a*float(M), 0.1)
                T_per = engine.thrust_with_lever(1.0, M, h)  # max lever
                if T_per is None:
                    continue
                T_tot = T_per * SystemConfiguration.N_ENGINES
                D = aero.get_drag(M, h)
                Ps[k, i] = ((T_tot - D) * V) / W
        return M_grid, H_grid, Ps
    
    @staticmethod
    def find_lever_for_thrust(eng: EngineWrapper, required_thrust_total: float,
                              mach: float, altitude_m: float,
                              lever_grid=None, allow_refine=True):
        """
        Determine lever to meet required total thrust (distributed evenly per engine).
        Returns (lever, T_per_engine, thrust_limited_flag) or (None, None, False) if not resolvable.
        """
        thrust_limited = False
        T_req = float(required_thrust_total) / float(SystemConfiguration.N_ENGINES)

        if lever_grid is None:
            lever_grid = np.linspace(0.0, 1.0, 21)

        def safe_thrust(lv):
            Tv = eng.thrust_with_lever(float(lv), float(mach), float(altitude_m))
            return Tv

        thrusts = [safe_thrust(lv) for lv in lever_grid]
        valid_idx = [i for i, Tv in enumerate(thrusts) if Tv is not None]

        if not valid_idx:
            return None, None, thrust_limited

        # enforce weak monotonicity
        for i in range(1, len(lever_grid)):
            if (thrusts[i] is not None) and (thrusts[i-1] is not None) and (thrusts[i] < thrusts[i-1]):
                thrusts[i] = thrusts[i-1]

        T0 = thrusts[0]; T1 = thrusts[-1]

        if (T0 is not None) and (T0 >= T_req):
            return float(lever_grid[0]), T0, thrust_limited

        if (T1 is not None) and (T1 <= T_req):
            thrust_limited = True
            return float(lever_grid[-1]), T1, thrust_limited

        # interpolate
        for i in range(len(lever_grid)-1):
            T_curr = thrusts[i]; T_next = thrusts[i+1]
            if (T_curr is not None) and (T_next is not None):
                if T_curr <= T_req <= T_next:
                    if allow_refine:
                        # refine with smaller grid
                        fine_grid = np.linspace(lever_grid[i], lever_grid[i+1], 11)
                        fine_thrusts = [safe_thrust(lv) for lv in fine_grid]
                        for j in range(len(fine_grid)-1):
                            if (fine_thrusts[j] is not None) and (fine_thrusts[j+1] is not None):
                                if fine_thrusts[j] <= T_req <= fine_thrusts[j+1]:
                                    t = (T_req - fine_thrusts[j]) / (fine_thrusts[j+1] - fine_thrusts[j])
                                    lever = fine_grid[j] + t * (fine_grid[j+1] - fine_grid[j])
                                    T_actual = safe_thrust(lever)
                                    return float(lever), T_actual, thrust_limited
                    else:
                        # simple linear interpolation
                        t = (T_req - T_curr) / (T_next - T_curr)
                        lever = lever_grid[i] + t * (lever_grid[i+1] - lever_grid[i])
                        T_actual = safe_thrust(lever)
                        return float(lever), T_actual, thrust_limited

        return None, None, thrust_limited

# Create global grid and plotting instance
_grid_and_plotting = GridAndPlotting()

# Backward compatibility functions
def compute_sep_grid_maxlever(aero: PyAerodynamicsWrapper, engine: EngineWrapper, ref_mass_kg: float,
                              M_grid: np.ndarray | None = None,
                              H_grid: np.ndarray | None = None):
    """Compute specific excess power Ps = ((T-D)V)/W at maximum lever for visualization backgrounds."""
    return GridAndPlotting.compute_sep_grid_maxlever(aero, engine, ref_mass_kg, M_grid, H_grid)

def find_lever_for_thrust(eng: EngineWrapper, required_thrust_total: float,
                          mach: float, altitude_m: float,
                          lever_grid=None, allow_refine=True):
    """Determine lever to meet required total thrust (distributed evenly per engine)."""
    return GridAndPlotting.find_lever_for_thrust(eng, required_thrust_total, mach, altitude_m, lever_grid, allow_refine)

# =========  4.5 - PLOTTING CONFIGURATION ========================
class PlottingConfig:
    """Configuration constants for visualization and graphical representation."""
    
    # Specific excess power contour levels for visualization
    PS_LEVELS = np.array(
        [-30,-25,-20,-15,-12,-10,-8,-6,-4,-2,-1,-0.5,
          0.5,1,2,3,4,5,6,8,10,12,15,20,24,25],
        dtype=float
    )
    
    # User interface visualization limits
    M_XMAX_UI = 1.25  # Maximum Mach number for SEP x-axis visualization

# Create global plotting config instance
_plotting_config = PlottingConfig()

# =========  4.6 - STRATEGY CONFIGURATION ========================
class StrategyConfig:
    """Configuration constants for strategy simulation and energy management."""
    
    # Strategy energy split magnitude
    E_DOT_CMD = 14  # [m/s] (split between climb & speed by each strategy)

# Create global strategy config instance
_strategy_config = StrategyConfig()

# =========  4.7 - GRID CONFIGURATION ========================
class GridConfig:
    """Configuration constants for grids, axes, and UI layout."""
    
    # Target and axis settings
    TARGET_ALT_M = 10000.0
    Y_AXIS_TOP_M = 14000.0
    
    # Grid resolution settings
    ALT_STEP_M = 200.0
    MACH_COLS = 81
    N_PLOT_STEPS = 50  # uniform # of points per trajectory

# Create global grid config instance
_grid_config = GridConfig()

# Backward compatibility constants 
PS_LEVELS = PlottingConfig.PS_LEVELS
M_XMAX_UI = PlottingConfig.M_XMAX_UI
E_DOT_CMD = StrategyConfig.E_DOT_CMD
TARGET_ALT_M = GridConfig.TARGET_ALT_M
Y_AXIS_TOP_M = GridConfig.Y_AXIS_TOP_M
ALT_STEP_M = GridConfig.ALT_STEP_M
MACH_COLS = GridConfig.MACH_COLS
N_PLOT_STEPS = GridConfig.N_PLOT_STEPS
MACH_TRAJECTORY_GUIDANCE = ClimbingCore.PenaltySystem.MACH_TRAJECTORY_GUIDANCE
LEVER_PENALTY_GUIDANCE = ClimbingCore.PenaltySystem.LEVER_PENALTY_GUIDANCE
TARGET_MACH_TOLERANCE = ClimbingCore.PenaltySystem.TARGET_MACH_TOLERANCE

# Backward compatibility for old PenaltySystem class
PenaltySystem = ClimbingCore.PenaltySystem

# Backward compatibility functions
def compute_mach_penalty(current_mach: float, target_mach: float, prev_mach: float = None, 
                         altitude_fraction: float = None) -> float:
 
    return ClimbingCore.PenaltySystem.compute_mach_penalty(current_mach, target_mach, prev_mach, altitude_fraction)

def compute_lever_penalty(current_lever: float, altitude_fraction: float = None) -> float:
 
    return ClimbingCore.PenaltySystem.compute_lever_penalty(current_lever, altitude_fraction)

# Backward compatibility for 3D Dynamic Programming functions
def solve_dp_3d_fixed_mass(aero: PyAerodynamicsWrapper, eng: EngineWrapper,
                          M_grid: np.ndarray, H_sched: np.ndarray,
                          lever_samples: int = 10,
                          target_mach: float = None,
                          target_mach_tolerance: float = 0.02,
                          start_mach: float = None,
                          start_lever: float = None):
    """Backward compatibility wrapper for ClimbingCore.solve_dp_3d_fixed_mass"""
    return ClimbingCore.solve_dp_3d_fixed_mass(aero, eng, M_grid, H_sched, lever_samples, 
                                             target_mach, target_mach_tolerance, start_mach, start_lever)

def compute_3d_cost(aero: PyAerodynamicsWrapper, eng: EngineWrapper, 
                   altitude: float, mach: float, lever: float,
                   target_mach: float = None, prev_mach: float = None,
                   altitude_fraction: float = None, mass_kg: float = None) -> float:
    """Backward compatibility wrapper for ClimbingCore.compute_3d_cost"""
    return ClimbingCore.compute_3d_cost(aero, eng, altitude, mach, lever, target_mach, prev_mach, altitude_fraction, mass_kg)

# Backward compatibility for Strategy System classes
StrategyProfiles = ClimbingCore.StrategyManager.StrategyProfiles
StrategyRun = ClimbingCore.EnergyCalculator.StrategyRun

# Backward compatibility wrappers
def compute_full_engine_envelope(aero: PyAerodynamicsWrapper, eng: EngineWrapper, M_grid: np.ndarray, H_sched: np.ndarray, lever_samples: int = 50):
    """Backward compatibility wrapper for ClimbingCore.compute_full_engine_envelope"""
    return ClimbingCore.compute_full_engine_envelope(aero, eng, M_grid, H_sched, lever_samples)

def check_envelope_exceedance(strategy, aero: PyAerodynamicsWrapper):
    """Backward compatibility wrapper for ClimbingCore.check_envelope_exceedance"""
    return ClimbingCore.check_envelope_exceedance(strategy, aero)

# =========  5 - AERODYNAMICS SYSTEM =================
# Note: AeroTables class removed - now using PyAerodynamicsWrapper from pyaerodynamics_wrapper.py



@dataclass
class MinFuelSchedule:
    alt_m: np.ndarray
    mach: np.ndarray
    fuel_est_kg: float
    J_kg_per_m: np.ndarray
    mdot_kgps: np.ndarray
    Ps_mps: np.ndarray
    T_total_N: np.ndarray
    D_N: np.ndarray
    lever: np.ndarray
    T_per_engine_N: np.ndarray
    mass_kg: np.ndarray
    thrust_limited: np.ndarray
    dt_s: np.ndarray
    dFuel_kg: np.ndarray
    cumFuel_kg: np.ndarray

# =========  8 - BACKGROUND COMPUTATION =============
# Note: compute_sep_grid_maxlever moved to GridAndPlotting class

# =========  9 - STRATEGY SYSTEM =====================
# Note: StrategyProfiles is now properly nested inside ClimbingCore class

# =========  10 - ENGINE CONTROL SYSTEM ==============
# Note: find_lever_for_thrust moved to GridAndPlotting class

# =========  11 - STRATEGY RUN CONTAINER =============
@dataclass
class StrategyRun:
    label: str
    alt_m: np.ndarray
    mach: np.ndarray
    time_s: np.ndarray
    lever: np.ndarray
    T_total_N: np.ndarray
    D_N: np.ndarray
    Ps_mps: np.ndarray
    mdot_kgps: np.ndarray
    dt_s: np.ndarray
    dFuel_kg: np.ndarray
    cumFuel_kg: np.ndarray
    thrust_limited: np.ndarray
    fuel_total_kg: float

# ========= BACKWARD COMPATIBILITY WRAPPERS =========
def resample_strategy_run(sr: StrategyRun, n_samples: int) -> StrategyRun:
    """Backward compatibility wrapper for ClimbingCore.resample_strategy_run"""
    return ClimbingCore.resample_strategy_run(sr, n_samples)

# =========  12 - STRATEGY INTEGRATOR ================
def simulate_strategy_path(*, label: str, aero: PyAerodynamicsWrapper, eng: EngineWrapper,
                           mass0_kg: float, h0_m: float, V0_ms: float,
                           target_alt_m: float, dt: float,
                           strategy_fn: Callable[[float,float,Optional[float]], tuple],
                           altitude_fraction: Optional[float]) -> StrategyRun:
    """Backward compatibility wrapper for ClimbingCore.simulate_strategy_path"""
    return ClimbingCore.simulate_strategy_path(
        label=label, aero=aero, eng=eng, mass0_kg=mass0_kg, h0_m=h0_m, V0_ms=V0_ms,
        target_alt_m=target_alt_m, dt=dt, strategy_fn=strategy_fn, altitude_fraction=altitude_fraction
    )

# =========  13 - STRATEGY BUILDER ===================
# Note: build_strategy_set moved to ClimbingCore.StrategyManager class