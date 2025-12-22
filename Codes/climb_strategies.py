# ========================================================================
# CLIMB STRATEGY COMPARISON MODULE
# ========================================================================
"""
Energy allocation strategies for climb trajectory comparison.

Mathematical basis: Energy rate equation Ė = mg·ḣ + mV·V̇
Energy allocation: Ė_cmd = w_c·Ė_cmd (climb) + w_s·Ė_cmd (speed)

Strategy types:
    - Linear: Constant weights w_c, w_s throughout climb
    - Exponential: Weights evolve as w(ξ) where ξ = h/h_target
    - Constant speed: V = const → w_c = 1, w_s = 0
    - Constant Mach: M = const → dV/dt = M·(da/dh)·(dh/dt)

Provides comparison baseline for DP-optimized trajectories.
Configurable via ENABLE_STRATEGY_COMPARISON in mission_config.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Callable, List

# Aircraft configuration: N_ENGINES, g_c, atmospheric models
from aircraft_config import (
    SystemConfiguration,
    a_from_altitude, _atmospheric_properties,
    G_C
)

# Mission parameters: Ė_cmd [m/s] (specific energy rate)
from mission_config import (
    E_DOT_CMD_CLIMB
)

# External model interfaces: aerodynamics D(M,h,m) and propulsion T(δ,M,h)
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper

# Utility functions: thrust solving, energy calculations, fuel flow
from mission_utils import (
    find_lever_for_thrust,
    calculate_specific_excess_power,
    calculate_fuel_flow_rate_safe,
    validate_tsfc
)

# Data structure and debug utilities from climb module
from climb import StrategyRun, dbg


# ========================================================================
# SECTION 1: STRATEGY PROFILE DEFINITIONS
# ========================================================================

class StrategyProfiles:
    """Energy allocation profiles for climb strategy comparison."""
    
    class FixedEnergy:
        """Constant energy allocation: w_c, w_s = const throughout climb."""
        
        class Linear:
            """Linear strategy: w_c = AF (constant), w_s = 1 - AF."""
            
            @staticmethod
            def profile(altitude, velocity, altitude_fraction):
                """
                Constant energy split: w_c = AF, w_s = 1-AF.
                
                Energy allocation:
                    ḣ = w_c · Ė_cmd
                    V̇ = (w_s · Ė_cmd · g) / V
                
                Parameters:
                    altitude: h [m] (unused, constant weights)
                    velocity: V [m/s] (unused, constant weights)
                    altitude_fraction: AF ∈ [0,1] - climb weight parameter
                    
                Returns:
                    (w_c, w_s): energy allocation weights
                """
                af = float(np.clip(altitude_fraction, 0.0, 1.0))
                cw = af          # w_c = AF
                sw = 1.0 - af    # w_s = 1 - AF
                return cw, sw
        
        class Exponential:
            """Exponential energy allocation: w_c = f(ξ), ξ = h/h_target."""
            
            @staticmethod
            def increasing_climb(altitude, velocity, altitude_fraction):
                """
                Exponential climb-biased: w_c increases with altitude.
                
                Formulation: w_c(ξ) = w_min + (w_max - w_min)·ξ^p
                where ξ = h/h_target, p = 1 + 2·AF
                
                Range: w_c ∈ [0.1, 0.9], w_s = 1 - w_c
                
                Parameters:
                    altitude: h [m] (unused)
                    velocity: V [m/s] (unused)
                    altitude_fraction: (AF, ξ) - tuple of (parameter, progress)
                    
                Returns:
                    (w_c, w_s): energy allocation weights
                """
                if not isinstance(altitude_fraction, tuple) or len(altitude_fraction) != 2:
                    raise ValueError(f"Expected tuple (AF, ξ), got {type(altitude_fraction)}")
                
                af_param, current_af = altitude_fraction
                af_param = float(np.clip(af_param, 0.0, 1.0))
                current_af = float(np.clip(current_af, 0.0, 1.0))
                
                # Weight bounds
                min_climb = 0.1
                max_climb = 0.9
                
                # Exponential parameter: p = 1 + 2·AF
                exponent = 1.0 + 2.0 * af_param
                exp_factor = current_af ** exponent
                
                # Map to [w_min, w_max]
                climb_norm = min_climb + (max_climb - min_climb) * exp_factor
                speed_norm = 1.0 - climb_norm
                
                return climb_norm, speed_norm
            
            @staticmethod
            def increasing_speed(altitude, velocity, altitude_fraction):
                """
                Exponential speed-biased: w_s increases with altitude.
                
                Formulation: w_s(ξ) = w_min + (w_max - w_min)·ξ^p
                where ξ = h/h_target, p = 1 + 2·AF
                
                Range: w_s ∈ [0.1, 0.9], w_c = 1 - w_s
                
                Parameters:
                    altitude: h [m] (unused)
                    velocity: V [m/s] (unused)
                    altitude_fraction: (AF, ξ) - tuple of (parameter, progress)
                    
                Returns:
                    (w_c, w_s): energy allocation weights
                """
                if not isinstance(altitude_fraction, tuple) or len(altitude_fraction) != 2:
                    raise ValueError(f"Expected tuple (AF, ξ), got {type(altitude_fraction)}")
                
                af_param, current_af = altitude_fraction
                af_param = float(np.clip(af_param, 0.0, 1.0))
                current_af = float(np.clip(current_af, 0.0, 1.0))
                
                # Weight bounds
                min_speed = 0.1
                max_speed = 0.9
                
                # Exponential parameter: p = 1 + 2·AF
                exponent = 1.0 + 2.0 * af_param
                exp_factor = current_af ** exponent
                
                # Map to [w_min, w_max]
                speed_norm = min_speed + (max_speed - min_speed) * exp_factor
                climb_norm = 1.0 - speed_norm
                
                return climb_norm, speed_norm
    
    class ConstantRates:
        """Constant kinematic quantities: V=const or M=const."""
        
        @staticmethod
        def constant_speed(altitude, velocity, altitude_fraction=None):
            """
            Constant velocity: V = const → dV/dt = 0.
            
            Energy allocation: All energy to climb
                w_c = 1, w_s = 0
                ḣ = Ė_cmd
                V̇ = 0
            
            Parameters:
                altitude: h [m] (unused)
                velocity: V [m/s] (maintained constant)
                altitude_fraction: unused
                
            Returns:
                (1.0, 0.0): all energy to climb
            """
            return 1.0, 0.0
        
        @staticmethod
        def constant_mach():
            """
            Constant Mach: M = const → dV/dt = M·(da/dh)·(dh/dt).
            
            Constraint: M = V/a = const
            Derivative: dV/dt = M·(da/dh)·(dh/dt)
            
            Returns:
                function: Strategy with _const_mach flag for special handling
            """
            def _const_mach(altitude, velocity, altitude_fraction=None):
                return 1.0, 0.0  # Primary energy to climb, V̇ computed from M=const
            _const_mach._const_mach = True  # Flag for special kinematic treatment
            return _const_mach


# ========================================================================
# SECTION 2: STRATEGY SIMULATION
# ========================================================================

class StrategyManager:
    """Strategy simulation and energy allocation computation."""
    
    @staticmethod
    def process_strategy_weights(strategy_fn: Callable, h: float, V: float, 
                                strategy_altitude_fraction, label: str, 
                                h_hist: list, current_altitude_fraction: float) -> tuple[float, float]:
        """
        Compute and validate energy allocation weights from strategy function.
        
        Algorithm:
            1. Call strategy: (w_c, w_s) = strategy_fn(h, V, AF)
            2. Validate: w_c, w_s ≥ 0, finite
            3. Normalize: (w_c, w_s) ← (w_c, w_s) / (w_c + w_s)
        
        Parameters:
            strategy_fn: Callable - strategy profile function
            h: h [m] - current altitude
            V: V [m/s] - current velocity
            strategy_altitude_fraction: AF or (AF,ξ) - strategy parameter
            label: str - strategy identifier
            h_hist: list - altitude history (for diagnostics)
            current_altitude_fraction: ξ - climb progress
            
        Returns:
            (w_c, w_s): normalized weights, w_c + w_s = 1
        """
        # Evaluate strategy function
        cw, sw = strategy_fn(h, V, strategy_altitude_fraction)
        
        # Validation: finite and non-negative
        if not np.isfinite(cw) or not np.isfinite(sw):
            dbg(f"[WARN] Non-finite weights for '{label}': w_c={cw}, w_s={sw} at h={h:.1f}m")
            cw, sw = 1.0, 0.0  # Fallback: climb-only
        
        if cw < 0 or sw < 0:
            dbg(f"[WARN] Negative weights for '{label}': w_c={cw}, w_s={sw} at h={h:.1f}m")
            cw, sw = max(0, cw), max(0, sw)
        
        # Normalization: ensure w_c + w_s = 1
        s = max(cw + sw, 1e-12)
        w_c, w_s = cw / s, sw / s
        
        # Diagnostic output (first few steps only)
        if len(h_hist) <= 3:
            if isinstance(strategy_altitude_fraction, tuple):
                af_str = f"(AF={strategy_altitude_fraction[0]:.3f}, ξ={strategy_altitude_fraction[1]:.3f})"
            elif strategy_altitude_fraction is not None:
                af_str = f"AF={strategy_altitude_fraction:.3f}"
            else:
                af_str = "N/A"
            dbg(f"[STRAT] {label} at h={h:.0f}m (ξ={current_altitude_fraction*100:.1f}%): w_c={w_c:.3f}, w_s={w_s:.3f}, {af_str}")
        
        return w_c, w_s
    
    @staticmethod
    def build_strategy_set() -> List[tuple[str, Callable, Optional[float]]]:
        """
        Construct full strategy catalog for comparative analysis.
        
        Strategy enumeration:
            - Linear: 5 AF values ∈ {0.10, 0.30, 0.50, 0.70, 0.90}
            - Exponential climb: 5 AF values (climb-biased)
            - Exponential speed: 5 AF values (speed-biased)
            - Constant speed: V = const
            - Constant Mach: M = const
        
        Returns:
            [(label, function, AF_param), ...]: strategy specifications
        """
        afs = [0.10, 0.30, 0.50, 0.70, 0.90]  # AF parameter sweep
        out = []
        
        # Linear strategies: w_c = AF
        for af in afs:
            out.append((f"Linear AF={af:.2f}", StrategyProfiles.FixedEnergy.Linear.profile, af))
        
        # Exponential climb-biased: w_c(ξ) = w_min + (w_max-w_min)·ξ^(1+2·AF)
        for af in afs:
            out.append((f"Exp climb AF={af:.2f}", StrategyProfiles.FixedEnergy.Exponential.increasing_climb, af))
        
        # Exponential speed-biased: w_s(ξ) = w_min + (w_max-w_min)·ξ^(1+2·AF)
        for af in afs:
            out.append((f"Exp speed AF={af:.2f}", StrategyProfiles.FixedEnergy.Exponential.increasing_speed, af))
        
        # Kinematic constraints
        out.append(("Constant speed", StrategyProfiles.ConstantRates.constant_speed, None))
        out.append(("Constant Mach",  StrategyProfiles.ConstantRates.constant_mach(), None))
        
        return out
    
    @staticmethod
    def simulate_strategy_path(*, label: str, aero: PyAerodynamicsWrapper, engine: EngineWrapper,
                              mass0_kg: float, h0_m: float, V0_ms: float,
                              target_alt_m: float, dt: float,
                              strategy_fn: Callable[[float,float,Optional[float]], tuple],
                              altitude_fraction: Optional[float]) -> StrategyRun:
        """
        Time-marching integration of climb strategy with fixed mass.
        
        Governing equations:
            Energy: Ė = w_c·Ė_cmd + w_s·Ė_cmd = mg·ḣ + mV·V̇
            Thrust: T = D + (mg·ḣ + mV·V̇)/V
            Kinematics: dh/dt = ḣ, dV/dt = V̇
        
        Integration: Forward Euler with time step Δt.
        Mass: m = m_0 (constant for strategy comparison).
        
        Parameters:
            label: str - strategy identifier
            aero: PyAerodynamicsWrapper - drag model D(M,h,m)
            engine: EngineWrapper - thrust model T(δ,M,h)
            mass0_kg: m_0 [kg] - initial mass (constant)
            h0_m: h_0 [m] - starting altitude
            V0_ms: V_0 [m/s] - starting velocity
            target_alt_m: h_target [m] - target altitude
            dt: Δt [s] - integration time step
            strategy_fn: Callable - energy allocation function
            altitude_fraction: AF - strategy parameter (or None)
            
        Returns:
            StrategyRun: complete trajectory X(t) with performance data
        """
        # Circular dependency resolution
        from climb import ClimbingCore
        from climb_plotting import GridConfig
        from aircraft_config import M_MIN_DEFAULT, M_MMO
        
        # Gravitational acceleration
        g0 = aero.G_C  # g_c [m/s²]

        # Initial velocity adjustment for constant strategies
        if "Constant speed" in label or "Constant Mach" in label:
            # Initialize at M=0.5 for constant strategies
            a = a_from_altitude(h0_m)
            V0_ms = 0.5 * a
            dbg(f"[STRAT] {label}: V_0={V0_ms:.1f} m/s (M_0={V0_ms/a:.3f}) at h_0={h0_m:.0f}m")

        # Initialize trajectory history arrays
        h_hist, V_hist, t_hist = [float(h0_m)], [float(V0_ms)], [0.0]
        lever_hist, Ttot_hist, D_hist, Ps_hist = [], [], [], []
        mdot_hist, dt_hist, dFuel_hist, cumFuel_hist, limited_hist = [], [], [], [0.0], []

        # Fixed mass for strategy comparison
        mass_kg = float(mass0_kg)  # m = m_0 (constant)

        # ════════════════════════════════════════════════════════════════
        # Time-Marching Integration Loop
        # ════════════════════════════════════════════════════════════════
        while h_hist[-1] < target_alt_m:
            h = float(h_hist[-1])      # h_k [m]
            V = float(V_hist[-1])      # V_k [m/s]
            t = float(t_hist[-1])      # t_k [s]
            a = a_from_altitude(h)     # a(h) [m/s]: speed of sound
            Mq = float(np.clip(V / max(a, 1e-9), M_MIN_DEFAULT, M_MMO))  # M [−]

            # Climb progress: ξ = (h - h_0) / (h_target - h_0)
            current_altitude_fraction = (h - h0_m) / (target_alt_m - h0_m) if target_alt_m > h0_m else 0.0
            current_altitude_fraction = np.clip(current_altitude_fraction, 0.0, 1.0)
            
            # Strategy parameter routing based on type
            if "Linear" in label:
                strategy_altitude_fraction = altitude_fraction  # AF (constant)
            elif "Exp" in label:
                strategy_altitude_fraction = (altitude_fraction, current_altitude_fraction)  # (AF, ξ)
            else:
                strategy_altitude_fraction = None  # Not used
            
            # Compute energy allocation weights (w_c, w_s)
            w_c, w_s = StrategyManager.process_strategy_weights(
                strategy_fn, h, V, strategy_altitude_fraction, label, h_hist, current_altitude_fraction
            )
            
            # Compute rates: ḣ, V̇
            dh_dt, dv_dt = StrategyManager.compute_energy_allocation(strategy_fn, w_c, w_s, h, V, a, Mq)
            
            # Aerodynamics and weight
            D = float(aero.get_drag(Mq, h, mass_kg))  # D [N]
            W = mass_kg * g0                          # W [N]
            
            # Required thrust: T = D + (mg·ḣ + mV·V̇)/V
            F_required_total = ClimbingCore.EnergyCalculator.compute_required_thrust(mass_kg, dh_dt, dv_dt, V, D)

            # Solve for lever position: find δ such that T(δ) = T_required
            lv, T_per, thrust_limited = find_lever_for_thrust(
                engine, F_required_total, Mq, h, 
                n_engines=SystemConfiguration.N_ENGINES, 
                lever_grid=None, allow_refine=True
            )
            
            if (lv is None) or (T_per is None):
                dbg(f"[WARN] No feasible lever at h={h:.1f}m, M={Mq:.3f}. Terminating '{label}'")
                if len(h_hist) < 2:
                    # Strategy failed at initialization
                    dbg(f"[ERROR] Strategy '{label}' infeasible at initial state")
                    return StrategyRun(
                        label=label, alt_m=np.array([h0_m]), mach=np.array([V0_ms/max(a_from_altitude(h0_m), 1e-6)]),
                        time_s=np.array([0.0]), lever=np.array([np.nan]), thrust_total_N=np.array([np.nan]),
                        D_N=np.array([np.nan]), Ps_mps=np.array([np.nan]), mdot_kgps=np.array([np.nan]),
                        dt_s=np.array([0.0]), dFuel_kg=np.array([0.0]), cumFuel_kg=np.array([0.0]),
                        thrust_limited=np.array([False]), fuel_total_kg=0.0
                    )
                break

            # Fuel flow: ṁ = TSFC · T
            tsfc = validate_tsfc(engine.tsfc_current(), fallback=np.nan)
            mdot_total = calculate_fuel_flow_rate_safe(tsfc, T_per, SystemConfiguration.N_ENGINES)

            # Specific excess power: Ps = (T-D)V/m
            T_tot = T_per * SystemConfiguration.N_ENGINES
            Ps = calculate_specific_excess_power(T_tot, D, mass_kg, V)

            # Adaptive time step: ensure h_new ≤ h_target
            dt_use = float(dt)
            if h + dh_dt * dt_use > target_alt_m and dh_dt > 0:
                dt_use = (target_alt_m - h) / dh_dt

            # Euler step: X_{k+1} = X_k + Ẋ_k · Δt
            h_new = h + dh_dt * dt_use    # h_{k+1} [m]
            V_new = max(V + dv_dt * dt_use, 1.0)  # V_{k+1} [m/s]
            t_new = t + dt_use            # t_{k+1} [s]

            # Fuel burned: Δm_fuel = ṁ · Δt
            burned = (mdot_total * dt_use) if np.isfinite(mdot_total) else 0.0

            # Append to history
            lever_hist.append(lv)
            Ttot_hist.append(T_per * SystemConfiguration.N_ENGINES)
            D_hist.append(D)
            Ps_hist.append(Ps)
            mdot_hist.append(mdot_total if np.isfinite(mdot_total) else np.nan)
            dt_hist.append(dt_use)
            dFuel_hist.append(burned)
            cumFuel_hist.append(cumFuel_hist[-1] + burned)
            limited_hist.append(bool(thrust_limited))

            h_hist.append(h_new)
            V_hist.append(V_new)
            t_hist.append(t_new)

        # ════════════════════════════════════════════════════════════════
        # Post-Processing: Convert to Arrays and Pad
        # ════════════════════════════════════════════════════════════════
        # Import padding utility
        from mission_utils import pad_array_to_length
        
        # State arrays
        alt = np.asarray(h_hist, float)   # h(t) [m]
        V = np.asarray(V_hist, float)     # V(t) [m/s]
        time = np.asarray(t_hist, float)  # t [s]
        
        # Mach calculation: M = V/a(h)
        def safe_mach_calc(hh):
            a = a_from_altitude(float(hh))
            return max(a, 1e-6)
        mach = V / np.vectorize(safe_mach_calc)(alt)
        
        # Pad performance arrays to match state array length
        base_length = len(h_hist)
        
        lever = pad_array_to_length(
            np.asarray(lever_hist, float), base_length, 
            lever_hist[-1] if lever_hist else np.nan
        )
        Ttot = pad_array_to_length(
            np.asarray(Ttot_hist, float), base_length,
            Ttot_hist[-1] if Ttot_hist else np.nan
        )
        Darr = pad_array_to_length(
            np.asarray(D_hist, float), base_length,
            D_hist[-1] if D_hist else np.nan
        )
        Psarr = pad_array_to_length(
            np.asarray(Ps_hist, float), base_length,
            Ps_hist[-1] if Ps_hist else np.nan
        )
        mdot = pad_array_to_length(
            np.asarray(mdot_hist, float), base_length,
            mdot_hist[-1] if mdot_hist else np.nan
        )
        limited = pad_array_to_length(
            np.asarray(limited_hist, bool), base_length,
            limited_hist[-1] if limited_hist else False
        )
        
        # Time and fuel increments (prepend zero for t_0, m_0)
        dtarr = np.asarray([0.0] + dt_hist, float)    # Δt_k [s]
        dFuel = np.asarray([0.0] + dFuel_hist, float) # Δm_fuel,k [kg]
        cumF = np.asarray(cumFuel_hist, float)        # m_fuel(t) [kg]
        fuel_total = float(cumF[-1])                  # Total fuel [kg]

        return StrategyRun(
            label=label,
            alt_m=alt,
            mach=mach,
            time_s=time,
            lever=lever,
            thrust_total_N=Ttot,
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
    def compute_energy_allocation(strategy_fn: Callable, w_c: float, w_s: float, 
                                 h: float, V: float, a: float, current_mach: float) -> tuple[float, float]:
        """
        Compute rates ḣ and V̇ from energy allocation weights.
        
        Standard formulation (V=const or exponential):
            Energy balance: Ė = mg·ḣ + mV·V̇ = (w_c + w_s)·Ė_cmd
            Climb rate: ḣ = w_c · Ė_cmd
            Acceleration: V̇ = (w_s · Ė_cmd · g) / V
        
        Constant Mach formulation:
            Constraint: M = V/a(h) = const
            Kinematic coupling: dV/dt = M · (da/dh) · (dh/dt)
            Climb rate: ḣ = w_c · Ė_cmd
            Acceleration: V̇ = M · (da/dh) · ḣ
        
        Parameters:
            strategy_fn: Callable - strategy function (checked for _const_mach flag)
            w_c: float - climb weight ∈ [0,1]
            w_s: float - speed weight ∈ [0,1]
            h: h [m] - current altitude
            V: V [m/s] - current velocity
            a: a [m/s] - speed of sound at h
            current_mach: M [-] - current Mach number
            
        Returns:
            (ḣ, V̇): climb rate [m/s], acceleration [m/s²]
        """
        from climb_plotting import GridConfig
        
        if getattr(strategy_fn, "_const_mach", False):
            # Constant Mach: compute da/dh via finite difference
            eps = GridConfig.TARGET_ALT_M / GridConfig.N_PLOT_STEPS
            a1 = _atmospheric_properties.a_from_altitude(h - eps/2)
            a2 = _atmospheric_properties.a_from_altitude(h + eps/2)
            dadh = (a2 - a1) / eps  # da/dh [m/s per m]
            
            # Rates with M=const constraint
            dh_dt = w_c * E_DOT_CMD_CLIMB         # ḣ [m/s]
            dv_dt = current_mach * dadh * dh_dt   # V̇ = M·(da/dh)·ḣ [m/s²]
        else:
            # Standard energy allocation
            # Ė = mg·ḣ + mV·V̇ where Ė_cmd = specific energy rate [m/s]·[m/s²]
            dh_dt = w_c * E_DOT_CMD_CLIMB  # ḣ [m/s]
            dv_dt = (w_s * E_DOT_CMD_CLIMB * G_C) / max(V, 1e-9)  # V̇ [m/s²]
        
        return dh_dt, dv_dt

