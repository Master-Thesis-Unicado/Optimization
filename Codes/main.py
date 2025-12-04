"""
Complete Mission Analysis with Climb, Cruise, and Descent Optimization

This script performs comprehensive three-phase mission analysis:
1. 3D Dynamic Programming climb optimization 
2. Steady-level cruise simulation 
3. 3D Dynamic Programming descent optimization

Output includes interactive plots, performance analysis, and complete mission summary.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import time

from aircraft_config import (
    INITIAL_MASS_KG,
    W_FUEL_KG, W_OE_KG, W_PL_KG,
    LEVER_MIN, LEVER_MAX
)
from atmosphere import a_from_altitude, isa_properties
from mission_config import (
    # Pre-computation grid parameters
    N_MACH_SAMPLES_PRECOMPUTE, N_ALTITUDE_SAMPLES_PRECOMPUTE, N_LEVER_SAMPLES_PRECOMPUTE,
    ALTITUDE_MAX_PRECOMPUTE_M,
    # Climb phase parameters
    TARGET_ALT_CLIMB_M,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE, STRATEGY_DT_CLIMB_S,
    # Cruise phase parameters
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S, 
    # Descent phase parameters
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT,
    MIN_DESCENT_MACH, MAX_DESCENT_MACH,
    # Other settings
    ENABLE_STRATEGY_COMPARISON
)
import climb
from climb import ClimbingCore
from climb_plotting import (
    compute_sep_grid_maxlever,
    compute_full_envelope
)
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper
from cruise import run_cruise_simulation
from cruise_plotting import plot_performance_2d as plot_cruise_performance_2d
from climb_plotting import (
    plot_3d_cost_space, plot_performance_2d as plot_climb_performance_2d,
    GridConfig
)

# Conditional import for strategy comparison
if ENABLE_STRATEGY_COMPARISON:
    from climb_strategies import StrategyManager
    from climb_strategies_plotting import plot_strategies_interactive, create_strategy_comparison_plots
from descent import run_optimization as run_descent_optimization, DescentCore
from descent_plotting import (
    plot_performance_2d as plot_descent_performance_2d, 
    plot_3d_cost_space as plot_descent_3d_cost_space
)
from mission_summary import (
    plot_mission_summary_dashboard, 
    plot_combined_performance_analysis,
    plot_complete_mission_3d
)
from mission_excel_export import export_mission_to_excel
from cg_plotting import plot_cg_analysis
from cg_x_calculation import record_mission_history

# ========= CONSTANTS AND SETTINGS =============================================

# Aircraft performance constants


def main():
    # Record simulation start time
    simulation_start_time = time.time()
    
    print("\n" + "="*80)
    print("AIRCRAFT CONFIGURATION")
    print("="*80)
    print(f"[CONFIG] W_FUEL_KG: {W_FUEL_KG:.2f} kg")
    print(f"[CONFIG] W_OE_KG: {W_OE_KG:.2f} kg")
    print(f"[CONFIG] W_PL_KG: {W_PL_KG:.2f} kg")
    print(f"[CONFIG] INITIAL_MASS_KG (W_TO_KG): {INITIAL_MASS_KG:.2f} kg")
    print(f"[CONFIG] = W_OE + W_PL + W_FUEL = {W_OE_KG + W_PL_KG + W_FUEL_KG:.2f} kg")
    print("="*80 + "\n")
    
    print("[MISSION] Centralized mission configuration initialized")
    print(f"[MISSION] Target altitude: {TARGET_ALT_CLIMB_M:.2f} m")
    print(f"[MISSION] Target Mach: {TARGET_MACH_CRUISE:.3f}")
    print(f"[MISSION] Cruise distance: {CRUISE_DISTANCE_KM:.2f} km")
    
    print("[CLIMB] Loading aerodynamics library")
    aero = PyAerodynamicsWrapper()

    # Pre-computation grids for engine and aerodynamic caching
    M_min, M_max = float(aero.mach_grid[0]), float(aero.mach_grid[-1])
    M_precompute = np.linspace(M_min, M_max, N_MACH_SAMPLES_PRECOMPUTE)
    H_precompute = np.linspace(START_ALTITUDE_CLIMB_M, 
                               ALTITUDE_MAX_PRECOMPUTE_M, 
                               N_ALTITUDE_SAMPLES_PRECOMPUTE)
    
    # Dense grids for visualization and background calculations
    M_dense = np.linspace(M_min, M_max, N_MACH_SAMPLES_CLIMB)
    H_plot = np.arange(START_ALTITUDE_CLIMB_M, 
                       GridConfig.Y_AXIS_TOP_M + 0.5*GridConfig.ALT_STEP_M, 
                       GridConfig.ALT_STEP_M)

    # Effective minimum Mach is set in aircraft_config
    print(f"[CLIMB] Effective minimum Mach: {climb.M_MIN_EFFECTIVE:.3f} (minimum Mach from data: {aero.mach_grid[0]:.3f})")

    print("[CLIMB] Loading engine model")
    
    eng = EngineWrapper("lls/stubs/engines/PW1127G-JM")

    # Performance optimization: Pre-compute engine grid for caching
    # Note: Drag is NOT pre-computed for DP optimization as it depends on dynamic mass
    # Drag cache populates naturally during mission with actual varying masses
    print("[CLIMB] Pre-computing engine grid for performance optimization")
    print(f"[CACHE] Engine pre-computation grid: {len(M_precompute)} Mach × {len(H_precompute)} Alt × {N_LEVER_SAMPLES_PRECOMPUTE+1} Lever")
    print(f"[CACHE] Grid spans: M=[{M_precompute[0]:.3f}, {M_precompute[-1]:.3f}], "
          f"h=[{H_precompute[0]:.0f}, {H_precompute[-1]:.0f}]m, δ=[0.0, 1.0]")
    
    # Create lever grid for pre-computation
    # Pre-computation grid covers all throttle positions for optimization and visualization
    lever_precompute = np.linspace(0.0, 1.0, N_LEVER_SAMPLES_PRECOMPUTE + 1)
    
    # Pre-compute engine values only (mass-independent)
    eng.precompute_grid(M_precompute, H_precompute, lever_precompute)
    
    # Drag pre-computation skipped: Drag = f(M, h, m) where mass varies during mission
    # Cache will populate naturally with actual dynamic masses during DP optimization
    print("[CACHE] Drag cache will populate dynamically during mission (mass-dependent)")

    print("[CLIMB] Computing background specific excess power grid at maximum lever position")
    mach_grid, H_plot, Ps_base = compute_sep_grid_maxlever(aero, eng, INITIAL_MASS_KG,
                                                        mach_grid=M_dense, H_grid=H_plot)

    # DP schedule on EXACT N_ALTITUDE_STEPS_CLIMB rows
    # Create uniform altitude steps using linspace for consistency with descent
    altitude_sched = np.linspace(START_ALTITUDE_CLIMB_M, 
                          TARGET_ALT_CLIMB_M, 
                          N_ALTITUDE_STEPS_CLIMB)
    print("[CLIMB] Solving 3D dynamic programming optimization for minimum fuel climb path")
    # Calculate starting Mach from takeoff velocity at start altitude
    a = a_from_altitude(START_ALTITUDE_CLIMB_M)
    start_mach = START_VELOCITY_CLIMB_MS / a
    
    dp_sched, dp_info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_dp(
        aero, eng, mach_grid, altitude_sched, 
        lever_samples=N_LEVER_SAMPLES_CLIMB, 
        target_mach=TARGET_MACH_CRUISE,
        target_mach_tolerance=TARGET_MACH_TOLERANCE, 
        start_mach=start_mach, 
        start_lever=START_LEVER_CLIMB
    )

    # ========= STRATEGY COMPARISON (OPTIONAL) =========================================
    if ENABLE_STRATEGY_COMPARISON:
        # Strategies (fixed mass), resampled to N_ALTITUDE_STEPS_CLIMB
        print("[CLIMB] Simulating alternative climb strategies")
        strategies: list[climb.StrategyRun] = []
        for name, fn, af in StrategyManager.build_strategy_set():
            sr = StrategyManager.simulate_strategy_path(
                label=name,
                aero=aero, engine=eng,
                mass0_kg=INITIAL_MASS_KG, 
                h0_m=START_ALTITUDE_CLIMB_M, 
                V0_ms=START_VELOCITY_CLIMB_MS,
                target_alt_m=TARGET_ALT_CLIMB_M, 
                dt=STRATEGY_DT_CLIMB_S,
                strategy_fn=fn, altitude_fraction=af
            )
            sr = ClimbingCore.resample_strategy_run(sr, N_ALTITUDE_STEPS_CLIMB)
            strategies.append(sr)

        # Add DP as a strategy
        # Convert MinFuelSchedule to StrategyRun
        alt = np.asarray(dp_sched.alt_m, float)
        mach = np.asarray(dp_sched.mach, float)
        dt = np.asarray(dp_sched.dt_s, float)
        time_array = np.cumsum(np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0))
        
        dp_run = climb.StrategyRun(
            label="3D DP (Global Optimization)",
            alt_m=alt,
            mach=mach,
            time_s=time_array,
            lever=np.asarray(dp_sched.lever, float),
            T_total_N=np.asarray(dp_sched.T_total_N, float),
            D_N=np.asarray(dp_sched.D_N, float),
            Ps_mps=np.asarray(dp_sched.Ps_mps, float),
            mdot_kgps=np.asarray(dp_sched.mdot_kgps, float),
            dt_s=dt,
            dFuel_kg=np.asarray(dp_sched.dFuel_kg, float),
            cumFuel_kg=np.asarray(dp_sched.cumFuel_kg, float),
            thrust_limited=np.asarray(dp_sched.thrust_limited, bool),
            fuel_total_kg=float(np.nan_to_num(dp_sched.cumFuel_kg, nan=0.0)[-1])
        )
        # Fix time to start from 0
        dp_run.time_s = dp_run.time_s - dp_run.time_s[0]  # Shift so first time is 0
        dp_run = ClimbingCore.resample_strategy_run(dp_run, N_ALTITUDE_STEPS_CLIMB)
        strategies.append(dp_run)
        
        # Align all strategies to start from the same Mach value as the minimum fuel path
        dp_start_mach = dp_run.mach[0] if len(dp_run.mach) > 0 else 0.2
        print(f"[CLIMB] Aligning all strategies to start from Mach {dp_start_mach:.3f}")
        
        # Align strategies to start from DP's first Mach value (skip DP and constant strategies)
        for strategy in strategies:
            if (strategy.label not in ["3D DP (Global Optimization)"] and 
                "Constant speed" not in strategy.label and 
                "Constant Mach" not in strategy.label):
                mach_array = np.array(strategy.mach, dtype=float)
                mach_offset = mach_array[0] - dp_start_mach
                strategy.mach = mach_array - mach_offset

        # Print comprehensive strategy comparison table
        print("\n" + "="*80)
        print("CLIMBING STRATEGIES COMPARISON")
        print("="*80)
        print(f"{'Strategy':<30} {'Fuel (kg)':<12} {'Time (min)':<12} {'Final Mach':<12} {'Envelope':<14}")
        print("-"*80)
        
        # Sort strategies by fuel consumption for ranking
        strategies_sorted = sorted(strategies, key=lambda s: s.fuel_total_kg)
        
        for i, strategy in enumerate(strategies_sorted):
            final_mach = strategy.mach[-1] if len(strategy.mach) > 0 else 0.0
            avg_ps = np.mean(strategy.Ps_mps) if len(strategy.Ps_mps) > 0 else 0.0
            time_min = strategy.time_s[-1] / 60.0 if len(strategy.time_s) > 0 else 0.0
            envelope_status = ClimbingCore.check_envelope_exceedance(strategy, aero)
            
            # Add ranking indicator
            rank_indicator = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1:2d}."
            
            print(f"{rank_indicator} {strategy.label:<27} {strategy.fuel_total_kg:<12.2f} {time_min:<12.2f} {final_mach:<12.3f} {envelope_status:<14}")
        
        print("-"*80)
        print("Ranking based on total fuel consumption (lower is better)")
        print("="*80)
        
        # Additional analysis
        print("\nDETAILED ANALYSIS:")
        print("-"*80)
        
        # Find best and worst strategies for different metrics
        best_fuel = min(strategies, key=lambda s: s.fuel_total_kg)
        worst_fuel = max(strategies, key=lambda s: s.fuel_total_kg)
        
        best_time = min(strategies, key=lambda s: s.time_s[-1] if len(s.time_s) > 0 else float('inf'))
        worst_time = max(strategies, key=lambda s: s.time_s[-1] if len(s.time_s) > 0 else 0)
        
        print(f"Most Fuel Efficient: {best_fuel.label} ({best_fuel.fuel_total_kg:.2f} kg)")
        print(f"Least Fuel Efficient: {worst_fuel.label} ({worst_fuel.fuel_total_kg:.2f} kg)")
        print(f"Fuel Difference: {worst_fuel.fuel_total_kg - best_fuel.fuel_total_kg:.2f} kg ({((worst_fuel.fuel_total_kg - best_fuel.fuel_total_kg) / best_fuel.fuel_total_kg * 100):.2f}%)")
        
        if len(best_time.time_s) > 0 and len(worst_time.time_s) > 0:
            print(f"Fastest: {best_time.label} ({best_time.time_s[-1]/60:.2f} min)")
            print(f"Slowest: {worst_time.label} ({worst_time.time_s[-1]/60:.2f} min)")
            print(f"Time Difference: {(worst_time.time_s[-1] - best_time.time_s[-1])/60:.2f} min")
        
        print("\n" + "="*80)

        # Create additional comparison plots
        print("[CLIMB] Creating strategy comparison plots")
        create_strategy_comparison_plots(strategies, aero)
        
        # Interactive strategy comparison window
        print("[CLIMB] Opening interactive strategy comparison window")
        _ = plot_strategies_interactive(
            mach_grid, H_plot, Ps_base, strategies,
            title_suffix=f"Target Alt={TARGET_ALT_CLIMB_M:.2f} m, Ref mass={INITIAL_MASS_KG:.2f} kg"
        )
    else:
        print("[CLIMB] Strategy comparison disabled (ENABLE_STRATEGY_COMPARISON = False)")
    
    # Create climb performance analysis plot
    print("[CLIMB] Creating climb performance analysis plots")
    plot_climb_performance_2d(dp_sched, dp_info)
    # Extract minimum-fuel path for overlay
    min_path = {
        'mach': np.asarray(dp_sched.mach, float),
        'alt': np.asarray(dp_sched.alt_m, float),
        'lever': np.asarray(dp_sched.lever, float),
    }
    # Compute full engine envelope for 3D visualization
    # Use initial mass as reference for envelope visualization
    print("[CLIMB] Computing full engine performance envelope for 3D visualization")
    lever_grid_envelope = np.linspace(LEVER_MIN, LEVER_MAX, N_LEVER_SAMPLES_CLIMB)
    J_envelope = compute_full_envelope(aero, eng, mach_grid, altitude_sched, lever_grid_envelope,
                                       mass_kg=INITIAL_MASS_KG)
    
    # Show the 3D J plot for 3D DP with full engine envelope in a separate window (browser)
    print("[CLIMB] Opening 3D dynamic programming visualization with full engine envelope")
    
    # Use full engine envelope for rich 3D visualization
    mach_grid_3d = mach_grid  # Use the full Mach grid for rich visualization
    H_grid_3d = altitude_sched  # Use the same altitude grid as 3D DP
    lever_grid_3d = lever_grid_envelope  # Use the envelope lever grid
    
    
    plot_3d_cost_space(mach_grid_3d, H_grid_3d, lever_grid_3d, J_envelope, min_path=min_path,
                       title="3D DP (Global Optimization)<br>Full Engine Envelope with Optimal Path")
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("="*80)
    engine_stats = eng.get_cache_stats()
    drag_stats = aero.get_cache_stats()
    
    print(f"Engine Cache Performance:")
    print(f"  Cache hits: {engine_stats['hits']:,}")
    print(f"  Cache misses: {engine_stats['misses']:,}")
    print(f"  Hit rate: {engine_stats['hit_rate']:.2%}")
    print(f"  Cache size: {engine_stats['cache_size']:,} entries")
    
    print(f"\nDrag Cache Performance:")
    print(f"  Cache hits: {drag_stats['hits']:,}")
    print(f"  Cache misses: {drag_stats['misses']:,}")
    print(f"  Hit rate: {drag_stats['hit_rate']:.2%}")
    print(f"  Cache size: {drag_stats['cache_size']:,} entries")
    
    total_hits = engine_stats['hits'] + drag_stats['hits']
    total_calls = engine_stats['hits'] + engine_stats['misses'] + drag_stats['hits'] + drag_stats['misses']
    overall_hit_rate = total_hits / total_calls if total_calls > 0 else 0
    
    print(f"\nOverall Cache Performance:")
    print(f"  Total cache hits: {total_hits:,}")
    print(f"  Total function calls: {total_calls:,}")
    print(f"  Overall hit rate: {overall_hit_rate:.2%}")
    print("="*80)
    
    # ========= CRUISE PHASE SIMULATION =========================================
    print("\n" + "="*80)
    print("CRUISE PHASE SIMULATION")
    print("="*80)
    
    # Cruise simulation parameters from mission configuration
    cruise_distance_km = CRUISE_DISTANCE_KM
    cruise_time_step_s = CRUISE_TIME_STEP_S
    
    # Run cruise simulation using 3D DP result
    try:
        cruise_results = run_cruise_simulation(
            climb_result=dp_sched,
            initial_mass_kg=INITIAL_MASS_KG,
            target_distance_km=cruise_distance_km,
            aero=aero,
            engine=eng,
            time_step_s=cruise_time_step_s,
            create_plots=True
        )
        
        # Print intermediate mission summary (climb + cruise only)
        print("\n" + "="*80)
        print("INTERMEDIATE MISSION SUMMARY (CLIMB + CRUISE PHASES)")
        print("="*80)
        
        # Climb summary
        climb_fuel = float(np.nan_to_num(dp_sched.cumFuel_kg, nan=0.0)[-1])
        climb_time_hours = float(np.sum(dp_sched.dt_s)) / 3600.0 if len(dp_sched.dt_s) > 0 else 0.0
        
        # Calculate climb distance and average fuel flow for consistency
        climb_distance_km = 0.0
        if len(dp_sched.dt_s) > 0 and len(dp_sched.mach) > 0:
            for i in range(len(dp_sched.dt_s)):
                if i < len(dp_sched.mach) and i < len(dp_sched.alt_m):
                    a = a_from_altitude(dp_sched.alt_m[i])
                    V_tas = dp_sched.mach[i] * a
                    climb_distance_km += V_tas * dp_sched.dt_s[i] / 1000.0
        climb_time_s = climb_time_hours * 3600.0
        climb_avg_fuel_flow_kg_h = (climb_fuel / climb_time_s * 3600.0) if climb_time_s > 0 else 0.0
        
        # Cruise summary
        cruise_summary = cruise_results.get_summary_dict()
        
        # Combined totals
        total_fuel = climb_fuel + cruise_summary['cruise_fuel_kg']
        total_time_hours = climb_time_hours + cruise_summary['cruise_time_hours']
        final_mission_mass = cruise_summary['final_mass_kg']
        
        print(f"CLIMB PHASE:")
        print(f"  Distance: {climb_distance_km:.2f} km")
        print(f"  Fuel consumed: {climb_fuel:.2f} kg")
        print(f"  Time: {climb_time_hours:.2f} hours ({climb_time_hours*60:.2f} min)")
        print(f"  Average fuel flow: {climb_avg_fuel_flow_kg_h:.2f} kg/h")
        print(f"  Final altitude: {dp_sched.alt_m[-1]:.2f} m")
        print(f"  Final Mach: {dp_sched.mach[-1]:.3f}")
        
        print(f"\nCRUISE PHASE:")
        print(f"  Distance: {cruise_summary['cruise_distance_km']:.2f} km")
        print(f"  Fuel consumed: {cruise_summary['cruise_fuel_kg']:.2f} kg") 
        print(f"  Time: {cruise_summary['cruise_time_hours']:.2f} hours ({cruise_summary['cruise_time_hours']*60:.2f} min)")
        print(f"  Average fuel flow: {cruise_summary['avg_fuel_flow_kg_h']:.2f} kg/h")
        print(f"  Final altitude: {cruise_results.altitude_m[-1]:.2f} m")
        print(f"  Final Mach: {cruise_results.mach_number[-1]:.3f}")
        
        print(f"\nCLIMB + CRUISE TOTALS (Descent phase to follow):")
        print(f"  Total fuel consumed: {total_fuel:.2f} kg ({total_fuel/INITIAL_MASS_KG*100:.2f}% of initial mass)")
        print(f"  Total time: {total_time_hours:.2f} hours ({total_time_hours*60:.2f} min)")
        print(f"  Mass after cruise: {final_mission_mass:.2f} kg")
        print(f"  Note: Complete mission totals will be displayed after descent phase")
        
        # Early feasibility check (partial mission)
        if total_fuel > W_FUEL_KG:
            print(f"\n  WARNING: Climb+Cruise fuel ({total_fuel:.2f} kg) exceeds capacity ({W_FUEL_KG:.2f} kg)")
            print(f"  Mission infeasibility anticipated after descent phase")
        
        print("="*80)
        
        # Create detailed cruise performance analysis
        print(f"[CRUISE] Creating detailed cruise performance analysis")
        plot_cruise_performance_2d(cruise_results)
        
        # ========= DESCENT PHASE SIMULATION =========================================
        print("\n" + "="*80)
        print("DESCENT PHASE OPTIMIZATION (3D DP with Penalty Guidance)")
        print("="*80)
        
        try:
            # Run 3D DP optimization for descent (with penalty guidance similar to climb)
            # Target: Approach conditions from mission configuration
            descent_result, descent_info = run_descent_optimization(
                cruise_results=cruise_results,
                climb_fuel_kg=climb_fuel,
                climb_time_s=climb_time_hours * 3600,
                aero=aero,
                engine=eng,
                target_altitude_m=TARGET_DESCENT_ALT_M,
                target_mach=TARGET_DESCENT_MACH,
                n_altitude_steps=N_ALTITUDE_STEPS_DESCENT,
                n_mach_samples=N_MACH_SAMPLES_DESCENT,
                lever_samples=N_LEVER_SAMPLES_DESCENT
            )
            
            # Extract grids for 3D visualization (same as DP used)
            descent_initial_mass = cruise_results.mass_kg[-1]  # Renamed for physics accuracy
            H_descent = np.linspace(cruise_results.altitude_m[-1], 
                                   TARGET_DESCENT_ALT_M, 
                                   N_ALTITUDE_STEPS_DESCENT)
            M_min_descent = max(MIN_DESCENT_MACH, 
                               TARGET_DESCENT_MACH - 0.1)
            M_max_descent = min(MAX_DESCENT_MACH, 
                               cruise_results.mach_number[-1] + 0.05)
            mach_grid_descent = np.linspace(M_min_descent, M_max_descent, 
                                        N_MACH_SAMPLES_DESCENT)
            
            print("\n" + "="*80)
            print("COMPLETE MISSION SUMMARY (CLIMB + CRUISE + DESCENT)")
            print("="*80)
            
            # Calculate total mission statistics
            total_mission_fuel = climb_fuel + cruise_summary['cruise_fuel_kg'] + descent_result.total_fuel_consumed_kg
            total_mission_time_hours = climb_time_hours + cruise_summary['cruise_time_hours'] + (descent_result.total_time_s / 3600.0)
            final_mission_mass = descent_result.final_mass_kg
            
            # Calculate descent distance
            descent_summary = descent_result.get_summary_dict()
            descent_distance_km = 0.0
            if len(descent_result.dt_s) > 0 and len(descent_result.mach) > 0:
                for i in range(len(descent_result.dt_s)):
                    if i < len(descent_result.mach) and i < len(descent_result.alt_m):
                        a = a_from_altitude(descent_result.alt_m[i])
                        V_tas = descent_result.mach[i] * a
                        descent_distance_km += V_tas * descent_result.dt_s[i] / 1000.0
            descent_time_hours = descent_result.total_time_s / 3600.0
            descent_avg_fuel_flow_kg_h = descent_summary.get('avg_fuel_flow_kg_h', 0.0)
            
            # Print all phases with consistent format
            print(f"CLIMB PHASE:")
            print(f"  Distance: {climb_distance_km:.2f} km")
            print(f"  Fuel consumed: {climb_fuel:.2f} kg")
            print(f"  Time: {climb_time_hours:.2f} hours ({climb_time_hours*60:.2f} min)")
            print(f"  Average fuel flow: {climb_avg_fuel_flow_kg_h:.2f} kg/h")
            print(f"  Final altitude: {dp_sched.alt_m[-1]:.2f} m")
            print(f"  Final Mach: {dp_sched.mach[-1]:.3f}")
            
            print(f"\nCRUISE PHASE:")
            print(f"  Distance: {cruise_summary['cruise_distance_km']:.2f} km")
            print(f"  Fuel consumed: {cruise_summary['cruise_fuel_kg']:.2f} kg") 
            print(f"  Time: {cruise_summary['cruise_time_hours']:.2f} hours ({cruise_summary['cruise_time_hours']*60:.2f} min)")
            print(f"  Average fuel flow: {cruise_summary['avg_fuel_flow_kg_h']:.2f} kg/h")
            print(f"  Final altitude: {cruise_results.altitude_m[-1]:.2f} m")
            print(f"  Final Mach: {cruise_results.mach_number[-1]:.3f}")
            
            print(f"\nDESCENT PHASE ({descent_result.strategy_name}):")
            print(f"  Distance: {descent_distance_km:.2f} km")
            print(f"  Fuel consumed: {descent_summary['descent_fuel_kg']:.2f} kg")
            print(f"  Time: {descent_time_hours:.2f} hours ({descent_time_hours*60:.2f} min)")
            print(f"  Average fuel flow: {descent_avg_fuel_flow_kg_h:.2f} kg/h")
            print(f"  Final altitude: {descent_result.alt_m[-1]:.2f} m")
            print(f"  Final Mach: {descent_result.mach[-1]:.3f}")
            
            print("\n" + "="*80)
            print("COMPLETE MISSION SUMMARY (ALL THREE PHASES)")
            print("="*80)
            print(f"TOTAL MISSION (CLIMB + CRUISE + DESCENT):")
            print(f"  Total fuel consumed: {total_mission_fuel:.2f} kg ({total_mission_fuel/INITIAL_MASS_KG*100:.2f}% of initial mass)")
            print(f"  Total time: {total_mission_time_hours:.2f} hours ({total_mission_time_hours*60:.2f} min)")
            print(f"  Initial mass: {INITIAL_MASS_KG:.2f} kg")
            print(f"  Final mass: {final_mission_mass:.2f} kg")
            print(f"  Mass reduction: {INITIAL_MASS_KG - final_mission_mass:.2f} kg")
            
            # Phase breakdown for reference
            descent_summary = descent_result.get_summary_dict()
            print(f"\nPHASE BREAKDOWN:")
            print(f"  Climb fuel:   {climb_fuel:.2f} kg ({climb_fuel/total_mission_fuel*100:.2f}% of total)")
            print(f"  Cruise fuel:  {cruise_summary['cruise_fuel_kg']:.2f} kg ({cruise_summary['cruise_fuel_kg']/total_mission_fuel*100:.2f}% of total)")
            print(f"  Descent fuel: {descent_summary['descent_fuel_kg']:.2f} kg ({descent_summary['descent_fuel_kg']/total_mission_fuel*100:.2f}% of total)")
            
            print("="*80)
            
            # CRITICAL: Validate fuel feasibility
            fuel_deficit = total_mission_fuel - W_FUEL_KG
            if fuel_deficit > 0:
                print("\n" + "="*80)
                print("MISSION INFEASIBILITY WARNING")
                print("="*80)
                print(f"  Maximum fuel capacity: {W_FUEL_KG:.2f} kg")
                print(f"  Required fuel consumption: {total_mission_fuel:.2f} kg")
                print(f"  Fuel deficit: {fuel_deficit:.2f} kg ({fuel_deficit/W_FUEL_KG*100:.2f}% over capacity)")
                print(f"\n  MISSION INFEASIBLE - Aircraft cannot carry sufficient fuel")
                print(f"  Possible solutions:")
                print(f"    1. Increase W_FUEL_KG in aircraft_config.py to at least {total_mission_fuel*1.05:.2f} kg")
                print(f"    2. Reduce cruise distance (currently {CRUISE_DISTANCE_KM:.2f} km) in mission_config.py")
                print(f"    3. Reduce payload or operating empty mass")
                print(f"    4. Utilize fuel optimizer (main_optimized.py) to determine minimum required fuel")
                print("="*80 + "\n")
            else:
                fuel_margin = W_FUEL_KG - total_mission_fuel
                print(f"\nFUEL FEASIBILITY CHECK: PASSED")
                print(f"  Maximum fuel capacity: {W_FUEL_KG:.2f} kg")
                print(f"  Required fuel consumption: {total_mission_fuel:.2f} kg")
                print(f"  Fuel margin: {fuel_margin:.2f} kg ({fuel_margin/W_FUEL_KG*100:.2f}% reserve)")
                print("="*80)
            
            # Compute full descent envelope for 3D visualization (similar to climb)
            print("\n[DESCENT] Computing full descent envelope for 3D visualization")
            lever_grid_descent = np.linspace(LEVER_MIN, LEVER_MAX, N_LEVER_SAMPLES_DESCENT)
            J_descent_envelope = DescentCore.compute_full_envelope(
                aero, eng, mach_grid_descent, H_descent, lever_grid_descent,
                mass_kg=descent_initial_mass,
                target_mach=TARGET_DESCENT_MACH
            )
            
            # Create descent path dict for 3D visualization
            descent_path = {
                'mach': np.asarray(descent_result.mach, float),
                'alt': np.asarray(descent_result.alt_m, float),
                'lever': np.asarray(descent_result.lever, float),
            }
            
            # Show the 3D J plot for descent DP with full envelope
            print("[DESCENT] Opening 3D visualization for Descent DP (Global Optimization) with full envelope")
            plot_descent_3d_cost_space(
                mach_grid_descent, H_descent, lever_grid_descent, J_descent_envelope,
                min_path=descent_path,
                title="3D DP Descent (Global Optimization)<br>Full Envelope with Optimal Path",
                mass_kg=descent_initial_mass
            )
            
            # Create descent visualization plots (Interactive Plotly - opens in browser)
            print(f"\n[DESCENT] Creating interactive descent visualization plots")
            
            # Plot DP optimal descent trajectory in detail (Interactive)
            print(f"[DESCENT] Opening optimal descent trajectory in browser")
            plot_descent_performance_2d(descent_result)
            
            # Create complete 3D visualization: Climb + Cruise + Descent
            print(f"[VISUALIZATION] Opening complete mission 3D visualization in browser")
            print(f"  Mission phases: Climb (blue) → Cruise (green) → Descent (red)")
            plot_complete_mission_3d(
                climb_result=dp_sched,
                cruise_result=cruise_results,
                descent_result=descent_result,
                climb_info=dp_info,
                descent_info=descent_info
            )
            
            # Calculate simulation execution time
            simulation_end_time = time.time()
            simulation_duration_min = (simulation_end_time - simulation_start_time) / 60.0
            
            # Create comprehensive mission summary dashboard
            print(f"\n[SUMMARY] Opening comprehensive mission summary dashboard in browser")
            print(f"  Dashboard displays all key mission metrics and performance indicators")
            print(f"  Simulation execution time: {simulation_duration_min:.2f} minutes")
            plot_mission_summary_dashboard(
                climb_result=dp_sched,
                cruise_result=cruise_results,
                descent_result=descent_result,
                initial_mass_kg=INITIAL_MASS_KG,
                simulation_duration_min=simulation_duration_min
            )
            
            # Create combined performance analysis
            print(f"\n[ANALYSIS] Creating combined performance analysis")
            try:
                plot_combined_performance_analysis(
                    climb_result=dp_sched,
                    cruise_result=cruise_results,
                    descent_result=descent_result,
                    initial_mass_kg=INITIAL_MASS_KG
                )
                print(f"[ANALYSIS] Combined performance analysis completed successfully")
            except Exception as e:
                print(f"[ERROR] Combined performance analysis failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Record mission history for CG plotting (MUST be done before Excel export)
            print(f"\n[CG] Recording mission history for CG analysis")
            try:
                record_mission_history(
                    climb_result=dp_sched,
                    cruise_result=cruise_results,
                    descent_result=descent_result
                )
            except Exception as e:
                print(f"[WARNING] Failed to record mission history: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Create CG analysis plots
            print(f"\n[CG] Creating CG movement analysis plots")
            try:
                plot_cg_analysis(save_plots=True, show_plots=True)
                print(f"[CG] CG analysis plots completed successfully")
            except Exception as e:
                print(f"[ERROR] CG analysis plotting failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Export all mission data to Excel (AFTER CG history is recorded)
            print(f"\n[EXPORT] Exporting mission data to Excel")
            try:
                excel_path = export_mission_to_excel(
                    climb_result=dp_sched,
                    cruise_result=cruise_results,
                    descent_result=descent_result,
                    initial_mass_kg=INITIAL_MASS_KG,
                    climb_info=dp_info,
                    descent_info=descent_info
                )
                print(f"[EXPORT] Mission data successfully exported to Excel")
            except Exception as e:
                print(f"[ERROR] Excel export failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"[ERROR] Descent simulation failed: {str(e)}")
            print("Analysis continues without descent phase")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"[ERROR] Cruise simulation failed: {str(e)}")
        print("Analysis continues with climb phase only")
        import traceback
        traceback.print_exc()
    
    plt.show(block=True)

if __name__ == "__main__":
    main()