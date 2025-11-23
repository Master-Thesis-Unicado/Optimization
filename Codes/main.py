"""
Mission Analysis with Climb and Cruise Simulation

This script runs:
1. 3D Dynamic Programming climb optimization 
2. Steady-level cruise simulation using specific excess power approach

Cruise Parameters (adjustable in main() function):
- cruise_distance_km: Distance to cruise in kilometers
- cruise_time_step_s: Time step for simulation (60s = 1 minute)

The cruise simulation:
- Uses exact final state from 3D climb optimization
- Maintains constant altitude and Mach number from climb end
- Updates weight based on fuel consumption during cruise
- Uses Ps = (T_total - D) * V / W for thrust balance calculations

Output includes interactive plots and complete mission summary.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import time

from aircraft_config import (
    INITIAL_MASS_KG, ENGINE_STUB_PATH,
    AtmosphericProperties, MAX_FUEL_KG, W_OE_KG, W_PL_KG
)
from mission_config import (
    TARGET_ALT_CLIMB_M,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE_CLIMB, STRATEGY_DT_CLIMB_S,
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S, 
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT,
    MIN_DESCENT_MACH, MAX_DESCENT_MACH
)
import climb
from climb import (
    compute_sep_grid_maxlever,
    dbg, compute_full_engine_envelope,
    ClimbingCore
)
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper
import cruise
from cruise import run_cruise_simulation
from cruise_plotting import plot_cruise_performance_detailed
from climb_plotting import (
    plot_strategies_interactive, plot_J_3d_plotly, create_strategy_comparison_plots, plot_climb_performance_detailed,
    GridConfig
)
import descent
from descent import run_descent_dp_optimization, compute_full_descent_envelope
from descent_plotting import (plot_descent_trajectory_interactive, 
                              plot_descent_3d_trajectory,
                              plot_complete_mission_3d_interactive,
                              plot_descent_J_3d_plotly)
from mission_summary import plot_mission_summary_dashboard, plot_combined_performance_analysis

# ========= CONSTANTS AND SETTINGS =============================================

# Aircraft performance constants

# Create atmospheric properties instance
atmospheric_props = AtmosphericProperties()


def main():
    # Record simulation start time
    simulation_start_time = time.time()
    
    print("\n" + "="*80)
    print("AIRCRAFT CONFIGURATION")
    print("="*80)
    print(f"[CONFIG] MAX_FUEL_KG: {MAX_FUEL_KG:.1f} kg")
    print(f"[CONFIG] W_OE_KG: {W_OE_KG:.1f} kg")
    print(f"[CONFIG] W_PL_KG: {W_PL_KG:.1f} kg")
    print(f"[CONFIG] INITIAL_MASS_KG (W_TO_KG): {INITIAL_MASS_KG:.1f} kg")
    print(f"[CONFIG] = W_OE + W_PL + MAX_FUEL = {W_OE_KG + W_PL_KG + MAX_FUEL_KG:.1f} kg")
    print("="*80 + "\n")
    
    print("[MISSION] Using centralized mission configuration")
    print(f"[MISSION] Target altitude: {TARGET_ALT_CLIMB_M:.0f} m")
    print(f"[MISSION] Target Mach: {TARGET_MACH_CRUISE:.3f}")
    print(f"[MISSION] Cruise distance: {CRUISE_DISTANCE_KM:.0f} km")
    
    print("[READ] Aerodynamics (Excel Sheet4) …")
    aero = PyAerodynamicsWrapper()

    # Dense grids for contours
    M_min, M_max = float(aero.mach_grid[0]), float(aero.mach_grid[-1])
    M_dense = np.linspace(M_min, M_max, N_MACH_SAMPLES_CLIMB)
    H_plot  = np.arange(START_ALTITUDE_CLIMB_M, 
                        GridConfig.Y_AXIS_TOP_M + 0.5*GridConfig.ALT_STEP_M, 
                        GridConfig.ALT_STEP_M)

    # Effective minimum Mach from sheet
    climb.M_MIN_EFFECTIVE = max(climb.M_MIN_DEFAULT, float(aero.mach_grid[0]))
    print(f"[INFO] Effective M_MIN set to {climb.M_MIN_EFFECTIVE:.3f} (sheet min Mach={aero.mach_grid[0]:.3f}).")

    print("[ENGINE] Loading engine stub …")
    
    eng = EngineWrapper("lls/stubs/engines/PW1127G-JM")

    # Performance optimization: Pre-compute grids for caching
    print("[OPTIMIZATION] Pre-computing engine and drag grids for performance...")
    
    # Create grids for pre-computation
    lever_grid = np.linspace(0.0, 1.0, 21)  # 21 lever positions (0.0 to 1.0 in steps of 0.05)
    
    # Pre-compute engine values
    eng.precompute_grid(M_dense, H_plot, lever_grid)
    
    # Pre-compute drag values at initial mass (reference weight for caching)
    aero.precompute_drag_grid(M_dense, H_plot, INITIAL_MASS_KG)

    print("[PS] Computing background Ps grid (max lever, ref mass) …")
    M_grid, H_plot, Ps_base = compute_sep_grid_maxlever(aero, eng, INITIAL_MASS_KG,
                                                        M_grid=M_dense, H_grid=H_plot)

    # DP schedule on EXACT N_ALTITUDE_STEPS_CLIMB rows
    # Create uniform altitude steps as requested
    uniform_step_size = TARGET_ALT_CLIMB_M / N_ALTITUDE_STEPS_CLIMB
    H_sched = np.arange(START_ALTITUDE_CLIMB_M, 
                        TARGET_ALT_CLIMB_M + uniform_step_size, 
                        uniform_step_size)
    print("[DP] Solving 3D fixed-mass DP …")
    # Calculate starting Mach from takeoff velocity at start altitude
    a = atmospheric_props.a_from_altitude(START_ALTITUDE_CLIMB_M)
    start_mach = START_VELOCITY_CLIMB_MS / a
    
    dp_sched, dp_info = ClimbingCore.DynamicProgrammingOptimizer.solve_3d_fixed_mass(
        aero, eng, M_grid, H_sched, 
        lever_samples=N_LEVER_SAMPLES_CLIMB, 
        target_mach=TARGET_MACH_CRUISE,
        target_mach_tolerance=TARGET_MACH_TOLERANCE_CLIMB, 
        start_mach=start_mach, 
        start_lever=START_LEVER_CLIMB
    )

    # Strategies (fixed mass), resampled to N_ALTITUDE_STEPS_CLIMB
    print("[STRAT] Simulating strategies …")
    strategies: list[climb.StrategyRun] = []
    for name, fn, af in ClimbingCore.StrategyManager.build_strategy_set():
        sr = ClimbingCore.StrategyManager.simulate_strategy_path(
            label=name,
            aero=aero, eng=eng,
            mass0_kg=INITIAL_MASS_KG, 
            h0_m=START_ALTITUDE_CLIMB_M, 
            V0_ms=START_VELOCITY_CLIMB_MS,
            target_alt_m=TARGET_ALT_CLIMB_M, 
            dt=STRATEGY_DT_CLIMB_S,
            strategy_fn=fn, altitude_fraction=af
        )
        sr = ClimbingCore.StrategyManager.resample_strategy_run(sr, N_ALTITUDE_STEPS_CLIMB)
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
    dp_run = ClimbingCore.StrategyManager.resample_strategy_run(dp_run, N_ALTITUDE_STEPS_CLIMB)
    strategies.append(dp_run)
    
    # 3D DP strategy already added above

    # Align all strategies to start from the same Mach value as the minimum fuel path
    dp_start_mach = dp_run.mach[0] if len(dp_run.mach) > 0 else 0.2
    print(f"[ALIGN] Aligning strategies to start from Mach {dp_start_mach:.3f}")
    
    # Align strategies to start from DP's first Mach value (skip DP and constant strategies)
    for strategy in strategies:
        if (strategy.label not in ["3D DP (Global Optimization)"] and 
            "Constant speed" not in strategy.label and 
            "Constant Mach" not in strategy.label):
            mach_array = np.array(strategy.mach, dtype=float)
            mach_offset = mach_array[0] - dp_start_mach
            strategy.mach = mach_array - mach_offset

    # Print comprehensive strategy comparison table
    print("\n" + "="*100)
    print("CLIMBING STRATEGIES COMPARISON")
    print("="*100)
    print(f"{'Strategy':<25} {'Fuel (kg)':<12} {'Time (min)':<12} {'Final Mach':<12} {'Avg Ps (m/s)':<12} {'Envelope':<15}")
    print("-"*100)
    
    # Sort strategies by fuel consumption for ranking
    strategies_sorted = sorted(strategies, key=lambda s: s.fuel_total_kg)
    
    for i, strategy in enumerate(strategies_sorted):
        final_mach = strategy.mach[-1] if len(strategy.mach) > 0 else 0.0
        avg_ps = np.mean(strategy.Ps_mps) if len(strategy.Ps_mps) > 0 else 0.0
        time_min = strategy.time_s[-1] / 60.0 if len(strategy.time_s) > 0 else 0.0
        envelope_status = ClimbingCore.check_envelope_exceedance(strategy, aero)
        
        # Add ranking indicator
        rank_indicator = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1:2d}."
        
        print(f"{rank_indicator} {strategy.label:<22} {strategy.fuel_total_kg:<12.1f} {time_min:<12.1f} {final_mach:<12.3f} {avg_ps:<12.2f} {envelope_status:<15}")
    
    print("-"*100)
    print("Ranking based on total fuel consumption (lower is better)")
    print("="*100)
    
    # Additional analysis
    print("\nDETAILED ANALYSIS:")
    print("-"*40)
    
    # Find best and worst strategies for different metrics
    best_fuel = min(strategies, key=lambda s: s.fuel_total_kg)
    worst_fuel = max(strategies, key=lambda s: s.fuel_total_kg)
    
    best_time = min(strategies, key=lambda s: s.time_s[-1] if len(s.time_s) > 0 else float('inf'))
    worst_time = max(strategies, key=lambda s: s.time_s[-1] if len(s.time_s) > 0 else 0)
    
    print(f"Most Fuel Efficient: {best_fuel.label} ({best_fuel.fuel_total_kg:.1f} kg)")
    print(f"Least Fuel Efficient: {worst_fuel.label} ({worst_fuel.fuel_total_kg:.1f} kg)")
    print(f"Fuel Difference: {worst_fuel.fuel_total_kg - best_fuel.fuel_total_kg:.1f} kg ({((worst_fuel.fuel_total_kg - best_fuel.fuel_total_kg) / best_fuel.fuel_total_kg * 100):.1f}%)")
    
    if len(best_time.time_s) > 0 and len(worst_time.time_s) > 0:
        print(f"Fastest: {best_time.label} ({best_time.time_s[-1]/60:.1f} min)")
        print(f"Slowest: {worst_time.label} ({worst_time.time_s[-1]/60:.1f} min)")
        print(f"Time Difference: {(worst_time.time_s[-1] - best_time.time_s[-1])/60:.1f} min")
    
    print("\n" + "="*80)

    # Create additional comparison plots
    print("[PLOT] Creating strategy comparison plots...")
    create_strategy_comparison_plots(strategies, aero)
    
    # Create climb performance analysis plot
    print("[PLOT] Creating climb performance analysis...")
    plot_climb_performance_detailed(dp_sched, dp_info)

    # Single window UI (strategies + DP strategy) - matplotlib
    print("[PLOT] Opening single interactive window …")
    _ = plot_strategies_interactive(
        M_grid, H_plot, Ps_base, strategies,
        title_suffix=f"Target Alt={TARGET_ALT_CLIMB_M:.0f} m, Ref mass={INITIAL_MASS_KG:.0f} kg"
    )
    # Extract minimum-fuel path for overlay
    min_path = {
        'mach': np.asarray(dp_sched.mach, float),
        'alt': np.asarray(dp_sched.alt_m, float),
        'lever': np.asarray(dp_sched.lever, float),
    }
    # Compute full engine envelope for 3D visualization
    print("[ENVELOPE] Computing full engine envelope for 3D visualization...")
    J_envelope, lever_grid_envelope = compute_full_engine_envelope(aero, eng, M_grid, H_sched, lever_samples=50)
    
    # Show the 3D J plot for 3D DP with full engine envelope in a separate window (browser)
    print("[PLOT] Opening 3D visualization for 3D DP (Simultaneous 3D Optimization) with full engine envelope...")
    
    # Use full engine envelope for rich 3D visualization
    M_grid_3d = M_grid  # Use the full Mach grid for rich visualization
    H_grid_3d = H_sched  # Use the same altitude grid as 3D DP
    lever_grid_3d = lever_grid_envelope  # Use the envelope lever grid
    
    
    plot_J_3d_plotly(M_grid_3d, H_grid_3d, lever_grid_3d, J_envelope, min_path=min_path,
                     title="3D DP (Global Optimization)<br>Full Engine Envelope with Optimal Path")
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("="*60)
    engine_stats = eng.get_cache_stats()
    drag_stats = aero.get_cache_stats()
    
    print(f"Engine Cache Performance:")
    print(f"  - Cache hits: {engine_stats['hits']:,}")
    print(f"  - Cache misses: {engine_stats['misses']:,}")
    print(f"  - Hit rate: {engine_stats['hit_rate']:.1%}")
    print(f"  - Cache size: {engine_stats['cache_size']:,} entries")
    
    print(f"\nDrag Cache Performance:")
    print(f"  - Cache hits: {drag_stats['hits']:,}")
    print(f"  - Cache misses: {drag_stats['misses']:,}")
    print(f"  - Hit rate: {drag_stats['hit_rate']:.1%}")
    print(f"  - Cache size: {drag_stats['cache_size']:,} entries")
    
    total_hits = engine_stats['hits'] + drag_stats['hits']
    total_calls = engine_stats['hits'] + engine_stats['misses'] + drag_stats['hits'] + drag_stats['misses']
    overall_hit_rate = total_hits / total_calls if total_calls > 0 else 0
    
    print(f"\nOverall Cache Performance:")
    print(f"  - Total cache hits: {total_hits:,}")
    print(f"  - Total function calls: {total_calls:,}")
    print(f"  - Overall hit rate: {overall_hit_rate:.1%}")
    print("="*60)
    
    # ========= CRUISE PHASE SIMULATION =========================================
    print("\n" + "="*60)
    print("STARTING CRUISE PHASE SIMULATION")
    print("="*60)
    
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
        
        # Cruise summary
        cruise_summary = cruise_results.get_summary_dict()
        
        # Combined totals
        total_fuel = climb_fuel + cruise_summary['cruise_fuel_kg']
        total_time_hours = climb_time_hours + cruise_summary['cruise_time_hours']
        final_mission_weight = cruise_summary['final_weight_kg']
        
        print(f"CLIMB PHASE:")
        print(f"  Fuel consumed: {climb_fuel:.1f} kg")
        print(f"  Time: {climb_time_hours:.2f} hours")
        print(f"  Final altitude: {dp_sched.alt_m[-1]:.0f} m")
        print(f"  Final Mach: {dp_sched.mach[-1]:.3f}")
        
        print(f"\nCRUISE PHASE:")
        print(f"  Distance: {cruise_summary['cruise_distance_km']:.0f} km")
        print(f"  Fuel consumed: {cruise_summary['cruise_fuel_kg']:.1f} kg") 
        print(f"  Time: {cruise_summary['cruise_time_hours']:.2f} hours")
        print(f"  Average fuel flow: {cruise_summary['avg_fuel_flow_kg_h']:.0f} kg/h")
        
        print(f"\nCLIMB + CRUISE TOTALS (Descent phase to follow):")
        print(f"  Total fuel consumed (climb+cruise): {total_fuel:.1f} kg ({total_fuel/INITIAL_MASS_KG*100:.1f}% of initial weight)")
        print(f"  Total time (climb+cruise): {total_time_hours:.2f} hours")
        print(f"  Weight after cruise: {final_mission_weight:.0f} kg")
        print(f"  Note: Complete mission totals will be shown after descent phase")
        
        # Early feasibility check (partial mission)
        if total_fuel > MAX_FUEL_KG:
            print(f"\n   WARNING: Climb+Cruise fuel ({total_fuel:.1f} kg) already exceeds capacity ({MAX_FUEL_KG:.1f} kg)")
            print(f"  Mission will be infeasible after descent phase")
        
        print("="*80)
        
        # Create detailed cruise performance analysis
        print(f"[CRUISE] Creating detailed cruise performance analysis...")
        plot_cruise_performance_detailed(cruise_results)
        
        # ========= DESCENT PHASE SIMULATION =========================================
        print("\n" + "="*60)
        print("STARTING DESCENT PHASE OPTIMIZATION (3D DP with Penalty Guidance)")
        print("="*60)
        
        try:
            # Run 3D DP optimization for descent (with penalty guidance similar to climb)
            # Target: Approach conditions from mission configuration
            descent_result, descent_info = run_descent_dp_optimization(
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
            descent_initial_weight = cruise_results.weight_kg[-1]
            H_descent = np.linspace(cruise_results.altitude_m[-1], 
                                   TARGET_DESCENT_ALT_M, 
                                   N_ALTITUDE_STEPS_DESCENT)
            M_min_descent = max(MIN_DESCENT_MACH, 
                               TARGET_DESCENT_MACH - 0.1)
            M_max_descent = min(MAX_DESCENT_MACH, 
                               cruise_results.mach_number[-1] + 0.05)
            M_grid_descent = np.linspace(M_min_descent, M_max_descent, 
                                        N_MACH_SAMPLES_DESCENT)
            
            print("\n" + "="*80)
            print("COMPLETE MISSION SUMMARY (CLIMB + CRUISE + DESCENT)")
            print("="*80)
            
            # Calculate total mission statistics
            total_mission_fuel = climb_fuel + cruise_summary['cruise_fuel_kg'] + descent_result.total_fuel_consumed_kg
            total_mission_time_hours = climb_time_hours + cruise_summary['cruise_time_hours'] + (descent_result.total_time_s / 3600.0)
            final_mission_weight = descent_result.final_weight_kg
            
            print(f"CLIMB PHASE:")
            print(f"  Fuel consumed: {climb_fuel:.1f} kg")
            print(f"  Time: {climb_time_hours:.2f} hours")
            print(f"  Final altitude: {dp_sched.alt_m[-1]:.0f} m")
            print(f"  Final Mach: {dp_sched.mach[-1]:.3f}")
            
            print(f"\nCRUISE PHASE:")
            print(f"  Distance: {cruise_summary['cruise_distance_km']:.0f} km")
            print(f"  Fuel consumed: {cruise_summary['cruise_fuel_kg']:.1f} kg") 
            print(f"  Time: {cruise_summary['cruise_time_hours']:.2f} hours")
            print(f"  Average fuel flow: {cruise_summary['avg_fuel_flow_kg_h']:.0f} kg/h")
            
            descent_summary = descent_result.get_summary_dict()
            print(f"\nDESCENT PHASE ({descent_result.strategy_name}):")
            print(f"  Fuel consumed: {descent_summary['descent_fuel_kg']:.2f} kg")
            print(f"  Time: {descent_summary['descent_time_minutes']:.1f} min")
            print(f"  Avg descent rate: {descent_summary['avg_descent_rate_mpm']:.0f} m/min")
            print(f"  Altitude change: {descent_summary['descent_altitude_change_m']:.0f} m")
            print(f"  Final Mach: {descent_result.mach[-1]:.3f} (target: {descent_result.target_mach:.3f})")
            print(f"  Final altitude: {descent_result.alt_m[-1]:.0f} m (target: {descent_result.target_altitude_m:.0f} m)")
            
            print("\n" + "="*80)
            print("COMPLETE MISSION SUMMARY (ALL THREE PHASES)")
            print("="*80)
            print(f"TOTAL MISSION (CLIMB + CRUISE + DESCENT):")
            print(f"  Total fuel consumed: {total_mission_fuel:.1f} kg ({total_mission_fuel/INITIAL_MASS_KG*100:.1f}% of initial weight)")
            print(f"  Total time: {total_mission_time_hours:.2f} hours ({total_mission_time_hours*60:.1f} minutes)")
            print(f"  Initial weight: {INITIAL_MASS_KG:.0f} kg")
            print(f"  Final weight: {final_mission_weight:.0f} kg")
            print(f"  Weight reduction: {INITIAL_MASS_KG - final_mission_weight:.1f} kg")
            
            # Phase breakdown for reference
            descent_summary = descent_result.get_summary_dict()
            print(f"\nPHASE BREAKDOWN:")
            print(f"  Climb fuel:   {climb_fuel:.1f} kg ({climb_fuel/total_mission_fuel*100:.1f}% of total)")
            print(f"  Cruise fuel:  {cruise_summary['cruise_fuel_kg']:.1f} kg ({cruise_summary['cruise_fuel_kg']/total_mission_fuel*100:.1f}% of total)")
            print(f"  Descent fuel: {descent_summary['descent_fuel_kg']:.1f} kg ({descent_summary['descent_fuel_kg']/total_mission_fuel*100:.1f}% of total)")
            
            print("="*80)
            
            # CRITICAL: Validate fuel feasibility
            fuel_deficit = total_mission_fuel - MAX_FUEL_KG
            if fuel_deficit > 0:
                print("\n" + "="*80)
                print(" MISSION INFEASIBILITY WARNING")
                print("="*80)
                print(f"  Maximum fuel capacity: {MAX_FUEL_KG:.1f} kg")
                print(f"  Required fuel consumption: {total_mission_fuel:.1f} kg")
                print(f"  Fuel deficit: {fuel_deficit:.1f} kg ({fuel_deficit/MAX_FUEL_KG*100:.1f}% over capacity)")
                print(f"\n   MISSION IS INFEASIBLE - Aircraft cannot carry sufficient fuel!")
                print(f"  Possible solutions:")
                print(f"    1. Increase MAX_FUEL_KG in aircraft_config.py to at least {total_mission_fuel*1.05:.1f} kg")
                print(f"    2. Reduce cruise distance (currently {CRUISE_DISTANCE_KM:.0f} km) in mission_config.py")
                print(f"    3. Reduce payload or operating empty weight")
                print(f"    4. Use fuel optimizer (main_optimized.py) to find minimum required fuel")
                print("="*80 + "\n")
            else:
                fuel_margin = MAX_FUEL_KG - total_mission_fuel
                print(f"\nFUEL FEASIBILITY CHECK: PASSED")
                print(f"  Maximum fuel capacity: {MAX_FUEL_KG:.1f} kg")
                print(f"  Required fuel consumption: {total_mission_fuel:.1f} kg")
                print(f"  Fuel margin: {fuel_margin:.1f} kg ({fuel_margin/MAX_FUEL_KG*100:.1f}% reserve)")
                print("="*80)
            
            # Compute full descent envelope for 3D visualization (similar to climb)
            print("\n[ENVELOPE-DESCENT] Computing full descent envelope for 3D visualization...")
            J_descent_envelope, lever_grid_descent = compute_full_descent_envelope(
                aero, eng, M_grid_descent, H_descent, 
                initial_weight_kg=descent_initial_weight,
                lever_samples=50,
                target_mach=0.25
            )
            
            # Create descent path dict for 3D visualization
            descent_path = {
                'mach': np.asarray(descent_result.mach, float),
                'alt': np.asarray(descent_result.alt_m, float),
                'lever': np.asarray(descent_result.lever, float),
            }
            
            # Show the 3D J plot for descent DP with full envelope (NEW!)
            print("[PLOT] Opening 3D visualization for Descent DP (Global Optimization) with full envelope...")
            plot_descent_J_3d_plotly(
                M_grid_descent, H_descent, lever_grid_descent, J_descent_envelope,
                min_path=descent_path,
                title="3D DP Descent (Global Optimization)<br>Full Envelope with Optimal Path",
                initial_weight_kg=descent_initial_weight
            )
            
            # Create descent visualization plots (Interactive Plotly - opens in browser)
            print(f"\n[DESCENT] Creating interactive descent visualization plots...")
            
            # Plot DP optimal descent trajectory in detail (Interactive)
            print(f"[DESCENT] Opening optimal descent trajectory in browser...")
            plot_descent_trajectory_interactive(descent_result)
            
            # Create complete 3D visualization: Climb + Cruise + Descent (NEW!)
            print(f"[3D VISUALIZATION] Opening complete mission 3D visualization in browser...")
            print(f"  This shows Climb (blue) → Cruise (green) → Descent (red) in 3D space")
            plot_complete_mission_3d_interactive(
                climb_result=dp_sched,
                cruise_result=cruise_results,
                descent_result=descent_result,
                climb_info=dp_info,
                descent_info=descent_info
            )
            
            # Calculate simulation execution time
            simulation_end_time = time.time()
            simulation_duration_min = (simulation_end_time - simulation_start_time) / 60.0
            
            # Create comprehensive mission summary dashboard (NEW!)
            print(f"\n[SUMMARY] Opening comprehensive mission summary dashboard in browser...")
            print(f"  Professional dashboard with all key mission metrics and performance indicators")
            print(f"  Simulation execution time: {simulation_duration_min:.2f} minutes")
            plot_mission_summary_dashboard(
                climb_result=dp_sched,
                cruise_result=cruise_results,
                descent_result=descent_result,
                initial_mass_kg=INITIAL_MASS_KG,
                simulation_duration_min=simulation_duration_min
            )
            
            # Create combined performance analysis
            print(f"\n[PLOT] Creating combined performance analysis...")
            try:
                plot_combined_performance_analysis(
                    climb_result=dp_sched,
                    cruise_result=cruise_results,
                    descent_result=descent_result,
                    initial_mass_kg=INITIAL_MASS_KG
                )
                print(f"[PLOT] Combined performance analysis completed successfully!")
            except Exception as e:
                print(f"[ERROR] Combined performance analysis failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"[ERROR] Descent simulation failed: {str(e)}")
            print("Continuing without descent analysis...")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"[ERROR] Cruise simulation failed: {str(e)}")
        print("Continuing with climb-only analysis...")
        import traceback
        traceback.print_exc()
    
    plt.show(block=True)

if __name__ == "__main__":
    main()