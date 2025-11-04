"""
Mission Analysis with Fuel Capacity Optimization

This script implements a convergent optimization loop to determine the minimum 
required fuel capacity for mission completion, replacing the static MTOF with 
a dynamically optimized value.

Features:
- Iterative fuel optimization (climb + cruise + descent)
- Convergence tracking with relative tolerance (0.1%)
- No intermediate plots during convergence
- Final plots generated only after convergence
- KPP evolution tracking across iterations

The optimization process:
1. Start with MAX_FUEL_KG as initial guess
2. Run full mission simulation (climb → cruise → descent)
3. Use consumed fuel as new initial fuel for next iteration3

4. Repeat until convergence
5. Apply 5% safety buffer to final result
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from aircraft_config import (
    INITIAL_MASS_KG, ENGINE_STUB_PATH, MAX_FUEL_KG, W_OE_KG, W_PL_KG,
    AtmosphericProperties, G_C
)
from mission_config import (
    TARGET_ALT_CLIMB_M, ALT_STEP_M, Y_AXIS_TOP_M,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE_CLIMB, STRATEGY_DT_CLIMB_S,
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S,
    THRUST_CONVERGENCE_TOL_CRUISE, MAX_ITERATIONS_CRUISE,
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
from climb_plotting import (plot_strategies_interactive, plot_J_3d_plotly, 
                           create_strategy_comparison_plots, plot_climb_performance_detailed)
import descent
from descent import run_descent_dp_optimization, compute_full_descent_envelope
from descent_plotting import (plot_descent_trajectory_interactive, 
                              plot_descent_3d_trajectory,
                              plot_complete_mission_3d_interactive,
                              plot_descent_J_3d_plotly)
from mission_summary import (plot_mission_summary_dashboard, 
                            plot_combined_performance_analysis)

# Import optimization modules
from fuel_optimizer import optimize_fuel_capacity, SAFETY_BUFFER_PERCENT
from fuel_plotting import visualize_convergence_analysis

# ========= CONSTANTS AND SETTINGS =============================================

# Create atmospheric properties instance
atmospheric_props = AtmosphericProperties()


def main():
    """
    Main mission analysis with fuel capacity optimization.
    
    Process:
    1. Run optimization loop to determine minimum fuel
    2. Use optimized fuel for final mission execution
    3. Generate all plots and visualizations
    """
    
    # ========= OPTIMIZATION LOOP =========================================
    
    print("\n" + "="*80)
    print("MISSION ANALYSIS WITH FUEL CAPACITY OPTIMIZATION")
    print("="*80)
    
    print("[READ] Loading aerodynamic and engine data...")
    aero = PyAerodynamicsWrapper()
    eng = EngineWrapper(ENGINE_STUB_PATH)
    
    # Dense grids for contours
    M_min, M_max = float(aero.mach_grid[0]), float(aero.mach_grid[-1])
    M_dense = np.linspace(M_min, M_max, N_MACH_SAMPLES_CLIMB)
    H_plot = np.arange(START_ALTITUDE_CLIMB_M, 
                        Y_AXIS_TOP_M + 0.5*ALT_STEP_M, 
                        ALT_STEP_M)
    
    # Pre-compute grids for performance
    print("[OPTIMIZATION] Pre-computing engine grids...")
    lever_grid = np.linspace(0.0, 1.0, 21)
    eng.precompute_grid(M_dense, H_plot, lever_grid)
    # Note: Drag grid precomputation skipped - drag is weight-dependent and varies during mission
    # The drag cache will populate naturally during optimization with correct dynamic weights
    
    # Run fuel optimization
    print("\n[OPTIMIZATION] Starting fuel capacity optimization loop...")
    optimal_result, convergence_history = optimize_fuel_capacity(
        aero=aero,
        eng=eng,
        M_grid=M_dense,
        H_plot=H_plot,
        lever_samples=N_LEVER_SAMPLES_CLIMB
    )
    
    # Visualize convergence
    visualize_convergence_analysis(convergence_history, save_plots=True)
    
    # ========= RUN FINAL MISSION WITH OPTIMIZED FUEL =========================================
    
    # Calculate optimized fuel capacity with safety buffer
    optimized_fuel = optimal_result.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT)
    optimized_mass = W_OE_KG + W_PL_KG + optimized_fuel
    
    print("\n" + "="*80)
    print("RUNNING FINAL MISSION WITH OPTIMIZED FUEL")
    print("="*80)
    print(f"[CONFIG] Optimized fuel capacity: {optimized_fuel:.1f} kg")
    print(f"[CONFIG] Optimized total mass: {optimized_mass:.1f} kg")
    print(f"[CONFIG] Savings vs original MAX_FUEL_KG ({MAX_FUEL_KG:.1f} kg): {MAX_FUEL_KG - optimized_fuel:.1f} kg ({(MAX_FUEL_KG - optimized_fuel) / MAX_FUEL_KG * 100:.1f}%)")
    print("="*80 + "\n")
    
    # Use the optimized result directly (already computed in the optimization loop)
    # This is the final mission with optimized fuel
    final_climb = optimal_result.climb_result
    final_cruise = optimal_result.cruise_result
    final_descent = optimal_result.descent_result
    
    print("[INFO] Using final optimized mission results for visualization...")
    print(f"[INFO] Climb: {optimal_result.climb_fuel_kg:.1f} kg, {optimal_result.climb_time_s/60:.1f} min")
    print(f"[INFO] Cruise: {optimal_result.cruise_fuel_kg:.1f} kg, {optimal_result.cruise_time_s/60:.1f} min")
    print(f"[INFO] Descent: {optimal_result.descent_fuel_kg:.2f} kg, {optimal_result.descent_time_s/60:.1f} min")
    print(f"[INFO] Total: {optimal_result.fuel_consumed_kg:.1f} kg, {optimal_result.total_time_s/60:.1f} min")
    
    
    # ========= COMPUTE GRIDS FOR VISUALIZATIONS =========================================
    
    print("\n[VISUALIZATION] Computing grids for final mission visualization...")
    
    # Compute Ps grid
    M_grid, H_plot, Ps_base = compute_sep_grid_maxlever(
        aero, eng, optimized_mass,
        M_grid=M_dense, H_grid=H_plot
    )
    
    # Climb phase visualization data
    uniform_step_size = TARGET_ALT_CLIMB_M / N_ALTITUDE_STEPS_CLIMB
    H_sched = np.arange(START_ALTITUDE_CLIMB_M, 
                        TARGET_ALT_CLIMB_M + uniform_step_size, 
                        uniform_step_size)
    
    # Descent phase visualization data
    H_descent = np.linspace(final_cruise.altitude_m[-1], 
                           TARGET_DESCENT_ALT_M, 
                           N_ALTITUDE_STEPS_DESCENT)
    M_min_descent = max(MIN_DESCENT_MACH, TARGET_DESCENT_MACH - 0.1)
    M_max_descent = min(MAX_DESCENT_MACH, 
                       final_cruise.mach_number[-1] + 0.05)
    M_grid_descent = np.linspace(M_min_descent, M_max_descent, N_MACH_SAMPLES_DESCENT)
    
    # ========= GENERATE ALL FINAL PLOTS =========================================
    
    print("\n[PLOT] Generating final mission visualizations...")
    
    # Climb plots
    print("[PLOT] Creating climb performance analysis...")
    plot_climb_performance_detailed(final_climb, None)
    
    # 3D climb visualization
    print("[PLOT] Opening 3D climb visualization...")
    J_envelope, lever_grid_envelope = compute_full_engine_envelope(
        aero, eng, M_grid, H_sched, lever_samples=50
    )
    
    min_path = {
        'mach': np.asarray(final_climb.mach, float),
        'alt': np.asarray(final_climb.alt_m, float),
        'lever': np.asarray(final_climb.lever, float),
    }
    
    plot_J_3d_plotly(
        M_grid, H_sched, lever_grid_envelope, J_envelope, 
        min_path=min_path,
        title="3D DP (Global Optimization)<br>Full Engine Envelope with Optimal Path"
    )
    
    # Cruise plots
    print("[PLOT] Creating cruise performance analysis...")
    plot_cruise_performance_detailed(final_cruise)
    
    # Descent plots
    print("[PLOT] Opening descent visualizations...")
    plot_descent_trajectory_interactive(final_descent)
    
    # Descent 3D visualization
    J_descent_envelope, lever_grid_descent = compute_full_descent_envelope(
        aero, eng, M_grid_descent, H_descent,
        initial_weight_kg=final_cruise.weight_kg[-1],
        lever_samples=50,
        target_mach=0.25
    )
    
    descent_path = {
        'mach': np.asarray(final_descent.mach, float),
        'alt': np.asarray(final_descent.alt_m, float),
        'lever': np.asarray(final_descent.lever, float),
    }
    
    plot_descent_J_3d_plotly(
        M_grid_descent, H_descent, lever_grid_descent, J_descent_envelope,
        min_path=descent_path,
        title="3D DP Descent (Global Optimization)<br>Full Envelope with Optimal Path",
        initial_weight_kg=final_cruise.weight_kg[-1]
    )
    
    # Complete mission 3D
    print("[PLOT] Opening complete mission 3D visualization...")
    plot_complete_mission_3d_interactive(
        climb_result=final_climb,
        cruise_result=final_cruise,
        descent_result=final_descent,
        climb_info=None,
        descent_info=None
    )
    
    # Mission summary dashboard
    print("[PLOT] Creating mission summary dashboard...")
    plot_mission_summary_dashboard(
        climb_result=final_climb,
        cruise_result=final_cruise,
        descent_result=final_descent,
        initial_mass_kg=optimized_mass
    )
    
    # Combined performance analysis
    print("[PLOT] Creating combined performance analysis...")
    plot_combined_performance_analysis(
        climb_result=final_climb,
        cruise_result=final_cruise,
        descent_result=final_descent,
        initial_mass_kg=optimized_mass
    )
    
    # ========= FINAL SUMMARY =========================================
    
    print("\n" + "="*80)
    print("FINAL MISSION SUMMARY")
    print("="*80)
    
    climb_fuel = final_climb.cumFuel_kg[-1] if len(final_climb.cumFuel_kg) > 0 else 0.0
    cruise_fuel = final_cruise.total_fuel_consumed_kg
    descent_fuel = final_descent.total_fuel_consumed_kg
    total_fuel = climb_fuel + cruise_fuel + descent_fuel
    
    climb_time = np.sum(final_climb.dt_s) if len(final_climb.dt_s) > 0 else 0.0
    cruise_time = final_cruise.total_time_s
    descent_time = final_descent.total_time_s
    total_time = climb_time + cruise_time + descent_time
    
    print(f"OPTIMIZATION RESULTS:")
    print(f"  Original MAX_FUEL_KG: {MAX_FUEL_KG:.1f} kg")
    print(f"  Optimized fuel capacity: {optimized_fuel:.1f} kg")
    print(f"  Fuel savings: {MAX_FUEL_KG - optimized_fuel:.1f} kg")
    print(f"  Savings percentage: {(MAX_FUEL_KG - optimized_fuel) / MAX_FUEL_KG * 100:.1f}%")
    
    print(f"\nMISSION PERFORMANCE:")
    print(f"  Climb: {climb_fuel:.1f} kg, {climb_time/60:.1f} min")
    print(f"  Cruise: {cruise_fuel:.1f} kg, {cruise_time/60:.1f} min")
    print(f"  Descent: {descent_fuel:.2f} kg, {descent_time/60:.1f} min")
    print(f"  Total: {total_fuel:.1f} kg, {total_time/3600:.2f} hours")
    
    print(f"\nMISSION PARAMETERS:")
    print(f"  Initial mass: {optimized_mass:.1f} kg")
    print(f"  Final mass: {final_descent.final_weight_kg:.1f} kg")
    print(f"  Total distance: {final_cruise.target_distance_km:.0f} km")
    
    print(f"\nNOTE:")
    print(f"  This analysis uses the dynamically optimized fuel: {optimized_fuel:.1f} kg")
    print(f"  aircraft_config.py retains original MAX_FUEL_KG: {MAX_FUEL_KG:.1f} kg")
    print(f"  To use optimized fuel in main.py, manually update MAX_FUEL_KG to {optimized_fuel:.1f} kg")
    
    print("="*80 + "\n")
    
    plt.show(block=True)


if __name__ == "__main__":
    main()

