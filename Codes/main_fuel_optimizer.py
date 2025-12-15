# ========= IMPORTS AND BASIC SETUP ===========================================
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ========= AIRCRAFT AND MISSION CONFIGURATION ================================
from aircraft_config import (
    INITIAL_MASS_KG,
    W_FUEL_KG, W_OE_KG, W_PL_KG,
    LEVER_MIN, LEVER_MAX
)
from pyengine_wrapper import ENGINE_STUB_PATH
from mission_config import (
    # Pre-computation grid parameters
    N_MACH_SAMPLES_PRECOMPUTE, N_ALTITUDE_SAMPLES_PRECOMPUTE, N_LEVER_SAMPLES_PRECOMPUTE,
    ALTITUDE_MAX_PRECOMPUTE_M,
    # Climb phase parameters
    TARGET_ALT_CLIMB_M,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    START_ALTITUDE_CLIMB_M,
    # Cruise phase parameters
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S,
    # Descent phase parameters
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT,
    MIN_DESCENT_MACH, MAX_DESCENT_MACH,
    # Feature flags
    ENABLE_EXCEL_EXPORT
)

# ========= CLIMB MODULE ===========================================
from climb_plotting import (
    GridConfig,
    compute_sep_grid_maxlever,
    compute_full_envelope
)

# ========= AERODYNAMICS AND ENGINE WRAPPERS ===========================
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper
from descent import DescentCore

# ========= PLOTTING MODULES (FINAL VISUALIZATION ONLY) ================
from cruise_plotting import plot_performance_2d as plot_cruise_performance_2d
from climb_plotting import (
    plot_3d_cost_space, 
    plot_performance_2d as plot_climb_performance_2d
)
from descent_plotting import (
    plot_descent_trajectory_interactive, 
    plot_descent_J_3d_plotly
)
from mission_summary import (
    plot_mission_summary_dashboard, 
    plot_combined_performance_analysis,
    plot_complete_mission_3d
)
from mission_excel_export import export_mission_to_excel

# ========= CG ANALYSIS MODULES ========================================
from cg_x_calculation import record_mission_history
from cg_plotting import plot_cg_analysis

# ========= FUEL OPTIMIZATION MODULES ==================================
from mission_fuel_optimizer import optimize_fuel_capacity, SAFETY_BUFFER_PERCENT
from mission_fuel_plotting import visualize_convergence_analysis


# ========= MAIN EXECUTION FUNCTION ========================================
def main():
    """
    Main execution function for mission analysis with fuel capacity optimization.
    
    The optimization process encompasses the following stages:
    1. Iterative optimization loop to determine minimum fuel requirement
    2. Convergence analysis visualization generation
    3. Final mission execution utilizing optimized fuel capacity
    4. Comprehensive mission visualization generation
    """
    
    # ========= OPTIMIZATION LOOP =========================================
    
    print("\n" + "="*80)
    print("MISSION ANALYSIS WITH FUEL CAPACITY OPTIMIZATION")
    print("="*80)
    
    print("[DATA] Loading aerodynamic and engine data")
    aero = PyAerodynamicsWrapper()
    eng = EngineWrapper(ENGINE_STUB_PATH)
    
    # Pre-computation grids for engine and aerodynamic caching
    M_min, M_max = float(aero.mach_grid[0]), float(aero.mach_grid[-1])
    M_precompute = np.linspace(M_min, M_max, N_MACH_SAMPLES_PRECOMPUTE)
    H_precompute = np.linspace(START_ALTITUDE_CLIMB_M, 
                               ALTITUDE_MAX_PRECOMPUTE_M, 
                               N_ALTITUDE_SAMPLES_PRECOMPUTE)
    
    # Dense grids for visualization
    M_dense = np.linspace(M_min, M_max, N_MACH_SAMPLES_CLIMB)
    H_plot = np.arange(START_ALTITUDE_CLIMB_M, 
                        GridConfig.Y_AXIS_TOP_M + 0.5*GridConfig.ALT_STEP_M, 
                        GridConfig.ALT_STEP_M)
    
    # Pre-compute engine grid for performance (mass-independent)
    print("[OPTIMIZATION] Pre-computing engine grid")
    print(f"[CACHE] Engine pre-computation grid: {len(M_precompute)} Mach × {len(H_precompute)} Alt × {N_LEVER_SAMPLES_PRECOMPUTE+1} Lever")
    lever_precompute = np.linspace(0.0, 1.0, N_LEVER_SAMPLES_PRECOMPUTE + 1)
    eng.precompute_grid(M_precompute, H_precompute, lever_precompute)
    
    # Drag pre-computation skipped: Drag = f(M, h, m) where mass varies during mission
    # Cache will populate naturally with actual dynamic masses during DP optimization
    print("[CACHE] Drag cache will populate dynamically during mission (mass-dependent)")
    
    # Run fuel optimization
    print("\n[OPTIMIZATION] Fuel capacity optimization loop initiated")
    optimal_result, convergence_history = optimize_fuel_capacity(
        aero=aero,
        eng=eng,
        mach_grid=M_dense,
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
    print("FINAL MISSION EXECUTION WITH OPTIMIZED FUEL")
    print("="*80)
    print(f"[CONFIG] Optimized fuel capacity: {optimized_fuel:.2f} kg")
    print(f"[CONFIG] Optimized total mass: {optimized_mass:.2f} kg")
    print(f"[CONFIG] Savings vs original W_FUEL_KG ({W_FUEL_KG:.2f} kg): {W_FUEL_KG - optimized_fuel:.2f} kg ({(W_FUEL_KG - optimized_fuel) / W_FUEL_KG * 100:.2f}%)")
    print("="*80 + "\n")
    
    # Use the optimized result directly (already computed in the optimization loop)
    # This is the final mission with optimized fuel
    final_climb = optimal_result.climb_result
    final_cruise = optimal_result.cruise_result
    final_descent = optimal_result.descent_result
    
    print("[MISSION] Optimized mission results prepared for visualization")
    print(f"[MISSION] Climb: {optimal_result.climb_fuel_kg:.2f} kg, {optimal_result.climb_time_s/60:.2f} min")
    print(f"[MISSION] Cruise: {optimal_result.cruise_fuel_kg:.2f} kg, {optimal_result.cruise_time_s/60:.2f} min")
    print(f"[MISSION] Descent: {optimal_result.descent_fuel_kg:.2f} kg, {optimal_result.descent_time_s/60:.2f} min")
    print(f"[MISSION] Total: {optimal_result.fuel_consumed_kg:.2f} kg, {optimal_result.total_time_s/60:.2f} min")
    
    
    # ========= COMPUTE GRIDS FOR VISUALIZATIONS =========================================
    
    print("\n[VISUALIZATION] Computing grids for final mission visualization")
    
    # Compute Ps grid
    mach_grid, H_plot, Ps_base = compute_sep_grid_maxlever(
        aero, eng, optimized_mass,
        mach_grid=M_dense, H_grid=H_plot
    )
    
    # Climb phase visualization data
    uniform_step_size = TARGET_ALT_CLIMB_M / N_ALTITUDE_STEPS_CLIMB
    altitude_sched = np.arange(START_ALTITUDE_CLIMB_M, 
                        TARGET_ALT_CLIMB_M + uniform_step_size, 
                        uniform_step_size)
    
    # Descent phase visualization data
    H_descent = np.linspace(final_cruise.altitude_m[-1], 
                           TARGET_DESCENT_ALT_M, 
                           N_ALTITUDE_STEPS_DESCENT)
    M_min_descent = max(MIN_DESCENT_MACH, TARGET_DESCENT_MACH - 0.1)
    M_max_descent = min(MAX_DESCENT_MACH, 
                       final_cruise.mach_number[-1] + 0.05)
    mach_grid_descent = np.linspace(M_min_descent, M_max_descent, N_MACH_SAMPLES_DESCENT)
    
    # ========= GENERATE ALL FINAL PLOTS =========================================
    
    print("\n[VISUALIZATION] Generating final mission visualizations")
    
    # Climb plots
    print("[VISUALIZATION] Creating climb performance analysis")
    plot_climb_performance_2d(final_climb, None)
    
    # 3D climb visualization
    # Use initial mass as reference for envelope visualization
    print("[VISUALIZATION] Opening 3D climb visualization")
    lever_grid_envelope = np.linspace(LEVER_MIN, LEVER_MAX, N_LEVER_SAMPLES_CLIMB)
    J_envelope = compute_full_envelope(
        aero, eng, mach_grid, altitude_sched, lever_grid_envelope,
        mass_kg=INITIAL_MASS_KG
    )
    
    min_path = {
        'mach': np.asarray(final_climb.mach, float),
        'alt': np.asarray(final_climb.alt_m, float),
        'lever': np.asarray(final_climb.lever, float),
    }
    
    plot_3d_cost_space(
        mach_grid, altitude_sched, lever_grid_envelope, J_envelope, 
        min_path=min_path,
        title="3D DP (Global Optimization)<br>Full Engine Envelope with Optimal Path",
        save_to_optimized=True
    )
    
    # Cruise plots
    print("[VISUALIZATION] Creating cruise performance analysis")
    plot_cruise_performance_2d(final_cruise)
    
    # Descent plots
    print("[VISUALIZATION] Opening descent visualizations")
    plot_descent_trajectory_interactive(final_descent)
    
    # Descent 3D visualization
    lever_grid_descent = np.linspace(LEVER_MIN, LEVER_MAX, N_LEVER_SAMPLES_DESCENT)
    J_descent_envelope = DescentCore.compute_full_envelope(
        aero, eng, mach_grid_descent, H_descent, lever_grid_descent,
        mass_kg=final_cruise.mass_kg[-1],
        target_mach=TARGET_DESCENT_MACH
    )
    
    descent_path = {
        'mach': np.asarray(final_descent.mach, float),
        'alt': np.asarray(final_descent.alt_m, float),
        'lever': np.asarray(final_descent.lever, float),
    }
    
    plot_descent_J_3d_plotly(
        mach_grid_descent, H_descent, lever_grid_descent, J_descent_envelope,
        min_path=descent_path,
        title="3D DP Descent (Global Optimization)<br>Full Envelope with Optimal Path",
        mass_kg=final_cruise.mass_kg[-1],
        save_to_optimized=True
    )
    
    # Complete mission 3D
    print("[VISUALIZATION] Opening complete mission 3D visualization")
    plot_complete_mission_3d(
        climb_result=final_climb,
        cruise_result=final_cruise,
        descent_result=final_descent,
        climb_info={},
        descent_info={},
        save_to_optimized=True
    )
    
    # Mission summary dashboard
    print("[SUMMARY] Creating mission summary dashboard")
    plot_mission_summary_dashboard(
        climb_result=final_climb,
        cruise_result=final_cruise,
        descent_result=final_descent,
        initial_mass_kg=optimized_mass,
        save_to_optimized=True
    )
    
    # Combined performance analysis
    print("[ANALYSIS] Creating combined performance analysis")
    plot_combined_performance_analysis(
        climb_result=final_climb,
        cruise_result=final_cruise,
        descent_result=final_descent,
        initial_mass_kg=optimized_mass,
        save_to_optimized=True
    )
    
    # Record mission history for CG analysis
    print(f"\n[CG] Recording mission history for CG analysis")
    try:
        record_mission_history(
            climb_result=final_climb,
            cruise_result=final_cruise,
            descent_result=final_descent
            ,
            initial_fuel_kg=optimized_fuel
        )
        print(f"[CG] Mission history recorded successfully")
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
    
    # Export all mission data to Excel (includes optimization history and CG data)
    if ENABLE_EXCEL_EXPORT:
        print(f"\n[EXPORT] Exporting mission data to Excel")
        try:
            excel_path = export_mission_to_excel(
                climb_result=final_climb,
                cruise_result=final_cruise,
                descent_result=final_descent,
                initial_mass_kg=optimized_mass,
                fuel_optimization_history=convergence_history
            )
            print(f"[EXPORT] Mission data successfully exported to Excel")
        except Exception as e:
            print(f"[ERROR] Excel export failed: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[EXPORT] Excel export disabled (ENABLE_EXCEL_EXPORT = False)")
    
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
    print(f"  Original W_FUEL_KG: {W_FUEL_KG:.2f} kg")
    print(f"  Optimized fuel capacity: {optimized_fuel:.2f} kg")
    print(f"  Fuel savings: {W_FUEL_KG - optimized_fuel:.2f} kg")
    print(f"  Savings percentage: {(W_FUEL_KG - optimized_fuel) / W_FUEL_KG * 100:.2f}%")
    
    print(f"\nMISSION PERFORMANCE:")
    print(f"  Climb: {climb_fuel:.2f} kg, {climb_time/60:.2f} min")
    print(f"  Cruise: {cruise_fuel:.2f} kg, {cruise_time/60:.2f} min")
    print(f"  Descent: {descent_fuel:.2f} kg, {descent_time/60:.2f} min")
    print(f"  Total: {total_fuel:.2f} kg, {total_time/3600:.2f} hours")
    
    print(f"\nMISSION PARAMETERS:")
    print(f"  Initial mass: {optimized_mass:.2f} kg")
    print(f"  Final mass: {final_descent.final_mass_kg:.2f} kg")
    print(f"  Total distance: {final_cruise.target_distance_km:.2f} km")
    
    print(f"\nIMPLEMENTATION NOTE:")
    print(f"  Analysis utilizes dynamically optimized fuel: {optimized_fuel:.2f} kg")
    print(f"  aircraft_config.py retains original W_FUEL_KG: {W_FUEL_KG:.2f} kg")
    print(f"  To implement optimized fuel in main.py, update W_FUEL_KG to {optimized_fuel:.2f} kg")
    
    print("="*80 + "\n")
    
    plt.show(block=True)


if __name__ == "__main__":
    main()

