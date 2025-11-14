# =========  1 - MODULE INITIALIZATION =================
# ========= IMPORTS AND BASIC SETUP ===========================================
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ========= AIRCRAFT AND MISSION CONFIGURATION ================================
from aircraft_config import (
    INITIAL_MASS_KG, ENGINE_STUB_PATH,
    AtmosphericProperties, G_C, MAX_FUEL_KG, W_OE_KG, W_PL_KG
)
from mission_config import (
    TARGET_ALT_CLIMB_M, ALT_STEP_M, Y_AXIS_TOP_M,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE_CLIMB, STRATEGY_DT_CLIMB_S,
    CRUISE_TIME_STEP_S,
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT,
    MIN_DESCENT_MACH, MAX_DESCENT_MACH,
    TARGET_MISSION_RANGE_KM, INITIAL_CRUISE_DISTANCE_KM,
    RANGE_OPTIMIZATION_TOLERANCE_KM, MAX_RANGE_OPTIMIZATION_ITERATIONS,
    RANGE_OPTIMIZATION_DAMPING_FACTOR
)

# ========= CLIMB MODULE ===========================================
import climb
from climb import (
    compute_sep_grid_maxlever,
    dbg,
    ClimbingCore
)

# ========= AERODYNAMICS AND ENGINE WRAPPERS ===========================
from pyaerodynamics_wrapper import PyAerodynamicsWrapper
from pyengine_wrapper import EngineWrapper

# ========= CRUISE MODULE ==============================================
import cruise
from cruise import run_cruise_simulation

# ========= DESCENT MODULE =============================================
import descent
from descent import run_descent_dp_optimization

# ========= RANGE OPTIMIZATION MODULES =================================
from mission_range_plotting import (
    create_optimization_dashboard,
    print_optimization_report
)
from mission_range_optimizer import (
    MissionRangeOptimizer,
    calculate_total_mission_distance_km,
    adjust_cruise_segment_extension,
    adjust_cruise_segment_truncation
)


# =========  2 - MAIN EXECUTION FUNCTION ========================================
def main():
    """
    Main execution function for mission range optimization.
    
    Executes complete mission simulation with iterative range optimization:
    1. Initialize aircraft, aerodynamics, and engine models
    2. Execute climb phase optimization (computed once)
    3. Iterative optimization loop for cruise + descent
    4. Generate comprehensive visualization at convergence
    """
    
    print("\n" + "="*80)
    print("MISSION RANGE OPTIMIZATION - ITERATIVE SIMULATION")
    print("="*80)
    
    # ========= AIRCRAFT CONFIGURATION =========================================
    print("\n" + "="*80)
    print("AIRCRAFT CONFIGURATION")
    print("="*80)
    print(f"[CONFIG] MAX_FUEL_KG: {MAX_FUEL_KG:.1f} kg")
    print(f"[CONFIG] W_OE_KG: {W_OE_KG:.1f} kg")
    print(f"[CONFIG] W_PL_KG: {W_PL_KG:.1f} kg")
    print(f"[CONFIG] INITIAL_MASS_KG (W_TO_KG): {INITIAL_MASS_KG:.1f} kg")
    print(f"[CONFIG] = W_OE + W_PL + MAX_FUEL = {W_OE_KG + W_PL_KG + MAX_FUEL_KG:.1f} kg")
    print("="*80 + "\n")
    
    # ========= MISSION RANGE OPTIMIZATION PARAMETERS ==========================
    print("="*80)
    print("RANGE OPTIMIZATION PARAMETERS")
    print("="*80)
    print(f"[RANGE] Target mission range: {TARGET_MISSION_RANGE_KM:.0f} km")
    print(f"[RANGE] Initial cruise distance estimate: {INITIAL_CRUISE_DISTANCE_KM:.0f} km")
    print(f"[RANGE] Convergence tolerance: ±{RANGE_OPTIMIZATION_TOLERANCE_KM:.0f} km")
    print(f"[RANGE] Damping factor: {RANGE_OPTIMIZATION_DAMPING_FACTOR:.2f}")
    print(f"[RANGE] Maximum iterations: {MAX_RANGE_OPTIMIZATION_ITERATIONS}")
    print("="*80 + "\n")
    
    print(f"[MISSION] Target altitude: {TARGET_ALT_CLIMB_M:.0f} m")
    print(f"[MISSION] Target Mach: {TARGET_MACH_CRUISE:.3f}")
    
    # ========= INITIALIZE AERODYNAMICS AND ENGINE =============================
    print("\n[READ] Aerodynamics (Excel Sheet4) …")
    aero = PyAerodynamicsWrapper()
    
    atmospheric_props = AtmosphericProperties()
    
    M_min, M_max = float(aero.mach_grid[0]), float(aero.mach_grid[-1])
    M_dense = np.linspace(M_min, M_max, N_MACH_SAMPLES_CLIMB)
    H_plot = np.arange(START_ALTITUDE_CLIMB_M,
                       Y_AXIS_TOP_M + 0.5*ALT_STEP_M,
                       ALT_STEP_M)
    
    climb.M_MIN_EFFECTIVE = max(climb.M_MIN_DEFAULT, float(aero.mach_grid[0]))
    print(f"[INFO] Effective M_MIN set to {climb.M_MIN_EFFECTIVE:.3f} (sheet min Mach={aero.mach_grid[0]:.3f}).")
    
    print("[ENGINE] Loading engine stub …")
    eng = EngineWrapper("lls/stubs/engines/PW1127G-JM")
    
    print("[OPTIMIZATION] Pre-computing engine and drag grids for performance...")
    lever_grid = np.linspace(0.0, 1.0, 21)
    eng.precompute_grid(M_dense, H_plot, lever_grid)
    aero.precompute_drag_grid(M_dense, H_plot, INITIAL_MASS_KG)
    
    print("[PS] Computing background Ps grid (max lever, ref mass) …")
    M_grid, H_plot, Ps_base = compute_sep_grid_maxlever(aero, eng, INITIAL_MASS_KG,
                                                        M_grid=M_dense, H_grid=H_plot)
    
    # ========= CLIMB PHASE OPTIMIZATION (COMPUTED ONCE) =======================
    print("\n" + "="*80)
    print("CLIMB PHASE OPTIMIZATION (FIXED - COMPUTED ONCE)")
    print("="*80)
    
    uniform_step_size = TARGET_ALT_CLIMB_M / N_ALTITUDE_STEPS_CLIMB
    H_sched = np.arange(START_ALTITUDE_CLIMB_M,
                        TARGET_ALT_CLIMB_M + uniform_step_size,
                        uniform_step_size)
    
    print("[DP] Solving 3D fixed-mass DP for climb phase…")
    
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
    
    climb_fuel = float(np.nan_to_num(dp_sched.cumFuel_kg, nan=0.0)[-1])
    climb_time_hours = float(np.sum(dp_sched.dt_s)) / 3600.0 if len(dp_sched.dt_s) > 0 else 0.0
    
    print(f"\n[CLIMB] Climb phase optimization complete:")
    print(f"  Fuel consumed: {climb_fuel:.1f} kg")
    print(f"  Time: {climb_time_hours:.2f} hours ({climb_time_hours*60:.1f} minutes)")
    print(f"  Final altitude: {dp_sched.alt_m[-1]:.0f} m")
    print(f"  Final Mach: {dp_sched.mach[-1]:.3f}")
    
    # ========= INITIALIZE RANGE OPTIMIZER =====================================
    print("\n" + "="*80)
    print("INITIALIZING RANGE OPTIMIZER")
    print("="*80)
    
    optimizer = MissionRangeOptimizer(
        target_range_km=TARGET_MISSION_RANGE_KM,
        tolerance_km=RANGE_OPTIMIZATION_TOLERANCE_KM,
        damping_factor=RANGE_OPTIMIZATION_DAMPING_FACTOR,
        max_iterations=MAX_RANGE_OPTIMIZATION_ITERATIONS
    )
    
    # ========= ITERATIVE OPTIMIZATION LOOP ====================================
    print("\n" + "="*80)
    print("STARTING ITERATIVE RANGE OPTIMIZATION")
    print("="*80)
    print(f"Target: {TARGET_MISSION_RANGE_KM:.0f} km ± {RANGE_OPTIMIZATION_TOLERANCE_KM:.0f} km\n")
    
    current_cruise_distance_km = INITIAL_CRUISE_DISTANCE_KM
    converged = False
    iteration = 0
    
    final_cruise_results = None
    final_descent_results = None
    final_descent_info = None
    
    base_cruise_results = None
    
    while not converged and iteration < MAX_RANGE_OPTIMIZATION_ITERATIONS:
        iteration += 1
        
        print(f"\n--- Iteration {iteration} ---")
        print(f"[CRUISE] Running cruise simulation with distance = {current_cruise_distance_km:.1f} km")
        
        # ========= CRUISE PHASE SIMULATION ====================================
        try:
            if iteration == 1:
                cruise_results = run_cruise_simulation(
                    climb_result=dp_sched,
                    initial_mass_kg=INITIAL_MASS_KG,
                    target_distance_km=current_cruise_distance_km,
                    aero=aero,
                    engine=eng,
                    time_step_s=CRUISE_TIME_STEP_S,
                    create_plots=False
                )
                base_cruise_results = cruise_results
                base_cruise_distance = current_cruise_distance_km
                print(f"[CRUISE] Base cruise established: {base_cruise_distance:.1f} km")
            else:
                print(f"[CRUISE] Base cruise distance: {base_cruise_distance:.1f} km")
                print(f"[CRUISE] Current iteration target: {current_cruise_distance_km:.1f} km")
                
                if current_cruise_distance_km > base_cruise_distance:
                    additional_distance_km = current_cruise_distance_km - base_cruise_distance
                    print(f"[CRUISE] → EXTENDING from base by {additional_distance_km:.1f} km")
                    print(f"[CRUISE]   Base cruise ends at mass: {base_cruise_results.weight_kg[-1]:.1f} kg")
                    
                    cruise_results = adjust_cruise_segment_extension(
                        cruise_result=base_cruise_results,
                        additional_distance_km=additional_distance_km,
                        aero=aero,
                        engine=eng,
                        time_step_s=CRUISE_TIME_STEP_S
                    )
                    
                    print(f"[CRUISE]   After extension, cruise ends at mass: {cruise_results.weight_kg[-1]:.1f} kg")
                elif current_cruise_distance_km < base_cruise_distance:
                    print(f"[CRUISE] → TRUNCATING base from {base_cruise_distance:.1f} km to {current_cruise_distance_km:.1f} km")
                    
                    cruise_results = adjust_cruise_segment_truncation(
                        cruise_result=base_cruise_results,
                        new_cruise_distance_km=current_cruise_distance_km
                    )
                    
                    print(f"[CRUISE]   After truncation: {len(cruise_results.weight_kg)} points, final mass: {cruise_results.weight_kg[-1]:.1f} kg")
                else:
                    print(f"[CRUISE] → REUSING base cruise (same distance)")
                    cruise_results = base_cruise_results
        
        except Exception as e:
            print(f"[ERROR] Cruise simulation failed: {str(e)}")
            break
        
        # ========= MASS CONTINUITY VERIFICATION ===============================
        climb_final_mass_kg = float(dp_sched.mass_kg[-1]) if len(dp_sched.mass_kg) > 0 else (INITIAL_MASS_KG - climb_fuel)
        cruise_initial_mass_kg = float(cruise_results.weight_kg[0])
        cruise_final_mass_kg = float(cruise_results.weight_kg[-1])
        cruise_fuel_consumed_kg = float(cruise_results.total_fuel_consumed_kg)
        
        print(f"\n[MASS] Phase-wise mass tracking:")
        print(f"  Initial takeoff mass: {INITIAL_MASS_KG:.1f} kg")
        print(f"  Climb ending mass:    {climb_final_mass_kg:.1f} kg (burned {climb_fuel:.1f} kg)")
        print(f"  Cruise starting mass: {cruise_initial_mass_kg:.1f} kg")
        
        climb_cruise_error = abs(climb_final_mass_kg - cruise_initial_mass_kg)
        if climb_cruise_error > 0.1:
            print(f"  [WARNING] Climb→Cruise mass mismatch: {climb_cruise_error:.2f} kg difference!")
        else:
            print(f"")
        
        print(f"  Cruise ending mass:   {cruise_final_mass_kg:.1f} kg (burned {cruise_fuel_consumed_kg:.1f} kg)")
        print(f"  → Descent will start with mass: {cruise_final_mass_kg:.1f} kg")
        
        # ========= DESCENT PHASE OPTIMIZATION =================================
        print(f"\n[DESCENT] Running descent optimization with initial mass {cruise_final_mass_kg:.1f} kg...")
        
        try:
            descent_results, descent_info = run_descent_dp_optimization(
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
            
            descent_initial_mass_kg = float(descent_results.weight_kg[0]) if len(descent_results.weight_kg) > 0 else cruise_final_mass_kg
            descent_final_mass_kg = float(descent_results.final_weight_kg)
            descent_fuel_kg = float(descent_results.total_fuel_consumed_kg)
            
            mass_continuity_error = abs(descent_initial_mass_kg - cruise_final_mass_kg)
            if mass_continuity_error > 0.1:
                print(f"[WARNING] Mass continuity issue detected!")
                print(f"  Cruise final mass: {cruise_final_mass_kg:.1f} kg")
                print(f"  Descent initial mass: {descent_initial_mass_kg:.1f} kg")
                print(f"  Difference: {mass_continuity_error:.2f} kg")
            else:
                print(f"[MASS] ✓ Mass continuity verified: Descent initial mass matches cruise final mass")
            
            print(f"  Descent ending mass: {descent_final_mass_kg:.1f} kg (burned {descent_fuel_kg:.1f} kg)")
            
        except Exception as e:
            print(f"[ERROR] Descent simulation failed: {str(e)}")
            break
        
        # ========= CALCULATE TOTAL MISSION DISTANCE ===========================
        total_distance_km, distance_breakdown = calculate_total_mission_distance_km(
            climb_result=dp_sched,
            cruise_result=cruise_results,
            descent_result=descent_results
        )
        
        # ========= CHECK CONVERGENCE ==========================================
        converged, error_km = optimizer.check_convergence(total_distance_km)
        
        optimizer.record_iteration(
            iteration=iteration,
            cruise_distance_km=current_cruise_distance_km,
            total_distance_km=total_distance_km,
            error_km=error_km,
            converged=converged,
            cruise_final_mass_kg=cruise_final_mass_kg,
            descent_initial_mass_kg=descent_initial_mass_kg,
            descent_final_mass_kg=descent_final_mass_kg
        )
        
        optimizer.print_iteration_status(
            iteration=iteration,
            cruise_distance_km=current_cruise_distance_km,
            total_distance_km=total_distance_km,
            error_km=error_km,
            converged=converged,
            cruise_final_mass_kg=cruise_final_mass_kg,
            descent_final_mass_kg=descent_final_mass_kg
        )
        
        if converged:
            final_cruise_results = cruise_results
            final_descent_results = descent_results
            final_descent_info = descent_info
            print(f"\n{'='*80}")
            print("CONVERGENCE ACHIEVED!")
            print(f"{'='*80}")
            break
        
        # ========= COMPUTE NEXT CRUISE DISTANCE ===============================
        if not converged:
            next_cruise_distance_km = optimizer.compute_next_cruise_distance(
                current_cruise_distance_km=current_cruise_distance_km,
                error_km=error_km
            )
            
            print(f"[OPTIMIZER] Adjusting cruise distance: {current_cruise_distance_km:.1f} -> {next_cruise_distance_km:.1f} km")
            current_cruise_distance_km = next_cruise_distance_km
    
    # ========= OPTIMIZATION SUMMARY ===========================================
    if len(optimizer.iteration_history) == 0:
        print(f"\n{'='*80}")
        print("OPTIMIZATION FAILED - NO ITERATIONS COMPLETED")
        print(f"{'='*80}")
        print("[ERROR] The optimization loop exited before completing any iterations.")
        print("Please check the error messages above for details.")
        print(f"{'='*80}\n")
        return
    
    if not converged:
        print(f"\n{'='*80}")
        print("MAXIMUM ITERATIONS REACHED WITHOUT CONVERGENCE")
        print(f"{'='*80}")
        print(f"Final error: {optimizer.iteration_history[-1].distance_error_km:+.1f} km")
        print(f"Consider adjusting damping factor or increasing max iterations.")
        
        final_cruise_results = cruise_results
        final_descent_results = descent_results
        final_descent_info = descent_info
    
    summary = optimizer.get_optimization_summary()
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Target range: {summary['target_range_km']:.0f} km")
    print(f"Final total distance: {summary['final_total_distance_km']:.1f} km")
    print(f"Final cruise distance: {summary['final_cruise_distance_km']:.1f} km")
    print(f"Final error: {summary['final_error_km']:+.1f} km")
    print(f"Total iterations: {summary['total_iterations']}")
    print(f"Converged: {'Yes' if summary['converged'] else 'No'}")
    print(f"{'='*80}\n")
    
    optimizer.print_mass_evolution_summary()
    
    # ========= FINAL MISSION SUMMARY ==========================================
    print("="*80)
    print("COMPLETE MISSION SUMMARY (FINAL CONVERGED SOLUTION)")
    print("="*80)
    
    cruise_summary = final_cruise_results.get_summary_dict()
    descent_summary = final_descent_results.get_summary_dict()
    
    total_mission_fuel = climb_fuel + cruise_summary['cruise_fuel_kg'] + descent_summary['descent_fuel_kg']
    total_mission_time_hours = climb_time_hours + cruise_summary['cruise_time_hours'] + (descent_summary['descent_time_minutes'] / 60.0)
    
    _, final_distance_breakdown = calculate_total_mission_distance_km(
        climb_result=dp_sched,
        cruise_result=final_cruise_results,
        descent_result=final_descent_results
    )
    
    print(f"\nCLIMB PHASE:")
    print(f"  Fuel consumed: {climb_fuel:.1f} kg")
    print(f"  Time: {climb_time_hours:.2f} hours ({climb_time_hours*60:.1f} min)")
    print(f"  Distance: {final_distance_breakdown['climb_km']:.1f} km")
    print(f"  Final altitude: {dp_sched.alt_m[-1]:.0f} m")
    print(f"  Final Mach: {dp_sched.mach[-1]:.3f}")
    
    print(f"\nCRUISE PHASE:")
    print(f"  Distance: {cruise_summary['cruise_distance_km']:.0f} km")
    print(f"  Fuel consumed: {cruise_summary['cruise_fuel_kg']:.1f} kg")
    print(f"  Time: {cruise_summary['cruise_time_hours']:.2f} hours ({cruise_summary['cruise_time_hours']*60:.1f} min)")
    print(f"  Average fuel flow: {cruise_summary['avg_fuel_flow_kg_h']:.0f} kg/h")
    
    print(f"\nDESCENT PHASE:")
    print(f"  Fuel consumed: {descent_summary['descent_fuel_kg']:.2f} kg")
    print(f"  Time: {descent_summary['descent_time_minutes']:.1f} min")
    print(f"  Distance: {final_distance_breakdown['descent_km']:.1f} km")
    print(f"  Avg descent rate: {descent_summary['avg_descent_rate_mpm']:.0f} m/min")
    print(f"  Final Mach: {final_descent_results.mach[-1]:.3f}")
    print(f"  Final altitude: {final_descent_results.alt_m[-1]:.0f} m")
    
    print(f"\nTOTAL MISSION:")
    print(f"  Total distance: {final_distance_breakdown['total_km']:.1f} km")
    print(f"  Total fuel consumed: {total_mission_fuel:.1f} kg ({total_mission_fuel/INITIAL_MASS_KG*100:.1f}% of initial weight)")
    print(f"  Total time: {total_mission_time_hours:.2f} hours ({total_mission_time_hours*60:.1f} minutes)")
    print(f"  Initial weight: {INITIAL_MASS_KG:.0f} kg")
    print(f"  Final weight: {final_descent_results.final_weight_kg:.0f} kg")
    print(f"  Weight reduction: {INITIAL_MASS_KG - final_descent_results.final_weight_kg:.1f} kg")
    
    print(f"\nPHASE BREAKDOWN:")
    print(f"  Climb distance:   {final_distance_breakdown['climb_km']:.1f} km ({final_distance_breakdown['climb_km']/final_distance_breakdown['total_km']*100:.1f}%)")
    print(f"  Cruise distance:  {final_distance_breakdown['cruise_km']:.1f} km ({final_distance_breakdown['cruise_km']/final_distance_breakdown['total_km']*100:.1f}%)")
    print(f"  Descent distance: {final_distance_breakdown['descent_km']:.1f} km ({final_distance_breakdown['descent_km']/final_distance_breakdown['total_km']*100:.1f}%)")
    
    print("="*80)
    
    # ========= RANGE OPTIMIZATION VISUALIZATION ===============================
    print("\n" + "="*80)
    print("GENERATING RANGE OPTIMIZATION VISUALIZATIONS")
    print("="*80)
    
    optimization_summary = optimizer.get_optimization_summary()
    print_optimization_report(optimization_summary)
    
    climb_distance_km = final_distance_breakdown['climb_km']
    descent_distance_km = final_distance_breakdown['descent_km']
    
    iteration_phase_data = []
    for record in optimizer.iteration_history:
        iteration_phase_data.append({
            'iteration': record.iteration,
            'climb_km': climb_distance_km,
            'cruise_km': record.cruise_distance_km,
            'descent_km': descent_distance_km,
            'total_km': record.total_distance_km
        })
    
    print("\n[DASHBOARD] Creating comprehensive optimization dashboard...")
    print("[DASHBOARD] Note: Dashboard will generate all plots (optimization + phases)")
    print("[DASHBOARD] to avoid duplicates and ensure proper order.\n")
    
    figures = create_optimization_dashboard(
        iteration_history=optimizer.iteration_history,
        iteration_data=iteration_phase_data,
        optimization_summary=optimization_summary,
        save_dir=None,
        climb_result=dp_sched,
        cruise_result=final_cruise_results,
        descent_result=final_descent_results,
        climb_info=dp_info,
        descent_info=final_descent_info,
        aero=aero,
        engine=eng,
        initial_mass_kg=INITIAL_MASS_KG
    )
    
    # ========= DISPLAY OPTIMIZATION PLOTS =====================================
    print("\n" + "="*80)
    print("DISPLAYING RANGE OPTIMIZATION PLOTS")
    print("="*80)
    
    if 'convergence' in figures and figures['convergence']:
        print("[DISPLAY] Opening convergence history plot...")
        figures['convergence'].show()
    
    if 'adjustment' in figures and figures['adjustment']:
        print("[DISPLAY] Opening cruise adjustment strategy plot...")
        figures['adjustment'].show()
    
    if 'breakdown' in figures and figures['breakdown']:
        print("[DISPLAY] Opening distance breakdown plot...")
        figures['breakdown'].show()
    
    if 'summary' in figures and figures['summary']:
        print("[DISPLAY] Opening optimization summary table...")
        figures['summary'].show()
    
    print("\n[DASHBOARD] All optimization plots displayed and saved!")
    
    plt.show(block=True)
    
    print("\n" + "="*80)
    print("MISSION RANGE OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
