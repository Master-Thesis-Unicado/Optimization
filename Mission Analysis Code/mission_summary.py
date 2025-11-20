"""
Mission Summary Dashboard Module

This module provides comprehensive mission analysis summary visualization
combining climb, cruise, and descent phases into a professional dashboard.

Author: Mission Analysis System
"""

from __future__ import annotations
import numpy as np
from typing import Optional
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set Plotly to open in browser
pio.renderers.default = "browser"

# Import necessary components
from aircraft_config import (
    a_from_altitude, N_ENGINES, S_REF_M2,
    W_AIRFRAME_KG, W_PROPULSION_KG, W_SYSTEMS_KG,
    PAYLOAD_PER_PERSON_KG, DEFAULT_PASSENGERS, MAX_FUEL_KG,
    W_OE_KG, W_PL_KG
)
from mission_config import (
    TARGET_ALT_CLIMB_M,
    START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE_CLIMB, STRATEGY_DT_CLIMB_S,
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S,
    TARGET_MISSION_RANGE_KM, INITIAL_CRUISE_DISTANCE_KM,
    RANGE_OPTIMIZATION_TOLERANCE_KM, MAX_RANGE_OPTIMIZATION_ITERATIONS,
    RANGE_OPTIMIZATION_DAMPING_FACTOR,
    ENABLE_CRUISE_CLIMB, CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION,
    CRUISE_CLIMB_ALTITUDE_INCREMENT_M, CRUISE_CLIMB_MACH_TOLERANCE,
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT
)
from climb import MinFuelSchedule
from cruise import CruiseResults
from descent import DescentResults
from pyaerodynamics_wrapper import PyAerodynamicsWrapper

# Import visualization configuration for consistent styling
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_axis_config,
    get_table_header_style, get_table_cell_style, HoverTemplates,
    get_or_create_run_directory
)


def calculate_aerodynamic_data_throughout_mission(climb_result: MinFuelSchedule,
                                                cruise_result: CruiseResults,
                                                descent_result: DescentResults,
                                                initial_mass_kg: float):
    """
    Calculate comprehensive aerodynamic data throughout the entire mission.
    
    Args:
        climb_result: Results from climb phase
        cruise_result: Results from cruise phase
        descent_result: Results from descent phase
        initial_mass_kg: Initial aircraft mass
        
    Returns:
        Dictionary containing aerodynamic data for each phase
    """
    aero = PyAerodynamicsWrapper()
    
    # Initialize data storage
    aero_data = {
        'climb': {'altitude': [], 'mach': [], 'weight': [], 'cd': [], 'cl': [], 'ld': [], 'drag': []},
        'cruise': {'altitude': [], 'mach': [], 'weight': [], 'cd': [], 'cl': [], 'ld': [], 'drag': []},
        'descent': {'altitude': [], 'mach': [], 'weight': [], 'cd': [], 'cl': [], 'ld': [], 'drag': []}
    }
    
    # Calculate climb aerodynamic data (sample every 5th point for efficiency)
    if len(climb_result.alt_m) > 0:
        sample_rate = max(1, len(climb_result.alt_m) // 20)  # Sample max 20 points
        print(f"[AERO] Processing {len(climb_result.alt_m)} climb points (sampling every {sample_rate}th point)...")
        for i in range(0, len(climb_result.alt_m), sample_rate):
            if i % (sample_rate * 5) == 0:  # Progress indicator every 5 samples
                print(f"[AERO] Climb progress: {i//sample_rate + 1}/{(len(climb_result.alt_m) + sample_rate - 1)//sample_rate}")
            
            alt = climb_result.alt_m[i]
            mach = climb_result.mach[i]
            weight = climb_result.mass_kg[i] if i < len(climb_result.mass_kg) else initial_mass_kg
            
            # Get comprehensive aerodynamic data
            aero_comp = aero.get_comprehensive_aerodynamics(mach, alt, weight)
            
            aero_data['climb']['altitude'].append(alt)
            aero_data['climb']['mach'].append(mach)
            aero_data['climb']['weight'].append(weight)
            aero_data['climb']['cd'].append(aero_comp['cd'])
            aero_data['climb']['cl'].append(aero_comp['cl'])
            aero_data['climb']['ld'].append(aero_comp['ld'])
            aero_data['climb']['drag'].append(aero_comp['drag_force_N'])
    
    # Calculate cruise aerodynamic data (sample at key points)
    if hasattr(cruise_result, 'altitude_m') and hasattr(cruise_result, 'mach'):
        cruise_alt = getattr(cruise_result, 'altitude_m', cruise_result.initial_state.altitude_m)
        cruise_mach = getattr(cruise_result, 'mach', cruise_result.initial_state.mach)
        cruise_weight = getattr(cruise_result, 'weight_kg', cruise_result.initial_state.weight_kg)
        
        # Sample at start, middle, and end of cruise
        for weight in [cruise_weight, cruise_weight * 0.95, cruise_weight * 0.9]:
            aero_comp = aero.get_comprehensive_aerodynamics(cruise_mach, cruise_alt, weight)
            
            aero_data['cruise']['altitude'].append(cruise_alt)
            aero_data['cruise']['mach'].append(cruise_mach)
            aero_data['cruise']['weight'].append(weight)
            aero_data['cruise']['cd'].append(aero_comp['cd'])
            aero_data['cruise']['cl'].append(aero_comp['cl'])
            aero_data['cruise']['ld'].append(aero_comp['ld'])
            aero_data['cruise']['drag'].append(aero_comp['drag_force_N'])
    
    # Calculate descent aerodynamic data (sample every 5th point for efficiency)
    if len(descent_result.alt_m) > 0:
        sample_rate = max(1, len(descent_result.alt_m) // 20)  # Sample max 20 points
        print(f"[AERO] Processing {len(descent_result.alt_m)} descent points (sampling every {sample_rate}th point)...")
        for i in range(0, len(descent_result.alt_m), sample_rate):
            if i % (sample_rate * 5) == 0:  # Progress indicator every 5 samples
                print(f"[AERO] Descent progress: {i//sample_rate + 1}/{(len(descent_result.alt_m) + sample_rate - 1)//sample_rate}")
            
            alt = descent_result.alt_m[i]
            mach = descent_result.mach[i]
            weight = descent_result.weight_kg[i] if i < len(descent_result.weight_kg) else initial_mass_kg * 0.85
            
            # Get comprehensive aerodynamic data
            aero_comp = aero.get_comprehensive_aerodynamics(mach, alt, weight)
            
            aero_data['descent']['altitude'].append(alt)
            aero_data['descent']['mach'].append(mach)
            aero_data['descent']['weight'].append(weight)
            aero_data['descent']['cd'].append(aero_comp['cd'])
            aero_data['descent']['cl'].append(aero_comp['cl'])
            aero_data['descent']['ld'].append(aero_comp['ld'])
            aero_data['descent']['drag'].append(aero_comp['drag_force_N'])
    
    return aero_data


def plot_mission_summary_dashboard(climb_result: MinFuelSchedule,
                                   cruise_result: CruiseResults,
                                   descent_result: DescentResults,
                                   initial_mass_kg: float,
                                   save_html: Optional[str] = None,
                                   simulation_duration_min: Optional[float] = None):
    """
    Create comprehensive mission summary dashboard with scientific visualization.
    
    Professional dashboard showing all key mission parameters, metrics, and performance
    indicators in a publication-quality format.
    
    Args:
        climb_result: Results from climb phase
        cruise_result: Results from cruise phase  
        descent_result: Results from descent phase
        initial_mass_kg: Initial aircraft mass
        save_html: Optional path to save HTML file
        simulation_duration_min: Optional simulation execution time in minutes
    """
    # Calculate aerodynamic data throughout mission
    print("[AERO] Calculating aerodynamic data throughout mission...")
    aero_data = calculate_aerodynamic_data_throughout_mission(
        climb_result, cruise_result, descent_result, initial_mass_kg
    )
    
    # Calculate comprehensive statistics
    climb_time_s = np.sum(climb_result.dt_s) if len(climb_result.dt_s) > 0 else 0.0
    climb_fuel = climb_result.cumFuel_kg[-1]
    
    # Calculate climb horizontal distance (ground distance covered during climb)
    # Using average true airspeed and time, accounting for vertical component
    if len(climb_result.dt_s) > 0 and len(climb_result.mach) > 0:
        climb_distances = []
        for i in range(len(climb_result.dt_s)):
            if i < len(climb_result.mach) and i < len(climb_result.alt_m):
                a = a_from_altitude(climb_result.alt_m[i])
                V_tas = climb_result.mach[i] * a  # True airspeed
                # Horizontal distance = TAS * dt (assuming small climb angles)
                climb_distances.append(V_tas * climb_result.dt_s[i] / 1000.0)  # Convert to km
        climb_distance_km = sum(climb_distances)
    else:
        climb_distance_km = 0.0
    
    cruise_time_s = cruise_result.total_time_s
    cruise_fuel = cruise_result.total_fuel_consumed_kg
    cruise_distance_km = cruise_result.target_distance_km
    
    descent_time_s = descent_result.total_time_s
    descent_fuel = descent_result.total_fuel_consumed_kg
    
    # Calculate descent horizontal distance (ground distance covered during descent)
    if len(descent_result.dt_s) > 0 and len(descent_result.mach) > 0:
        descent_distances = []
        for i in range(len(descent_result.dt_s)):
            if i < len(descent_result.mach) and i < len(descent_result.alt_m):
                a = a_from_altitude(descent_result.alt_m[i])
                V_tas = descent_result.mach[i] * a  # True airspeed
                # Horizontal distance = TAS * dt (assuming small descent angles)
                descent_distances.append(V_tas * descent_result.dt_s[i] / 1000.0)  # Convert to km
        descent_distance_km = sum(descent_distances)
    else:
        descent_distance_km = 0.0
    
    total_time_s = climb_time_s + cruise_time_s + descent_time_s
    total_fuel = climb_fuel + cruise_fuel + descent_fuel
    total_distance_km = climb_distance_km + cruise_distance_km + descent_distance_km
    
    # Create figure with subplots including aerodynamic data and fuel management
    fig = make_subplots(
        rows=9, cols=3,
        subplot_titles=(
            '<b>Mission Profile</b>',
            '<b>Phase Breakdown</b>',
            '<b>Fuel Consumption</b>',
            '<b>Weight Evolution</b>',
            '<b>Drag Coefficient (CD)</b>',
            '<b>Lift Coefficient (CL)</b>',
            '<b>Lift-to-Drag Ratio (L/D)</b>',
            '<b>Mission Statistics</b>',
            '<b>Key Parameters</b>',
            '<b>Fuel Management</b>',
            '<b>Climb Configuration</b>',
            '<b>Cruise Configuration</b>',
            '<b>Descent Configuration</b>',
            '<b>Range Optimization</b>',
            '<b>Cruise Climb</b>',
            '<b>Aircraft Configuration</b>'
        ),
        specs=[
            [{"colspan": 3, "type": "scatter"}, None, None],
            [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"colspan": 2, "type": "table"}, None, {"type": "table"}],
            [{"colspan": 3, "type": "table"}, None, None],
            [{"type": "table"}, {"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}, {"type": "table"}],
            [None, None, None]
        ],
        row_heights=[0.07, 0.07, 0.07, 0.09, 0.16, 0.24, 0.24, 0.00, 0.00],
        vertical_spacing=0.02,
        horizontal_spacing=0.10
    )
    
    # ========= ROW 1: MISSION PROFILE =========
    # Complete altitude profile over time
    climb_time = np.cumsum(climb_result.dt_s)
    cruise_time_array = cruise_result.time_s + climb_time[-1]
    descent_time_array = descent_result.time_s + cruise_time_array[-1]
    
    # Convert to minutes
    climb_time_min = climb_time / 60.0
    cruise_time_min = cruise_time_array / 60.0
    descent_time_min = descent_time_array / 60.0
    
    # Plot altitude profile
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_result.alt_m,
            mode='lines',
            name='Climb',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.2)',
            hovertemplate=HoverTemplates.altitude('Climb', 'Time', 'min')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_result.altitude_m,
            mode='lines',
            name='Cruise',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.2)',
            hovertemplate=HoverTemplates.altitude('Cruise', 'Time', 'min')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_result.alt_m,
            mode='lines',
            name='Descent',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.2)',
            hovertemplate=HoverTemplates.altitude('Descent', 'Time', 'min')
        ),
        row=1, col=1
    )
    
    # Add phase transition lines
    fig.add_vline(x=climb_time_min[-1], line_dash=LineStyles.DASH, line_color=Colors.TRANSITION_LINE, 
                 line_width=LineStyles.MEDIUM, row=1, col=1, annotation_text="Climb→Cruise",
                 annotation_position="top")
    fig.add_vline(x=cruise_time_min[-1], line_dash=LineStyles.DASH, line_color=Colors.TRANSITION_LINE,
                 line_width=LineStyles.MEDIUM, row=1, col=1, annotation_text="Cruise→Descent",
                 annotation_position="top")
    
    # ========= ROW 2 COL 1: PHASE BREAKDOWN (Bar Chart) =========
    phases = ['Climb', 'Cruise', 'Descent']
    phase_times_min = [climb_time_s/60, cruise_time_s/60, descent_time_s/60]
    phase_fuels = [climb_fuel, cruise_fuel, descent_fuel]
    phase_colors = [Colors.CLIMB, Colors.CRUISE, Colors.DESCENT]
    
    fig.add_trace(
        go.Bar(
            x=phases,
            y=phase_times_min,
            name='Time (min)',
            marker_color=phase_colors,
            text=[f'{t:.1f} min<br>{f:.1f} kg' for t, f in zip(phase_times_min, phase_fuels)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Time: %{y:.1f} min<br><extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # ========= ROW 2 COL 2: FUEL CONSUMPTION =========
    climb_fuel_array = climb_result.cumFuel_kg
    cruise_fuel_array = cruise_result.fuel_consumed_kg + climb_fuel
    descent_fuel_array = descent_result.cumFuel_kg + cruise_fuel_array[-1]
    
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_fuel_array,
            mode='lines',
            line=dict(color=Colors.CLIMB, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate=HoverTemplates.fuel('Climb', 'Time', 'min')
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_fuel_array,
            mode='lines',
            line=dict(color=Colors.CRUISE, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate=HoverTemplates.fuel('Cruise', 'Time', 'min')
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_fuel_array,
            mode='lines',
            line=dict(color=Colors.DESCENT, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate=HoverTemplates.fuel('Descent', 'Time', 'min')
        ),
        row=2, col=2
    )
    
    # ========= ROW 2 COL 3: WEIGHT EVOLUTION =========
    climb_weight = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic weight from DP optimization
    cruise_weight = cruise_result.weight_kg
    descent_weight = descent_result.weight_kg
    
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_weight,
            mode='lines',
            line=dict(color=Colors.CLIMB, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate='<b>Climb</b><br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_weight,
            mode='lines',
            line=dict(color=Colors.CRUISE, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate='<b>Cruise</b><br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_weight,
            mode='lines',
            line=dict(color=Colors.DESCENT, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate='<b>Descent</b><br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    # ========= ROW 3: AERODYNAMIC DATA PLOTS =========
    # Drag Coefficient (CD) vs Altitude
    if len(aero_data['climb']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['climb']['altitude'],
                y=aero_data['climb']['cd'],
                mode='lines+markers',
                name='Climb CD',
                line=dict(color=Colors.CLIMB, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Climb</b><br>Altitude: %{x:.0f} m<br>CD: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    if len(aero_data['cruise']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['cruise']['altitude'],
                y=aero_data['cruise']['cd'],
                mode='lines+markers',
                name='Cruise CD',
                line=dict(color=Colors.CRUISE, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Cruise</b><br>Altitude: %{x:.0f} m<br>CD: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    if len(aero_data['descent']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['descent']['altitude'],
                y=aero_data['descent']['cd'],
                mode='lines+markers',
                name='Descent CD',
                line=dict(color=Colors.DESCENT, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Descent</b><br>Altitude: %{x:.0f} m<br>CD: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Lift Coefficient (CL) vs Altitude
    if len(aero_data['climb']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['climb']['altitude'],
                y=aero_data['climb']['cl'],
                mode='lines+markers',
                name='Climb CL',
                line=dict(color=Colors.CLIMB, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Climb</b><br>Altitude: %{x:.0f} m<br>CL: %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=3, col=2
        )
    
    if len(aero_data['cruise']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['cruise']['altitude'],
                y=aero_data['cruise']['cl'],
                mode='lines+markers',
                name='Cruise CL',
                line=dict(color=Colors.CRUISE, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Cruise</b><br>Altitude: %{x:.0f} m<br>CL: %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=3, col=2
        )
    
    if len(aero_data['descent']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['descent']['altitude'],
                y=aero_data['descent']['cl'],
                mode='lines+markers',
                name='Descent CL',
                line=dict(color=Colors.DESCENT, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Descent</b><br>Altitude: %{x:.0f} m<br>CL: %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=3, col=2
        )
    
    # Lift-to-Drag Ratio (L/D) vs Altitude
    if len(aero_data['climb']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['climb']['altitude'],
                y=aero_data['climb']['ld'],
                mode='lines+markers',
                name='Climb L/D',
                line=dict(color=Colors.CLIMB, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Climb</b><br>Altitude: %{x:.0f} m<br>L/D: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=3, col=3
        )
    
    if len(aero_data['cruise']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['cruise']['altitude'],
                y=aero_data['cruise']['ld'],
                mode='lines+markers',
                name='Cruise L/D',
                line=dict(color=Colors.CRUISE, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Cruise</b><br>Altitude: %{x:.0f} m<br>L/D: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=3, col=3
        )
    
    if len(aero_data['descent']['altitude']) > 0:
        fig.add_trace(
            go.Scatter(
                x=aero_data['descent']['altitude'],
                y=aero_data['descent']['ld'],
                mode='lines+markers',
                name='Descent L/D',
                line=dict(color=Colors.DESCENT, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Descent</b><br>Altitude: %{x:.0f} m<br>L/D: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=3, col=3
        )
    
    # ========= ROW 4 COL 1: MISSION STATISTICS TABLE =========
    header_style = get_table_header_style()
    cell_style = get_table_cell_style()
    
    # Override cell_style to remove fixed height and prevent scrollbar
    cell_style_no_height = {k: v for k, v in cell_style.items() if k != 'height'}
    
    # Verify consistency between weight tracking and fuel consumption
    weight_loss_calculated = initial_mass_kg - descent_result.final_weight_kg
    
    # Diagnostic print to identify discrepancy
    print(f"\n[MISSION SUMMARY] Fuel and Weight Consistency Check:")
    print(f"  Climb fuel: {climb_fuel:.2f} kg")
    print(f"  Cruise fuel: {cruise_fuel:.2f} kg")
    print(f"  Descent fuel: {descent_fuel:.2f} kg")
    print(f"  Total fuel (sum): {total_fuel:.2f} kg")
    print(f"  Initial mass: {initial_mass_kg:.2f} kg")
    print(f"  Final weight: {descent_result.final_weight_kg:.2f} kg")
    print(f"  Weight loss (diff): {weight_loss_calculated:.2f} kg")
    print(f"  Discrepancy: {abs(total_fuel - weight_loss_calculated):.2f} kg")
    
    # Use weight loss as the definitive value for total fuel to ensure consistency
    # Weight tracking is more accurate as it follows the actual trajectory
    total_fuel_consistent = weight_loss_calculated
    
    # CRITICAL: Validate fuel feasibility - check if mission requires more fuel than available
    from aircraft_config import MAX_FUEL_KG
    fuel_deficit = total_fuel_consistent - MAX_FUEL_KG
    
    if fuel_deficit > 0:
        print(f"\n{'='*80}")
        print(f"⚠️  MISSION INFEASIBILITY WARNING")
        print(f"{'='*80}")
        print(f"  Maximum fuel capacity: {MAX_FUEL_KG:.1f} kg")
        print(f"  Required fuel consumption: {total_fuel_consistent:.1f} kg")
        print(f"  Fuel deficit: {fuel_deficit:.1f} kg ({fuel_deficit/MAX_FUEL_KG*100:.1f}% over capacity)")
        print(f"\n  ❌ MISSION IS INFEASIBLE - Aircraft cannot carry sufficient fuel!")
        print(f"  Possible solutions:")
        print(f"    1. Increase MAX_FUEL_KG in aircraft_config.py to at least {total_fuel_consistent*1.05:.1f} kg")
        print(f"    2. Reduce cruise distance in mission_config.py")
        print(f"    3. Reduce payload or operating empty weight")
        print(f"    4. Use fuel optimizer (main_optimized.py) to find minimum required fuel")
        print(f"{'='*80}\n")
    else:
        fuel_margin = MAX_FUEL_KG - total_fuel_consistent
        print(f"\n✅ Fuel Feasibility Check: PASSED")
        print(f"  Maximum fuel capacity: {MAX_FUEL_KG:.1f} kg")
        print(f"  Required fuel consumption: {total_fuel_consistent:.1f} kg")
        print(f"  Fuel margin: {fuel_margin:.1f} kg ({fuel_margin/MAX_FUEL_KG*100:.1f}% reserve)")
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Phase</b>', '<b>Time</b>', '<b>Fuel (kg)</b>', '<b>Distance (km)</b>'],
                **header_style
            ),
            cells=dict(
                values=[
                    ['<b>Climb</b>', '<b>Cruise</b>', '<b>Descent</b>', '<b>TOTAL</b>'],
                    [f'{climb_time_s/60:.1f} min', f'{cruise_time_s/60:.1f} min', 
                     f'{descent_time_s/60:.1f} min', f'<b>{total_time_s/60:.1f} min</b>'],
                    [f'{climb_fuel:.1f}', f'{cruise_fuel:.1f}', 
                     f'{descent_fuel:.2f}', f'<b>{total_fuel_consistent:.1f}</b>'],
                    [f'{climb_distance_km:.1f}', f'{cruise_distance_km:.0f}', 
                     f'{descent_distance_km:.1f}', f'<b>{total_distance_km:.0f}</b>']
                ],
                fill_color=[['white', 'lightgray', 'white', 'lightblue']],
                **cell_style_no_height
            )
        ),
        row=4, col=1
    )
    
    # ========= ROW 3 COL 2: EFFICIENCY INDICATORS =========
    # Calculate efficiency metrics using consistent fuel value
    fuel_efficiency_kg_km = total_fuel_consistent / total_distance_km if total_distance_km > 0 else 0
    time_efficiency_min_km = (total_time_s/60) / total_distance_km if total_distance_km > 0 else 0
    avg_fuel_flow = total_fuel_consistent / (total_time_s/3600) if total_time_s > 0 else 0  # kg/h
    fuel_fraction = (total_fuel_consistent / initial_mass_kg) * 100
    
    # Efficiency Indicators plot removed per user request
    
    # ========= ROW 3 COL 3: KEY PARAMETERS TABLE =========
    avg_climb_rate_mpm = np.mean(np.abs(climb_result.Ps_mps)) * 60.0 if len(climb_result.Ps_mps) > 0 else 0
    avg_descent_rate_mpm = np.mean(np.abs(descent_result.descent_rate_mps)) * 60.0 if len(descent_result.descent_rate_mps) > 0 else 0
    cruise_mach = cruise_result.mach_number[-1]
    cruise_altitude_ft = cruise_result.altitude_m[-1] * 3.28084
    # Use provided simulation duration or default to mission time if not provided
    if simulation_duration_min is None:
        simulation_duration_min = total_time_s / 60.0
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Parameter</b>', '<b>Value</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    ['<b>Initial Weight</b>', '<b>Final Weight</b>', '<b>Weight Loss</b>',
                     '<b>Cruise Altitude</b>', '<b>Cruise Mach</b>', 
                     '<b>Avg Climb Rate</b>', '<b>Avg Descent Rate</b>', '<b>Simulation Duration</b>'],
                    [f'{initial_mass_kg:.0f} kg', 
                     f'{descent_result.final_weight_kg:.0f} kg',
                     f'{initial_mass_kg - descent_result.final_weight_kg:.0f} kg',
                     f'{cruise_altitude_ft:.0f} ft',
                     f'{cruise_mach:.3f}',
                     f'{avg_climb_rate_mpm:.0f} m/min',
                     f'{avg_descent_rate_mpm:.0f} m/min',
                     f'{simulation_duration_min:.1f} min']
                ],
                fill_color=[['white', 'lightgray'] * 4],
                align=['left', 'right'],
                font=dict(size=10, family=Typography.FONT_FAMILY)  # Smaller font to fit all rows
            )
        ),
        row=4, col=3
    )
    
    # ========= LAYOUT AND STYLING =========
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("Altitude (m)"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("Phase"), row=2, col=1)
    fig.update_yaxes(**get_axis_config("Time (min)"), row=2, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=2)
    fig.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"), row=2, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=3)
    # Set y-axis range to zoom in on weight changes for better visibility
    all_weights = np.concatenate([climb_weight, cruise_weight, descent_weight])
    weight_min, weight_max = np.min(all_weights), np.max(all_weights)
    weight_margin = (weight_max - weight_min) * 0.2  # Add 20% margin
    fig.update_yaxes(**get_axis_config("Weight (kg)"), 
                     range=[weight_min - weight_margin, weight_max + weight_margin], 
                     row=2, col=3)
    
    # Row 3: Aerodynamic plots
    fig.update_xaxes(**get_axis_config("Altitude (m)"), row=3, col=1)
    fig.update_yaxes(**get_axis_config("CD"), row=3, col=1)
    
    fig.update_xaxes(**get_axis_config("Altitude (m)"), row=3, col=2)
    fig.update_yaxes(**get_axis_config("CL"), row=3, col=2)
    
    fig.update_xaxes(**get_axis_config("Altitude (m)"), row=3, col=3)
    fig.update_yaxes(**get_axis_config("L/D"), row=3, col=3)
    
    # ========= ROW 5: FUEL MANAGEMENT TABLE =========
    # Calculate comprehensive fuel management metrics
    fuel_remaining = MAX_FUEL_KG - total_fuel_consistent
    fuel_used_percent = (total_fuel_consistent / MAX_FUEL_KG) * 100
    fuel_margin_percent = (fuel_remaining / MAX_FUEL_KG) * 100
    
    # Fuel consumption rates
    avg_fuel_rate_kg_hr = (total_fuel_consistent / total_time_s) * 3600 if total_time_s > 0 else 0
    avg_fuel_rate_kg_km = total_fuel_consistent / total_distance_km if total_distance_km > 0 else 0
    
    # Determine feasibility status
    if fuel_deficit > 0:
        feasibility_status = f'<span style="color:red;font-weight:bold">❌ INFEASIBLE</span>'
        status_color = 'mistyrose'
    elif fuel_margin_percent < 5:
        feasibility_status = f'<span style="color:orange;font-weight:bold">⚠️ CRITICAL</span>'
        status_color = 'lightyellow'
    elif fuel_margin_percent < 10:
        feasibility_status = f'<span style="color:orange;">⚠️ LOW</span>'
        status_color = 'lightyellow'
    else:
        feasibility_status = f'<span style="color:green;font-weight:bold">✅ FEASIBLE</span>'
        status_color = 'lightgreen'
    
    # Build fuel management table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Fuel Parameter</b>', '<b>Value</b>', '<b>Percentage</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    # Parameter names
                    ['<b>Initial Fuel Capacity</b>', 
                     '<b>Fuel Consumed</b>', 
                     '<b>Fuel Remaining</b>',
                     '<b>Average Fuel Rate (Time)</b>',
                     '<b>Average Fuel Rate (Distance)</b>'],
                    # Values with units embedded
                    [f'{MAX_FUEL_KG:.1f} kg',
                     f'{total_fuel_consistent:.1f} kg',
                     f'{fuel_remaining:.1f} kg' if fuel_remaining >= 0 else f'<span style="color:red">{fuel_remaining:.1f} kg</span>',
                     f'{avg_fuel_rate_kg_hr:.1f} kg/h',
                     f'{avg_fuel_rate_kg_km:.2f} kg/km'],
                    # Percentages
                    ['100.0%',
                     f'{fuel_used_percent:.1f}%',
                     f'{fuel_margin_percent:.1f}%' if fuel_remaining >= 0 else f'<span style="color:red">{fuel_margin_percent:.1f}%</span>',
                     '-',
                     '-']
                ],
                fill_color=[
                    ['white', 'lightgray', 'white', 'lightgray', 'white']
                ],
                align=['left', 'right', 'center'],
                font=dict(size=10, family=Typography.FONT_FAMILY)
            )
        ),
        row=5, col=1
    )
    
    # ========= ROW 6: MISSION CONFIGURATION TABLE =========
    # Create phase-specific background colors for better visual distinction
    # Climb: light blue tint (rgba(173, 216, 230, 0.3) = lightblue with transparency)
    # Cruise: light green tint (rgba(144, 238, 144, 0.3) = lightgreen with transparency)
    # Descent: light red/pink tint (rgba(255, 182, 193, 0.3) = lightpink with transparency)
    
    # Row indices: 0=Climb header, 1-10=Climb params, 11=Cruise header, 12-13=Cruise params,
    # 14=Range Opt header, 15-19=Range Opt params, 20=Cruise Climb header, 21-24=Cruise Climb params,
    # 25=Descent header, 26-30=Descent params (total 31 rows)
    
    # ========= ROW 6 COL 1: CLIMB CONFIGURATION TABLE =========
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Climb Parameter</b>', '<b>Value</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    ['<b>Target Altitude</b>', '<b>Start Altitude</b>', '<b>Start Velocity</b>', '<b>Start Lever</b>',
                     '<b>Mach Samples</b>', '<b>Altitude Steps</b>', '<b>Lever Samples</b>',
                     '<b>Target Mach</b>', '<b>Mach Tolerance</b>', '<b>Strategy Time Step</b>'],
                    [f'{TARGET_ALT_CLIMB_M:.0f} m', f'{START_ALTITUDE_CLIMB_M:.1f} m', 
                     f'{START_VELOCITY_CLIMB_MS:.1f} m/s', f'{START_LEVER_CLIMB:.2f}',
                     f'{N_MACH_SAMPLES_CLIMB}', f'{N_ALTITUDE_STEPS_CLIMB}', f'{N_LEVER_SAMPLES_CLIMB}',
                     f'{TARGET_MACH_CRUISE:.3f}', f'{TARGET_MACH_TOLERANCE_CLIMB:.3f}', f'{STRATEGY_DT_CLIMB_S:.1f} s']
                ],
                fill_color=[['white', 'white', 'lightgray', 'lightgray'] * 5],  # 10 rows: white/grey alternating rows
                align=['left', 'right'],
                font=dict(size=10, family=Typography.FONT_FAMILY)
            )
        ),
        row=6, col=1
    )
    
    # ========= ROW 6 COL 2: CRUISE CONFIGURATION TABLE =========
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Cruise Parameter</b>', '<b>Value</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    ['<b>Cruise Distance</b>', '<b>Cruise Time Step</b>'],
                    [f'{CRUISE_DISTANCE_KM:.1f} km', f'{CRUISE_TIME_STEP_S:.1f} s']
                ],
                fill_color=[['white', 'white', 'lightgray', 'lightgray']],  # 2 rows: white/grey alternating rows
                align=['left', 'right'],
                font=dict(size=10, family=Typography.FONT_FAMILY)
            )
        ),
        row=6, col=2
    )
    
    # ========= ROW 6 COL 3: DESCENT CONFIGURATION TABLE =========
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Descent Parameter</b>', '<b>Value</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    ['<b>Target Altitude</b>', '<b>Target Mach</b>',
                     '<b>Mach Samples</b>', '<b>Altitude Steps</b>', '<b>Lever Samples</b>'],
                    [f'{TARGET_DESCENT_ALT_M:.1f} m', f'{TARGET_DESCENT_MACH:.3f}',
                     f'{N_MACH_SAMPLES_DESCENT}', f'{N_ALTITUDE_STEPS_DESCENT}', f'{N_LEVER_SAMPLES_DESCENT}']
                ],
                fill_color=[['white', 'white', 'lightgray', 'lightgray'] * 2 + ['white', 'white']],  # 5 rows: white/grey alternating rows
                align=['left', 'right'],
                font=dict(size=10, family=Typography.FONT_FAMILY)
            )
        ),
        row=6, col=3
    )
    
    # ========= ROW 7: RANGE OPTIMIZATION TABLE =========
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Range Optimization Parameter</b>', '<b>Value</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    ['<b>Target Mission Range</b>', '<b>Initial Cruise Distance</b>',
                     '<b>Range Tolerance</b>', '<b>Max Iterations</b>', '<b>Damping Factor</b>'],
                    [f'{TARGET_MISSION_RANGE_KM:.1f} km', f'{INITIAL_CRUISE_DISTANCE_KM:.1f} km',
                     f'{RANGE_OPTIMIZATION_TOLERANCE_KM:.1f} km', f'{MAX_RANGE_OPTIMIZATION_ITERATIONS}', f'{RANGE_OPTIMIZATION_DAMPING_FACTOR:.2f}']
                ],
                fill_color=[['white', 'white', 'lightgray', 'lightgray'] * 2 + ['white', 'white']],  # 5 rows: white/grey alternating rows
                align=['left', 'right'],
                font=dict(size=10, family=Typography.FONT_FAMILY)
            )
        ),
        row=7, col=1
    )
    
    # ========= ROW 7 COL 2: CRUISE CLIMB TABLE =========
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Cruise Climb Parameter</b>', '<b>Value</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    ['<b>Enable Cruise Climb</b>', '<b>Trigger Distance Fraction</b>',
                     '<b>Altitude Increment</b>', '<b>Mach Tolerance</b>'],
                    ['Yes' if ENABLE_CRUISE_CLIMB else 'No',
                     f'{CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION:.2f}',
                     f'{CRUISE_CLIMB_ALTITUDE_INCREMENT_M:.0f} m', f'{CRUISE_CLIMB_MACH_TOLERANCE:.3f}']
                ],
                fill_color=[['white', 'white', 'lightgray', 'lightgray'] * 2],  # 4 rows: white/grey alternating rows
                align=['left', 'right'],
                font=dict(size=10, family=Typography.FONT_FAMILY)
            )
        ),
        row=7, col=2
    )
    
    # ========= ROW 7 COL 3: AIRCRAFT CONFIGURATION TABLE =========
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Aircraft Parameter</b>', '<b>Value</b>'],
                fill_color=Colors.CRUISE,
                align='center',
                font=dict(color='white', size=Typography.AXIS_LABEL_SIZE, family=Typography.FONT_FAMILY)
            ),
            cells=dict(
                values=[
                    # Parameter names
                    ['<b>Number of Engines</b>', '<b>Reference Area</b>',
                     '<b>Airframe Weight</b>', '<b>Propulsion Weight</b>', '<b>Systems Weight</b>',
                     '<b>Operating Empty Weight</b>',
                     '<b>Payload per Person</b>', '<b>Default Passengers</b>', '<b>Total Payload</b>',
                     '<b>Maximum Fuel</b>'],
                    # Values
                    [f'{N_ENGINES}', f'{S_REF_M2:.2f} m²',
                     f'{W_AIRFRAME_KG:.1f} kg', f'{W_PROPULSION_KG:.1f} kg', f'{W_SYSTEMS_KG:.1f} kg',
                     f'{W_OE_KG:.1f} kg',
                     f'{PAYLOAD_PER_PERSON_KG:.1f} kg', f'{DEFAULT_PASSENGERS}', f'{W_PL_KG:.1f} kg',
                     f'{MAX_FUEL_KG:.1f} kg']
                ],
                fill_color=[['white', 'white', 'lightgray', 'lightgray'] * 4 + ['white', 'white']],  # 9 rows: white/grey alternating rows
                align=['left', 'right'],
                font=dict(size=10, family=Typography.FONT_FAMILY)
            )
        ),
        row=7, col=3
    )
    
    # Main title with comprehensive summary - add warning if infeasible
    feasibility_warning = ""
    if fuel_deficit > 0:
        feasibility_warning = f"<br><span style='color:red;font-weight:bold'>⚠️ MISSION INFEASIBLE: Fuel deficit {fuel_deficit:.1f} kg ({fuel_deficit/MAX_FUEL_KG*100:.1f}% over capacity)</span>"
    
    subtitle = (
        f"Total Distance: {total_distance_km:.0f} km | "
        f"Total Time: {total_time_s/3600:.2f} hours ({total_time_s/60:.1f} min) | "
        f"Total Fuel: {total_fuel_consistent:.1f} kg ({fuel_fraction:.1f}% of initial mass) | "
        f"Fuel Efficiency: {fuel_efficiency_kg_km:.2f} kg/km"
        f"{feasibility_warning}"
    )
    
    # Increase height to accommodate additional configuration tables
    dashboard_height = Layout.DASHBOARD_HEIGHT * 2.5  # 150% increase for new tables and better visibility
    
    layout_config = get_standard_layout(
        "COMPLETE MISSION ANALYSIS SUMMARY",
        subtitle,
        height=dashboard_height,
        width=Layout.DASHBOARD_WIDTH
    )
    
    # Add extra right margin for legend positioning
    layout_config['margin'] = dict(l=80, r=150, t=120, b=80)
    
    fig.update_layout(
        **layout_config,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    
    # Save to root timestamped folder (combines all phases)
    run_dir = get_or_create_run_directory()
    output_path_html = os.path.join(run_dir, 'mission_summary_dashboard.html')
    output_path_png = os.path.join(run_dir, 'mission_summary_dashboard.png')
    
    fig.write_html(output_path_html)
    
    # Also save as PNG
    try:
        fig.write_image(output_path_png, width=1800, height=1600, scale=2)
        print(f"[EXPORT] Mission summary dashboard saved to: {output_path_html} (interactive) and {output_path_png} (PNG)")
    except Exception as e:
        print(f"[EXPORT] Mission summary dashboard saved to: {output_path_html} (HTML only)")
        print(f"[WARNING] Could not save PNG version: {e}")
    
    # Also save if custom path requested
    if save_html:
        fig.write_html(save_html)
    
    # Show in browser
    fig.show()
    
    return fig


def plot_combined_performance_analysis(climb_result: MinFuelSchedule,
                                     cruise_result: CruiseResults,
                                     descent_result: DescentResults,
                                     initial_mass_kg: float,
                                     save_html: Optional[str] = None):
    """
    Create combined performance analysis showing all three phases side-by-side.
    
    Shows 3x2 grid for each phase (Climb, Cruise, Descent) with aligned metrics:
    - Fuel Flow Rate
    - Thrust vs Drag
    - Weight Evolution
    - Lever Position
    - True Airspeed
    - Cumulative Fuel Consumption
    
    Args:
        climb_result: Results from climb phase
        cruise_result: Results from cruise phase
        descent_result: Results from descent phase
        initial_mass_kg: Initial aircraft mass
        save_html: Optional path to save HTML file
    """
    
    # Calculate aerodynamic data throughout mission
    print("[AERO] Calculating aerodynamic data for combined analysis...")
    aero_data = calculate_aerodynamic_data_throughout_mission(
        climb_result, cruise_result, descent_result, initial_mass_kg
    )
    
    # Create combined subplots: 3 phases × 6 metrics = 3 rows × 6 columns
    fig = make_subplots(
        rows=3, cols=6,
        subplot_titles=(
            # Row 1: Climb Phase
            '<b>Climb: Fuel Flow</b>', '<b>Climb: Thrust/Drag</b>', '<b>Climb: Weight</b>',
            '<b>Climb: Lever</b>', '<b>Climb: Airspeed</b>', '<b>Climb: Fuel</b>',
            # Row 2: Cruise Phase  
            '<b>Cruise: Fuel Flow</b>', '<b>Cruise: Thrust/Drag</b>', '<b>Cruise: Weight</b>',
            '<b>Cruise: Lever</b>', '<b>Cruise: Airspeed</b>', '<b>Cruise: Fuel</b>',
            # Row 3: Descent Phase
            '<b>Descent: Fuel Flow</b>', '<b>Descent: Thrust/Drag</b>', '<b>Descent: Weight</b>',
            '<b>Descent: Lever</b>', '<b>Descent: Airspeed</b>', '<b>Descent: Fuel</b>'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.02
    )
    
    # ========= CLIMB PHASE DATA =========
    climb_time_s = np.cumsum(np.nan_to_num(climb_result.dt_s, nan=0.0, posinf=0.0, neginf=0.0))
    climb_time_min = climb_time_s / 60.0
    climb_alt_m = np.asarray(climb_result.alt_m, float)
    climb_mach = np.asarray(climb_result.mach, float)
    climb_lever = np.asarray(climb_result.lever, float)
    climb_thrust_N = np.asarray(climb_result.T_total_N, float)
    climb_drag_N = np.asarray(climb_result.D_N, float)
    climb_fuel_kg = np.asarray(climb_result.cumFuel_kg, float)
    climb_weight_kg = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic weight from DP optimization
    
    # Calculate climb fuel flow rate
    climb_fuel_flow_kgh = []
    for i in range(len(climb_result.dt_s)):
        if i == 0:
            climb_fuel_flow_kgh.append(0.0)
        else:
            dt_hours = climb_result.dt_s[i] / 3600.0
            if dt_hours > 0:
                fuel_consumed = climb_fuel_kg[i] - climb_fuel_kg[i-1]
                climb_fuel_flow_kgh.append(fuel_consumed / dt_hours)
            else:
                climb_fuel_flow_kgh.append(0.0)
    climb_fuel_flow_kgh = np.array(climb_fuel_flow_kgh)
    
    # Calculate climb true airspeed
    climb_tas_ms = []
    for i in range(len(climb_mach)):
        a = a_from_altitude(float(climb_alt_m[i]))
        climb_tas_ms.append(climb_mach[i] * a)
    climb_tas_ms = np.array(climb_tas_ms)
    
    # ========= CRUISE PHASE DATA =========
    cruise_time_min = cruise_result.time_s / 60.0
    cruise_fuel_flow_kgh = cruise_result.fuel_flow_kgps * 3600
    cruise_thrust_kn = cruise_result.thrust_total_N / 1000
    cruise_drag_kn = cruise_result.drag_N / 1000
    cruise_weight_kg = cruise_result.weight_kg
    cruise_lever = cruise_result.lever_position * 100
    cruise_fuel_consumed = cruise_result.fuel_consumed_kg
    cruise_tas_ms = cruise_result.true_airspeed_mps
    
    # ========= DESCENT PHASE DATA =========
    descent_time_min = descent_result.time_s / 60.0
    descent_fuel_flow_kgh = descent_result.fuel_flow_kgps * 3600
    descent_thrust_kn = descent_result.thrust_total_N / 1000
    descent_drag_kn = descent_result.drag_N / 1000
    descent_weight_kg = descent_result.weight_kg
    descent_lever = descent_result.lever * 100
    descent_cum_fuel_kg = descent_result.cumFuel_kg
    
    # Calculate descent true airspeed
    descent_tas_ms = []
    for i in range(len(descent_result.mach)):
        a = a_from_altitude(float(descent_result.alt_m[i]))
        descent_tas_ms.append(descent_result.mach[i] * a)
    descent_tas_ms = np.array(descent_tas_ms)
    
    # ========= ADD TRACES FOR ALL PHASES =========
    
    # CLIMB PHASE (Row 1)
    # 1. Fuel Flow Rate
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_fuel_flow_kgh,
            mode='lines',
            name='Climb Fuel Flow',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Thrust vs Drag
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_thrust_N / 1000,
            mode='lines',
            name='Climb Thrust',
            line=dict(color='darkblue', width=LineStyles.THICK),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_drag_N / 1000,
            mode='lines',
            name='Climb Drag',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Weight
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_weight_kg,
            mode='lines',
            name='Climb Weight',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            showlegend=False
        ),
        row=1, col=3
    )
    
    # 4. Lever Position
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_lever * 100,
            mode='lines',
            name='Climb Lever',
            line=dict(color='steelblue', width=LineStyles.THICK),
            showlegend=False
        ),
        row=1, col=4
    )
    
    # 5. True Airspeed
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_tas_ms,
            mode='lines',
            name='Climb Airspeed',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            showlegend=False
        ),
        row=1, col=5
    )
    
    # 6. Cumulative Fuel
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_fuel_kg,
            mode='lines',
            name='Climb Fuel',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            showlegend=False
        ),
        row=1, col=6
    )
    
    # CRUISE PHASE (Row 2)
    # 1. Fuel Flow Rate
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_fuel_flow_kgh,
            mode='lines',
            name='Cruise Fuel Flow',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.15)',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 2. Thrust vs Drag
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_thrust_kn,
            mode='lines',
            name='Cruise Thrust',
            line=dict(color='darkgreen', width=LineStyles.THICK),
            showlegend=False
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_drag_kn,
            mode='lines',
            name='Cruise Drag',
            line=dict(color='lightcoral', width=LineStyles.THICK, dash=LineStyles.DASH),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 3. Weight
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_weight_kg,
            mode='lines',
            name='Cruise Weight',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.15)',
            showlegend=False
        ),
        row=2, col=3
    )
    
    # 4. Lever Position
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_lever,
            mode='lines',
            name='Cruise Lever',
            line=dict(color='olive', width=LineStyles.THICK),
            showlegend=False
        ),
        row=2, col=4
    )
    
    # 5. True Airspeed
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_tas_ms,
            mode='lines',
            name='Cruise Airspeed',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            showlegend=False
        ),
        row=2, col=5
    )
    
    # 6. Cumulative Fuel
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_fuel_consumed,
            mode='lines',
            name='Cruise Fuel',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.15)',
            showlegend=False
        ),
        row=2, col=6
    )
    
    # DESCENT PHASE (Row 3)
    # 1. Fuel Flow Rate
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_fuel_flow_kgh,
            mode='lines',
            name='Descent Fuel Flow',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 2. Thrust vs Drag
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_thrust_kn,
            mode='lines',
            name='Descent Thrust',
            line=dict(color='darkred', width=LineStyles.THICK),
            showlegend=False
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_drag_kn,
            mode='lines',
            name='Descent Drag',
            line=dict(color='salmon', width=LineStyles.THICK, dash=LineStyles.DASH),
            showlegend=False
        ),
        row=3, col=2
    )
    
    # 3. Weight
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_weight_kg,
            mode='lines',
            name='Descent Weight',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            showlegend=False
        ),
        row=3, col=3
    )
    
    # 4. Lever Position
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_lever,
            mode='lines',
            name='Descent Lever',
            line=dict(color='firebrick', width=LineStyles.THICK),
            showlegend=False
        ),
        row=3, col=4
    )
    
    # 5. True Airspeed
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_tas_ms,
            mode='lines',
            name='Descent Airspeed',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            showlegend=False
        ),
        row=3, col=5
    )
    
    # 6. Cumulative Fuel
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_cum_fuel_kg,
            mode='lines',
            name='Descent Fuel',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            showlegend=False
        ),
        row=3, col=6
    )
    
    # Calculate mission totals
    # Use weight tracking for consistent fuel calculation
    total_fuel_sum = climb_fuel_kg[-1] + cruise_fuel_consumed[-1] + descent_cum_fuel_kg[-1]
    total_fuel_weight = initial_mass_kg - descent_result.final_weight_kg
    
    # Diagnostic check for consistency
    if abs(total_fuel_sum - total_fuel_weight) > 0.1:
        print(f"[COMBINED ANALYSIS] Warning: Fuel discrepancy detected: {abs(total_fuel_sum - total_fuel_weight):.2f} kg")
        print(f"  Fuel sum: {total_fuel_sum:.2f} kg, Weight loss: {total_fuel_weight:.2f} kg")
    
    # Use weight-based calculation for consistency with dashboard
    total_fuel = total_fuel_weight
    total_time_min = climb_time_min[-1] + cruise_time_min[-1] + descent_time_min[-1]
    
    # ========= AERODYNAMIC SUMMARY TABLE =========
    # Get table styling
    header_style = get_table_header_style()
    cell_style = get_table_cell_style()
    
    # Calculate aerodynamic statistics
    aero_stats = []
    
    # Climb statistics
    if len(aero_data['climb']['cd']) > 0:
        climb_cd_avg = np.mean(aero_data['climb']['cd'])
        climb_cl_avg = np.mean(aero_data['climb']['cl'])
        climb_ld_avg = np.mean(aero_data['climb']['ld'])
        climb_drag_avg = np.mean(aero_data['climb']['drag'])
        aero_stats.append(['Climb', f'{climb_cd_avg:.4f}', f'{climb_cl_avg:.4f}', f'{climb_ld_avg:.2f}', f'{climb_drag_avg:.0f}'])
    
    # Cruise statistics
    if len(aero_data['cruise']['cd']) > 0:
        cruise_cd_avg = np.mean(aero_data['cruise']['cd'])
        cruise_cl_avg = np.mean(aero_data['cruise']['cl'])
        cruise_ld_avg = np.mean(aero_data['cruise']['ld'])
        cruise_drag_avg = np.mean(aero_data['cruise']['drag'])
        aero_stats.append(['Cruise', f'{cruise_cd_avg:.4f}', f'{cruise_cl_avg:.4f}', f'{cruise_ld_avg:.2f}', f'{cruise_drag_avg:.0f}'])
    
    # Descent statistics
    if len(aero_data['descent']['cd']) > 0:
        descent_cd_avg = np.mean(aero_data['descent']['cd'])
        descent_cl_avg = np.mean(aero_data['descent']['cl'])
        descent_ld_avg = np.mean(aero_data['descent']['ld'])
        descent_drag_avg = np.mean(aero_data['descent']['drag'])
        aero_stats.append(['Descent', f'{descent_cd_avg:.4f}', f'{descent_cl_avg:.4f}', f'{descent_ld_avg:.2f}', f'{descent_drag_avg:.0f}'])
    
    # Aerodynamic summary table removed - layout has only 3 rows
    # The aerodynamic data is already visualized in the mission summary dashboard
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>COMBINED MISSION PERFORMANCE ANALYSIS</b><br>" +
                 f"<sup>Climb (Blue) → Cruise (Green) → Descent (Red) | " +
                 f"Total Fuel: {total_fuel:.1f} kg | Total Time: {total_time_min:.1f} min</sup>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=1000,
        width=1800,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=11),
        showlegend=False,
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    # Update axes labels
    for row in range(1, 4):
        for col in range(1, 7):
            fig.update_xaxes(title_text="Time (min)", row=row, col=col, gridcolor='lightgray')
            
            if col == 1:  # Fuel Flow
                fig.update_yaxes(title_text="Fuel Flow (kg/h)", row=row, col=col, gridcolor='lightgray')
            elif col == 2:  # Thrust/Drag
                fig.update_yaxes(title_text="Force (kN)", row=row, col=col, gridcolor='lightgray')
            elif col == 3:  # Weight
                # Set y-axis range to zoom in on weight changes for better visibility
                if row == 1:  # Climb
                    weight_min, weight_max = np.min(climb_weight_kg), np.max(climb_weight_kg)
                    weight_margin = (weight_max - weight_min) * 0.2
                    fig.update_yaxes(title_text="Weight (kg)", row=row, col=col, gridcolor='lightgray',
                                   range=[weight_min - weight_margin, weight_max + weight_margin])
                elif row == 2:  # Cruise
                    weight_min, weight_max = np.min(cruise_weight_kg), np.max(cruise_weight_kg)
                    weight_margin = (weight_max - weight_min) * 0.2
                    fig.update_yaxes(title_text="Weight (kg)", row=row, col=col, gridcolor='lightgray',
                                   range=[weight_min - weight_margin, weight_max + weight_margin])
                elif row == 3:  # Descent
                    weight_min, weight_max = np.min(descent_weight_kg), np.max(descent_weight_kg)
                    weight_margin = (weight_max - weight_min) * 0.2
                    fig.update_yaxes(title_text="Weight (kg)", row=row, col=col, gridcolor='lightgray',
                                   range=[weight_min - weight_margin, weight_max + weight_margin])
                else:
                    fig.update_yaxes(title_text="Weight (kg)", row=row, col=col, gridcolor='lightgray')
            elif col == 4:  # Lever
                fig.update_yaxes(title_text="Lever (%)", row=row, col=col, gridcolor='lightgray')
            elif col == 5:  # Airspeed
                fig.update_yaxes(title_text="TAS (m/s)", row=row, col=col, gridcolor='lightgray')
            elif col == 6:  # Fuel
                fig.update_yaxes(title_text="Fuel (kg)", row=row, col=col, gridcolor='lightgray')
    
    # Add phase labels on the left
    fig.add_annotation(
        x=-0.08, y=0.83, xref='paper', yref='paper',
        text="<b>CLIMB</b>", showarrow=False, font=dict(size=16, color=Colors.CLIMB)
    )
    fig.add_annotation(
        x=-0.08, y=0.5, xref='paper', yref='paper',
        text="<b>CRUISE</b>", showarrow=False, font=dict(size=16, color=Colors.CRUISE)
    )
    fig.add_annotation(
        x=-0.08, y=0.17, xref='paper', yref='paper',
        text="<b>DESCENT</b>", showarrow=False, font=dict(size=16, color=Colors.DESCENT)
    )
    
    # Save to root timestamped folder (combines all phases)
    run_dir = get_or_create_run_directory()
    output_path_html = os.path.join(run_dir, 'combined_performance_analysis.html')
    output_path_png = os.path.join(run_dir, 'combined_performance_analysis.png')
    
    fig.write_html(output_path_html)
    
    # Also save as PNG
    try:
        fig.write_image(output_path_png, width=2400, height=1400, scale=2)
        print(f"[EXPORT] Combined performance analysis saved to: {output_path_html} (interactive) and {output_path_png} (PNG)")
    except Exception as e:
        print(f"[EXPORT] Combined performance analysis saved to: {output_path_html} (HTML only)")
        print(f"[WARNING] Could not save PNG version: {e}")
    
    # Also save if custom path requested
    if save_html:
        fig.write_html(save_html)
    
    # Show the plot
    fig.show()
    
    return fig