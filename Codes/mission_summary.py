# ========================================================================
# COMPLETE MISSION SUMMARY VISUALIZATION MODULE
# ========================================================================
"""
Comprehensive mission analysis dashboard combining all flight phases.

Dashboard components:
    1. Mission profile: h(t) for climb, cruise, descent
    2. Phase breakdown: Δt, Δm_fuel by phase
    3. Performance evolution: m_fuel(t), m(t), ṁ(t)
    4. Aerodynamic data: CD(h), CL(h), L/D(h) throughout mission
    5. Configuration tables: Mission parameters, aircraft specifications
    6. Fuel management: m_fuel,capacity, m_fuel,consumed, m_fuel,remaining
    7. Feasibility check: m_fuel,required ≤ m_fuel,capacity

Combined analysis:
    3×6 grid showing all phases side-by-side:
    - Fuel flow, thrust/drag, mass, lever, airspeed, cumulative fuel
    
Mathematical context:
    Distance: s = ∫V dt ≈ Σ V_TAS,i·Δt_i where V_TAS = M·a(h)
    Fuel: Δm_total = m_0 - m_f
    Feasibility: Δm_total ≤ m_fuel,capacity
    Efficiency: η = Δm_total/s [kg/km]

All plots use Plotly interactive graphics with HTML/PNG export.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"

# Aircraft parameters and atmospheric model
from aircraft_config import (
    a_from_altitude, N_ENGINES, S_REF_M2,
    W_AIRFRAME_KG, W_PROPULSION_KG, W_SYSTEMS_KG,
    PAYLOAD_PER_PERSON_KG, DEFAULT_PASSENGERS, W_FUEL_KG,
    W_OE_KG, W_PL_KG
)

# Mission configuration parameters
from mission_config import (
    TARGET_ALT_CLIMB_M,
    START_ALTITUDE_CLIMB_M, START_VELOCITY_CLIMB_MS, START_LEVER_CLIMB,
    N_MACH_SAMPLES_CLIMB, N_ALTITUDE_STEPS_CLIMB, N_LEVER_SAMPLES_CLIMB,
    TARGET_MACH_CRUISE, TARGET_MACH_TOLERANCE, STRATEGY_DT_CLIMB_S,
    CRUISE_DISTANCE_KM, CRUISE_TIME_STEP_S,
    TARGET_MISSION_RANGE_KM, INITIAL_CRUISE_DISTANCE_KM,
    RANGE_OPTIMIZATION_TOLERANCE_KM, MAX_RANGE_OPTIMIZATION_ITERATIONS,
    RANGE_OPTIMIZATION_DAMPING_FACTOR,
    ENABLE_CRUISE_CLIMB, CRUISE_CLIMB_TRIGGER_DISTANCE_FRACTION,
    CRUISE_CLIMB_ALTITUDE_INCREMENT_M,
    TARGET_DESCENT_ALT_M, TARGET_DESCENT_MACH,
    N_MACH_SAMPLES_DESCENT, N_ALTITUDE_STEPS_DESCENT, N_LEVER_SAMPLES_DESCENT
)

# Mission phase data structures
from climb import MinFuelSchedule
from cruise import CruiseResults
from descent import DescentResults, calculate_min_descent_mach

# Aerodynamics model
from pyaerodynamics_wrapper import PyAerodynamicsWrapper

# Visualization styling
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_axis_config,
    get_table_header_style, get_table_cell_style, HoverTemplates,
    get_or_create_run_directory
)


# ========================================================================
# SECTION 1: AERODYNAMIC DATA COMPUTATION
# ========================================================================

def calculate_aerodynamic_data_throughout_mission(climb_result: MinFuelSchedule,
                                                cruise_result: CruiseResults,
                                                descent_result: DescentResults,
                                                initial_mass_kg: float):
    """
    Compute aerodynamic coefficients throughout complete mission.
    
    Computed quantities:
        - CD(M, h, m): Drag coefficient
        - CL(M, h, m): Lift coefficient
        - L/D: Aerodynamic efficiency
        - D [N]: Drag force
    
    Sampling strategy: Subsample trajectory (max 20 points/phase) for efficiency.
    
    Parameters:
        climb_result: MinFuelSchedule - climb trajectory
        cruise_result: CruiseResults - cruise trajectory
        descent_result: DescentResults - descent trajectory
        initial_mass_kg: m_0 [kg] - initial mass
        
    Returns:
        dict: {'climb', 'cruise', 'descent'} → {h, M, m, CD, CL, L/D, D}
    """
    aero = PyAerodynamicsWrapper()
    
    # ────────────────────────────────────────────────────────────────────
    # Data Structure Initialization
    # ────────────────────────────────────────────────────────────────────
    aero_data = {
        'climb': {'altitude': [], 'mach': [], 'mass': [], 'cd': [], 'cl': [], 'ld': [], 'drag': []},
        'cruise': {'altitude': [], 'mach': [], 'mass': [], 'cd': [], 'cl': [], 'ld': [], 'drag': []},
        'descent': {'altitude': [], 'mach': [], 'mass': [], 'cd': [], 'cl': [], 'ld': [], 'drag': []}
    }
    
    # ────────────────────────────────────────────────────────────────────
    # Climb Phase Aerodynamics: CD(M,h,m), CL(M,h,m), L/D
    # ────────────────────────────────────────────────────────────────────
    if len(climb_result.alt_m) > 0:
        sample_rate = max(1, len(climb_result.alt_m) // 20)  # Max 20 samples
        print(f"[AERO] Processing {len(climb_result.alt_m)} climb points (sampling every {sample_rate}th point)...")
        for i in range(0, len(climb_result.alt_m), sample_rate):
            if i % (sample_rate * 5) == 0:
                print(f"[AERO] Climb progress: {i//sample_rate + 1}/{(len(climb_result.alt_m) + sample_rate - 1)//sample_rate}")
            
            alt = climb_result.alt_m[i]
            mach = climb_result.mach[i]
            mass = climb_result.mass_kg[i] if i < len(climb_result.mass_kg) else initial_mass_kg
            
            # Get comprehensive aerodynamic data
            aero_comp = aero.get_comprehensive_aerodynamics(mach, alt, mass)
            
            aero_data['climb']['altitude'].append(alt)
            aero_data['climb']['mach'].append(mach)
            aero_data['climb']['mass'].append(mass)
            aero_data['climb']['cd'].append(aero_comp['cd'])
            aero_data['climb']['cl'].append(aero_comp['cl'])
            aero_data['climb']['ld'].append(aero_comp['ld'])
            aero_data['climb']['drag'].append(aero_comp['drag_force_N'])
    
    # Calculate cruise aerodynamic data using actual trajectory mass values
    if hasattr(cruise_result, 'altitude_m') and hasattr(cruise_result, 'mach') and hasattr(cruise_result, 'mass_kg'):
        # Use actual arrays from cruise trajectory
        cruise_alt_array = cruise_result.altitude_m if isinstance(cruise_result.altitude_m, np.ndarray) else np.array([cruise_result.altitude_m])
        cruise_mach_array = cruise_result.mach_number if hasattr(cruise_result, 'mach_number') else (cruise_result.mach if isinstance(cruise_result.mach, np.ndarray) else np.array([cruise_result.mach]))
        cruise_mass_array = cruise_result.mass_kg if isinstance(cruise_result.mass_kg, np.ndarray) else np.array([cruise_result.mass_kg])
        
        # Sample at key points: start, middle, and end using actual trajectory data
        n_points = len(cruise_mass_array)
        if n_points > 0:
            sample_indices = [0, n_points // 2, -1] if n_points > 2 else [0, -1] if n_points > 1 else [0]
            
            for idx in sample_indices:
                cruise_alt = cruise_alt_array[idx] if len(cruise_alt_array) > 1 else cruise_alt_array[0]
                cruise_mach = cruise_mach_array[idx] if len(cruise_mach_array) > 1 else cruise_mach_array[0]
                cruise_mass = cruise_mass_array[idx]
                
                aero_comp = aero.get_comprehensive_aerodynamics(cruise_mach, cruise_alt, cruise_mass)
                
                aero_data['cruise']['altitude'].append(cruise_alt)
                aero_data['cruise']['mach'].append(cruise_mach)
                aero_data['cruise']['mass'].append(cruise_mass)
                aero_data['cruise']['cd'].append(aero_comp['cd'])
                aero_data['cruise']['cl'].append(aero_comp['cl'])
                aero_data['cruise']['ld'].append(aero_comp['ld'])
                aero_data['cruise']['drag'].append(aero_comp['drag_force_N'])
    
    # Calculate descent aerodynamic data (sample every nth point for efficiency)
    if len(descent_result.alt_m) > 0:
        sample_rate = max(1, len(descent_result.alt_m) // 20)  # Sample max 20 points
        print(f"[AERO] Processing {len(descent_result.alt_m)} descent points (sampling every {sample_rate}th point)...")
        for i in range(0, len(descent_result.alt_m), sample_rate):
            if i % (sample_rate * 5) == 0:  # Progress indicator every 5 samples
                print(f"[AERO] Descent progress: {i//sample_rate + 1}/{(len(descent_result.alt_m) + sample_rate - 1)//sample_rate}")
            
            alt = descent_result.alt_m[i]
            mach = descent_result.mach[i]
            mass = descent_result.mass_kg[i] if i < len(descent_result.mass_kg) else initial_mass_kg * 0.85
            
            # Get comprehensive aerodynamic data using correct mass variable
            aero_comp = aero.get_comprehensive_aerodynamics(mach, alt, mass)
            
            aero_data['descent']['altitude'].append(alt)
            aero_data['descent']['mach'].append(mach)
            aero_data['descent']['mass'].append(mass)
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
            '<b>Mass Evolution</b>',
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
    
    # ========= ROW 2 COL 3: MASS EVOLUTION =========
    climb_mass = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic mass from DP optimization
    cruise_mass = cruise_result.mass_kg
    descent_mass = descent_result.mass_kg
    
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_mass,
            mode='lines',
            line=dict(color=Colors.CLIMB, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate='<b>Climb</b><br>Mass: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_mass,
            mode='lines',
            line=dict(color=Colors.CRUISE, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate='<b>Cruise</b><br>Mass: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_mass,
            mode='lines',
            line=dict(color=Colors.DESCENT, width=LineStyles.MEDIUM),
            showlegend=False,
            hovertemplate='<b>Descent</b><br>Mass: %{y:.0f} kg<extra></extra>'
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
    
    # Verify consistency between mass tracking and fuel consumption
    mass_loss_calculated = initial_mass_kg - descent_result.final_mass_kg
    
    # Diagnostic print to identify discrepancy
    print(f"\n[MISSION SUMMARY] Fuel and Mass Consistency Check:")
    print(f"  Climb fuel: {climb_fuel:.2f} kg")
    print(f"  Cruise fuel: {cruise_fuel:.2f} kg")
    print(f"  Descent fuel: {descent_fuel:.2f} kg")
    print(f"  Total fuel (sum): {total_fuel:.2f} kg")
    print(f"  Initial mass: {initial_mass_kg:.2f} kg")
    print(f"  Final mass: {descent_result.final_mass_kg:.2f} kg")
    print(f"  Mass loss (fuel burned): {mass_loss_calculated:.2f} kg")
    print(f"  Discrepancy: {abs(total_fuel - mass_loss_calculated):.2f} kg")
    
    # Use mass loss as the definitive value for total fuel to ensure consistency
    # Mass tracking is more accurate as it follows the actual trajectory
    total_fuel_consistent = mass_loss_calculated
    
    # CRITICAL: Validate fuel feasibility - check if mission requires more fuel than available
    from aircraft_config import W_FUEL_KG
    fuel_deficit = total_fuel_consistent - W_FUEL_KG
    
    if fuel_deficit > 0:
        print(f"\n{'='*80}")
        print(f"  MISSION INFEASIBILITY WARNING")
        print(f"{'='*80}")
        print(f"  Maximum fuel capacity: {W_FUEL_KG:.1f} kg")
        print(f"  Required fuel consumption: {total_fuel_consistent:.1f} kg")
        print(f"  Fuel deficit: {fuel_deficit:.1f} kg ({fuel_deficit/W_FUEL_KG*100:.1f}% over capacity)")
        print(f"\n   MISSION IS INFEASIBLE - Aircraft cannot carry sufficient fuel!")
        print(f"  Possible solutions:")
        print(f"    1. Increase W_FUEL_KG in aircraft_config.py to at least {total_fuel_consistent*1.05:.1f} kg")
        print(f"    2. Reduce cruise distance in mission_config.py")
        print(f"    3. Reduce payload or operating empty mass")
        print(f"    4. Use fuel optimizer (main_optimized.py) to find minimum required fuel")
        print(f"{'='*80}\n")
    else:
        fuel_margin = W_FUEL_KG - total_fuel_consistent
        print(f"\n Fuel Feasibility Check: PASSED")
        print(f"  Maximum fuel capacity: {W_FUEL_KG:.1f} kg")
        print(f"  Required fuel consumption: {total_fuel_consistent:.1f} kg")
        print(f"  Fuel margin: {fuel_margin:.1f} kg ({fuel_margin/W_FUEL_KG*100:.1f}% reserve)")
    
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
    fuel_fraction = (total_fuel_consistent / initial_mass_kg) * 100
    
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
                    ['<b>Initial Mass</b>', '<b>Final Mass</b>', '<b>Mass Loss (Fuel)</b>',
                     '<b>Cruise Altitude</b>', '<b>Cruise Mach</b>', 
                     '<b>Avg Climb Rate</b>', '<b>Avg Descent Rate</b>', '<b>Simulation Duration</b>'],
                    [f'{initial_mass_kg:.0f} kg', 
                     f'{descent_result.final_mass_kg:.0f} kg',
                     f'{initial_mass_kg - descent_result.final_mass_kg:.0f} kg',
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
    # Set y-axis range to zoom in on mass changes for better visibility
    all_masses = np.concatenate([climb_mass, cruise_mass, descent_mass])
    mass_min, mass_max = np.min(all_masses), np.max(all_masses)
    mass_margin = (mass_max - mass_min) * 0.2  # Add 20% margin
    fig.update_yaxes(**get_axis_config("Mass (kg)"),
                     range=[mass_min - mass_margin, mass_max + mass_margin], 
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
    fuel_remaining = W_FUEL_KG - total_fuel_consistent
    fuel_used_percent = (total_fuel_consistent / W_FUEL_KG) * 100
    fuel_margin_percent = (fuel_remaining / W_FUEL_KG) * 100
    
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
                    [f'{W_FUEL_KG:.1f} kg',
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
                    f'{TARGET_MACH_CRUISE:.3f}', f'{TARGET_MACH_TOLERANCE:.3f}', f'{STRATEGY_DT_CLIMB_S:.1f} s']
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
                     f'{CRUISE_CLIMB_ALTITUDE_INCREMENT_M:.0f} m', f'{TARGET_MACH_TOLERANCE:.3f}']
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
                     '<b>Airframe Mass</b>', '<b>Propulsion Mass</b>', '<b>Systems Mass</b>',
                     '<b>Operating Empty Mass</b>',
                     '<b>Payload per Person</b>', '<b>Default Passengers</b>', '<b>Total Payload</b>',
                     '<b>Maximum Fuel</b>'],
                    # Values
                    [f'{N_ENGINES}', f'{S_REF_M2:.2f} m²',
                     f'{W_AIRFRAME_KG:.1f} kg', f'{W_PROPULSION_KG:.1f} kg', f'{W_SYSTEMS_KG:.1f} kg',
                     f'{W_OE_KG:.1f} kg',
                     f'{PAYLOAD_PER_PERSON_KG:.1f} kg', f'{DEFAULT_PASSENGERS}', f'{W_PL_KG:.1f} kg',
                     f'{W_FUEL_KG:.1f} kg']
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
        feasibility_warning = f"<br><span style='color:red;font-weight:bold'>⚠️ MISSION INFEASIBLE: Fuel deficit {fuel_deficit:.1f} kg ({fuel_deficit/W_FUEL_KG*100:.1f}% over capacity)</span>"
    
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
    
    # Save to Mission Summary subfolder
    run_dir = get_or_create_run_directory(phase="MissionSummary")
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
    
    # ════════════════════════════════════════════════════════════════
    # Save individual plots as separate PNG files
    # ════════════════════════════════════════════════════════════════
    try:
        save_prefix = "mission_summary"
        
        # 1. Mission Profile (Altitude vs Time - all phases)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=climb_time_min, y=climb_result.alt_m, mode='lines', name='Climb',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.2)',
            hovertemplate=HoverTemplates.altitude('Climb', 'Time', 'min')))
        fig1.add_trace(go.Scatter(x=cruise_time_min, y=cruise_result.altitude_m, mode='lines', name='Cruise',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.2)',
            hovertemplate=HoverTemplates.altitude('Cruise', 'Time', 'min')))
        fig1.add_trace(go.Scatter(x=descent_time_min, y=descent_result.alt_m, mode='lines', name='Descent',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.2)',
            hovertemplate=HoverTemplates.altitude('Descent', 'Time', 'min')))
        fig1.add_vline(x=climb_time_min[-1], line_dash=LineStyles.DASH, line_color=Colors.TRANSITION_LINE, 
                      line_width=LineStyles.MEDIUM, annotation_text="Climb→Cruise", annotation_position="top")
        fig1.add_vline(x=cruise_time_min[-1], line_dash=LineStyles.DASH, line_color=Colors.TRANSITION_LINE,
                      line_width=LineStyles.MEDIUM, annotation_text="Cruise→Descent", annotation_position="top")
        fig1.update_layout(**get_standard_layout("MISSION SUMMARY - Mission Profile", subtitle, height=600, width=1200))
        fig1.update_xaxes(**get_axis_config("Time (min)")); fig1.update_yaxes(**get_axis_config("Altitude (m)"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_mission_profile.png'), width=1400, height=800, scale=2)
        
        # 2. Phase Breakdown (Bar Chart)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=phases, y=phase_times_min, name='Time (min)', marker_color=phase_colors,
            text=[f'{t:.1f} min<br>{f:.1f} kg' for t, f in zip(phase_times_min, phase_fuels)],
            textposition='auto', hovertemplate='<b>%{x}</b><br>Time: %{y:.1f} min<br><extra></extra>'))
        fig2.update_layout(**get_standard_layout("MISSION SUMMARY - Phase Breakdown", subtitle, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("Phase")); fig2.update_yaxes(**get_axis_config("Time (min)"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_phase_breakdown.png'), width=1200, height=800, scale=2)
        
        # 3. Fuel Consumption (all phases)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=climb_time_min, y=climb_fuel_array, mode='lines', name='Climb',
            line=dict(color=Colors.CLIMB, width=LineStyles.MEDIUM), hovertemplate=HoverTemplates.fuel('Climb', 'Time', 'min')))
        fig3.add_trace(go.Scatter(x=cruise_time_min, y=cruise_fuel_array, mode='lines', name='Cruise',
            line=dict(color=Colors.CRUISE, width=LineStyles.MEDIUM), hovertemplate=HoverTemplates.fuel('Cruise', 'Time', 'min')))
        fig3.add_trace(go.Scatter(x=descent_time_min, y=descent_fuel_array, mode='lines', name='Descent',
            line=dict(color=Colors.DESCENT, width=LineStyles.MEDIUM), hovertemplate=HoverTemplates.fuel('Descent', 'Time', 'min')))
        fig3.update_layout(**get_standard_layout("MISSION SUMMARY - Fuel Consumption", subtitle, height=600, width=1200))
        fig3.update_xaxes(**get_axis_config("Time (min)")); fig3.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"))
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_consumption.png'), width=1400, height=800, scale=2)
        
        # 4. Mass Evolution (all phases)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=climb_time_min, y=climb_mass, mode='lines', name='Climb',
            line=dict(color=Colors.CLIMB, width=LineStyles.MEDIUM), hovertemplate='<b>Climb</b><br>Mass: %{y:.0f} kg<extra></extra>'))
        fig4.add_trace(go.Scatter(x=cruise_time_min, y=cruise_mass, mode='lines', name='Cruise',
            line=dict(color=Colors.CRUISE, width=LineStyles.MEDIUM), hovertemplate='<b>Cruise</b><br>Mass: %{y:.0f} kg<extra></extra>'))
        fig4.add_trace(go.Scatter(x=descent_time_min, y=descent_mass, mode='lines', name='Descent',
            line=dict(color=Colors.DESCENT, width=LineStyles.MEDIUM), hovertemplate='<b>Descent</b><br>Mass: %{y:.0f} kg<extra></extra>'))
        fig4.update_layout(**get_standard_layout("MISSION SUMMARY - Mass Evolution", subtitle, height=600, width=1200))
        # Set y-axis range to zoom in on mass changes
        fig4.update_xaxes(**get_axis_config("Time (min)"))
        fig4.update_yaxes(**get_axis_config("Mass (kg)"), range=[mass_min - mass_margin, mass_max + mass_margin])
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_mass_evolution.png'), width=1400, height=800, scale=2)
        
        # 5. Drag Coefficient (CD) vs Altitude
        fig5 = go.Figure()
        if len(aero_data['climb']['altitude']) > 0:
            fig5.add_trace(go.Scatter(x=aero_data['climb']['altitude'], y=aero_data['climb']['cd'],
                mode='lines+markers', name='Climb CD', line=dict(color=Colors.CLIMB, width=2), marker=dict(size=4),
                hovertemplate='<b>Climb</b><br>Altitude: %{x:.0f} m<br>CD: %{y:.4f}<extra></extra>'))
        if len(aero_data['cruise']['altitude']) > 0:
            fig5.add_trace(go.Scatter(x=aero_data['cruise']['altitude'], y=aero_data['cruise']['cd'],
                mode='lines+markers', name='Cruise CD', line=dict(color=Colors.CRUISE, width=2), marker=dict(size=4),
                hovertemplate='<b>Cruise</b><br>Altitude: %{x:.0f} m<br>CD: %{y:.4f}<extra></extra>'))
        if len(aero_data['descent']['altitude']) > 0:
            fig5.add_trace(go.Scatter(x=aero_data['descent']['altitude'], y=aero_data['descent']['cd'],
                mode='lines+markers', name='Descent CD', line=dict(color=Colors.DESCENT, width=2), marker=dict(size=4),
                hovertemplate='<b>Descent</b><br>Altitude: %{x:.0f} m<br>CD: %{y:.4f}<extra></extra>'))
        fig5.update_layout(**get_standard_layout("MISSION SUMMARY - Drag Coefficient", subtitle, height=600, width=1200))
        fig5.update_xaxes(**get_axis_config("Altitude (m)")); fig5.update_yaxes(**get_axis_config("CD"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_drag_coefficient.png'), width=1400, height=800, scale=2)
        
        # 6. Lift Coefficient (CL) vs Altitude
        fig6 = go.Figure()
        if len(aero_data['climb']['altitude']) > 0:
            fig6.add_trace(go.Scatter(x=aero_data['climb']['altitude'], y=aero_data['climb']['cl'],
                mode='lines+markers', name='Climb CL', line=dict(color=Colors.CLIMB, width=2), marker=dict(size=4),
                hovertemplate='<b>Climb</b><br>Altitude: %{x:.0f} m<br>CL: %{y:.4f}<extra></extra>'))
        if len(aero_data['cruise']['altitude']) > 0:
            fig6.add_trace(go.Scatter(x=aero_data['cruise']['altitude'], y=aero_data['cruise']['cl'],
                mode='lines+markers', name='Cruise CL', line=dict(color=Colors.CRUISE, width=2), marker=dict(size=4),
                hovertemplate='<b>Cruise</b><br>Altitude: %{x:.0f} m<br>CL: %{y:.4f}<extra></extra>'))
        if len(aero_data['descent']['altitude']) > 0:
            fig6.add_trace(go.Scatter(x=aero_data['descent']['altitude'], y=aero_data['descent']['cl'],
                mode='lines+markers', name='Descent CL', line=dict(color=Colors.DESCENT, width=2), marker=dict(size=4),
                hovertemplate='<b>Descent</b><br>Altitude: %{x:.0f} m<br>CL: %{y:.4f}<extra></extra>'))
        fig6.update_layout(**get_standard_layout("MISSION SUMMARY - Lift Coefficient", subtitle, height=600, width=1200))
        fig6.update_xaxes(**get_axis_config("Altitude (m)")); fig6.update_yaxes(**get_axis_config("CL"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_lift_coefficient.png'), width=1400, height=800, scale=2)
        
        # 7. Lift-to-Drag Ratio (L/D) vs Altitude
        fig7 = go.Figure()
        if len(aero_data['climb']['altitude']) > 0:
            fig7.add_trace(go.Scatter(x=aero_data['climb']['altitude'], y=aero_data['climb']['ld'],
                mode='lines+markers', name='Climb L/D', line=dict(color=Colors.CLIMB, width=2), marker=dict(size=4),
                hovertemplate='<b>Climb</b><br>Altitude: %{x:.0f} m<br>L/D: %{y:.2f}<extra></extra>'))
        if len(aero_data['cruise']['altitude']) > 0:
            fig7.add_trace(go.Scatter(x=aero_data['cruise']['altitude'], y=aero_data['cruise']['ld'],
                mode='lines+markers', name='Cruise L/D', line=dict(color=Colors.CRUISE, width=2), marker=dict(size=4),
                hovertemplate='<b>Cruise</b><br>Altitude: %{x:.0f} m<br>L/D: %{y:.2f}<extra></extra>'))
        if len(aero_data['descent']['altitude']) > 0:
            fig7.add_trace(go.Scatter(x=aero_data['descent']['altitude'], y=aero_data['descent']['ld'],
                mode='lines+markers', name='Descent L/D', line=dict(color=Colors.DESCENT, width=2), marker=dict(size=4),
                hovertemplate='<b>Descent</b><br>Altitude: %{x:.0f} m<br>L/D: %{y:.2f}<extra></extra>'))
        fig7.update_layout(**get_standard_layout("MISSION SUMMARY - Lift-to-Drag Ratio", subtitle, height=600, width=1200))
        fig7.update_xaxes(**get_axis_config("Altitude (m)")); fig7.update_yaxes(**get_axis_config("L/D"))
        fig7.write_image(os.path.join(run_dir, f'{save_prefix}_lift_drag_ratio.png'), width=1400, height=800, scale=2)
        
        print(f"[EXPORT] Individual mission summary plots saved as PNG to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save individual mission summary plots: {e}")
    
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
    - Mass Evolution
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
            '<b>Climb: Fuel Flow</b>', '<b>Climb: Thrust/Drag</b>', '<b>Climb: Mass</b>',
            '<b>Climb: Lever</b>', '<b>Climb: Airspeed</b>', '<b>Climb: Fuel</b>',
            # Row 2: Cruise Phase  
            '<b>Cruise: Fuel Flow</b>', '<b>Cruise: Thrust/Drag</b>', '<b>Cruise: Mass</b>',
            '<b>Cruise: Lever</b>', '<b>Cruise: Airspeed</b>', '<b>Cruise: Fuel</b>',
            # Row 3: Descent Phase
            '<b>Descent: Fuel Flow</b>', '<b>Descent: Thrust/Drag</b>', '<b>Descent: Mass</b>',
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
    climb_mass_kg = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic mass from DP optimization
    
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
    cruise_mass_kg = cruise_result.mass_kg
    cruise_lever = cruise_result.lever_position * 100
    cruise_fuel_consumed = cruise_result.fuel_consumed_kg
    cruise_tas_ms = cruise_result.true_airspeed_mps
    
    # ========= DESCENT PHASE DATA =========
    descent_time_min = descent_result.time_s / 60.0
    descent_fuel_flow_kgh = descent_result.fuel_flow_kgps * 3600
    descent_thrust_kn = descent_result.thrust_total_N / 1000
    descent_drag_kn = descent_result.drag_N / 1000
    descent_mass_kg = descent_result.mass_kg
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
    
    # 3. Mass
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_mass_kg,
            mode='lines',
            name='Climb Mass',
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
    
    # 3. Mass
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_mass_kg,
            mode='lines',
            name='Cruise Mass',
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
    
    # 3. Mass
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_mass_kg,
            mode='lines',
            name='Descent Mass',
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
    # Use mass tracking for consistent fuel calculation
    total_fuel_sum = climb_fuel_kg[-1] + cruise_fuel_consumed[-1] + descent_cum_fuel_kg[-1]
    total_fuel_mass = initial_mass_kg - descent_result.final_mass_kg
    
    # Diagnostic check for consistency with detailed breakdown
    if abs(total_fuel_sum - total_fuel_mass) > 0.1:
        print(f"\n[COMBINED ANALYSIS] Warning: Fuel discrepancy detected: {abs(total_fuel_sum - total_fuel_mass):.2f} kg")
        print(f"  Fuel sum method: {total_fuel_sum:.2f} kg")
        print(f"    - Climb fuel: {climb_fuel_kg[-1]:.2f} kg")
        print(f"    - Cruise fuel: {cruise_fuel_consumed[-1]:.2f} kg")
        print(f"    - Descent fuel: {descent_cum_fuel_kg[-1]:.2f} kg")
        print(f"  Mass loss method: {total_fuel_mass:.2f} kg")
        print(f"    - Initial mass: {initial_mass_kg:.2f} kg")
        print(f"    - Final mass: {descent_result.final_mass_kg:.2f} kg")
        print(f"    - Mass loss (fuel): {total_fuel_mass:.2f} kg")
        
        # Check mass continuity between phases
        climb_final_mass = climb_result.mass_kg[-1] if len(climb_result.mass_kg) > 0 else initial_mass_kg - climb_fuel_kg[-1]
        cruise_initial_mass = cruise_result.mass_kg[0] if len(cruise_result.mass_kg) > 0 else climb_final_mass
        cruise_final_mass = cruise_result.mass_kg[-1] if len(cruise_result.mass_kg) > 0 else cruise_initial_mass - cruise_fuel_consumed[-1]
        descent_initial_mass = descent_result.mass_kg[0] if len(descent_result.mass_kg) > 0 else cruise_final_mass
        
        print(f"\n  Mass continuity check:")
        print(f"    - Climb final mass: {climb_final_mass:.2f} kg")
        print(f"    - Cruise initial mass: {cruise_initial_mass:.2f} kg (diff: {abs(climb_final_mass - cruise_initial_mass):.2f} kg)")
        print(f"    - Cruise final mass: {cruise_final_mass:.2f} kg")
        print(f"    - Descent initial mass: {descent_initial_mass:.2f} kg (diff: {abs(cruise_final_mass - descent_initial_mass):.2f} kg)")
        print(f"    - Descent final mass: {descent_result.final_mass_kg:.2f} kg")
        
        # Check if mass changes match fuel consumption in each phase
        climb_mass_loss = initial_mass_kg - climb_final_mass
        cruise_mass_loss = cruise_initial_mass - cruise_final_mass
        descent_mass_loss = descent_initial_mass - descent_result.final_mass_kg
        
        print(f"\n  Phase-wise mass loss (fuel) vs fuel consumption:")
        print(f"    - Climb: mass loss = {climb_mass_loss:.2f} kg, fuel = {climb_fuel_kg[-1]:.2f} kg (diff: {abs(climb_mass_loss - climb_fuel_kg[-1]):.2f} kg)")
        print(f"    - Cruise: mass loss = {cruise_mass_loss:.2f} kg, fuel = {cruise_fuel_consumed[-1]:.2f} kg (diff: {abs(cruise_mass_loss - cruise_fuel_consumed[-1]):.2f} kg)")
        print(f"    - Descent: mass loss = {descent_mass_loss:.2f} kg, fuel = {descent_cum_fuel_kg[-1]:.2f} kg (diff: {abs(descent_mass_loss - descent_cum_fuel_kg[-1]):.2f} kg)")
    
    # Use mass-based calculation for consistency with dashboard
    total_fuel = total_fuel_mass
    total_time_min = climb_time_min[-1] + cruise_time_min[-1] + descent_time_min[-1]
    
    # ========= AERODYNAMIC SUMMARY TABLE =========
    # Get table styling
    header_style = get_table_header_style()
    cell_style = get_table_cell_style()
    
    
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
            elif col == 3:  # Mass
                # Set y-axis range to zoom in on mass changes for better visibility
                if row == 1:  # Climb
                    mass_min_local, mass_max_local = np.min(climb_mass_kg), np.max(climb_mass_kg)
                    mass_margin_local = (mass_max_local - mass_min_local) * 0.2
                    fig.update_yaxes(title_text="Mass (kg)", row=row, col=col, gridcolor='lightgray',
                                   range=[mass_min_local - mass_margin_local, mass_max_local + mass_margin_local])
                elif row == 2:  # Cruise
                    mass_min_local, mass_max_local = np.min(cruise_mass_kg), np.max(cruise_mass_kg)
                    mass_margin_local = (mass_max_local - mass_min_local) * 0.2
                    fig.update_yaxes(title_text="Mass (kg)", row=row, col=col, gridcolor='lightgray',
                                   range=[mass_min_local - mass_margin_local, mass_max_local + mass_margin_local])
                elif row == 3:  # Descent
                    mass_min_local, mass_max_local = np.min(descent_mass_kg), np.max(descent_mass_kg)
                    mass_margin_local = (mass_max_local - mass_min_local) * 0.2
                    fig.update_yaxes(title_text="Mass (kg)", row=row, col=col, gridcolor='lightgray',
                                   range=[mass_min_local - mass_margin_local, mass_max_local + mass_margin_local])
                else:
                    fig.update_yaxes(title_text="Mass (kg)", row=row, col=col, gridcolor='lightgray')
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
    
    # Save to CombinedPerformance subfolder
    run_dir = get_or_create_run_directory(phase="CombinedPerformance")
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
    
    # ════════════════════════════════════════════════════════════════
    # Save individual phase performance plots (grouped by phase)
    # ════════════════════════════════════════════════════════════════
    try:
        save_prefix = "combined_performance"
        
        # CLIMB PHASE - All 6 metrics in one figure
        fig_climb = make_subplots(rows=2, cols=3,
            subplot_titles=('<b>Fuel Flow</b>', '<b>Thrust/Drag</b>', '<b>Mass</b>',
                           '<b>Lever</b>', '<b>Airspeed</b>', '<b>Fuel</b>'),
            vertical_spacing=0.12, horizontal_spacing=0.12)
        
        # Fuel Flow
        fig_climb.add_trace(go.Scatter(x=climb_time_min, y=climb_fuel_flow_kgh, mode='lines', name='Fuel Flow',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            showlegend=False), row=1, col=1)
        # Thrust/Drag
        fig_climb.add_trace(go.Scatter(x=climb_time_min, y=climb_thrust_N / 1000, mode='lines', name='Thrust',
            line=dict(color='darkblue', width=LineStyles.THICK), showlegend=False), row=1, col=2)
        fig_climb.add_trace(go.Scatter(x=climb_time_min, y=climb_drag_N / 1000, mode='lines', name='Drag',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH), showlegend=False), row=1, col=2)
        # Mass
        fig_climb.add_trace(go.Scatter(x=climb_time_min, y=climb_mass_kg, mode='lines', name='Mass',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            showlegend=False), row=1, col=3)
        # Lever
        fig_climb.add_trace(go.Scatter(x=climb_time_min, y=climb_lever * 100, mode='lines', name='Lever',
            line=dict(color='steelblue', width=LineStyles.THICK), showlegend=False), row=2, col=1)
        # Airspeed
        fig_climb.add_trace(go.Scatter(x=climb_time_min, y=climb_tas_ms, mode='lines', name='Airspeed',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), showlegend=False), row=2, col=2)
        # Fuel
        fig_climb.add_trace(go.Scatter(x=climb_time_min, y=climb_fuel_kg, mode='lines', name='Fuel',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            showlegend=False), row=2, col=3)
        
        fig_climb.update_layout(title=dict(text="<b>CLIMB PHASE - Performance Analysis</b>", x=0.5, xanchor='center'),
            height=800, width=1400, template='plotly_white', showlegend=False)
        fig_climb.update_xaxes(title_text="Time (min)", row=1, col=1); fig_climb.update_yaxes(title_text="Fuel Flow (kg/h)", row=1, col=1)
        fig_climb.update_xaxes(title_text="Time (min)", row=1, col=2); fig_climb.update_yaxes(title_text="Force (kN)", row=1, col=2)
        fig_climb.update_xaxes(title_text="Time (min)", row=1, col=3); fig_climb.update_yaxes(title_text="Mass (kg)", row=1, col=3)
        fig_climb.update_xaxes(title_text="Time (min)", row=2, col=1); fig_climb.update_yaxes(title_text="Lever (%)", row=2, col=1)
        fig_climb.update_xaxes(title_text="Time (min)", row=2, col=2); fig_climb.update_yaxes(title_text="TAS (m/s)", row=2, col=2)
        fig_climb.update_xaxes(title_text="Time (min)", row=2, col=3); fig_climb.update_yaxes(title_text="Fuel (kg)", row=2, col=3)
        fig_climb.write_image(os.path.join(run_dir, f'{save_prefix}_climb_phase.png'), width=1800, height=1000, scale=2)
        
        # CRUISE PHASE - All 6 metrics in one figure
        fig_cruise = make_subplots(rows=2, cols=3,
            subplot_titles=('<b>Fuel Flow</b>', '<b>Thrust/Drag</b>', '<b>Mass</b>',
                           '<b>Lever</b>', '<b>Airspeed</b>', '<b>Fuel</b>'),
            vertical_spacing=0.12, horizontal_spacing=0.12)
        
        # Fuel Flow
        fig_cruise.add_trace(go.Scatter(x=cruise_time_min, y=cruise_fuel_flow_kgh, mode='lines', name='Fuel Flow',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            showlegend=False), row=1, col=1)
        # Thrust/Drag
        fig_cruise.add_trace(go.Scatter(x=cruise_time_min, y=cruise_thrust_kn, mode='lines', name='Thrust',
            line=dict(color='darkgreen', width=LineStyles.THICK), showlegend=False), row=1, col=2)
        fig_cruise.add_trace(go.Scatter(x=cruise_time_min, y=cruise_drag_kn, mode='lines', name='Drag',
            line=dict(color='lightcoral', width=LineStyles.THICK, dash=LineStyles.DASH), showlegend=False), row=1, col=2)
        # Mass
        fig_cruise.add_trace(go.Scatter(x=cruise_time_min, y=cruise_mass_kg, mode='lines', name='Mass',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            showlegend=False), row=1, col=3)
        # Lever
        fig_cruise.add_trace(go.Scatter(x=cruise_time_min, y=cruise_lever, mode='lines', name='Lever',
            line=dict(color='olive', width=LineStyles.THICK), showlegend=False), row=2, col=1)
        # Airspeed
        fig_cruise.add_trace(go.Scatter(x=cruise_time_min, y=cruise_tas_ms, mode='lines', name='Airspeed',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), showlegend=False), row=2, col=2)
        # Fuel
        fig_cruise.add_trace(go.Scatter(x=cruise_time_min, y=cruise_fuel_consumed, mode='lines', name='Fuel',
            line=dict(color=Colors.CRUISE, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(0, 128, 0, 0.15)',
            showlegend=False), row=2, col=3)
        
        fig_cruise.update_layout(title=dict(text="<b>CRUISE PHASE - Performance Analysis</b>", x=0.5, xanchor='center'),
            height=800, width=1400, template='plotly_white', showlegend=False)
        fig_cruise.update_xaxes(title_text="Time (min)", row=1, col=1); fig_cruise.update_yaxes(title_text="Fuel Flow (kg/h)", row=1, col=1)
        fig_cruise.update_xaxes(title_text="Time (min)", row=1, col=2); fig_cruise.update_yaxes(title_text="Force (kN)", row=1, col=2)
        fig_cruise.update_xaxes(title_text="Time (min)", row=1, col=3); fig_cruise.update_yaxes(title_text="Mass (kg)", row=1, col=3)
        fig_cruise.update_xaxes(title_text="Time (min)", row=2, col=1); fig_cruise.update_yaxes(title_text="Lever (%)", row=2, col=1)
        fig_cruise.update_xaxes(title_text="Time (min)", row=2, col=2); fig_cruise.update_yaxes(title_text="TAS (m/s)", row=2, col=2)
        fig_cruise.update_xaxes(title_text="Time (min)", row=2, col=3); fig_cruise.update_yaxes(title_text="Fuel (kg)", row=2, col=3)
        fig_cruise.write_image(os.path.join(run_dir, f'{save_prefix}_cruise_phase.png'), width=1800, height=1000, scale=2)
        
        # DESCENT PHASE - All 6 metrics in one figure
        fig_descent = make_subplots(rows=2, cols=3,
            subplot_titles=('<b>Fuel Flow</b>', '<b>Thrust/Drag</b>', '<b>Mass</b>',
                           '<b>Lever</b>', '<b>Airspeed</b>', '<b>Fuel</b>'),
            vertical_spacing=0.12, horizontal_spacing=0.12)
        
        # Fuel Flow
        fig_descent.add_trace(go.Scatter(x=descent_time_min, y=descent_fuel_flow_kgh, mode='lines', name='Fuel Flow',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            showlegend=False), row=1, col=1)
        # Thrust/Drag
        fig_descent.add_trace(go.Scatter(x=descent_time_min, y=descent_thrust_kn, mode='lines', name='Thrust',
            line=dict(color='darkred', width=LineStyles.THICK), showlegend=False), row=1, col=2)
        fig_descent.add_trace(go.Scatter(x=descent_time_min, y=descent_drag_kn, mode='lines', name='Drag',
            line=dict(color='salmon', width=LineStyles.THICK, dash=LineStyles.DASH), showlegend=False), row=1, col=2)
        # Mass
        fig_descent.add_trace(go.Scatter(x=descent_time_min, y=descent_mass_kg, mode='lines', name='Mass',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            showlegend=False), row=1, col=3)
        # Lever
        fig_descent.add_trace(go.Scatter(x=descent_time_min, y=descent_lever, mode='lines', name='Lever',
            line=dict(color='firebrick', width=LineStyles.THICK), showlegend=False), row=2, col=1)
        # Airspeed
        fig_descent.add_trace(go.Scatter(x=descent_time_min, y=descent_tas_ms, mode='lines', name='Airspeed',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), showlegend=False), row=2, col=2)
        # Fuel
        fig_descent.add_trace(go.Scatter(x=descent_time_min, y=descent_cum_fuel_kg, mode='lines', name='Fuel',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            showlegend=False), row=2, col=3)
        
        fig_descent.update_layout(title=dict(text="<b>DESCENT PHASE - Performance Analysis</b>", x=0.5, xanchor='center'),
            height=800, width=1400, template='plotly_white', showlegend=False)
        fig_descent.update_xaxes(title_text="Time (min)", row=1, col=1); fig_descent.update_yaxes(title_text="Fuel Flow (kg/h)", row=1, col=1)
        fig_descent.update_xaxes(title_text="Time (min)", row=1, col=2); fig_descent.update_yaxes(title_text="Force (kN)", row=1, col=2)
        fig_descent.update_xaxes(title_text="Time (min)", row=1, col=3); fig_descent.update_yaxes(title_text="Mass (kg)", row=1, col=3)
        fig_descent.update_xaxes(title_text="Time (min)", row=2, col=1); fig_descent.update_yaxes(title_text="Lever (%)", row=2, col=1)
        fig_descent.update_xaxes(title_text="Time (min)", row=2, col=2); fig_descent.update_yaxes(title_text="TAS (m/s)", row=2, col=2)
        fig_descent.update_xaxes(title_text="Time (min)", row=2, col=3); fig_descent.update_yaxes(title_text="Fuel (kg)", row=2, col=3)
        fig_descent.write_image(os.path.join(run_dir, f'{save_prefix}_descent_phase.png'), width=1800, height=1000, scale=2)
        
        print(f"[EXPORT] Individual combined performance plots saved as PNG to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save individual combined performance plots: {e}")
    
    # Also save if custom path requested
    if save_html:
        fig.write_html(save_html)
    
    # Show the plot
    fig.show()
    
    return fig


# ========================================================================
# SECTION 3: 3D MISSION TRAJECTORY VISUALIZATION
# ========================================================================

def plot_complete_mission_3d(climb_result: MinFuelSchedule,
                            cruise_result: CruiseResults,
                            descent_result: DescentResults,
                            climb_info: Dict[str, Any],
                            descent_info: Dict[str, Any],
                            save_html: Optional[str] = None):
    """
    Generate complete mission trajectory in 3D state space (δ,M,h).
    
    Visualization: Three-phase trajectory with flight envelope constraints
        - Climb: X_climb(t) in (δ,M,h) space (blue)
        - Cruise: X_cruise(t) ≈ constant (δ,M,h) (green)
        - Descent: X_descent(t) in (δ,M,h) space (red)
        - Envelope limits: M_MMO, M_stall(h), h_ceiling
    
    Axes: δ (throttle) × M (Mach) × h (altitude)
    Interactive: Camera presets, phase markers, hover tooltips
    
    Parameters:
        climb_result: MinFuelSchedule - climb trajectory
        cruise_result: CruiseResults - cruise trajectory
        descent_result: DescentResults - descent trajectory
        climb_info: dict - climb metadata
        descent_info: dict - descent metadata
        save_html: str - optional custom save path
        
    Returns:
        Plotly figure object
    """
    from aircraft_config import M_MMO
    
    fig = go.Figure()
    
    # ════════════════════════════════════════════════════════════════════
    # Climb Phase Trajectory: X_climb(t) = (δ(t), M(t), h(t))
    # ════════════════════════════════════════════════════════════════════
    climb_lever = climb_result.lever
    climb_mach = climb_result.mach
    climb_alt = climb_result.alt_m
    
    fig.add_trace(go.Scatter3d(
        x=climb_lever,
        y=climb_mach,
        z=climb_alt,
        mode='lines+markers',
        line=dict(color='royalblue', width=8),
        marker=dict(size=4, color='blue'),
        name='Climb Phase (Optimal)',
        hovertemplate='<b>CLIMB</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='climb'
    ))
    
    # Takeoff marker: X_0 = (δ_0, M_0, h_0)
    fig.add_trace(go.Scatter3d(
        x=[climb_lever[0]],
        y=[climb_mach[0]],
        z=[climb_alt[0]],
        mode='markers',
        marker=dict(size=14, color='darkblue', symbol='diamond', line=dict(color='white', width=2)),
        name='Takeoff',
        hovertemplate='<b>Takeoff</b><br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='climb'
    ))
    
    # Cruise transition marker: X_cruise,0 = X_climb,f
    fig.add_trace(go.Scatter3d(
        x=[climb_lever[-1]],
        y=[climb_mach[-1]],
        z=[climb_alt[-1]],
        mode='markers',
        marker=dict(size=12, color='cyan', symbol='diamond', line=dict(color='white', width=2)),
        name='Climb End / Cruise Start',
        hovertemplate='<b>Climb End / Cruise Start</b><br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='climb'
    ))
    
    # ════════════════════════════════════════════════════════════════════
    # Climb Design Space: Visualize J(δ,M,h) neighborhood
    # ════════════════════════════════════════════════════════════════════
    climb_J = climb_result.cumFuel_kg
    
    # Generate point cloud: Sample neighborhood around X*(t)
    climb_cloud_x = []
    climb_cloud_y = []
    climb_cloud_z = []
    climb_cloud_j = []
    
    n_climb_points = len(climb_mach)
    # Sample more densely (50 points) and ensure we start from index 0
    sample_indices = list(range(0, n_climb_points, max(1, n_climb_points // 50)))
    if 0 not in sample_indices:
        sample_indices = [0] + sample_indices
    
    for i in sample_indices:
        # Add variations around each point
        lever_var = np.linspace(max(0.5, climb_lever[i] - 0.15), 
                               min(1.0, climb_lever[i] + 0.15), 6)
        mach_var = np.linspace(max(0.2, climb_mach[i] - 0.05), 
                              min(0.85, climb_mach[i] + 0.05), 6)
        
        for lv in lever_var:
            for mv in mach_var:
                climb_cloud_x.append(lv)
                climb_cloud_y.append(mv)
                climb_cloud_z.append(climb_alt[i])
                dist = abs(lv - climb_lever[i]) + abs(mv - climb_mach[i]) * 10
                climb_cloud_j.append(climb_J[i] * (1 + dist * 0.5))
    
    climb_cloud_x = np.array(climb_cloud_x)
    climb_cloud_y = np.array(climb_cloud_y)
    climb_cloud_z = np.array(climb_cloud_z)
    climb_cloud_j = np.array(climb_cloud_j)
    
    # Normalize J values for colormap: J_norm = (J - J_min) / (J_max - J_min)
    climb_j_norm = (climb_cloud_j - np.min(climb_cloud_j)) / (np.max(climb_cloud_j) - np.min(climb_cloud_j) + 1e-9)
    
    # Design space scatter plot
    fig.add_trace(go.Scatter3d(
        x=climb_cloud_x,
        y=climb_cloud_y,
        z=climb_cloud_z,
        mode='markers',
        marker=dict(
            size=4,
            color=climb_j_norm,
            colorscale='Blues',
            opacity=0.5,
            showscale=False
        ),
        name='Climb Design Space',
        hovertemplate='<b>Design Space</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='climb',
        showlegend=True
    ))
    
    # ════════════════════════════════════════════════════════════════════
    # Cruise Phase Trajectory: X_cruise(t) ≈ (δ_const, M_const, h_const)
    # ════════════════════════════════════════════════════════════════════
    cruise_lever_avg = 0.60  # Typical cruise thrust lever (approximate)
    cruise_mach = cruise_result.mach_number
    cruise_alt = cruise_result.altitude_m
    
    fig.add_trace(go.Scatter3d(
        x=[cruise_lever_avg] * len(cruise_mach),
        y=cruise_mach,
        z=cruise_alt,
        mode='lines+markers',
        line=dict(color='green', width=8),
        marker=dict(size=4, color='green'),
        name='Cruise Phase',
        hovertemplate='<b>CRUISE</b><br>' +
                     'Lever: ~%{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='cruise'
    ))
    
    # Descent transition marker: X_descent,0 = X_cruise,f
    fig.add_trace(go.Scatter3d(
        x=[cruise_lever_avg],
        y=[cruise_mach[-1]],
        z=[cruise_alt[-1]],
        mode='markers',
        marker=dict(size=12, color='lime', symbol='diamond', line=dict(color='white', width=2)),
        name='Cruise End / Descent Start',
        hovertemplate='<b>Cruise End / Descent Start</b><br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='cruise'
    ))
    
    # ════════════════════════════════════════════════════════════════════
    # Cruise Design Space: Visualize J(δ,M,h) neighborhood
    # ════════════════════════════════════════════════════════════════════
    # Calculate approximate cruise fuel consumption for visualization
    cruise_fuel_consumed = cruise_result.fuel_consumed_kg
    climb_final_fuel = climb_result.cumFuel_kg[-1]
    
    # Generate point cloud: Sample neighborhood around X*(t)
    cruise_cloud_x = []
    cruise_cloud_y = []
    cruise_cloud_z = []
    cruise_cloud_j = []
    
    n_cruise_points = len(cruise_mach)
    # Sample more densely (50 points) and ensure we start from index 0
    cruise_sample_indices = list(range(0, n_cruise_points, max(1, n_cruise_points // 50)))
    if 0 not in cruise_sample_indices:
        cruise_sample_indices = [0] + cruise_sample_indices
    
    for i in cruise_sample_indices:
        # Add variations around each point (smaller range for cruise)
        lever_var = np.linspace(max(0.4, cruise_lever_avg - 0.15), 
                               min(0.8, cruise_lever_avg + 0.15), 6)
        mach_var = np.linspace(max(0.2, cruise_mach[i] - 0.03), 
                              min(0.85, cruise_mach[i] + 0.03), 6)
        # Cruise may have altitude changes (cruise climb)
        alt_var = np.linspace(max(cruise_alt[0] - 500, cruise_alt[i] - 200), 
                             min(cruise_alt[-1] + 500, cruise_alt[i] + 200), 4)
        
        for lv in lever_var:
            for mv in mach_var:
                for av in alt_var:
                    cruise_cloud_x.append(lv)
                    cruise_cloud_y.append(mv)
                    cruise_cloud_z.append(av)
                    dist = abs(lv - cruise_lever_avg) + abs(mv - cruise_mach[i]) * 10 + abs(av - cruise_alt[i]) / 1000
                    cruise_cloud_j.append((climb_final_fuel + cruise_fuel_consumed[i]) * (1 + dist * 0.3))
    
    cruise_cloud_x = np.array(cruise_cloud_x)
    cruise_cloud_y = np.array(cruise_cloud_y)
    cruise_cloud_z = np.array(cruise_cloud_z)
    cruise_cloud_j = np.array(cruise_cloud_j)
    
    # Normalize J values for colormap: J_norm = (J - J_min) / (J_max - J_min)
    cruise_j_norm = (cruise_cloud_j - np.min(cruise_cloud_j)) / (np.max(cruise_cloud_j) - np.min(cruise_cloud_j) + 1e-9)
    
    # Design space scatter plot
    fig.add_trace(go.Scatter3d(
        x=cruise_cloud_x,
        y=cruise_cloud_y,
        z=cruise_cloud_z,
        mode='markers',
        marker=dict(
            size=4,
            color=cruise_j_norm,
            colorscale='Greens',
            opacity=0.5,
            showscale=False
        ),
        name='Cruise Design Space',
        hovertemplate='<b>Design Space</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='cruise',
        showlegend=True
    ))
    
    # ════════════════════════════════════════════════════════════════════
    # Descent Design Space: Visualize J(δ,M,h) neighborhood
    # ════════════════════════════════════════════════════════════════════
    descent_mach = descent_result.mach
    descent_alt = descent_result.alt_m
    descent_lever = descent_result.lever
    descent_J = descent_result.cumFuel_kg
    
    # Generate point cloud: Sample neighborhood around X*(t)
    cloud_points_x = []
    cloud_points_y = []
    cloud_points_z = []
    cloud_points_j = []
    
    n_points = len(descent_mach)
    # Sample more densely (50 points) and ensure we start from index 0
    descent_sample_indices = list(range(0, n_points, max(1, n_points // 50)))
    if 0 not in descent_sample_indices:
        descent_sample_indices = [0] + descent_sample_indices
    
    for i in descent_sample_indices:
        # Add variations around each point
        lever_var = np.linspace(max(0, descent_lever[i] - 0.1), 
                               min(0.3, descent_lever[i] + 0.1), 6)
        mach_var = np.linspace(max(0.2, descent_mach[i] - 0.05), 
                              min(0.85, descent_mach[i] + 0.05), 6)
        
        for lv in lever_var:
            for mv in mach_var:
                cloud_points_x.append(lv)
                cloud_points_y.append(mv)
                cloud_points_z.append(descent_alt[i])
                dist = abs(lv - descent_lever[i]) + abs(mv - descent_mach[i]) * 10
                cloud_points_j.append(descent_J[i] * (1 + dist * 0.5))
    
    cloud_points_x = np.array(cloud_points_x)
    cloud_points_y = np.array(cloud_points_y)
    cloud_points_z = np.array(cloud_points_z)
    cloud_points_j = np.array(cloud_points_j)
    
    # Normalize J values for colormap: J_norm = (J - J_min) / (J_max - J_min)
    cloud_j_norm = (cloud_points_j - np.min(cloud_points_j)) / (np.max(cloud_points_j) - np.min(cloud_points_j) + 1e-9)
    
    # Design space scatter plot
    fig.add_trace(go.Scatter3d(
        x=cloud_points_x,
        y=cloud_points_y,
        z=cloud_points_z,
        mode='markers',
        marker=dict(
            size=4,
            color=cloud_j_norm,
            colorscale='Reds',
            opacity=0.5,
            showscale=False
        ),
        name='Descent Design Space',
        hovertemplate='<b>Design Space</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='descent',
        showlegend=True
    ))
    
    # ════════════════════════════════════════════════════════════════════
    # Optimal Descent Path: X*(t) from DP optimization
    # ════════════════════════════════════════════════════════════════════
    fig.add_trace(go.Scatter3d(
        x=descent_lever,
        y=descent_mach,
        z=descent_alt,
        mode='lines+markers',
        line=dict(color='crimson', width=8),
        marker=dict(size=5, color='red'),
        name='Descent Phase (Optimal)',
        hovertemplate='<b>DESCENT</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='descent'
    ))
    
    # Landing marker: X_f = (δ_f, M_f, h_f)
    fig.add_trace(go.Scatter3d(
        x=[descent_lever[-1]],
        y=[descent_mach[-1]],
        z=[descent_alt[-1]],
        mode='markers',
        marker=dict(size=14, color='darkred', symbol='diamond', line=dict(color='white', width=2)),
        name='Approach/Landing',
        hovertemplate='<b>Approach/Landing</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='descent'
    ))
    
    # ════════════════════════════════════════════════════════════════════
    # Flight Envelope Constraints
    # ════════════════════════════════════════════════════════════════════
    MAX_SERVICE_CEILING_M = 13994.1  # h_max [m]: service ceiling at δ=1.0, M=0.900
    
    # Constraint 1: Maximum Mach M ≤ M_MMO (vertical plane)
    max_alt = max(climb_alt[-1], cruise_alt[0], 12000)
    lever_range = np.linspace(0, 1.0, 10)
    alt_range = np.linspace(0, max_alt, 20)
    L_mmo, Z_mmo = np.meshgrid(lever_range, alt_range)
    M_mmo = np.full_like(L_mmo, M_MMO)  # M_MMO = 0.9392
    
    fig.add_trace(go.Surface(
        x=L_mmo,
        y=M_mmo,
        z=Z_mmo,
        colorscale=[[0, 'rgba(220, 20, 60, 0.2)'], [1, 'rgba(220, 20, 60, 0.2)']],
        showscale=False,
        name='Max Engine Mach Limit',
        hovertemplate='<b>Max Engine Mach Limit</b><br>M = ' + f'{M_MMO:.3f}<br>' +
                     'Altitude: %{z:.0f} m<extra></extra>',
        legendgroup='limits'
    ))
    
    # Constraint 2: Service ceiling h ≤ h_max (horizontal plane)
    if MAX_SERVICE_CEILING_M <= max_alt + 1000:
        lever_range_ceiling = np.linspace(0, 1.0, 10)
        mach_range_ceiling = np.linspace(0.1, M_MMO, 10)
        L_ceiling, M_ceiling = np.meshgrid(lever_range_ceiling, mach_range_ceiling)
        Z_ceiling = np.full_like(L_ceiling, MAX_SERVICE_CEILING_M)
        
        fig.add_trace(go.Surface(
            x=L_ceiling,
            y=M_ceiling,
            z=Z_ceiling,
            colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
            opacity=0.45,
            showscale=False,
            name=f'Max Service Ceiling ({MAX_SERVICE_CEILING_M/1000:.2f} km)',
            hovertemplate='<b>Max Service Ceiling</b><br>Altitude: ' + f'{MAX_SERVICE_CEILING_M:.0f} m<br>' +
                         'Mach: %{y:.3f}<extra></extra>',
            legendgroup='limits'
        ))
    
    # ════════════════════════════════════════════════════════════════════
    # Layout Configuration and Export
    # ════════════════════════════════════════════════════════════════════
    # Mission statistics for subtitle
    total_fuel = climb_result.cumFuel_kg[-1] + cruise_result.total_fuel_consumed_kg + descent_result.total_fuel_consumed_kg
    climb_time_total = np.sum(climb_result.dt_s) if len(climb_result.dt_s) > 0 else 0.0
    total_time = (climb_time_total + cruise_result.total_time_s + descent_result.total_time_s) / 60.0
    
    fig.update_layout(
        title=dict(
            text="<b>Complete Mission 3D Trajectory: Climb → Cruise → Descent</b><br>" +
                 f"<sup>Blue (Climb) → Green (Cruise) → Red (Descent) | " +
                 f"Total Fuel: {total_fuel:.1f} kg | Total Time: {total_time:.1f} min</sup>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='<b>Throttle Lever Position</b>',
            yaxis_title='<b>Mach Number</b>',
            zaxis_title='<b>Altitude (m)</b>',
            camera=dict(
                eye=dict(x=1.8, y=-1.8, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(
                gridcolor='lightgray',
                backgroundcolor='rgba(240, 240, 255, 0.9)',
                range=[0, 1.05]
            ),
            yaxis=dict(
                gridcolor='lightgray',
                backgroundcolor='rgba(240, 240, 255, 0.9)',
                range=[0.2, 0.85]
            ),
            zaxis=dict(
                gridcolor='lightgray',
                backgroundcolor='rgba(240, 240, 255, 0.9)',
                range=[0, max_alt * 1.1]
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1.5, z=2.5)
        ),
        width=1400,
        height=1000,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=11),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10)
        ),
        hovermode='closest'
    )
    
    # Camera view presets for trajectory inspection
    camera_buttons = [
        dict(
            label="Front View",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=2.5, y=0, z=0.5)}]
        ),
        dict(
            label="Side View",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=0, y=2.5, z=0.5)}]
        ),
        dict(
            label="Top View",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=0, y=0, z=3)}]
        ),
        dict(
            label="Isometric",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=1.8, y=-1.8, z=1.5)}]
        ),
        dict(
            label="Climb Focus",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=1.5, y=-1.5, z=0.8)}]
        ),
        dict(
            label="Descent Focus",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=1.5, y=-1.5, z=1.8)}]
        )
    ]
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=camera_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.02,
                yanchor="bottom",
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )
    
    # Save to Mission3D subfolder
    run_dir = get_or_create_run_directory(phase="Mission3D")
    output_path_html = os.path.join(run_dir, 'complete_mission_3d.html')
    output_path_png = os.path.join(run_dir, 'complete_mission_3d.png')
    
    fig.write_html(output_path_html)
    
    # Also save as PNG
    try:
        fig.write_image(output_path_png, width=1800, height=1200, scale=2)
        print(f"[EXPORT] Complete mission 3D saved to: {output_path_html} (interactive) and {output_path_png} (PNG)")
    except Exception as e:
        print(f"[EXPORT] Complete mission 3D saved to: {output_path_html} (HTML only)")
        print(f"[WARNING] Could not save PNG version: {e}")
    
    # Optional custom save path
    if save_html:
        fig.write_html(save_html)
    
    fig.show()
    
    return fig