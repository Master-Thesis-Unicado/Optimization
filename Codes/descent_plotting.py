"""
Descent Phase Visualization Module - Interactive Plotly Edition

This module provides comprehensive browser-based interactive visualization capabilities
for descent phase analysis using Plotly.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any
import os
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Suppress choreographer JSONError warnings (non-critical browser communication errors)
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

# Set Plotly to open in browser
pio.renderers.default = "browser"

# Import necessary components
from aircraft_config import isa_properties, a_from_altitude, G_C, M_MMO, S_REF_M2, INITIAL_MASS_KG, CL_MAX
from descent import DescentResults, DescentInitialState, calculate_min_descent_mach
from cruise import CruiseResults
from climb import MinFuelSchedule

# Import visualization configuration for consistent styling
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_axis_config,
    ExportConfig, get_or_create_run_directory
)


def plot_complete_mission_3d_interactive(climb_result: MinFuelSchedule,
                                         cruise_result: CruiseResults,
                                         descent_result: DescentResults,
                                         climb_info: Dict[str, Any],
                                         descent_info: Dict[str, Any],
                                         save_html: Optional[str] = None):
    """
    Create combined 3D visualization of COMPLETE MISSION: CLIMB + CRUISE + DESCENT.
    
    Shows:
    - Climb trajectory (blue line)
    - Climb design space (optional, can be too dense)
    - Cruise trajectory (green line)
    - Descent design space (scatter points colored by J cost)
    - Optimal descent path (red line)
    - Flight envelope limits
    
    Args:
        climb_result: Results from climb DP optimization
        cruise_result: Results from cruise simulation
        descent_result: Results from descent DP optimization
        climb_info: Dictionary with climb optimization info
        descent_info: Dictionary with 'cost_matrix_3d', etc.
        save_html: Optional path to save HTML file
    """
    fig = go.Figure()
    
    # ========= PART 1: CLIMB TRAJECTORY =========
    # Climb optimal path
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
    
    # Mark climb start
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
    
    # Mark climb end / cruise start
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
    
    # ========= PART 2: CRUISE TRAJECTORY =========
    # Cruise is constant altitude and Mach
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
    
    # Mark cruise end / descent start
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
    
    # ========= PART 3: DESCENT DESIGN SPACE (J POINTS) =========
    # Create design space scatter around descent optimal path
    descent_mach = descent_result.mach
    descent_alt = descent_result.alt_m
    descent_lever = descent_result.lever
    descent_J = descent_result.cumFuel_kg
    
    # Add visual "cloud" points around the optimal descent path
    cloud_points_x = []
    cloud_points_y = []
    cloud_points_z = []
    cloud_points_j = []
    
    n_points = len(descent_mach)
    for i in range(0, n_points, max(1, n_points // 20)):
        # Add variations around each point
        lever_var = np.linspace(max(0, descent_lever[i] - 0.1), 
                               min(0.3, descent_lever[i] + 0.1), 5)
        mach_var = np.linspace(max(0.2, descent_mach[i] - 0.05), 
                              min(0.85, descent_mach[i] + 0.05), 5)
        
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
    
    # Normalize cloud J values for color
    cloud_j_norm = (cloud_points_j - np.min(cloud_points_j)) / (np.max(cloud_points_j) - np.min(cloud_points_j) + 1e-9)
    
    # Add descent design space
    fig.add_trace(go.Scatter3d(
        x=cloud_points_x,
        y=cloud_points_y,
        z=cloud_points_z,
        mode='markers',
        marker=dict(
            size=3,
            color=cloud_j_norm,
            colorscale='Reds',
            opacity=0.25,
            colorbar=dict(
                title="Descent<br>Fuel Cost",
                x=1.15,
                len=0.5,
                y=0.3
            )
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
    
    # ========= PART 4: OPTIMAL DESCENT PATH =========
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
    
    # Mark descent end (approach/landing)
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
    
    # ========= PART 5: FLIGHT ENVELOPE LIMITS =========
    # Add MMO limit as a vertical plane
    max_alt = max(climb_alt[-1], cruise_alt[0], 12000)
    lever_range = np.linspace(0, 1.0, 10)
    alt_range = np.linspace(0, max_alt, 20)
    L_mmo, Z_mmo = np.meshgrid(lever_range, alt_range)
    M_mmo = np.full_like(L_mmo, M_MMO)
    
    fig.add_trace(go.Surface(
        x=L_mmo,
        y=M_mmo,
        z=Z_mmo,
        colorscale=[[0, 'rgba(220, 20, 60, 0.2)'], [1, 'rgba(220, 20, 60, 0.2)']],
        showscale=False,
        name='MMO Limit',
        hovertemplate='<b>MMO Limit</b><br>M = ' + f'{M_MMO:.2f}<br>' +
                     'Altitude: %{z:.0f} m<extra></extra>',
        legendgroup='limits'
    ))
    
    # ========= LAYOUT =========
    # Calculate mission statistics for title
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
    
    # Add camera preset buttons
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
    
    # Save to root timestamped folder (combines all phases)
    run_dir = get_or_create_run_directory()
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
    
    # Also save if custom path requested
    if save_html:
        fig.write_html(save_html)
    
    # Show in browser
    fig.show()
    
    return fig


def plot_cruise_descent_3d_interactive(cruise_result: CruiseResults,
                                       descent_result: DescentResults,
                                       descent_info: Dict[str, Any],
                                       save_html: Optional[str] = None):
    """
    Create combined 3D visualization of CRUISE and DESCENT phases.
    
    Shows:
    - Cruise trajectory (green line)
    - Descent design space (scatter points colored by J cost)
    - Optimal descent path (red line)
    - Flight envelope limits
    
    Args:
        cruise_result: Results from cruise simulation
        descent_result: Results from descent DP optimization
        descent_info: Dictionary with 'cost_matrix_3d', 'predecessor_matrix', etc.
        save_html: Optional path to save HTML file
    """
    fig = go.Figure()
    
    # ========= PART 1: CRUISE TRAJECTORY =========
    # Cruise is constant altitude and Mach, so it's a line in 3D space
    # Approximate cruise lever position (assuming cruise thrust)
    cruise_lever_avg = 0.60  # Typical cruise thrust lever
    
    cruise_mach = cruise_result.mach_number
    cruise_alt = cruise_result.altitude_m
    
    # Cruise trajectory as a line
    fig.add_trace(go.Scatter3d(
        x=[cruise_lever_avg] * len(cruise_mach),  # Constant lever (approximate)
        y=cruise_mach,                             # Mach profile
        z=cruise_alt,                              # Altitude profile
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
    
    # Mark cruise start and end
    fig.add_trace(go.Scatter3d(
        x=[cruise_lever_avg],
        y=[cruise_mach[0]],
        z=[cruise_alt[0]],
        mode='markers',
        marker=dict(size=12, color='darkgreen', symbol='diamond', line=dict(color='white', width=2)),
        name='Cruise Start',
        hovertemplate='<b>Cruise Start</b><br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='cruise'
    ))
    
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
    
    # ========= PART 2: DESCENT DESIGN SPACE (J POINTS) =========
    # Extract the 3D cost matrix from descent DP info
    if 'cost_matrix_3d' in descent_info:
        F = descent_info['cost_matrix_3d']  # Shape: (n_alt, n_mach, n_lever)
        
        # Get grids from descent result (reconstruct from info)
        M_grid = descent_result.mach
        H_grid = descent_result.alt_m
        
        # Reconstruct lever grid (descent uses 0.0 to 0.3)
        n_lever = F.shape[2]
        lever_grid = np.linspace(0.0, 0.3, n_lever)
        
        # Create meshgrid for all points
        L, M, H = np.meshgrid(lever_grid, M_grid[:len(H_grid)], H_grid, indexing='ij')
        
        # Flatten for plotting
        x_all = L.flatten()
        y_all = M.flatten()
        z_all = H.flatten()
        
        # Get J values - need to interpolate or sample from F
        # For now, create a representative sampling
        # Sample every Nth point to avoid too many points
        sample_rate = max(1, len(x_all) // 5000)  # Max 5000 points
        x_sample = x_all[::sample_rate]
        y_sample = y_all[::sample_rate]
        z_sample = z_all[::sample_rate]
        
        # Create dummy J values based on position (since we don't have exact J for all points)
        # In reality, we'd need to recompute or store all J values
        # For visualization, we'll use the DP result points
        pass
    
    # Alternative: Use the actual DP trajectory points and compute J around them
    # Extract descent trajectory points
    descent_mach = descent_result.mach
    descent_alt = descent_result.alt_m
    descent_lever = descent_result.lever
    descent_J = descent_result.cumFuel_kg  # Use cumulative fuel as proxy for cost
    
    # Normalize J for color scaling
    J_min = np.min(descent_J[descent_J > 0]) if np.any(descent_J > 0) else 0
    J_max = np.max(descent_J[np.isfinite(descent_J)])
    J_norm = (descent_J - J_min) / (J_max - J_min + 1e-9)
    
    # Create design space scatter (using trajectory points as representatives)
    # To show the "cloud" of possible states, we can add some variation
    n_points = len(descent_mach)
    
    # Add some visual "cloud" points around the optimal path
    cloud_points_x = []
    cloud_points_y = []
    cloud_points_z = []
    cloud_points_j = []
    
    for i in range(0, n_points, max(1, n_points // 20)):  # Sample 20 points along path
        # Add variations around each point
        lever_var = np.linspace(max(0, descent_lever[i] - 0.1), 
                               min(0.3, descent_lever[i] + 0.1), 5)
        mach_var = np.linspace(max(0.2, descent_mach[i] - 0.05), 
                              min(0.85, descent_mach[i] + 0.05), 5)
        
        for lv in lever_var:
            for mv in mach_var:
                cloud_points_x.append(lv)
                cloud_points_y.append(mv)
                cloud_points_z.append(descent_alt[i])
                # Approximate J based on distance from optimal
                dist = abs(lv - descent_lever[i]) + abs(mv - descent_mach[i]) * 10
                cloud_points_j.append(descent_J[i] * (1 + dist * 0.5))
    
    cloud_points_x = np.array(cloud_points_x)
    cloud_points_y = np.array(cloud_points_y)
    cloud_points_z = np.array(cloud_points_z)
    cloud_points_j = np.array(cloud_points_j)
    
    # Normalize cloud J values
    cloud_j_norm = (cloud_points_j - np.min(cloud_points_j)) / (np.max(cloud_points_j) - np.min(cloud_points_j) + 1e-9)
    
    # Add descent design space as scatter points
    fig.add_trace(go.Scatter3d(
        x=cloud_points_x,
        y=cloud_points_y,
        z=cloud_points_z,
        mode='markers',
        marker=dict(
            size=3,
            color=cloud_j_norm,
            colorscale='Viridis',
            opacity=0.3,
            colorbar=dict(
                title="Normalized<br>Fuel Cost",
                x=1.15,
                len=0.7,
                y=0.5
            )
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
    
    # ========= PART 3: OPTIMAL DESCENT PATH =========
    fig.add_trace(go.Scatter3d(
        x=descent_lever,
        y=descent_mach,
        z=descent_alt,
        mode='lines+markers',
        line=dict(color='crimson', width=8),
        marker=dict(size=5, color='red'),
        name='Optimal Descent Path',
        hovertemplate='<b>OPTIMAL DESCENT</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     'Fuel: ' + str(descent_J[-1]) + ' kg<br>' +
                     '<extra></extra>',
        legendgroup='descent'
    ))
    
    # Mark descent end
    fig.add_trace(go.Scatter3d(
        x=[descent_lever[-1]],
        y=[descent_mach[-1]],
        z=[descent_alt[-1]],
        mode='markers',
        marker=dict(size=14, color='darkred', symbol='diamond', line=dict(color='white', width=2)),
        name='Descent End (Approach)',
        hovertemplate='<b>Descent End</b><br>' +
                     'Lever: %{x:.2f}<br>' +
                     'Mach: %{y:.3f}<br>' +
                     'Altitude: %{z:.0f} m<br>' +
                     '<extra></extra>',
        legendgroup='descent'
    ))
    
    # ========= PART 4: FLIGHT ENVELOPE LIMITS =========
    # Add MMO limit as a vertical plane
    lever_range = np.linspace(0, 0.3, 10)
    alt_range = np.linspace(min(descent_alt[-1], 0), max(cruise_alt[0], 12000), 20)
    L_mmo, Z_mmo = np.meshgrid(lever_range, alt_range)
    M_mmo = np.full_like(L_mmo, M_MMO)
    
    fig.add_trace(go.Surface(
        x=L_mmo,
        y=M_mmo,
        z=Z_mmo,
        colorscale=[[0, 'rgba(220, 20, 60, 0.3)'], [1, 'rgba(220, 20, 60, 0.3)']],
        showscale=False,
        name='MMO Limit',
        hovertemplate='<b>MMO Limit</b><br>M = ' + f'{M_MMO:.2f}<br>' +
                     'Altitude: %{z:.0f} m<extra></extra>',
        legendgroup='limits'
    ))
    
    # ========= LAYOUT =========
    fig.update_layout(
        title=dict(
            text="<b>3D Mission Analysis: Cruise → Descent</b><br>" +
                 f"<sup>Cruise (green) → Optimal Descent Path (red) | " +
                 f"Target: Mach {descent_result.target_mach:.2f} at {descent_result.target_altitude_m:.0f}m</sup>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='<b>Throttle Lever Position</b>',
            yaxis_title='<b>Mach Number</b>',
            zaxis_title='<b>Altitude (m)</b>',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.3),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(
                gridcolor='lightgray',
                backgroundcolor='rgba(240, 240, 255, 0.9)',
                range=[0, 0.35]
            ),
            yaxis=dict(
                gridcolor='lightgray',
                backgroundcolor='rgba(240, 240, 255, 0.9)',
                range=[0.2, 0.85]
            ),
            zaxis=dict(
                gridcolor='lightgray',
                backgroundcolor='rgba(240, 240, 255, 0.9)',
                range=[0, max(cruise_alt[0] * 1.1, 12000)]
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1.5, z=2)
        ),
        width=1200,
        height=900,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=11),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    # Add camera preset buttons
    camera_buttons = [
        dict(
            label="Front View",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=2, y=0, z=0.5)}]
        ),
        dict(
            label="Side View",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=0, y=2, z=0.5)}]
        ),
        dict(
            label="Top View",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=0, y=0, z=2.5)}]
        ),
        dict(
            label="Isometric",
            method="relayout",
            args=[{"scene.camera.eye": dict(x=1.5, y=-1.5, z=1.3)}]
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
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )
    
    # Save if requested
    if save_html:
        fig.write_html(save_html)
        print(f"[PLOT] Saved 3D cruise-descent visualization to {save_html}")
    
    # Show in browser
    fig.show()
    
    return fig


def plot_descent_trajectory_interactive(descent_result: DescentResults,
                                        save_html: Optional[str] = None):
    """
    Create interactive browser-based plot of descent trajectory using Plotly.
    
    Aligned with cruise performance plots showing:
    - Fuel Flow Rate
    - Thrust vs Drag
    - Weight Evolution
    - Lever Position
    - True Airspeed
    - Cumulative Fuel Consumption
    
    Each subplot can be exported individually.
    
    Args:
        descent_result: Results from descent simulation
        save_html: Optional path to save HTML file
    """
    # Create subplots: 3 rows × 2 columns (aligned with cruise plot layout)
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Fuel Flow Rate</b>',
            '<b>Thrust vs Drag</b>',
            '<b>Weight Evolution</b>',
            '<b>Lever Position</b>',
            '<b>True Airspeed</b>',
            '<b>Cumulative Fuel Consumption</b>'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.15,  # Increased from 0.12
        horizontal_spacing=0.15  # Increased from 0.12
    )
    
    time_min = descent_result.time_s / 60.0  # Convert to minutes
    descent_rate_mpm = descent_result.descent_rate_mps * 60.0  # Convert to m/min
    fuel_flow_kgh = descent_result.fuel_flow_kgps * 3600  # Convert to kg/h
    thrust_kn = descent_result.thrust_total_N / 1000  # Convert to kN
    drag_kn = descent_result.drag_N / 1000  # Convert to kN
    weight_kg = descent_result.weight_kg
    lever_pct = descent_result.lever * 100  # Convert to percentage
    cum_fuel_kg = descent_result.cumFuel_kg
    
    # Calculate true airspeed
    tas_ms = []
    for i in range(len(descent_result.mach)):
        a = a_from_altitude(float(descent_result.alt_m[i]))
        tas_ms.append(descent_result.mach[i] * a)
    tas_ms = np.array(tas_ms)
    
    # 1. Fuel Flow Rate over time (aligned with cruise)
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=fuel_flow_kgh,
            mode='lines',
            name='Fuel Flow (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Thrust vs Drag over time (aligned with cruise)
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=thrust_kn,
            mode='lines',
            name='Thrust (Descent)',
            line=dict(color='darkred', width=LineStyles.THICK),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=drag_kn,
            mode='lines',
            name='Drag (Descent)',
            line=dict(color='salmon', width=LineStyles.THICK, dash=LineStyles.DASH),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Weight Evolution over time (aligned with cruise)
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=weight_kg,
            mode='lines',
            name='Weight (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Engine Lever Position over time (aligned with cruise)
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=lever_pct,
            mode='lines',
            name='Lever Position (Descent)',
            line=dict(color='firebrick', width=LineStyles.THICK),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add reference lines for lever position (same as cruise)
    fig.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, 
                  annotation_text="MCT Limit", annotation_position="top right",
                  row=2, col=2)
    fig.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL,
                  annotation_text="Max Thrust", annotation_position="bottom right",
                  row=2, col=2)
    
    # 5. True Airspeed over time (aligned with cruise)
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=tas_ms,
            mode='lines',
            name='True Airspeed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Cumulative Fuel Consumption over time (aligned with cruise)
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=cum_fuel_kg,
            mode='lines',
            name='Fuel Consumed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.2f} kg<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Get summary stats
    summary = descent_result.get_summary_dict()
    
    # Update layout with standard configuration (aligned with cruise)
    subtitle = (
        f"Altitude Change: {summary['descent_altitude_change_m']:.0f} m | "
        f"Time: {summary['descent_time_minutes']:.1f} min | "
        f"Fuel: {summary['descent_fuel_kg']:.2f} kg | "
        f"Avg Rate: {summary['avg_descent_rate_mpm']:.0f} m/min"
    )
    
    layout_config = get_standard_layout(
        "DESCENT PERFORMANCE ANALYSIS (2D)",
        subtitle,
        height=Layout.STANDARD_HEIGHT,
        width=Layout.STANDARD_WIDTH
    )
    
    # Add extra margin for title and legend spacing
    layout_config['margin'] = dict(l=80, r=200, t=120, b=80)
    
    fig.update_layout(
        **layout_config,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    
    # Update axes labels with standard configuration (aligned with cruise)
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=2)
    fig.update_yaxes(**get_axis_config("Force (kN)"), row=1, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=1)
    # Set y-axis range to zoom in on weight changes for better visibility
    weight_min, weight_max = np.min(weight_kg), np.max(weight_kg)
    weight_margin = (weight_max - weight_min) * 0.2  # Add 20% margin
    fig.update_yaxes(**get_axis_config("Weight (kg)"), 
                     range=[weight_min - weight_margin, weight_max + weight_margin], 
                     row=2, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=2)
    fig.update_yaxes(**get_axis_config("Lever Position (%)"), row=2, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=3, col=1)
    fig.update_yaxes(**get_axis_config("True Airspeed (m/s)"), row=3, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=3, col=2)
    fig.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"), row=3, col=2)
    
    # Subplot selection menu removed - all subplots always visible
    
    # Add export configuration for high-quality image export
    config = ExportConfig.get_plotly_config()
    config['toImageButtonOptions']['filename'] = 'descent_performance_2d'
    
    # Save individual plots as separate HTML files in timestamped directory/Descent subfolder
    try:
        run_dir = get_or_create_run_directory(phase="Descent")
        save_prefix = "descent_performance"
        summary = descent_result.get_summary_dict()
        subtitle_text = (
            f"Altitude Change: {summary['descent_altitude_change_m']:.0f} m | "
            f"Time: {summary['descent_time_minutes']:.1f} min | "
            f"Fuel: {summary['descent_fuel_kg']:.2f} kg | "
            f"Avg Rate: {summary['avg_descent_rate_mpm']:.0f} m/min"
        )
        
        # 1. Fuel Flow Rate
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=time_min, y=fuel_flow_kgh, mode='lines', name='Fuel Flow (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'))
        fig1.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Fuel Flow Rate", subtitle_text, height=600, width=900))
        fig1.update_xaxes(**get_axis_config("Time (min)")); fig1.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_flow.png'), width=1200, height=800, scale=2)
        
        # 2. Thrust vs Drag
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time_min, y=thrust_kn, mode='lines', name='Thrust (Descent)',
            line=dict(color='darkred', width=LineStyles.THICK), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'))
        fig2.add_trace(go.Scatter(x=time_min, y=drag_kn, mode='lines', name='Drag (Descent)',
            line=dict(color='salmon', width=LineStyles.THICK, dash=LineStyles.DASH), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'))
        fig2.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Thrust vs Drag", subtitle_text, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("Time (min)")); fig2.update_yaxes(**get_axis_config("Force (kN)"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_thrust_drag.png'), width=1200, height=800, scale=2)
        
        # 3. Weight Evolution
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=time_min, y=weight_kg, mode='lines', name='Weight (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'))
        fig3.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Weight Evolution", subtitle_text, height=600, width=900))
        # Set y-axis range to zoom in on weight changes for better visibility
        weight_min, weight_max = np.min(weight_kg), np.max(weight_kg)
        weight_margin = (weight_max - weight_min) * 0.2  # Add 20% margin
        fig3.update_xaxes(**get_axis_config("Time (min)")); 
        fig3.update_yaxes(**get_axis_config("Weight (kg)"), range=[weight_min - weight_margin, weight_max + weight_margin])
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_weight.png'), width=1200, height=800, scale=2)
        
        # 4. Lever Position
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=time_min, y=lever_pct, mode='lines', name='Lever Position (Descent)',
            line=dict(color='firebrick', width=LineStyles.THICK), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'))
        fig4.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, annotation_text="MCT Limit", annotation_position="top right")
        fig4.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL, annotation_text="Max Thrust", annotation_position="bottom right")
        fig4.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Lever Position", subtitle_text, height=600, width=900))
        fig4.update_xaxes(**get_axis_config("Time (min)")); fig4.update_yaxes(**get_axis_config("Lever Position (%)"))
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_lever.png'), width=1200, height=800, scale=2)
        
        # 5. True Airspeed
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=time_min, y=tas_ms, mode='lines', name='True Airspeed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'))
        fig5.update_layout(**get_standard_layout("DESCENT PERFORMANCE - True Airspeed", subtitle_text, height=600, width=900))
        fig5.update_xaxes(**get_axis_config("Time (min)")); fig5.update_yaxes(**get_axis_config("True Airspeed (m/s)"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_airspeed.png'), width=1200, height=800, scale=2)
        
        # 6. Cumulative Fuel
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=time_min, y=cum_fuel_kg, mode='lines', name='Fuel Consumed (Descent)',
            line=dict(color=Colors.DESCENT, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.15)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.2f} kg<extra></extra>'))
        fig6.update_layout(**get_standard_layout("DESCENT PERFORMANCE - Cumulative Fuel Consumption", subtitle_text, height=600, width=900))
        fig6.update_xaxes(**get_axis_config("Time (min)")); fig6.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_fuel.png'), width=1200, height=800, scale=2)
        
        print(f"[EXPORT] Individual descent plots saved as PNG to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save individual descent plots: {e}")
    
    # Save if requested
    if save_html:
        fig.write_html(save_html, config=config)
        print(f"[PLOT] Saved descent trajectory to {save_html}")
    
    # Show in browser
    fig.show(config=config)
    
    return fig


# Helper function removed - subplot selection menu no longer needed


def plot_complete_mission_profile_interactive(climb_result: MinFuelSchedule,
                                             cruise_result: CruiseResults,
                                             descent_result: DescentResults,
                                             initial_mass_kg: float,
                                             save_html: Optional[str] = None):
    """
    Create interactive complete mission profile (climb + cruise + descent) using Plotly.
    
    Args:
        climb_result: Results from climb phase
        cruise_result: Results from cruise phase
        descent_result: Results from descent phase
        initial_mass_kg: Initial aircraft mass
        save_html: Optional path to save HTML file
    """
    # Calculate cumulative times for each phase
    climb_time = np.cumsum(climb_result.dt_s)
    cruise_time = climb_time[-1] + cruise_result.time_s
    descent_time = cruise_time[-1] + descent_result.time_s
    
    # Calculate cumulative fuel for each phase
    climb_fuel = climb_result.cumFuel_kg
    cruise_fuel = climb_fuel[-1] + cruise_result.fuel_consumed_kg
    descent_fuel = cruise_fuel[-1] + descent_result.cumFuel_kg
    
    # Convert times to minutes
    climb_time_min = climb_time / 60.0
    cruise_time_min = cruise_time / 60.0
    descent_time_min = descent_time / 60.0
    
    # Create subplots: 2 rows × 4 columns
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=(
            'Complete Mission Altitude Profile',
            'Mach Number Profile',
            'Aircraft Weight Profile',
            'Cumulative Fuel Consumption',
            'Fuel Flow Profile',
            'Thrust Profile',
            'Mission Statistics',
            ''  # Empty for stats
        ),
        specs=[
            [{"colspan": 4, "secondary_y": False}, None, None, None],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
        row_heights=[0.4, 0.6]
    )
    
    # 1. Complete Altitude Profile (spans full width)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_result.alt_m,
            mode='lines',
            name='Climb',
            line=dict(color='royalblue', width=3),
            hovertemplate='<b>Climb</b><br>Time: %{x:.2f} min<br>Altitude: %{y:.0f} m<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_result.altitude_m,
            mode='lines',
            name='Cruise',
            line=dict(color='green', width=3),
            hovertemplate='<b>Cruise</b><br>Time: %{x:.2f} min<br>Altitude: %{y:.0f} m<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_result.alt_m,
            mode='lines',
            name='Descent',
            line=dict(color='crimson', width=3),
            hovertemplate='<b>Descent</b><br>Time: %{x:.2f} min<br>Altitude: %{y:.0f} m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add phase transition markers
    fig.add_vline(x=climb_time_min[-1], line_dash="dash", line_color="gray", 
                 annotation_text="Climb→Cruise", row=1, col=1)
    fig.add_vline(x=cruise_time_min[-1], line_dash="dash", line_color="gray",
                 annotation_text="Cruise→Descent", row=1, col=1)
    
    # 2. Mach Profile
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_result.mach,
            mode='lines',
            name='Climb',
            line=dict(color='royalblue', width=2),
            showlegend=False,
            hovertemplate='<b>Climb</b><br>Time: %{x:.2f} min<br>Mach: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_result.mach_number,
            mode='lines',
            name='Cruise',
            line=dict(color='green', width=2),
            showlegend=False,
            hovertemplate='<b>Cruise</b><br>Time: %{x:.2f} min<br>Mach: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_result.mach,
            mode='lines',
            name='Descent',
            line=dict(color='crimson', width=2),
            showlegend=False,
            hovertemplate='<b>Descent</b><br>Time: %{x:.2f} min<br>Mach: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_vline(x=climb_time_min[-1], line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_vline(x=cruise_time_min[-1], line_dash="dash", line_color="gray", row=2, col=1)
    
    # 3. Weight Profile
    climb_weight = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic weight from DP optimization
    cruise_weight = cruise_result.weight_kg
    descent_weight = descent_result.weight_kg
    
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_weight,
            mode='lines',
            name='Climb',
            line=dict(color='royalblue', width=2),
            showlegend=False,
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.1)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.2f} min<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_weight,
            mode='lines',
            name='Cruise',
            line=dict(color='green', width=2),
            showlegend=False,
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.1)',
            hovertemplate='<b>Cruise</b><br>Time: %{x:.2f} min<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_weight,
            mode='lines',
            name='Descent',
            line=dict(color='crimson', width=2),
            showlegend=False,
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.1)',
            hovertemplate='<b>Descent</b><br>Time: %{x:.2f} min<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_vline(x=climb_time_min[-1], line_dash="dash", line_color="gray", row=2, col=2)
    fig.add_vline(x=cruise_time_min[-1], line_dash="dash", line_color="gray", row=2, col=2)
    
    # 4. Fuel Consumption
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_fuel,
            mode='lines',
            name='Climb',
            line=dict(color='royalblue', width=2),
            showlegend=False,
            hovertemplate='<b>Climb</b><br>Time: %{x:.2f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=cruise_time_min,
            y=cruise_fuel,
            mode='lines',
            name='Cruise',
            line=dict(color='green', width=2),
            showlegend=False,
            hovertemplate='<b>Cruise</b><br>Time: %{x:.2f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=descent_time_min,
            y=descent_fuel,
            mode='lines',
            name='Descent',
            line=dict(color='crimson', width=2),
            showlegend=False,
            hovertemplate='<b>Descent</b><br>Time: %{x:.2f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=2, col=3
    )
    
    fig.add_vline(x=climb_time_min[-1], line_dash="dash", line_color="gray", row=2, col=3)
    fig.add_vline(x=cruise_time_min[-1], line_dash="dash", line_color="gray", row=2, col=3)
    
    # 5. Mission Statistics Table
    total_time_min = descent_time_min[-1]
    total_fuel = descent_fuel[-1]
    final_weight = descent_weight[-1]
    
    climb_fuel_total = climb_fuel[-1]
    cruise_fuel_total = cruise_fuel[-1] - climb_fuel[-1]
    descent_fuel_total = descent_fuel[-1] - cruise_fuel[-1]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Phase</b>', '<b>Time (min)</b>', '<b>Fuel (kg)</b>'],
                fill_color='royalblue',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    ['<b>Climb</b>', '<b>Cruise</b>', '<b>Descent</b>', '<b>TOTAL</b>'],
                    [f'{climb_time_min[-1]:.1f}', 
                     f'{(cruise_time_min[-1] - climb_time_min[-1]):.1f}',
                     f'{(descent_time_min[-1] - cruise_time_min[-1]):.1f}',
                     f'<b>{total_time_min:.1f}</b>'],
                    [f'{climb_fuel_total:.1f}',
                     f'{cruise_fuel_total:.1f}',
                     f'{descent_fuel_total:.2f}',
                     f'<b>{total_fuel:.1f}</b>']
                ],
                fill_color=['white', 'lightgray'],
                align='center',
                font=dict(size=11),
                height=30
            )
        ),
        row=2, col=4
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (min)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1, gridcolor='lightgray')
    
    fig.update_xaxes(title_text="Time (min)", row=2, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time (min)", row=2, col=2, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time (min)", row=2, col=3, gridcolor='lightgray')
    
    fig.update_yaxes(title_text="Mach Number", row=2, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Weight (kg)", row=2, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text="Cumulative Fuel (kg)", row=2, col=3, gridcolor='lightgray')
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Complete Mission Profile (Climb + Cruise + Descent)</b><br>" +
                 f"<sup>Total Fuel: {total_fuel:.1f} kg ({total_fuel/initial_mass_kg*100:.1f}% of initial) | " +
                 f"Total Time: {total_time_min:.1f} min | " +
                 f"Initial Weight: {initial_mass_kg:.0f} kg | Final Weight: {final_weight:.0f} kg | " +
                 f"Descent: {descent_result.strategy_name}</sup>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=1100,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save if requested
    if save_html:
        fig.write_html(save_html)
        print(f"[PLOT] Saved complete mission profile to {save_html}")
    
    # Show in browser
    fig.show()
    
    return fig


def plot_descent_3d_trajectory(descent_result: DescentResults,
                              save_html: Optional[str] = None):
    """
    Create interactive 3D plot of descent trajectory (Mach-Altitude-Time) using Plotly.
    
    Args:
        descent_result: Results from descent simulation
        save_html: Optional path to save HTML file
    """
    # Create 3D trajectory plot
    time_normalized = (descent_result.time_s - descent_result.time_s[0]) / \
                     (descent_result.time_s[-1] - descent_result.time_s[0] + 1e-9)
    
    fig = go.Figure()
    
    # Add descent trajectory
    fig.add_trace(go.Scatter3d(
        x=descent_result.mach,
        y=descent_result.alt_m,
        z=descent_result.time_s / 60.0,
        mode='lines+markers',
        line=dict(
            color=time_normalized,
            colorscale='Viridis',
            width=6,
            colorbar=dict(title="Normalized<br>Time", x=1.1)
        ),
        marker=dict(size=4, color=time_normalized, colorscale='Viridis'),
        name='Descent Trajectory',
        hovertemplate='<b>Descent Trajectory</b><br>' +
                     'Mach: %{x:.3f}<br>' +
                     'Altitude: %{y:.0f} m<br>' +
                     'Time: %{z:.1f} min<br>' +
                     '<extra></extra>'
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[descent_result.mach[0]],
        y=[descent_result.alt_m[0]],
        z=[descent_result.time_s[0] / 60.0],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Start (Cruise End)',
        hovertemplate='<b>Start</b><br>' +
                     'Mach: %{x:.3f}<br>' +
                     'Altitude: %{y:.0f} m<br>' +
                     '<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[descent_result.mach[-1]],
        y=[descent_result.alt_m[-1]],
        z=[descent_result.time_s[-1] / 60.0],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='End (Landing)',
        hovertemplate='<b>End</b><br>' +
                     'Mach: %{x:.3f}<br>' +
                     'Altitude: %{y:.0f} m<br>' +
                     '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>3D Descent Trajectory - {descent_result.strategy_name}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='Mach Number',
            yaxis_title='Altitude (m)',
            zaxis_title='Time (min)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            xaxis=dict(gridcolor='lightgray', backgroundcolor='white'),
            yaxis=dict(gridcolor='lightgray', backgroundcolor='white'),
            zaxis=dict(gridcolor='lightgray', backgroundcolor='white')
        ),
        width=1000,
        height=800,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    # Save if requested
    if save_html:
        fig.write_html(save_html)
        print(f"[PLOT] Saved 3D descent trajectory to {save_html}")
    
    # Show in browser
    fig.show()
    
    return fig


def plot_descent_J_3d_plotly(M_grid: np.ndarray, H_sched: np.ndarray, 
                             lever_grid: np.ndarray, J_grid_3d: np.ndarray,
                             min_path: Optional[Dict] = None, 
                             title: Optional[str] = None,
                             initial_weight_kg: Optional[float] = None):
    """
    Visualize descent J values in 3D (Mach, Altitude, Lever) using Plotly.
    
    Similar to climb's plot_J_3d_plotly but adapted for descent phase.
    Shows the 3D cost landscape and optimal descent path with flight envelope limits.
    
    Args:
        M_grid: Mach number grid
        H_sched: Altitude schedule (descending from high to low)
        lever_grid: Lever position grid
        J_grid_3d: 3D array of J values (shape: len(M_grid) × len(H_sched) × len(lever_grid))
        min_path: Optional dict with keys 'mach', 'alt', 'lever' for optimal path overlay
        title: Optional custom title
        initial_weight_kg: Initial weight for stall speed calculation
    """
    if initial_weight_kg is None:
        initial_weight_kg = INITIAL_MASS_KG
    
    # Prepare meshgrid for scatter
    M, H, L = np.meshgrid(M_grid, H_sched, lever_grid, indexing='ij')
    
    # Flatten arrays for plotting - NEW AXIS ASSIGNMENT (same as climb)
    x = L.flatten()  # Lever (X-axis)
    y = M.flatten()  # Mach (Y-axis) 
    z = H.flatten()  # Altitude (Z-axis)
    J_flat = J_grid_3d.flatten()
    
    # Only plot finite J values
    mask = np.isfinite(J_flat)
    x, y, z, J_flat = x[mask], y[mask], z[mask], J_flat[mask]
    
    # Enhanced hover text with more detailed information
    hover_text = []
    for lx, my, hz, jv in zip(x, y, z, J_flat):
        # Calculate additional metrics for hover
        _, _, rho = isa_properties(float(hz))
        a = a_from_altitude(float(hz))
        V = my * a
        hover_text.append(
            f"<b>Point Details</b><br>"
            f"Lever: {lx:.3f}<br>"
            f"Mach: {my:.3f}<br>"
            f"Altitude: {hz:.1f} m<br>"
            f"J: {jv:.4g} kg/m<br>"
            f"Velocity: {V:.1f} m/s<br>"
            f"Air Density: {rho:.3f} kg/m³"
        )

    fig = go.Figure()
    
    # Add flight envelope limits
    # 1. MMO (Maximum Mach Operating) limit - vertical plane at M_MMO
    lever_range = np.linspace(0, 1.0, 10)  # Full lever range for envelope visualization
    alt_range = np.linspace(H_sched[-1], H_sched[0], 10)  # From low to high
    L_mmo, H_mmo = np.meshgrid(lever_range, alt_range)
    M_mmo = np.full_like(L_mmo, M_MMO)
    
    fig.add_trace(go.Surface(
        x=L_mmo,
        y=M_mmo,
        z=H_mmo,
        colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
        opacity=0.45,
        showscale=False,
        name=f'MMO Limit (M={M_MMO:.2f})'
    ))
    
    # 2. CLmax (stall) limit - compute stall curve for descent altitudes
    def _compute_mstall_curve_descent():
        W = initial_weight_kg * G_C
        # Use CL_MAX from aircraft_config (set by PyAerodynamicsWrapper), fallback to 1.4 if not set
        cl_max_value = CL_MAX if CL_MAX is not None else 1.4
        out = np.full_like(H_sched, np.nan, float)
        for k, h in enumerate(H_sched):
            _, _, rho = isa_properties(float(h))
            a = a_from_altitude(float(h))
            if cl_max_value > 0:
                q_req = W / (S_REF_M2 * cl_max_value)
                if rho > 0:
                    V = np.sqrt(2*q_req/max(rho,1e-12))
                    out[k] = V / max(a,1e-12)
            else:
                # Use dynamic calculation if CL_MAX not available
                out[k] = calculate_min_descent_mach(float(h), initial_weight_kg)
        return out
    
    M_stall = _compute_mstall_curve_descent()
    if np.isfinite(M_stall).any():
        # Create stall surface - vertical plane at stall Mach
        lever_range = np.linspace(0, 1.0, 10)  # Full lever range
        alt_range = np.linspace(H_sched[-1], H_sched[0], 10)
        L_stall, H_stall = np.meshgrid(lever_range, alt_range)
        M_stall_surface = np.full_like(L_stall, np.nan)
        
        # Fill in stall Mach values where they exist
        for i, h in enumerate(alt_range):
            h_idx = np.argmin(np.abs(H_sched - h))
            if h_idx < len(M_stall) and np.isfinite(M_stall[h_idx]):
                M_stall_surface[i, :] = M_stall[h_idx]
        
        # Only plot where we have valid stall data
        valid_mask = np.isfinite(M_stall_surface)
        if np.any(valid_mask):
            fig.add_trace(go.Surface(
                x=L_stall,
                y=M_stall_surface,
                z=H_stall,
                colorscale=[[0, Colors.ENVELOPE_LIMIT], [1, Colors.ENVELOPE_LIMIT]],
                opacity=0.45,
                showscale=False,
                name='Flight Envelope Limit (Stall)'
            ))
    
    # 3. Operating envelope boundaries (between stall and MMO) - as lines
    if np.isfinite(M_stall).any():
        cond = np.isfinite(M_stall) & (M_stall < M_MMO)
        if np.any(cond):
            # Create operating envelope boundary lines at different lever positions
            for lever_val in [0.0, 0.5, 1.0]:  # Show at different lever positions (full range)
                # Stall edge line
                fig.add_trace(go.Scatter3d(
                    x=[lever_val] * int(np.sum(cond)),
                    y=M_stall[cond],
                    z=H_sched[cond],
                    mode='lines',
                    line=dict(color=Colors.ENVELOPE_LIMIT, width=LineStyles.THICK),
                    name='Operating Envelope' if lever_val == 0.0 else None,
                    showlegend=(lever_val == 0.0)
                ))
                # MMO edge line
                fig.add_trace(go.Scatter3d(
                    x=[lever_val] * int(np.sum(cond)),
                    y=(M_MMO * np.ones_like(M_stall[cond])),
                    z=H_sched[cond],
                    mode='lines',
                    line=dict(color=Colors.ENVELOPE_LIMIT, width=LineStyles.THICK),
                    showlegend=False
                ))
    
    # Add J values scatter plot (orange/red for descent)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=Colors.J_VALUES_DESCENT,
            opacity=0.7,
            line=dict(width=0)
        ),
        name='J values (Descent)',
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))

    # Overlay minimum-fuel path if provided
    if min_path is not None:
        # min_path should be a dict with keys: 'mach', 'alt', 'lever'
        fig.add_trace(go.Scatter3d(
            x=min_path['lever'],  # Lever on X-axis
            y=min_path['mach'],   # Mach on Y-axis
            z=min_path['alt'],    # Altitude on Z-axis
            mode='lines+markers',
            line=dict(color=Colors.OPTIMAL_PATH_DESCENT, width=6),
            marker=dict(size=6, color=Colors.OPTIMAL_PATH_DESCENT),
            name='Optimal Descent Path',
            hovertemplate='<b>Optimal Path</b><br>Lever: %{x:.3f}<br>Mach: %{y:.3f}<br>Alt: %{z:.1f} m<extra></extra>'
        ))

    # Enhanced layout with user-friendly features (same as climb)
    fig.update_layout(
        scene=dict(
            xaxis_title='Lever',
            yaxis_title='Mach',
            zaxis_title='Altitude (m)',
            # Camera presets
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Default isometric view
                center=dict(x=0, y=0, z=0)
            ),
            # Enhanced axis formatting using standard config
            xaxis=dict(
                title=dict(font=dict(size=Typography.AXIS_LABEL_SIZE)),
                tickfont=dict(size=Typography.AXIS_TICK_SIZE),
                gridcolor='rgba(128,128,128,0.3)',
                range=[0, 1.05]  # Full lever range (same as climb)
            ),
            yaxis=dict(
                title=dict(font=dict(size=Typography.AXIS_LABEL_SIZE)),
                tickfont=dict(size=Typography.AXIS_TICK_SIZE),
                gridcolor='rgba(128,128,128,0.3)'
            ),
            zaxis=dict(
                title=dict(font=dict(size=Typography.AXIS_LABEL_SIZE)),
                tickfont=dict(size=Typography.AXIS_TICK_SIZE),
                gridcolor='rgba(128,128,128,0.3)'
            )
        ),
        title=dict(
            text=title if title is not None else '3D Visualization of Descent J (Fuel/Energy)',
            font=dict(size=Typography.MAIN_TITLE_SIZE, family=Typography.FONT_FAMILY),
            x=0.5
        ),
        # Add control buttons (same as climb)
        updatemenus=[
            # Camera preset buttons
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}],
                        label="Front View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0}}],
                        label="Side View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 2.5, "z": 0}}],
                        label="Top View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.5}}],
                        label="Isometric",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
            # Data filtering buttons
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True] * len(fig.data)}],
                        label="Show All",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and "J values" in trace.name else False for trace in fig.data]}],
                        label="J Values Only",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and "Optimal" in trace.name else False for trace in fig.data]}],
                        label="Optimal Path Only",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and ("MMO Limit" in trace.name or "Flight Envelope" in trace.name or "Operating Envelope" in trace.name) else False for trace in fig.data]}],
                        label="Envelope Only",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.25,
                xanchor="left",
                y=1.02,
                yanchor="top"
            )
        ],
        # Enhanced legend
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=Typography.LEGEND_SIZE)
        ),
        # Add margin for controls
        margin=dict(l=0, r=Layout.MARGIN_RIGHT, t=Layout.MARGIN_TOP, b=Layout.MARGIN_BOTTOM)
    )

    # Set template
    fig.update_layout(template="plotly_white")

    # Write to HTML file with enhanced features using standard export config
    export_config = ExportConfig.get_plotly_config()
    export_config['toImageButtonOptions']['filename'] = 'descent_3d_plot_export'
    export_config['modeBarButtonsToAdd'] = [
        "drawline", "drawopenpath", "drawclosedpath",
        "drawcircle", "drawrect", "eraseshape"
    ]
    
    # Save to Descent folder (both HTML for interaction and PNG for static)
    descent_dir = get_or_create_run_directory(phase="Descent")
    output_path_html = os.path.join(descent_dir, 'descent_J_3d_plot.html')
    output_path_png = os.path.join(descent_dir, 'descent_J_3d_plot.png')
    
    pio.write_html(
        fig, 
        file=output_path_html, 
        auto_open=True,
        config=export_config
    )
    
    # Also save as PNG
    try:
        fig.write_image(output_path_png, width=1600, height=1200, scale=2)
        print(f"[EXPORT] Descent 3D J plot saved to: {output_path_html} (interactive) and {output_path_png} (PNG)")
    except Exception as e:
        print(f"[EXPORT] Descent 3D J plot saved to: {output_path_html} (HTML only)")
        print(f"[WARNING] Could not save PNG version: {e}")
    
    return fig
