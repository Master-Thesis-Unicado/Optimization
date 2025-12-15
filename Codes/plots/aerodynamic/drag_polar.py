"""Plotly drag polar visualizations for aerodynamic model.

Generates interactive HTML and high-DPI PNG figures for:
1) Drag polar (C_D vs C_L) at multiple aircraft masses
2) Demonstrates C_D = C_{D,0} + C_{D,i}(C_L) relationship
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Repository root insertion for imports
ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "plots" / "aerodynamic"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircraft_config import W_TO_KG, W_OE_KG, W_FUEL_KG  # noqa: E402
from pyaerodynamics_wrapper import PyAerodynamicsWrapper  # noqa: E402


def base_layout(title: str) -> dict:
    """Standardized academic layout configuration."""
    return dict(
        title=dict(text=title, x=0.5, y=0.97),
        template="plotly_white",
        font=dict(family="Times New Roman", size=14),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=40, t=90, b=110),
        width=1100,
        height=720,
    )


def export_figure(fig: go.Figure, filename_stem: str) -> None:
    """Save HTML and PNG outputs to plots directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = PLOTS_DIR / f"{filename_stem}.html"
    png_path = PLOTS_DIR / f"{filename_stem}.png"

    pio.write_html(fig, file=html_path, auto_open=False, include_plotlyjs="cdn")
    try:
        fig.write_image(png_path, width=1400, height=900, scale=2)
    except Exception as exc:
        print(f"[WARNING] PNG export failed for {filename_stem}: {exc}")


def plot_drag_polar(
    aero: PyAerodynamicsWrapper,
    mach: float,
    altitude_m: float,
    mass_values: Sequence[float],
) -> go.Figure:
    """Plot drag polar C_L vs C_D using direct aerodynamic wrapper data."""
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig = go.Figure()
    
    for idx, mass_kg in enumerate(mass_values):
        # Generate CL/CD pairs by systematically varying flight conditions
        # Use wrapper directly without assumptions
        cl_values = []
        cd_values = []
        ld_values = []
        
        # Vary Mach at fixed altitude to get different trim conditions
        mach_range = np.linspace(0.3, 0.85, 80)
        
        for m in mach_range:
            try:
                aero_data = aero.get_comprehensive_aerodynamics(m, altitude_m, mass_kg)
                if aero_data and 'cl' in aero_data and 'cd' in aero_data:
                    cl = aero_data['cl']
                    cd = aero_data['cd']
                    if cl > 0 and cd > 0:
                        cl_values.append(cl)
                        cd_values.append(cd)
                        if 'ld' in aero_data and aero_data['ld'] is not None:
                            ld_values.append(aero_data['ld'])
                        else:
                            ld_values.append(cl / cd if cd > 0 else 0)
            except Exception:
                continue
        
        if cl_values and cd_values:
            # Sort by CD for smooth curve
            sorted_pairs = sorted(zip(cd_values, cl_values, ld_values))
            cd_sorted, cl_sorted, ld_sorted = zip(*sorted_pairs)
            
            # Convert to arrays for analysis
            cd_array = np.array(cd_sorted)
            cl_array = np.array(cl_sorted)
            ld_array = np.array(ld_sorted)
            
            # Main drag polar curve
            fig.add_trace(
                go.Scatter(
                    x=cd_array,
                    y=cl_array,
                    mode="lines",
                    name=f"m = {mass_kg/1000:.0f} t",
                    line=dict(color=colors[idx % len(colors)], width=3),
                    showlegend=False,
                )
            )
            
            # Find key points from actual data (no extrapolation)
            if len(cd_array) > 0:
                # C_D_MIN: minimum drag coefficient from data
                cd_min_idx = np.argmin(cd_array)
                cd_min = cd_array[cd_min_idx]
                cl_mindrag = cl_array[cd_min_idx]
                
                # (C_L/C_D)_MAX: maximum lift-to-drag ratio from data
                ld_max_idx = np.argmax(ld_array)
                ld_max = ld_array[ld_max_idx]
                cl_ldmax = cl_array[ld_max_idx]
                cd_ldmax = cd_array[ld_max_idx]
                
                # C_D_MIN point
                fig.add_trace(
                    go.Scatter(
                        x=[cd_min],
                        y=[cl_mindrag],
                        mode="markers",
                        marker=dict(size=10, color="orange", symbol="circle"),
                        name="C_D_MIN",
                        showlegend=False,
                    )
                )
                
                # C_D_MIN vertical line
                fig.add_trace(
                    go.Scatter(
                        x=[cd_min, cd_min],
                        y=[0, max(cl_array) * 1.1],
                        mode="lines",
                        line=dict(color="red", width=2, dash="dash"),
                        name="C_D_MIN line",
                        showlegend=False,
                    )
                )
                
                # (C_L/C_D)_MAX point
                fig.add_trace(
                    go.Scatter(
                        x=[cd_ldmax],
                        y=[cl_ldmax],
                        mode="markers",
                        marker=dict(size=10, color="blue", symbol="circle"),
                        name="(C_L/C_D)_MAX",
                        showlegend=False,
                    )
                )
                
                # (C_L/C_D)_MAX tangent line from origin
                if cd_ldmax > 0 and ld_max > 0:
                    cl_tangent = np.linspace(0, cl_ldmax * 1.2, 10)
                    cd_tangent = cl_tangent / ld_max
                    fig.add_trace(
                        go.Scatter(
                            x=cd_tangent,
                            y=cl_tangent,
                            mode="lines",
                            line=dict(color="orange", width=2, dash="dot"),
                            name="(C_L/C_D)_MAX line",
                            showlegend=False,
                        )
                    )
    
    fig.update_layout(
        **base_layout(f"Drag Polar | m = {mass_values[0]/1000:.0f} t"),
        xaxis=dict(title="Drag coefficient C_D [-]", range=[0, None], showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Lift coefficient C_L [-]", rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def main(save: bool = True, show: bool = False) -> None:
    aero = PyAerodynamicsWrapper()
    
    # Single mass: 75t
    mass_values = [
        75000.0,  # 75 t
    ]
    
    # Fixed flight condition for polar
    mach = 0.78
    altitude_m = 10000.0
    
    fig_polar = plot_drag_polar(aero, mach, altitude_m, mass_values)
    
    if save:
        export_figure(fig_polar, "drag_polar")
    
    if show:
        fig_polar.show()


if __name__ == "__main__":
    main()
