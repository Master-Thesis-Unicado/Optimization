"""Plotly drag coefficient visualizations for aerodynamic model.

Generates interactive HTML and high-DPI PNG figures for:
1) Drag coefficient vs altitude at multiple Mach numbers
2) Drag coefficient vs Mach number at multiple altitudes
3) Demonstrates mass-dependent drag characteristics
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

from aircraft_config import W_TO_KG, M_MMO  # noqa: E402
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


def plot_cd_vs_altitude(
    aero: PyAerodynamicsWrapper,
    altitudes_m: Sequence[float],
    mach_values: Sequence[float],
    mass_kg: float,
) -> go.Figure:
    """Plot drag coefficient vs altitude at several Mach numbers."""
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2"]
    fig = go.Figure()
    
    for idx, m in enumerate(mach_values):
        cd_values = []
        for h in altitudes_m:
            try:
                aero_data = aero.get_comprehensive_aerodynamics(m, h, mass_kg)
                if aero_data and 'cd' in aero_data:
                    cd_values.append(aero_data['cd'])
                else:
                    cd_values.append(np.nan)
            except Exception:
                cd_values.append(np.nan)
        
        fig.add_trace(
            go.Scatter(
                x=np.array(altitudes_m) / 1000.0,
                y=cd_values,
                mode="lines",
                name=f"M = {m:.2f}",
                line=dict(color=colors[idx % len(colors)], width=3),
            )
        )
    
    fig.update_layout(
        **base_layout(f"Drag Coefficient vs Altitude | m = {mass_kg/1000:.1f} t"),
        xaxis=dict(title="Altitude [km]", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Drag coefficient C_D [-]", rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def plot_cd_vs_mach(
    aero: PyAerodynamicsWrapper,
    mach_grid: Sequence[float],
    altitudes_m: Sequence[float],
    mass_kg: float,
) -> go.Figure:
    """Plot drag coefficient vs Mach number at several altitudes."""
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig = go.Figure()
    
    for idx, h in enumerate(altitudes_m):
        cd_values = []
        for m in mach_grid:
            try:
                aero_data = aero.get_comprehensive_aerodynamics(m, h, mass_kg)
                if aero_data and 'cd' in aero_data:
                    cd_values.append(aero_data['cd'])
                else:
                    cd_values.append(np.nan)
            except Exception:
                cd_values.append(np.nan)
        
        fig.add_trace(
            go.Scatter(
                x=mach_grid,
                y=cd_values,
                mode="lines",
                name=f"h = {h/1000:.0f} km",
                line=dict(color=colors[idx % len(colors)], width=3),
            )
        )
    
    fig.update_layout(
        **base_layout(f"Drag Coefficient vs Mach | m = {mass_kg/1000:.1f} t"),
        xaxis=dict(title="Mach number [-]", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Drag coefficient C_D [-]", rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def main(save: bool = True, show: bool = False) -> None:
    aero = PyAerodynamicsWrapper()
    
    # Use takeoff mass for representative analysis
    mass_kg = W_TO_KG
    
    # Drag coefficient vs altitude at multiple Mach values
    altitudes = np.linspace(0.0, 12000.0, 13)  # 0–12 km
    mach_values = np.linspace(0.5, min(M_MMO, 0.85), 6)
    fig_alt = plot_cd_vs_altitude(aero, altitudes, mach_values, mass_kg)
    
    # Drag coefficient vs Mach at multiple altitudes
    mach_grid = np.linspace(0.3, min(M_MMO, 0.90), 25)
    altitude_slices = [0.0, 3000.0, 6000.0, 9000.0, 12000.0]
    fig_mach = plot_cd_vs_mach(aero, mach_grid, altitude_slices, mass_kg)
    
    if save:
        export_figure(fig_alt, "cd_vs_altitude")
        export_figure(fig_mach, "cd_vs_mach")
    
    if show:
        fig_alt.show()
        fig_mach.show()


if __name__ == "__main__":
    main()
