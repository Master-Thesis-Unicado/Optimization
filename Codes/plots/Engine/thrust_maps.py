"""Plotly thrust performance visualizations for PW1127G-JM.

Generates interactive HTML and high-DPI PNG figures for:
1) Thrust vs altitude at multiple Mach numbers (δ = 1.00)
2) Thrust vs Mach number at multiple altitudes (δ = 0.85)
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
PLOTS_DIR = ROOT / "plots" / "Engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircraft_config import LEVER_MAX, M_MMO  # noqa: E402
from pyengine_wrapper import EngineWrapper  # noqa: E402


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


def thrust_map_altitude(
    engine: EngineWrapper,
    altitudes_m: Sequence[float],
    mach_values: Sequence[float],
    lever: float,
) -> go.Figure:
    """Plot thrust vs altitude at several Mach numbers."""
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2"]
    fig = go.Figure()
    for idx, m in enumerate(mach_values):
        thrust = []
        for h in altitudes_m:
            val = engine.thrust_with_lever(lever, m, h)
            thrust.append(val if val is not None else np.nan)
        fig.add_trace(
            go.Scatter(
                x=np.array(altitudes_m) / 1000.0,
                y=np.array(thrust) / 1000.0,
                mode="lines",
                name=f"M = {m:.2f}",
                line=dict(color=colors[idx % len(colors)], width=3),
            )
        )

    fig.update_layout(
        **base_layout(f"Thrust vs Altitude | Lever δ = {lever:.2f}"),
        xaxis=dict(title="Altitude [km]", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Per-engine thrust [kN]", rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def thrust_map_mach(
    engine: EngineWrapper,
    mach_grid: Sequence[float],
    altitudes_m: Sequence[float],
    lever: float,
) -> go.Figure:
    """Plot thrust vs Mach number at several altitudes."""
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig = go.Figure()
    for idx, h in enumerate(altitudes_m):
        thrust = []
        for m in mach_grid:
            val = engine.thrust_with_lever(lever, m, h)
            thrust.append(val if val is not None else np.nan)
        fig.add_trace(
            go.Scatter(
                x=mach_grid,
                y=np.array(thrust) / 1000.0,
                mode="lines",
                name=f"h = {h/1000:.0f} km",
                line=dict(color=colors[idx % len(colors)], width=3),
            )
        )

    fig.update_layout(
        **base_layout(f"Thrust vs Mach | Lever δ = {lever:.2f}"),
        xaxis=dict(title="Mach number [-]", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Per-engine thrust [kN]", rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def main(save: bool = True, show: bool = False) -> None:
    engine = EngineWrapper()

    # Altitude sweep at multiple Mach values (full power)
    altitudes = np.linspace(0.0, 12000.0, 13)
    mach_values = np.linspace(0.2, min(M_MMO, 0.86), 8)
    fig_alt = thrust_map_altitude(engine, altitudes, mach_values, LEVER_MAX)

    # Mach sweep at multiple altitudes (climb/cruise lever)
    mach_grid = np.linspace(0.2, min(M_MMO, 0.90), 28)
    altitude_slices = [0.0, 3000.0, 6000.0, 9000.0, 12000.0]
    fig_mach = thrust_map_mach(engine, mach_grid, altitude_slices, lever=0.85)

    if save:
        export_figure(fig_alt, "thrust_vs_altitude")
        export_figure(fig_mach, "thrust_vs_mach")

    if show:
        fig_alt.show()
        fig_mach.show()


if __name__ == "__main__":
    main()
