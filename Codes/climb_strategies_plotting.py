# =========================================================================
# CLIMB STRATEGY COMPARISON PLOTTING MODULE
# =========================================================================
"""
Independent module for climb strategy comparison visualization.

This module provides plotting and visualization capabilities for climb strategy
comparison analysis. Functions include interactive SEP diagrams, comparison
plots, and browser-based tables.

The strategy comparison visualization can be enabled/disabled via the
ENABLE_STRATEGY_COMPARISON flag in mission_config.py.

Key Functions:
- plot_strategies_interactive: Interactive SEP diagram with strategy paths
- create_strategy_comparison_plots: Comparison plots for multiple strategies
- create_browser_comparison_table: Browser-based comparison table
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import patheffects as pe
from typing import List
import os
import warnings
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

# Suppress choreographer JSONError warnings (non-critical browser communication errors)
warnings.filterwarnings('ignore', category=UserWarning, module='choreographer')
try:
    import logging
    logging.getLogger('choreographer').setLevel(logging.ERROR)
except:
    pass

pio.renderers.default = "browser"

# Import needed types, helpers, and constants
from aircraft_config import (
    isa_properties, a_from_altitude, INITIAL_MASS_KG,
    M_MMO, S_REF_M2, CL_MAX
)
from climb import ClimbingCore, StrategyRun
from climb_plotting import GridConfig, PlottingConfig

from visualization_config import (
    get_or_create_run_directory
)


# =========================================================================
# INTERACTIVE STRATEGY PLOTTING
# =========================================================================
def plot_strategies_interactive(
    mach_grid, H_plot, Ps_base,
    strategies_runs: List[StrategyRun],
    *,
    title_suffix=""
):
    """
    Create interactive SEP diagram with multiple strategy paths.
    
    Provides step-by-step visualization of climb strategies with:
    - Specific Excess Power (Ps) contours
    - Flight envelope boundaries (CLmax, MMO, service ceiling)
    - Interactive step navigation
    - Strategy switching
    - Detailed state information at each point
    
    Args:
        mach_grid: Mach number grid for contour plots
        H_plot: Altitude grid for contour plots
        Ps_base: Specific excess power grid data
        strategies_runs: List of StrategyRun objects to visualize
        title_suffix: Optional title suffix for context
        
    Returns:
        Figure object with interactive controls
    """
    plt.style.use("classic")
    fig = plt.figure(figsize=(16, 10), dpi=110, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[3.5, 2.0, 1.8], width_ratios=[2.7, 2.0])

    ax_left = fig.add_subplot(gs[:, 0])   # SEP + path
    ax_time = fig.add_subplot(gs[0, 1])   # Altitude vs time/step
    ax_lever = fig.add_subplot(gs[1, 1])  # Lever vs time/step
    ax_text = fig.add_subplot(gs[2, 1])   # Info box
    ax_text.axis("off")

    # --- Buttons (at bottom of text panel to avoid overlap) ---
    fig.canvas.draw_idle(); plt.pause(0.001)
    panel = ax_text.get_position()
    L = panel.x0 + 0.03 * panel.width
    R = panel.x1 - 0.03 * panel.width
    B = panel.y0 + 0.02 * panel.height  # Small margin from bottom of text panel
    H = 0.12 * panel.height  # Reasonable button height within panel bounds
    Wavail = max(1e-6, R - L)
    gap = 0.02 * Wavail
    bw = min(0.23, (Wavail - 3*gap) / 4)  # 4 buttons

    x0=L; x1=L+bw+gap; x2=x1+bw+gap; x3=x2+bw+gap
    ax_prev_step = fig.add_axes([x0, B, bw, H])
    ax_next_step = fig.add_axes([x1, B, bw, H])
    ax_prev_str  = fig.add_axes([x2, B, bw, H])
    ax_next_str  = fig.add_axes([x3, B, bw, H])

    btn_prev_step = Button(ax_prev_step, "◀ Step")
    btn_next_step = Button(ax_next_step, "Step ▶")
    btn_prev_str  = Button(ax_prev_str,  "◀ Strategy")
    btn_next_str  = Button(ax_next_str,  "Strategy ▶")
    for b in (btn_prev_step, btn_next_step, btn_prev_str, btn_next_str):
        b.label.set_fontsize(10)

    # --- Keep references to avoid GC (CRITICAL) ---
    fig._controls = {
        "axes": (ax_prev_step, ax_next_step, ax_prev_str, ax_next_str),
        "buttons": (btn_prev_step, btn_next_step, btn_prev_str, btn_next_str),
    }

    # --- Background Es contours (light) ---
    Hm, Mm = np.meshgrid(H_plot, mach_grid, indexing="ij")
    # Get gravity constant from Atmosphere class (strategies_runs don't have aero directly)
    from atmosphere import Atmosphere
    g = Atmosphere.G_C
    Es = np.zeros_like(Ps_base)
    for k, h in enumerate(H_plot):
        a = a_from_altitude(float(h))
        V = a * np.asarray(mach_grid, float)
        Es[k, :] = 0.5 * V * V + g * h
    Es_levels = np.linspace(np.nanmin(Es), np.nanmax(Es), 8)
    ax_left.contour(Mm, Hm, Es, levels=Es_levels, colors=[(0.6, 0.6, 0.6)], linewidths=0.8, zorder=0)

    # Engine envelope limits (from engine envelope analysis)
    # Maximum service ceiling altitude: 13994.1 m at lever=1.0, Mach=0.900
    MAX_SERVICE_CEILING_M = 13994.1
    # Maximum operational Mach from engine envelope: 0.9392 at lever=1.0, altitude=10500 m
    MAX_ENGINE_MACH = 0.9392
    # Minimum operational Mach from engine envelope test: 0.200 (tested range)
    MIN_ENGINE_MACH = 0.200
    
    # CLmax + Engine Mach limits + operating fill
    def _compute_mstall_curve():
        W = INITIAL_MASS_KG * g
        # Use CL_MAX from aircraft_config (fixed value)
        cl_max_value = CL_MAX
        out = np.full_like(H_plot, np.nan, float)
        for k, h in enumerate(H_plot):
            _, _, rho = isa_properties(float(h)); a = a_from_altitude(float(h))
            q_req = W / (S_REF_M2 * cl_max_value)
            if rho > 0:
                V = np.sqrt(2*q_req/max(rho,1e-12))
                out[k] = V / max(a,1e-12)
        return out
    M_stall = _compute_mstall_curve()
    ax_left.axvline(MAX_ENGINE_MACH, color="red", linewidth=2.0, linestyle="--", label=f"Max Engine Mach={MAX_ENGINE_MACH:.3f}", zorder=2)
    if np.isfinite(M_stall).any():
        ax_left.plot(M_stall, H_plot, linestyle="--", color="red", linewidth=2.0, label="CLmax", zorder=2)
        cond = np.isfinite(M_stall) & (M_stall < MAX_ENGINE_MACH)
        if np.any(cond):
            ax_left.fill_betweenx(H_plot[cond], M_stall[cond], MAX_ENGINE_MACH, alpha=0.12, color="tab:green", zorder=1)
    
    # Add maximum service ceiling limit (horizontal line)
    # Always show if within reasonable range (even if slightly above GridConfig.Y_AXIS_TOP_M for visibility)
    if MAX_SERVICE_CEILING_M <= GridConfig.Y_AXIS_TOP_M + 1000:  # Allow slight margin
        ax_left.axhline(MAX_SERVICE_CEILING_M, color="red", linewidth=2.0, linestyle="--", 
                        label=f"Max Service Ceiling={MAX_SERVICE_CEILING_M/1000:.2f} km", zorder=2)
    
    # Add minimum engine Mach limit (vertical line)
    if MIN_ENGINE_MACH >= mach_grid[0]:
        ax_left.axvline(MIN_ENGINE_MACH, color="red", linewidth=2.0, linestyle="--", 
                       label=f"Min Engine Mach={MIN_ENGINE_MACH:.3f}", zorder=2)

    # Ps contours
    neg = [lv for lv in PlottingConfig.PS_LEVELS if lv < 0]; pos = [lv for lv in PlottingConfig.PS_LEVELS if lv > 0]
    if neg:
        cs_neg = ax_left.contour(Mm, Hm, Ps_base, levels=neg, colors="0.55", linestyles="--", linewidths=1.1, zorder=4)
        for t in ax_left.clabel(cs_neg, fmt=lambda v: f"Ps={v:g}", inline=True, fontsize=8):
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    if pos:
        cs_pos = ax_left.contour(Mm, Hm, Ps_base, levels=pos, colors="k", linewidths=1.2, zorder=4)
        for t in ax_left.clabel(cs_pos, fmt=lambda v: f"Ps={v:g}", inline=True, fontsize=8):
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    try:
        cs0 = ax_left.contour(Mm, Hm, Ps_base, levels=[0.0], colors="k", linestyles="--", linewidths=2.0, zorder=5)
        ax_left.clabel(cs0, fmt={0.0: "Ps=0"}, inline=True, fontsize=9)
    except Exception:
        pass

    # Axes cosmetics (cap at Mach 1.25)
    ax_left.set_xlabel("Mach No.")
    ax_left.set_ylabel("Altitude [m]")
    ax_left.set_xlim(mach_grid[0] - 0.02, float(PlottingConfig.M_XMAX_UI))
    ax_left.set_ylim(H_plot[0] - 200, GridConfig.Y_AXIS_TOP_M + 200)
    ax_left.set_aspect("auto", adjustable="box")
    ax_left.set_autoscale_on(False)
    ax_left.margins(x=0, y=0)

    # ---------- State & initial data ----------
    if not strategies_runs:
        raise ValueError("strategies_runs is empty.")
    s = 0           # strategy index
    k = 0           # step index

    def _coerce_run(run: StrategyRun):
        return (run.label, run.alt_m, run.mach, run.lever, run.time_s, run.T_total_N,
                run.D_N, run.Ps_mps, run.mdot_kgps, run.dt_s, run.dFuel_kg, run.cumFuel_kg, run.thrust_limited)

    (label, alt_m, mach, lever, time_s, Ttot, D, Ps, mdot, dt_s, dFuel, cumFuel, limited) = _coerce_run(strategies_runs[s])
    K = len(alt_m)

    # Left path & dot
    path_line, = ax_left.plot(mach, alt_m, color="k", linewidth=2.4, zorder=6, label=label)
    dot_left,  = ax_left.plot([mach[k]], [alt_m[k]], 'o', markersize=8, color='red', zorder=7)

    # Right top: altitude vs time (or step)
    if (dt_s is None) or (not np.isfinite(dt_s).any()):
        t_vec = np.arange(K, dtype=float)
        ax_time.set_xlabel("Step")
    else:
        t_vec = time_s
        ax_time.set_xlabel("Time [s]")
    line_time, = ax_time.plot(t_vec, alt_m, linewidth=1.8)
    dot_time,  = ax_time.plot([t_vec[k]], [alt_m[k]], 'o', color='red')
    ax_time.set_ylabel("Altitude [m]")
    ax_time.grid(True, linestyle=":", linewidth=0.6)
    ax_time.set_ylim(H_plot[0] - 200, GridConfig.Y_AXIS_TOP_M + 200)

    # Right middle: lever vs time (or step)
    line_lever, = ax_lever.plot(t_vec, lever, linewidth=1.8, color='tab:orange')
    dot_lever,  = ax_lever.plot([t_vec[k]], [lever[k]], 'o', color='red')
    ax_lever.set_ylabel("Lever Position")
    ax_lever.set_xlabel("Time [s]" if ((dt_s is not None) and np.isfinite(dt_s).any()) else "Step")
    ax_lever.grid(True, linestyle=":", linewidth=0.6)
    ax_lever.set_ylim(-0.05, 1.05)  # Lever ranges from 0 to 1

    def _update_time_limits():
        if len(t_vec) and np.isfinite(t_vec).any():
            tmax = float(np.nanmax(t_vec))
            ax_time.set_xlim(-0.02 * tmax, 1.02 * tmax)
            ax_lever.set_xlim(-0.02 * tmax, 1.02 * tmax)
        else:
            ax_time.set_xlim(0, max(1.0, K - 1))
            ax_lever.set_xlim(0, max(1.0, K - 1))

    _update_time_limits()

    # Bottom-right: text panel (positioned higher to give more space for buttons)
    info = ax_text.text(0.02, 0.99, "", va="top", ha="left", family="monospace")

    def _format_info():
        tsfc = (mdot[k] / Ttot[k]) if (np.isfinite(mdot[k]) and np.isfinite(Ttot[k]) and Ttot[k] > 0) else np.nan
        J = (mdot[k] / Ps[k]) if (np.isfinite(mdot[k]) and np.isfinite(Ps[k]) and Ps[k] > 0) else np.nan
        t_here = t_vec[k] if (len(t_vec) and k < len(t_vec)) else float(k)
        return (f"{label}\n"
                f"Step {k+1} / {K}\n"
                f"Alt: {alt_m[k]:.1f} m ({alt_m[k]*3.28084:.0f} ft)\n"
                f"Mach: {mach[k]:.3f}   Lever: {lever[k]:.3f}   Limited: {bool(limited[k])}\n"
                f"ṁ: {mdot[k]:.3f} kg/s   J: {J:.4f} kg/m   Ps: {Ps[k]:.3f} m/s\n"
                f"TSFC: {tsfc:.6g} kg/(N*s)\n"
                f"Ttot: {Ttot[k]:.0f} N   D: {D[k]:.0f} N\n"
                f"Δt: {dt_s[k]:.2f} s   ΔFuel: {dFuel[k]:.2f} kg   CumFuel: {cumFuel[k]:.2f} kg\n"
                f"t: {t_here:.2f}")

    info.set_text(_format_info())
    ax_left.set_title("SPECIFIC EXCESS POWER DIAGRAM" + (f" — {title_suffix}" if title_suffix else "") + f"  [{label}]")
    ax_left.legend(loc="upper left", frameon=False)

    # ---------- Helpers ----------
    def _redraw_step():
        dot_left.set_data([mach[k]], [alt_m[k]])
        dot_time.set_data([t_vec[k]], [alt_m[k]])
        dot_lever.set_data([t_vec[k]], [lever[k]])
        info.set_text(_format_info())
        fig.canvas.draw_idle()

    def _switch_strategy(new_s: int):
        nonlocal s, k, label, alt_m, mach, lever, time_s, Ttot, D, Ps, mdot, dt_s, dFuel, cumFuel, limited, K, t_vec
        s = new_s % len(strategies_runs)
        (label, alt_m, mach, lever, time_s, Ttot, D, Ps, mdot, dt_s, dFuel, cumFuel, limited) = _coerce_run(strategies_runs[s])
        K = len(alt_m)
        k = min(k, K-1)

        path_line.set_data(mach, alt_m)
        path_line.set_label(label)

        if (dt_s is None) or (not np.isfinite(dt_s).any()):
            t_vec = np.arange(K, dtype=float)
            ax_time.set_xlabel("Step")
            ax_lever.set_xlabel("Step")
        else:
            t_vec = time_s
            ax_time.set_xlabel("Time [s]")
            ax_lever.set_xlabel("Time [s]")

        line_time.set_data(t_vec, alt_m)
        line_lever.set_data(t_vec, lever)
        _update_time_limits()

        ax_left.set_title("SPECIFIC EXCESS POWER DIAGRAM"
                          + (f" — {title_suffix}" if title_suffix else "")
                          + f"  [{label}]")
        ax_left.legend(loc="upper left", frameon=False)

        _redraw_step()

    # ---------- Callbacks ----------
    def on_prev_step(event):
        nonlocal k
        k = max(0, k-1)
        _redraw_step()

    def on_next_step(event):
        nonlocal k
        k = min(K-1, k+1)
        _redraw_step()

    def on_prev_str(event):
        _switch_strategy(s-1)

    def on_next_str(event):
        _switch_strategy(s+1)

    def on_click(event):
        nonlocal k
        if event.inaxes is ax_left and (event.ydata is not None):
            yi = float(event.ydata)
            if len(alt_m):
                idx = int(np.argmin(np.abs(alt_m - yi)))
                if event.xdata is not None:
                    xd = float(event.xdata)
                    ties = np.where(np.isclose(alt_m, alt_m[idx], atol=1e-9))[0]
                    if ties.size > 1:
                        idx = int(min(ties, key=lambda _k: abs(mach[_k] - xd)))
                k = idx; _redraw_step(); return
        if event.inaxes in (ax_time, ax_lever) and (event.xdata is not None):
            xt = float(event.xdata)
            idx = int(np.nanargmin(np.abs(t_vec - xt)))
            k = max(0, min(K-1, idx)); _redraw_step(); return

    # connect and KEEP the connection ids too
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_keys  = fig.canvas.mpl_connect(
        'key_press_event',
        lambda e: on_prev_step(e) if e.key in ("left","a") else (
                  on_next_step(e) if e.key in ("right","d") else (
                  on_prev_str(e)  if e.key in ("up","w") else (
                  on_next_str(e)  if e.key in ("down","s") else None))))
    fig._controls["cids"] = (cid_click, cid_keys)

    # wire buttons (and keep refs on fig)
    btn_prev_step.on_clicked(on_prev_step)
    btn_next_step.on_clicked(on_next_step)
    btn_prev_str.on_clicked(on_prev_str)
    btn_next_str.on_clicked(on_next_str)

    # Save SEP diagrams for ALL strategies to Climb folder
    try:
        climb_dir = get_or_create_run_directory(phase="Climb")
        
        print(f"[EXPORT] Saving SEP diagrams for all {len(strategies_runs)} strategies...")
        
        # Loop through all strategies and save each one
        for strategy_idx, strategy_run in enumerate(strategies_runs):
            # Switch to this strategy (update the plot)
            _switch_strategy(strategy_idx)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.1)  # Give time for plot to update
            
            # Create safe filename from strategy label
            safe_label = strategy_run.label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(":", "")
            
            # Save complete diagram for this strategy
            sep_path = os.path.join(climb_dir, f"SEP_diagram_{safe_label}.png")
            fig.savefig(sep_path, dpi=300, bbox_inches='tight')
        
        print(f"[EXPORT] Saved {len(strategies_runs)} SEP diagrams to: {climb_dir}")
        print(f"  → One PNG for each strategy showing its specific path")
    except Exception as e:
        print(f"[WARNING] Could not save all SEP diagrams: {e}")

    return fig


# =========================================================================
# STRATEGY COMPARISON PLOTS
# =========================================================================
def create_strategy_comparison_plots(strategies, aero):
    """
    Create additional comparison plots for strategies using Plotly.
    
    Generates comprehensive comparison visualizations:
    - Fuel consumption bar chart (sorted)
    - Time vs fuel trade-off scatter plot
    - Average specific excess power comparison
    - Final Mach number comparison
    - Browser-based comparison table
    
    Args:
        strategies: List of StrategyRun objects
        aero: Aerodynamics wrapper for envelope checks
    """
    
    strategy_names = [s.label for s in strategies]
    fuel_consumption = [s.fuel_total_kg for s in strategies]
    times = [s.time_s[-1]/60.0 if len(s.time_s) > 0 else 0 for s in strategies]  # Convert to minutes
    avg_ps = [np.mean(s.Ps_mps) if len(s.Ps_mps) > 0 else 0 for s in strategies]
    final_machs = [s.mach[-1] if len(s.mach) > 0 else 0 for s in strategies]
    
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Fuel Consumption Comparison', 'Time vs Fuel Trade-off', 
                       'Average Specific Excess Power', 'Final Mach Number Comparison'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Fuel Consumption Comparison (Bar Chart)
    sorted_indices = np.argsort(fuel_consumption)
    sorted_names = [strategy_names[i] for i in sorted_indices]
    sorted_fuel = [fuel_consumption[i] for i in sorted_indices]
    
    fig.add_trace(
        go.Bar(
            x=sorted_names,
            y=sorted_fuel,
            name='Fuel Consumption',
            marker_color='skyblue',
            text=[f'{fuel:.1f}' for fuel in sorted_fuel],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Time vs Fuel Trade-off (Scatter Plot)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=fuel_consumption,
            mode='markers+text',
            text=strategy_names,
            textposition='top center',
            marker=dict(
                size=12,
                color=list(range(len(strategies))),
                colorscale='viridis',
                opacity=0.7
            ),
            name='Time vs Fuel',
            hovertemplate='<b>%{text}</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Average Specific Excess Power Comparison
    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=avg_ps,
            name='Average Ps',
            marker_color='lightgreen',
            text=[f'{ps:.2f}' for ps in avg_ps],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Avg Ps: %{y:.2f} m/s<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Final Mach Number Comparison
    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=final_machs,
            name='Final Mach',
            marker_color='orange',
            text=[f'{mach:.3f}' for mach in final_machs],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Final Mach: %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Strategy Comparison Analysis",
        title_x=0.5,
        title_font_size=20,
        showlegend=False,
        height=800,
        width=1200
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Strategy", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="Total Fuel Consumption (kg)", row=1, col=1)
    
    fig.update_xaxes(title_text="Total Time (minutes)", row=1, col=2)
    fig.update_yaxes(title_text="Total Fuel Consumption (kg)", row=1, col=2)
    
    fig.update_xaxes(title_text="Strategy", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Average Ps (m/s)", row=2, col=1)
    
    fig.update_xaxes(title_text="Strategy", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Final Mach Number", row=2, col=2)
    
    # Save the plot to Climb folder as PNG
    try:
        climb_dir = get_or_create_run_directory(phase="Climb")
        fig.write_image(os.path.join(climb_dir, 'strategy_comparison.png'), width=1600, height=1000, scale=2)
        print(f"[EXPORT] Strategy comparison plot saved to: {climb_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save strategy comparison plot: {e}")
    
    # Show the plot in browser
    fig.show()
    
    # Create comparison table in browser
    create_browser_comparison_table(strategies, aero)


def create_browser_comparison_table(strategies, aero):
    """
    Create a browser-based comparison table for strategies.
    
    Generates an interactive table with:
    - Ranking by fuel consumption
    - Complete strategy metrics
    - Envelope compliance status
    
    Args:
        strategies: List of StrategyRun objects
        aero: Aerodynamics wrapper for envelope checks
    """
    
    # Sort strategies by fuel consumption for ranking
    strategies_sorted = sorted(strategies, key=lambda s: s.fuel_total_kg)
    
    # Prepare data for the table
    table_data = []
    for i, strategy in enumerate(strategies_sorted):
        final_mach = strategy.mach[-1] if len(strategy.mach) > 0 else 0.0
        avg_ps = np.mean(strategy.Ps_mps) if len(strategy.Ps_mps) > 0 else 0.0
        time_min = strategy.time_s[-1] / 60.0 if len(strategy.time_s) > 0 else 0.0
        envelope_status = ClimbingCore.check_envelope_exceedance(strategy, aero)
        
        table_data.append([
            f"{i+1:2d}.",  # Rank
            strategy.label,  # Strategy name
            f"{strategy.fuel_total_kg:.1f}",  # Fuel (kg)
            f"{time_min:.1f}",  # Time (min)
            f"{final_mach:.3f}",  # Final Mach
            f"{avg_ps:.2f}",  # Avg Ps (m/s)
            envelope_status  # Envelope status
        ])
    
    # Create table using Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'Strategy', 'Fuel (kg)', 'Time (min)', 'Final Mach', 'Avg Ps (m/s)', 'Envelope'],
            fill_color='lightblue',
            align='center',
            font=dict(size=14, color='black', family='Arial Black')
        ),
        cells=dict(
            values=list(zip(*table_data)),
            fill_color=[
                ['lightgray' if i % 2 == 0 else 'white' for i in range(len(table_data))] for _ in range(7)
            ],
            align='center',
            font=dict(size=12, color='black', family='Arial')
        )
    )])
    
    fig.update_layout(
        title="Climbing Strategies Comparison",
        title_x=0.5,
        title_font_size=20,
        height=600,
        width=1200,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Save the table to Climb folder as PNG
    try:
        climb_dir = get_or_create_run_directory(phase="Climb")
        fig.write_image(os.path.join(climb_dir, 'strategy_comparison_table.png'), width=1600, height=800, scale=2)
        print(f"[EXPORT] Strategy comparison table saved to: {climb_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save strategy comparison table: {e}")
    
    # Show the table in browser
    fig.show()

