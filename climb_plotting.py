from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import patheffects as pe
from typing import List
import os
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
pio.renderers.default = "browser"

# import needed types, helpers, and constants from logic
from aircraft_config import (
    isa_properties, a_from_altitude, INITIAL_MASS_KG,
    M_MMO, S_REF_M2, G_C
)
from climb import (
    PlottingConfig, Y_AXIS_TOP_M, TARGET_ALT_M, ClimbingCore
)

# Import visualization configuration for consistent styling
from visualization_config import (
    Colors, Typography, Layout, LineStyles,
    get_standard_layout, get_standard_legend, get_axis_config,
    ExportConfig, get_or_create_run_directory
)


def plot_strategies_interactive(
    M_grid, H_plot, Ps_base,
    strategies_runs: List[ClimbingCore.StrategyManager.StrategyRun],
    *,
    title_suffix=""
):
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
    Hm, Mm = np.meshgrid(H_plot, M_grid, indexing="ij")
    g = G_C
    Es = np.zeros_like(Ps_base)
    for k, h in enumerate(H_plot):
        a = a_from_altitude(float(h))
        V = a * np.asarray(M_grid, float)
        Es[k, :] = 0.5 * V * V + g * h
    Es_levels = np.linspace(np.nanmin(Es), np.nanmax(Es), 8)
    ax_left.contour(Mm, Hm, Es, levels=Es_levels, colors=[(0.6, 0.6, 0.6)], linewidths=0.8, zorder=0)

    # CLmax + MMO + operating fill
    def _compute_mstall_curve():
        W = INITIAL_MASS_KG * g
        CL_MAX = 1.4  # Typical CL_MAX for commercial aircraft
        out = np.full_like(H_plot, np.nan, float)
        for k, h in enumerate(H_plot):
            _, _, rho = isa_properties(float(h)); a = a_from_altitude(float(h))
            q_req = W / (S_REF_M2 * CL_MAX)
            if rho > 0:
                V = np.sqrt(2*q_req/max(rho,1e-12))
                out[k] = V / max(a,1e-12)
        return out
    M_stall = _compute_mstall_curve()
    ax_left.axvline(M_MMO, color="red", linewidth=2.0, linestyle="--", label=f"MMO={M_MMO:.2f}", zorder=2)
    if np.isfinite(M_stall).any():
        ax_left.plot(M_stall, H_plot, linestyle="--", color="red", linewidth=1.6, label="CLmax", zorder=2)
        cond = np.isfinite(M_stall) & (M_stall < M_MMO)
        if np.any(cond):
            ax_left.fill_betweenx(H_plot[cond], M_stall[cond], M_MMO, alpha=0.12, color="tab:green", zorder=1)

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
    ax_left.set_xlim(M_grid[0] - 0.02, float(PlottingConfig.M_XMAX_UI))
    ax_left.set_ylim(H_plot[0] - 200, Y_AXIS_TOP_M + 200)
    ax_left.set_aspect("auto", adjustable="box")
    ax_left.set_autoscale_on(False)
    ax_left.margins(x=0, y=0)

    # ---------- State & initial data ----------
    if not strategies_runs:
        raise ValueError("strategies_runs is empty.")
    s = 0           # strategy index
    k = 0           # step index

    def _coerce_run(run: ClimbingCore.StrategyManager.StrategyRun):
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
    ax_time.set_ylim(H_plot[0] - 200, Y_AXIS_TOP_M + 200)

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

def plot_J_3d_plotly(M_grid, H_sched, lever_grid, J_grid_3d, min_path=None, title=None):
    """
    Visualize J values in 3D (Mach, Altitude, Lever) using Plotly.
    Optionally overlay the minimum-fuel path as a line if min_path is provided.
    Includes flight envelope limits (MMO, CLmax, operating envelope).
    Enhanced with user-friendly features: camera presets, data filtering, and export.
    """
    # Prepare meshgrid for scatter
    M, H, L = np.meshgrid(M_grid, H_sched, lever_grid, indexing='ij')
    # Flatten arrays for plotting - NEW AXIS ASSIGNMENT
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
    # Create a vertical plane at M_MMO across all altitudes and lever positions
    lever_range = np.linspace(0, 1, 10)
    alt_range = np.linspace(H_sched[0], H_sched[-1], 10)
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
    
    # 2. CLmax (stall) limit - compute stall curve
    def _compute_mstall_curve():
        W = INITIAL_MASS_KG * G_C
        CL_MAX = 1.4  # Default CL_MAX for commercial aircraft
        out = np.full_like(H_sched, np.nan, float)
        for k, h in enumerate(H_sched):
            _, _, rho = isa_properties(float(h))
            a = a_from_altitude(float(h))
            q_req = W / (S_REF_M2 * CL_MAX)
            if rho > 0:
                V = np.sqrt(2*q_req/max(rho,1e-12))
                out[k] = V / max(a,1e-12)
        return out
    
    M_stall = _compute_mstall_curve()
    if np.isfinite(M_stall).any():
        # Create stall surface - vertical plane at stall Mach
        lever_range = np.linspace(0, 1, 10)
        alt_range = np.linspace(H_sched[0], H_sched[-1], 10)
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
                name='Flight Envelope Limit'
            ))
    
    # 3. Operating envelope boundaries (between stall and MMO) - as lines
    if np.isfinite(M_stall).any():
        cond = np.isfinite(M_stall) & (M_stall < M_MMO)
        if np.any(cond):
            # Create operating envelope boundary lines (separate traces to avoid connecting lines)
            for lever_val in [0.0, 0.5, 1.0]:  # Show at different lever positions
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
    # Add J values scatter plot (dark purple)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=Colors.J_VALUES_CLIMB,
            opacity=0.7,
            line=dict(width=0)
        ),
        name='J values',
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
            line=dict(color=Colors.OPTIMAL_PATH, width=6),
            marker=dict(size=6, color=Colors.OPTIMAL_PATH),
            name='Optimal Path',
            hovertemplate='<b>Optimal Path</b><br>Lever: %{x:.3f}<br>Mach: %{y:.3f}<br>Alt: %{z:.1f} m<extra></extra>'
        ))

    # Enhanced layout with user-friendly features
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
                gridcolor='rgba(128,128,128,0.3)'
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
            text=title if title is not None else '3D Visualization of J (Fuel/Energy) - Enhanced View',
            font=dict(size=Typography.MAIN_TITLE_SIZE, family=Typography.FONT_FAMILY),
            x=0.5
        ),
        # Add control buttons
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
                        args=[{"visible": [True if trace.name and "Optimal Path" in trace.name else False for trace in fig.data]}],
                        label="Optimal Path Only",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True if trace.name and ("MMO Limit" in trace.name or "Flight Envelope Limit" in trace.name or "Operating Envelope" in trace.name) else False for trace in fig.data]}],
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
        # Add margin for controls using standard config
        margin=dict(l=0, r=Layout.MARGIN_RIGHT, t=Layout.MARGIN_TOP, b=Layout.MARGIN_BOTTOM)
    )

    # Set template
    fig.update_layout(template="plotly_white")

    # Write to HTML file with enhanced features using standard export config
    export_config = ExportConfig.get_plotly_config()
    export_config['toImageButtonOptions']['filename'] = '3d_plot_export'
    export_config['modeBarButtonsToAdd'] = [
        "drawline", "drawopenpath", "drawclosedpath",
        "drawcircle", "drawrect", "eraseshape"
    ]
    
    # Save to Climb folder (both HTML for interaction and PNG for static)
    climb_dir = get_or_create_run_directory(phase="Climb")
    output_path_html = os.path.join(climb_dir, 'climb_J_3d_plot.html')
    output_path_png = os.path.join(climb_dir, 'climb_J_3d_plot.png')
    
    pio.write_html(
        fig, 
        file=output_path_html, 
        auto_open=True,
        config=export_config
    )
    
    # Also save as PNG
    try:
        fig.write_image(output_path_png, width=1600, height=1200, scale=2)
        print(f"[EXPORT] 3D J plot saved to: {output_path_html} (interactive) and {output_path_png} (PNG)")
    except Exception as e:
        print(f"[EXPORT] 3D J plot saved to: {output_path_html} (HTML only)")
        print(f"[WARNING] Could not save PNG version: {e}")

def check_envelope_exceedance(strategy, aero):
    """Check if a strategy exceeds the flight envelope (MMO or CLmax)."""
    # ClimbingCore already imported at top
    return ClimbingCore.check_envelope_exceedance(strategy, aero)

def create_strategy_comparison_plots(strategies, aero):
    """Create additional comparison plots for strategies using Plotly."""
    
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
    """Create a browser-based comparison table for strategies."""
    
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


def plot_climb_performance_detailed(climb_result, climb_info: dict = None) -> None:
    """
    Create detailed climb performance analysis plot (climb-specific data only).
    
    Aligned with cruise and descent performance plots showing:
    - Fuel Flow Rate
    - Thrust vs Drag
    - Weight Evolution
    - Lever Position
    - True Airspeed
    - Cumulative Fuel Consumption
    
    Args:
        climb_result: Results from climb DP optimization (MinFuelSchedule)
        climb_info: Optional dictionary with climb optimization info
    """
    
    # Extract climb data
    climb_time_s = np.cumsum(np.nan_to_num(climb_result.dt_s, nan=0.0, posinf=0.0, neginf=0.0))
    climb_time_min = climb_time_s / 60.0  # Convert to minutes
    climb_alt_m = np.asarray(climb_result.alt_m, float)
    climb_mach = np.asarray(climb_result.mach, float)
    climb_lever = np.asarray(climb_result.lever, float)
    climb_thrust_N = np.asarray(climb_result.T_total_N, float)
    climb_drag_N = np.asarray(climb_result.D_N, float)
    climb_fuel_kg = np.asarray(climb_result.cumFuel_kg, float)
    climb_weight_kg = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic weight from DP optimization
    
    # Calculate fuel flow rate (kg/h) - approximate from fuel consumption
    fuel_flow_kgh = []
    for i in range(len(climb_result.dt_s)):
        if i == 0:
            fuel_flow_kgh.append(0.0)
        else:
            dt_hours = climb_result.dt_s[i] / 3600.0
            if dt_hours > 0:
                fuel_consumed = climb_fuel_kg[i] - climb_fuel_kg[i-1]
                fuel_flow_kgh.append(fuel_consumed / dt_hours)
            else:
                fuel_flow_kgh.append(0.0)
    fuel_flow_kgh = np.array(fuel_flow_kgh)
    
    # Calculate true airspeed
    tas_ms = []
    for i in range(len(climb_mach)):
        a = a_from_altitude(float(climb_alt_m[i]))
        tas_ms.append(climb_mach[i] * a)
    tas_ms = np.array(tas_ms)
    
    # Convert forces to kN
    thrust_kn = climb_thrust_N / 1000
    drag_kn = climb_drag_N / 1000
    
    # Create climb performance plot (3x2 grid) with consistent styling
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Fuel Flow Rate</b>', '<b>Thrust vs Drag</b>', 
            '<b>Weight Evolution</b>', '<b>Lever Position</b>',
            '<b>True Airspeed</b>', '<b>Cumulative Fuel Consumption</b>'
        ),
        vertical_spacing=0.15,  # Same as cruise and descent
        horizontal_spacing=0.15
    )
    
    # 1. Fuel Flow Rate over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=fuel_flow_kgh,
            mode='lines',
            name='Fuel Flow (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Thrust vs Drag over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=thrust_kn,
            mode='lines',
            name='Thrust (Climb)',
            line=dict(color='darkblue', width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=drag_kn,
            mode='lines',
            name='Drag (Climb)',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Weight Evolution over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_weight_kg,
            mode='lines',
            name='Weight (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Engine Lever Position over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_lever * 100,  # Convert to percentage
            mode='lines',
            name='Lever Position (Climb)',
            line=dict(color='steelblue', width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add reference lines for lever position (same as cruise and descent)
    fig.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, 
                  annotation_text="MCT Limit", annotation_position="top right",
                  row=2, col=2)
    fig.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL,
                  annotation_text="Max Thrust", annotation_position="bottom right",
                  row=2, col=2)
    
    # 5. True Airspeed over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=tas_ms,
            mode='lines',
            name='True Airspeed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Cumulative Fuel Consumption over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_fuel_kg,
            mode='lines',
            name='Fuel Consumed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Calculate summary statistics
    total_time_min = climb_time_min[-1] if len(climb_time_min) > 0 else 0.0
    total_fuel = climb_fuel_kg[-1] if len(climb_fuel_kg) > 0 else 0.0
    avg_fuel_flow = np.mean(fuel_flow_kgh[fuel_flow_kgh > 0]) if np.any(fuel_flow_kgh > 0) else 0.0
    altitude_gain = climb_alt_m[-1] - climb_alt_m[0] if len(climb_alt_m) > 0 else 0.0
    
    # Update layout with standard configuration (aligned with cruise and descent)
    subtitle = (
        f"Altitude Gain: {altitude_gain:.0f} m | Time: {total_time_min:.1f} min | "
        f"Fuel: {total_fuel:.1f} kg | Avg Fuel Flow: {avg_fuel_flow:.0f} kg/h"
    )
    
    layout_config = get_standard_layout(
        "CLIMB PERFORMANCE ANALYSIS (2D)",
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
    
    # Update axes labels with standard configuration (aligned with cruise and descent)
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=2)
    fig.update_yaxes(**get_axis_config("Force (kN)"), row=1, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=1)
    # Set y-axis range to zoom in on weight changes for better visibility
    weight_min, weight_max = np.min(climb_weight_kg), np.max(climb_weight_kg)
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
    config['toImageButtonOptions']['filename'] = 'climb_performance_2d'
    
    # Save individual plots as separate HTML files in timestamped directory/Climb subfolder
    try:
        run_dir = get_or_create_run_directory(phase="Climb")
        save_prefix = "climb_performance"
        
        # 1. Fuel Flow Rate
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=climb_time_min, y=fuel_flow_kgh, mode='lines', name='Fuel Flow (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'))
        fig1.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Fuel Flow Rate", subtitle, height=600, width=900))
        fig1.update_xaxes(**get_axis_config("Time (min)")); fig1.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_flow.png'), width=1200, height=800, scale=2)
        
        # 2. Thrust vs Drag
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=climb_time_min, y=thrust_kn, mode='lines', name='Thrust (Climb)',
            line=dict(color='darkblue', width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'))
        fig2.add_trace(go.Scatter(x=climb_time_min, y=drag_kn, mode='lines', name='Drag (Climb)',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'))
        fig2.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Thrust vs Drag", subtitle, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("Time (min)")); fig2.update_yaxes(**get_axis_config("Force (kN)"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_thrust_drag.png'), width=1200, height=800, scale=2)
        
        # 3. Weight Evolution
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=climb_time_min, y=climb_weight_kg, mode='lines', name='Weight (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'))
        fig3.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Weight Evolution", subtitle, height=600, width=900))
        # Set y-axis range to zoom in on weight changes for better visibility
        weight_min, weight_max = np.min(climb_weight_kg), np.max(climb_weight_kg)
        weight_margin = (weight_max - weight_min) * 0.2  # Add 20% margin
        fig3.update_xaxes(**get_axis_config("Time (min)")); 
        fig3.update_yaxes(**get_axis_config("Weight (kg)"), range=[weight_min - weight_margin, weight_max + weight_margin])
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_weight.png'), width=1200, height=800, scale=2)
        
        # 4. Lever Position
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=climb_time_min, y=climb_lever * 100, mode='lines', name='Lever Position (Climb)',
            line=dict(color='steelblue', width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'))
        fig4.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, annotation_text="MCT Limit", annotation_position="top right")
        fig4.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL, annotation_text="Max Thrust", annotation_position="bottom right")
        fig4.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Lever Position", subtitle, height=600, width=900))
        fig4.update_xaxes(**get_axis_config("Time (min)")); fig4.update_yaxes(**get_axis_config("Lever Position (%)"))
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_lever.png'), width=1200, height=800, scale=2)
        
        # 5. True Airspeed
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=climb_time_min, y=tas_ms, mode='lines', name='True Airspeed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'))
        fig5.update_layout(**get_standard_layout("CLIMB PERFORMANCE - True Airspeed", subtitle, height=600, width=900))
        fig5.update_xaxes(**get_axis_config("Time (min)")); fig5.update_yaxes(**get_axis_config("True Airspeed (m/s)"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_airspeed.png'), width=1200, height=800, scale=2)
        
        # 6. Cumulative Fuel
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=climb_time_min, y=climb_fuel_kg, mode='lines', name='Fuel Consumed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'))
        fig6.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Cumulative Fuel Consumption", subtitle, height=600, width=900))
        fig6.update_xaxes(**get_axis_config("Time (min)")); fig6.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_fuel.png'), width=1200, height=800, scale=2)
        
        print(f"[EXPORT] Individual climb plots saved as PNG to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save individual climb plots: {e}")
    
    # Show the plot
    fig.show(config=config)


def create_strategy_comparison_plots(strategies, aero):
    """Create additional comparison plots for strategies using Plotly."""
    
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
    """Create a browser-based comparison table for strategies."""
    # check_envelope_exceedance already available via ClimbingCore
    
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


def plot_climb_performance_detailed(climb_result, climb_info: dict = None) -> None:
    """
    Create detailed climb performance analysis plot (climb-specific data only).
    
    Aligned with cruise and descent performance plots showing:
    - Fuel Flow Rate
    - Thrust vs Drag
    - Weight Evolution
    - Lever Position
    - True Airspeed
    - Cumulative Fuel Consumption
    
    Args:
        climb_result: Results from climb DP optimization (MinFuelSchedule)
        climb_info: Optional dictionary with climb optimization info
    """
    
    # Extract climb data
    climb_time_s = np.cumsum(np.nan_to_num(climb_result.dt_s, nan=0.0, posinf=0.0, neginf=0.0))
    climb_time_min = climb_time_s / 60.0  # Convert to minutes
    climb_alt_m = np.asarray(climb_result.alt_m, float)
    climb_mach = np.asarray(climb_result.mach, float)
    climb_lever = np.asarray(climb_result.lever, float)
    climb_thrust_N = np.asarray(climb_result.T_total_N, float)
    climb_drag_N = np.asarray(climb_result.D_N, float)
    climb_fuel_kg = np.asarray(climb_result.cumFuel_kg, float)
    climb_weight_kg = np.asarray(climb_result.mass_kg, float)  # Use actual dynamic weight from DP optimization
    
    # Calculate fuel flow rate (kg/h) - approximate from fuel consumption
    fuel_flow_kgh = []
    for i in range(len(climb_result.dt_s)):
        if i == 0:
            fuel_flow_kgh.append(0.0)
        else:
            dt_hours = climb_result.dt_s[i] / 3600.0
            if dt_hours > 0:
                fuel_consumed = climb_fuel_kg[i] - climb_fuel_kg[i-1]
                fuel_flow_kgh.append(fuel_consumed / dt_hours)
            else:
                fuel_flow_kgh.append(0.0)
    fuel_flow_kgh = np.array(fuel_flow_kgh)
    
    # Calculate true airspeed
    tas_ms = []
    for i in range(len(climb_mach)):
        a = a_from_altitude(float(climb_alt_m[i]))
        tas_ms.append(climb_mach[i] * a)
    tas_ms = np.array(tas_ms)
    
    # Convert forces to kN
    thrust_kn = climb_thrust_N / 1000
    drag_kn = climb_drag_N / 1000
    
    # Create climb performance plot (3x2 grid) with consistent styling
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Fuel Flow Rate</b>', '<b>Thrust vs Drag</b>', 
            '<b>Weight Evolution</b>', '<b>Lever Position</b>',
            '<b>True Airspeed</b>', '<b>Cumulative Fuel Consumption</b>'
        ),
        vertical_spacing=0.15,  # Same as cruise and descent
        horizontal_spacing=0.15
    )
    
    # 1. Fuel Flow Rate over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=fuel_flow_kgh,
            mode='lines',
            name='Fuel Flow (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Thrust vs Drag over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=thrust_kn,
            mode='lines',
            name='Thrust (Climb)',
            line=dict(color='darkblue', width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=drag_kn,
            mode='lines',
            name='Drag (Climb)',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Weight Evolution over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_weight_kg,
            mode='lines',
            name='Weight (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Engine Lever Position over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_lever * 100,  # Convert to percentage
            mode='lines',
            name='Lever Position (Climb)',
            line=dict(color='steelblue', width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add reference lines for lever position (same as cruise and descent)
    fig.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, 
                  annotation_text="MCT Limit", annotation_position="top right",
                  row=2, col=2)
    fig.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL,
                  annotation_text="Max Thrust", annotation_position="bottom right",
                  row=2, col=2)
    
    # 5. True Airspeed over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=tas_ms,
            mode='lines',
            name='True Airspeed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Cumulative Fuel Consumption over time (aligned with cruise and descent)
    fig.add_trace(
        go.Scatter(
            x=climb_time_min,
            y=climb_fuel_kg,
            mode='lines',
            name='Fuel Consumed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Calculate summary statistics
    total_time_min = climb_time_min[-1] if len(climb_time_min) > 0 else 0.0
    total_fuel = climb_fuel_kg[-1] if len(climb_fuel_kg) > 0 else 0.0
    avg_fuel_flow = np.mean(fuel_flow_kgh[fuel_flow_kgh > 0]) if np.any(fuel_flow_kgh > 0) else 0.0
    altitude_gain = climb_alt_m[-1] - climb_alt_m[0] if len(climb_alt_m) > 0 else 0.0
    
    # Update layout with standard configuration (aligned with cruise and descent)
    subtitle = (
        f"Altitude Gain: {altitude_gain:.0f} m | Time: {total_time_min:.1f} min | "
        f"Fuel: {total_fuel:.1f} kg | Avg Fuel Flow: {avg_fuel_flow:.0f} kg/h"
    )
    
    layout_config = get_standard_layout(
        "CLIMB PERFORMANCE ANALYSIS (2D)",
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
    
    # Update axes labels with standard configuration (aligned with cruise and descent)
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=1)
    fig.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"), row=1, col=1)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=1, col=2)
    fig.update_yaxes(**get_axis_config("Force (kN)"), row=1, col=2)
    
    fig.update_xaxes(**get_axis_config("Time (min)"), row=2, col=1)
    # Set y-axis range to zoom in on weight changes for better visibility
    weight_min, weight_max = np.min(climb_weight_kg), np.max(climb_weight_kg)
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
    config['toImageButtonOptions']['filename'] = 'climb_performance_2d'
    
    # Save individual plots as separate HTML files in timestamped directory/Climb subfolder
    try:
        run_dir = get_or_create_run_directory(phase="Climb")
        save_prefix = "climb_performance"
        
        # 1. Fuel Flow Rate
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=climb_time_min, y=fuel_flow_kgh, mode='lines', name='Fuel Flow (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel Flow: %{y:.1f} kg/h<extra></extra>'))
        fig1.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Fuel Flow Rate", subtitle, height=600, width=900))
        fig1.update_xaxes(**get_axis_config("Time (min)")); fig1.update_yaxes(**get_axis_config("Fuel Flow (kg/h)"))
        fig1.write_image(os.path.join(run_dir, f'{save_prefix}_fuel_flow.png'), width=1200, height=800, scale=2)
        
        # 2. Thrust vs Drag
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=climb_time_min, y=thrust_kn, mode='lines', name='Thrust (Climb)',
            line=dict(color='darkblue', width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Thrust: %{y:.1f} kN<extra></extra>'))
        fig2.add_trace(go.Scatter(x=climb_time_min, y=drag_kn, mode='lines', name='Drag (Climb)',
            line=dict(color='lightblue', width=LineStyles.THICK, dash=LineStyles.DASH), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Drag: %{y:.1f} kN<extra></extra>'))
        fig2.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Thrust vs Drag", subtitle, height=600, width=900))
        fig2.update_xaxes(**get_axis_config("Time (min)")); fig2.update_yaxes(**get_axis_config("Force (kN)"))
        fig2.write_image(os.path.join(run_dir, f'{save_prefix}_thrust_drag.png'), width=1200, height=800, scale=2)
        
        # 3. Weight Evolution
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=climb_time_min, y=climb_weight_kg, mode='lines', name='Weight (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Weight: %{y:.0f} kg<extra></extra>'))
        fig3.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Weight Evolution", subtitle, height=600, width=900))
        # Set y-axis range to zoom in on weight changes for better visibility
        weight_min, weight_max = np.min(climb_weight_kg), np.max(climb_weight_kg)
        weight_margin = (weight_max - weight_min) * 0.2  # Add 20% margin
        fig3.update_xaxes(**get_axis_config("Time (min)")); 
        fig3.update_yaxes(**get_axis_config("Weight (kg)"), range=[weight_min - weight_margin, weight_max + weight_margin])
        fig3.write_image(os.path.join(run_dir, f'{save_prefix}_weight.png'), width=1200, height=800, scale=2)
        
        # 4. Lever Position
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=climb_time_min, y=climb_lever * 100, mode='lines', name='Lever Position (Climb)',
            line=dict(color='steelblue', width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Lever: %{y:.1f}%<extra></extra>'))
        fig4.add_hline(y=85, line_dash=LineStyles.DASH, line_color=Colors.WARNING, annotation_text="MCT Limit", annotation_position="top right")
        fig4.add_hline(y=100, line_dash=LineStyles.DOT, line_color=Colors.CRITICAL, annotation_text="Max Thrust", annotation_position="bottom right")
        fig4.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Lever Position", subtitle, height=600, width=900))
        fig4.update_xaxes(**get_axis_config("Time (min)")); fig4.update_yaxes(**get_axis_config("Lever Position (%)"))
        fig4.write_image(os.path.join(run_dir, f'{save_prefix}_lever.png'), width=1200, height=800, scale=2)
        
        # 5. True Airspeed
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=climb_time_min, y=tas_ms, mode='lines', name='True Airspeed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>TAS: %{y:.1f} m/s<extra></extra>'))
        fig5.update_layout(**get_standard_layout("CLIMB PERFORMANCE - True Airspeed", subtitle, height=600, width=900))
        fig5.update_xaxes(**get_axis_config("Time (min)")); fig5.update_yaxes(**get_axis_config("True Airspeed (m/s)"))
        fig5.write_image(os.path.join(run_dir, f'{save_prefix}_airspeed.png'), width=1200, height=800, scale=2)
        
        # 6. Cumulative Fuel
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=climb_time_min, y=climb_fuel_kg, mode='lines', name='Fuel Consumed (Climb)',
            line=dict(color=Colors.CLIMB, width=LineStyles.THICK), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.15)',
            hovertemplate='<b>Climb</b><br>Time: %{x:.1f} min<br>Fuel: %{y:.1f} kg<extra></extra>'))
        fig6.update_layout(**get_standard_layout("CLIMB PERFORMANCE - Cumulative Fuel Consumption", subtitle, height=600, width=900))
        fig6.update_xaxes(**get_axis_config("Time (min)")); fig6.update_yaxes(**get_axis_config("Cumulative Fuel (kg)"))
        fig6.write_image(os.path.join(run_dir, f'{save_prefix}_fuel.png'), width=1200, height=800, scale=2)
        
        print(f"[EXPORT] Individual climb plots saved as PNG to: {run_dir}")
    except Exception as e:
        print(f"[WARNING] Could not save individual climb plots: {e}")
    
    # Show the plot
    fig.show(config=config)