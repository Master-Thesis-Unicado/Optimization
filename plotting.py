# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import pandas as pd
from tkinter import Tk, filedialog
import re


def _legend_outside(ax):
    handles, labels = ax.get_legend_handles_labels()
    filt = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if filt:
        h, l = zip(*filt)
        ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def _safe_sheet_name(name: str) -> str:
    """
    Excel sheet name constraints: max 31 chars, no []:*?/\\
    We'll also collapse spaces and trim.
    """
    name = re.sub(r'[\[\]\:\*\?\/\\]', '_', str(name))
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:31] if len(name) > 31 else name


def interactive_plot(generate_strategy_func, simulate_func, target_altitude):
    """
    Interactive plotting UI.

    Parameters
    ----------
    generate_strategy_func : callable
        Function(profile: str) -> list[(af, strategy_function)]
    simulate_func : callable
        Function(strategy_function, altitude_fraction) -> (t, h, V, lever_positions, final_results, diagnostics)
    target_altitude : float or None
        Target altitude [m] for a guide line.
    """
    profiles = [
        "linear",
        "exponential_increasing_climb",
        "exponential_decreasing_climb",
        "exponential_increasing_speed",
        "exponential_decreasing_speed",
        "constant_speed",
        "constant_mach",
    ]

    # Figure & layout
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1, 3])

    ax_alt = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_alt)
    ax_lev = fig.add_subplot(gs[2, 1], sharex=ax_alt)

    left_gs = gs[:, 0].subgridspec(3, 1, height_ratios=[5, 1, 1])
    radio_ax = fig.add_subplot(left_gs[0])
    btn_ax_clear = fig.add_subplot(left_gs[1])
    btn_ax_save = fig.add_subplot(left_gs[2])

    radio = RadioButtons(radio_ax, profiles, active=0)
    radio_ax.set_title("Strategy Profile", fontsize=10)
    btn_clear = Button(btn_ax_clear, "Clear Plots")
    btn_save = Button(btn_ax_save, "Export Excel")

    # storage for per-scenario exports
    # key: (profile, af_str), value: dict with "df" (timeseries) and "final" (dict of finals)
    scenarios_export = {}

    def _format_axes():
        ax_alt.set_title("Altitude vs Time")
        ax_alt.set_xlabel("Time [s]"); ax_alt.set_ylabel("Altitude [m]")
        ax_vel.set_title("True Airspeed vs Time")
        ax_vel.set_xlabel("Time [s]"); ax_vel.set_ylabel("Speed [m/s]")
        ax_lev.set_title("Lever Position vs Time")
        ax_lev.set_xlabel("Time [s]"); ax_lev.set_ylabel("Lever [0–1]")
        for ax in (ax_alt, ax_vel, ax_lev):
            ax.grid(True, alpha=0.3)

    def _add_target_alt_line():
        if target_altitude is not None:
            ax_alt.axhline(target_altitude, linestyle="--", linewidth=1.0, label="Target Alt")

    def _clear_axes():
        ax_alt.cla(); ax_vel.cla(); ax_lev.cla()
        _format_axes()
        _add_target_alt_line()

    _clear_axes()

    def run_profile(profile):
        nonlocal scenarios_export
        scenarios_export = {}

        strategies = generate_strategy_func(profile)
        if not strategies:
            for ax in (ax_alt, ax_vel, ax_lev):
                _legend_outside(ax)
            fig.canvas.draw_idle()
            return

        for af, strat_fn in strategies:
            try:
                t, h, V, lever_positions, final_results, diagnostics = simulate_func(strat_fn, af)
                label_suffix = "AF=—" if af is None else f"AF={af:.2f}"

                # --- plotting
                ax_alt.plot(t, h, label=f"{profile} | {label_suffix} | Alt")
                ax_vel.plot(t, V, label=f"{profile} | {label_suffix} | V")
                t_lev = t[:len(lever_positions)]
                ax_lev.plot(t_lev, lever_positions, label=f"{profile} | {label_suffix} | Lever")

                # --- build per-scenario timeseries DataFrame (with aligned lengths)
                n = max(len(t), len(h), len(V), len(lever_positions),
                        len(diagnostics.get("fuel_flow_kg_s", [])),
                        len(diagnostics.get("mass_kg", [])))

                def _pad(arr, n):
                    arr = np.asarray(arr, dtype=float)
                    out = np.full(n, np.nan)
                    out[:len(arr)] = arr
                    return out

                df_ts = pd.DataFrame({
                    "t_s":     _pad(t, n),
                    "h_m":     _pad(h, n),
                    "V_mps":   _pad(V, n),
                    "lever":   _pad(lever_positions, n),
                    "fuel_flow_kg_s": _pad(diagnostics.get("fuel_flow_kg_s", []), n),
                    "mass_kg": _pad(diagnostics.get("mass_kg", []), n),
                })

                # store
                af_str = "NA" if af is None else f"{af:.2f}"
                scenarios_export[(profile, af_str)] = {
                    "df": df_ts,
                    "final": {
                        "profile": profile,
                        "altitude_fraction": (np.nan if af is None else float(af)),
                        "final_time_s": final_results.get("Total Climb Time", np.nan),
                        "final_altitude_m": final_results.get("Final Altitude", np.nan),
                        "final_velocity_mps": final_results.get("Final Velocity", np.nan),
                        "final_mass_kg": final_results.get("Final Mass (kg)", np.nan),
                        "total_fuel_burn_kg": final_results.get("Total Fuel Burned (kg)", np.nan),
                        "engines": final_results.get("Engines", np.nan),
                    }
                }

            except Exception as e:
                print(f"[ERROR] Skipped strategy {('AF=' + f'{af:.2f}' if af is not None else 'AF=—')} "
                      f"due to simulation failure: {e}")

        _legend_outside(ax_alt)
        _legend_outside(ax_vel)
        _legend_outside(ax_lev)
        fig.canvas.draw_idle()

    def on_profile_change(label):
        _clear_axes()
        run_profile(label)

    def on_clear_clicked(event):
        _clear_axes()
        # keep scenarios_export as-is so user can still export if they want
        fig.canvas.draw_idle()

    def on_save_clicked(event):
        if not scenarios_export:
            print("[INFO] Nothing to export yet.")
            return
        try:
            Tk().withdraw()
            path = filedialog.asksaveasfilename(
                title="Save simulation data",
                defaultextension=".xlsx",
                filetypes=[("Excel workbook", "*.xlsx")]
            )
            if not path:
                return

            # Build a summary first
            summary_rows = [v["final"] for v in scenarios_export.values()]
            df_summary = pd.DataFrame(summary_rows)

            # Write each scenario to its own sheet + summary
            with pd.ExcelWriter(path) as writer:
                # summary first
                df_summary.to_excel(writer, sheet_name=_safe_sheet_name("Summary"), index=False)

                # then one sheet per scenario
                for (profile, af_str), data in scenarios_export.items():
                    sheet = _safe_sheet_name(f"{profile[:20]}_{af_str}")
                    data["df"].to_excel(writer, sheet_name=sheet, index=False)

            print(f"[INFO] Exported Excel workbook: {path}")

        except Exception as e:
            print(f"[ERROR] Failed to export: {e}")

    radio.on_clicked(on_profile_change)
    btn_clear.on_clicked(on_clear_clicked)
    btn_save.on_clicked(on_save_clicked)

    run_profile(profiles[0])
    plt.show()
