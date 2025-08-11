import numpy as np
import pandas as pd
from pathlib import Path
import pyengine as engine

# === USER SETTINGS ===
STUB = Path(r"D:\Icloud\iCloudDrive\Master Thesis\Mission Analysis Code\lls\stubs\engines\PW1127G-JM")
OUT_DIR = Path(r"D:\Icloud\iCloudDrive\Master Thesis\Mission Analysis Code\lls\envelope_scan")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE  = OUT_DIR / "engine_envelope.csv"
SUMMARY_ALT  = OUT_DIR / "engine_envelope_summary_by_alt.csv"
SUMMARY_LEV  = OUT_DIR / "engine_envelope_summary_by_lever.csv"
SUMMARY_ALL  = OUT_DIR / "engine_envelope_overall_bounds.csv"

MACH_MIN, MACH_MAX, MACH_STEP = 0.0, 1.0, 0.02
ALT_MIN_FT, ALT_MAX_FT, ALT_STEP_FT = 0, 40000, 2000
LEVERS = np.linspace(0.0, 1.0, 6)  # 0.0, 0.2, ..., 1.0

# === ENGINE ===
eng = engine.Engine(str(STUB))

# === Stable grids (avoid float arange drift) ===
num_mach = int(round((MACH_MAX - MACH_MIN) / MACH_STEP)) + 1
mach_grid = np.linspace(MACH_MIN, MACH_MAX, num_mach)
alt_grid  = np.arange(ALT_MIN_FT, ALT_MAX_FT + ALT_STEP_FT, ALT_STEP_FT, dtype=float)

# === Scan ===
rows = []
for lever in LEVERS:
    for alt_ft in alt_grid:
        for mach in mach_grid:
            try:
                thrust = eng.get_thrust_with_lever_position(float(lever), float(mach), float(alt_ft))
                tsfc = eng.get_tsfc()  # uses last evaluated state (lever, mach, altitude)
                # Heuristic unit check for TSFC: if too large, assume kg/(N·hr) and convert to kg/(N·s)
                tsfc_unit = "kg/(N·s)"
                if tsfc > 1e-3:
                    tsfc /= 3600.0
                    tsfc_unit = "kg/(N·hr)->kg/(N·s)"
                valid = (np.isfinite(thrust) and np.isfinite(tsfc) and thrust >= 0.0 and tsfc >= 0.0)
                rows.append({
                    "Lever": float(lever),
                    "Altitude_ft": float(alt_ft),
                    "Mach": float(mach),
                    "Thrust_N": float(thrust) if valid else None,
                    "TSFC_kg_per_Ns": float(tsfc) if valid else None,
                    "TSFC_unit_flag": tsfc_unit if valid else "",
                    "Valid": int(valid),
                    "Error": "" if valid else "non-finite or negative",
                })
            except Exception as e:
                rows.append({
                    "Lever": float(lever),
                    "Altitude_ft": float(alt_ft),
                    "Mach": float(mach),
                    "Thrust_N": None,
                    "TSFC_kg_per_Ns": None,
                    "TSFC_unit_flag": "",
                    "Valid": 0,
                    "Error": str(e)[:200],
                })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"[INFO] Detailed envelope saved: {OUTPUT_FILE}  ({len(df)} rows)")

# === Overall boundary summary (only where Valid==1) ===
valid = df[df["Valid"] == 1].copy()
if not valid.empty:
    overall = pd.DataFrame([{
        "mach_min": float(valid["Mach"].min()),
        "mach_max": float(valid["Mach"].max()),
        "altitude_min_ft": float(valid["Altitude_ft"].min()),
        "altitude_max_ft": float(valid["Altitude_ft"].max()),
        "lever_min": float(valid["Lever"].min()),
        "lever_max": float(valid["Lever"].max()),
        "tsfc_min_kg_per_Ns": float(valid["TSFC_kg_per_Ns"].min()),
        "tsfc_max_kg_per_Ns": float(valid["TSFC_kg_per_Ns"].max()),
        "num_valid_points": int(valid.shape[0]),
        "num_total_points": int(df.shape[0]),
    }])
else:
    overall = pd.DataFrame([{
        "mach_min": None, "mach_max": None,
        "altitude_min_ft": None, "altitude_max_ft": None,
        "lever_min": None, "lever_max": None,
        "tsfc_min_kg_per_Ns": None, "tsfc_max_kg_per_Ns": None,
        "num_valid_points": 0,
        "num_total_points": int(df.shape[0]),
    }])

overall.to_csv(SUMMARY_ALL, index=False)
print(f"[INFO] Overall bounds saved: {SUMMARY_ALL}")

# === Per-altitude summary: Mach fully valid across all levers ===
df["Mach_bin"] = df["Mach"].round(3)
summary_alt = []
for alt_ft, group in df.groupby("Altitude_ft", sort=True):
    fully_valid_by_mach = group.groupby("Mach_bin")["Valid"].sum() == len(LEVERS)
    mach_valid = fully_valid_by_mach[fully_valid_by_mach].index.to_numpy()
    sub = valid[valid["Altitude_ft"] == alt_ft]
    tsfc_min = float(sub["TSFC_kg_per_Ns"].min()) if not sub.empty else None
    tsfc_max = float(sub["TSFC_kg_per_Ns"].max()) if not sub.empty else None
    summary_alt.append({
        "Altitude_ft": float(alt_ft),
        "num_points": int(len(group)),
        "valid_points": int(group["Valid"].sum()),
        "mach_min_full_valid": float(mach_valid.min()) if mach_valid.size else None,
        "mach_max_full_valid": float(mach_valid.max()) if mach_valid.size else None,
        "tsfc_min_kg_per_Ns_at_alt": tsfc_min,
        "tsfc_max_kg_per_Ns_at_alt": tsfc_max,
        "any_valid": int(group["Valid"].any()),
    })
pd.DataFrame(summary_alt).sort_values("Altitude_ft").to_csv(SUMMARY_ALT, index=False)
print(f"[INFO] Summary by altitude saved: {SUMMARY_ALT}")

# === Per-lever summary: Mach/alt ranges where valid for each lever ===
summary_lev = []
for lever, group in valid.groupby("Lever", sort=True):
    summary_lev.append({
        "Lever": float(lever),
        "mach_min_valid": float(group["Mach"].min()),
        "mach_max_valid": float(group["Mach"].max()),
        "altitude_min_ft_valid": float(group["Altitude_ft"].min()),
        "altitude_max_ft_valid": float(group["Altitude_ft"].max()),
        "tsfc_min_kg_per_Ns_at_lever": float(group["TSFC_kg_per_Ns"].min()),
        "tsfc_max_kg_per_Ns_at_lever": float(group["TSFC_kg_per_Ns"].max()),
        "valid_points_at_lever": int(group.shape[0]),
    })
pd.DataFrame(summary_lev).sort_values("Lever").to_csv(SUMMARY_LEV, index=False)
print(f"[INFO] Summary by lever saved: {SUMMARY_LEV}")

# === Console recap ===
if not valid.empty:
    print("[BOUNDS] Mach:", overall.loc[0, "mach_min"], "→", overall.loc[0, "mach_max"])
    print("[BOUNDS] Altitude_ft:", overall.loc[0, "altitude_min_ft"], "→", overall.loc[0, "altitude_max_ft"])
    print("[BOUNDS] Lever:", overall.loc[0, "lever_min"], "→", overall.loc[0, "lever_max"])
    print("[BOUNDS] TSFC_kg_per_Ns:", overall.loc[0, "tsfc_min_kg_per_Ns"], "→", overall.loc[0, "tsfc_max_kg_per_Ns"])
else:
    print("[BOUNDS] No valid points found.")
