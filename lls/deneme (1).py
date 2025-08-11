from pathlib import Path
import pyengine as engine
import numpy as np
import matplotlib.pyplot as plt

# ── 1. build an absolute path to the stub directory ──
STUB = Path(__file__).parent / "stubs" / "engines" / "PW1127G-JM"

# ── 2. pass it to Engine as a string ─────────────────
eng = engine.Engine(str(STUB))          # or engine.Engine(STUB.as_posix())

# ── 3. analysis parameters ───────────────────────────
mach        = 0.76
altitude_ft = 10_000
levers      = np.linspace(0.0, 0.99, 100)

# ── 4. gather data without repeated np.append ────────
thrust_N = []
tsfc     = []

for lv in levers:
    thrust_N.append(eng.get_thrust_with_lever_position(lv, mach, altitude_ft))
    tsfc.append(eng.get_tsfc())

tsfc = np.asarray(tsfc)

# ── 5. plot ──────────────────────────────────────────
plt.plot(levers * 100, tsfc)
plt.xlabel("Throttle lever position (%)")
plt.ylabel("TSFC (kg·s⁻¹·N⁻¹)")
plt.title(f"PW1127G-JM TSFC vs Throttle  (Mach {mach}, {altitude_ft:,} ft)")
plt.grid(True)
plt.tight_layout()
plt.show()
