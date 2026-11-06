# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""GPU-evaluated PCHIP (and not-a-knot cubic) samples of deno/plot1d.ts vs scipy.

    uv run deno/tools/check_pchip.py

For each function and n: slopes vs `PchipInterpolator(x, y)(x, 1)`; then the samples
`NurbsCurveSampler` evaluated on the GPU (f32, read back by `plot1d.ts samples`) vs scipy at the
same abscissae (sensitive to the f32 rounding of x where the slope is steep) and at the
samples' own parameters u (the evaluation error proper), relative to max |y|; and x(u) vs u.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

DENO = Path(__file__).resolve().parents[1]


def main():
    tmp = Path(tempfile.mkdtemp())
    worst = 0.0
    for name in ["bumps", "runge", "step", "blocks"]:
        for n in [8, 32, 128, 1024]:
            ctl, out = tmp / "c.json", tmp / "s.json"
            ctl.write_text(json.dumps({"function": name, "n": n}))
            subprocess.run(["deno", "run", "-A", str(DENO / "plot1d.ts"), "samples", str(ctl), str(out)],
                           check=True, cwd=DENO)
            r = json.loads(out.read_text())
            xs, ys = np.array(r["xs"]), np.array(r["ys"])
            ref = PchipInterpolator(xs, ys)
            d_ref = ref(xs, 1)
            dslope = np.abs(np.array(r["slopes"]) - d_ref).max() / max(np.abs(d_ref).max(), 1e-300)
            yscale = np.abs(ys).max()
            a, b = r["domain"]
            line = f"{name:7s} n={n:5d}: slopes vs scipy {dslope:.1e}"
            for curve, interp in (("pchip", ref), ("cubic", CubicSpline(xs, ys))):
                if curve not in r:
                    continue
                s = np.array(r[curve], dtype=np.float64)
                x, y, u = s[:, 0], s[:, 1], s[:, 2]
                dy = np.abs(y - interp(x)).max() / yscale
                xu = a + u * (b - a)  # the sample's own parameter (x(u) = u), in f64
                dyu = np.abs(y - interp(xu)).max() / yscale
                dx = np.abs(x - xu).max() / (b - a)
                if curve == "pchip":
                    worst = max(worst, dyu)
                line += (f" | {curve}: {len(s)} samples, |Δy|/max|y| at x {dy:.1e}, at u {dyu:.1e},"
                         f" |Δx(u)|/L {dx:.1e}")
            print(line)
    print(f"worst PCHIP relative error at the samples' parameters: {worst:.1e}")


if __name__ == "__main__":
    main()
