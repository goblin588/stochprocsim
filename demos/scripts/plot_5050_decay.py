"""Measured vs. expected per-loop-pass counts under the 50/50 calibration
optics (calibrate_loss's '5050' stage) -- the same channel-to-channel decay
fit that loss_per_loop_pass_5050 comes from (see autoqstochmeasure's
_fit_loop_decay), just plotted instead of collapsed to one ratio.

Expected curve is anchored at loop 1's own measured rate and decays by the
fitted ratio per pass; both series normalised to loop 1 = 1.0.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("/home/brendan/Documents/PhD/Programs/stochprocsim/data")
LOOP_CHS = [4, 6, 8, 10, 12, 14]  # loop pass 1..6, see autoqstochmeasure/src/libraries/settings.py
FIT_PASSES = 4  # loss balloons past pass 4, dragging the fit -- see autoqstochmeasure's calibrate_loss


def fit_loop_decay(bin_rates_by_ch):
    """Same log(rate)-vs-pass least-squares fit as autoqstochmeasure's
    _fit_loop_decay: returns (ratio_per_pass, ratio_err)."""
    points = [(k, math.log(r)) for k, r in enumerate(bin_rates_by_ch, start=1) if r > 0]
    n = len(points)
    kbar = sum(k for k, _ in points) / n
    ybar = sum(y for _, y in points) / n
    sxx = sum((k - kbar) ** 2 for k, _ in points)
    sxy = sum((k - kbar) * (y - ybar) for k, y in points)
    slope = sxy / sxx
    ratio = math.exp(slope)
    if n > 2:
        resid_sq = sum((y - (ybar + slope * (k - kbar))) ** 2 for k, y in points)
        s_err = (resid_sq / (n - 2)) ** 0.5
        ratio_err = ratio * (s_err / sxx ** 0.5)
    else:
        ratio_err = 0.0
    return ratio, ratio_err


def main():
    cal_path = sorted(DATA_DIR.glob("*_loss_calibration.json"))[-1]
    cal = json.loads(cal_path.read_text())
    bins = cal["stages"]["5050"]["counts"]["bins"]

    mean_rate, sem_rate = [], []
    for ch in LOOP_CHS:
        rates = [b[f"coinc_ch{ch}"] / b["int_time"] for b in bins]
        mean_rate.append(float(np.mean(rates)))
        sem_rate.append(float(np.std(rates, ddof=1) / len(rates) ** 0.5))

    ratio, ratio_err = fit_loop_decay(mean_rate[:FIT_PASSES])
    print(f"calibration file: {cal_path.name}")
    print(f"fitted decay ratio per pass = {ratio:.4f} +/- {ratio_err:.4f}  "
          f"(loss_per_loop_pass_5050 = ratio/0.5 = {ratio / 0.5:.4f})")
    print(f"saved value in file: {cal['losses'].get('loss_per_loop_pass_5050'):.4f}")

    k = np.arange(1, 7)
    measured = np.array(mean_rate) / mean_rate[0]
    measured_err = np.array(sem_rate) / mean_rate[0]
    expected = ratio ** (k - 1)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 5))
        _w = 0.35
        ax.bar(k - _w / 2, measured, width=_w, yerr=measured_err, capsize=4,
               color="gold", ecolor="black", label="Measured (5050 stage)")
        ax.bar(k + _w / 2, expected, width=_w, fill=False, edgecolor="#7A0C13",
               linestyle="dashed", linewidth=1.5,
               label=f"5050 expected (fit ratio={ratio:.3f}/pass)")
        ax.set_xticks(k)
        ax.set_xlabel("Loop pass")
        ax.set_ylabel("Counts (normalised to loop 1)")
        ax.set_title("50/50 calibration: measured vs. expected counts per loop pass")
        ax.legend()

    out = DATA_DIR.parent / "src/stochprocsim/Figures/5050_decay_per_loop.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
