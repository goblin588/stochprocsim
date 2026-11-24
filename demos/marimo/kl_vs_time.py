import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from stochprocsim.utils import (
        background_rate_by_channel,
        drop_count_outliers,
        efficiency_for_channel,
        loop_pass_k,
        loss_rates_for,
        kl_rate_from_output_distribution,
        kl_rate_mc_error,
    )
    return (
        Path,
        background_rate_by_channel,
        drop_count_outliers,
        efficiency_for_channel,
        kl_rate_from_output_distribution,
        kl_rate_mc_error,
        loop_pass_k,
        loss_rates_for,
        mo,
        np,
        pd,
        plt,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    filepath = "/home/brendan/Documents/PhD/Programs/stochprocsim/data/"
    refresh_button = mo.ui.run_button(label="Refresh file list")
    mo.hstack([filepath, refresh_button], justify="start")
    return filepath, refresh_button


@app.cell(hide_code=True)
def _(Path, filepath, mo, re, refresh_button):
    refresh_button  # ponytail: re-scan trigger, value itself unused
    _DATA_DIR = Path(filepath)
    _files = sorted(_DATA_DIR.glob("*measurement_N*.csv"),
                     key=lambda p: re.search(r"\d{8}_\d{6}", p.name).group())
    mo.stop(not _files, mo.md(f"No measurement CSV files found in `{_DATA_DIR}`"))
    file_picker = mo.ui.dropdown(
        options={str(p.relative_to(_DATA_DIR)): p for p in _files},
        value=str(_files[-1].relative_to(_DATA_DIR)),
        label="Measurement file",
    )
    file_picker
    return (file_picker,)


@app.cell(hide_code=True)
def _(file_picker, mo, pd, re):
    DATA_PATH = file_picker.value
    N = int(re.search(r"_N(\d+)_", DATA_PATH.name).group(1))

    _cols = pd.read_csv(DATA_PATH, nrows=0).columns
    _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
    BINS = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
    if "coinc_ch7" in _cols:
        BINS = BINS + ["coinc_ch7"]

    # default 50, not 1 -- the MC error band needs >=2 rows/window to
    # compute a distribution std at all (see the MC cell below); r=1 still
    # works for the plain KL-vs-time line, just with a zero/meaningless
    # error band if the MC checkbox is on
    r_control = mo.ui.number(value=50, start=1, label="recompute KL every r rows")
    mc_control = mo.ui.checkbox(value=False, label="show Monte Carlo error band (slower)")
    n_samples_control = mo.ui.number(value=200, start=20, step=20, label="MC samples")
    mo.vstack([mo.md(f"**{DATA_PATH.name}** — N={N}, bins={BINS}"),
               r_control, mo.hstack([mc_control, n_samples_control])])
    return BINS, DATA_PATH, N, mc_control, n_samples_control, r_control


@app.cell(hide_code=True)
def _(
    BINS,
    DATA_PATH,
    background_rate_by_channel,
    efficiency_for_channel,
    loss_rates_for,
    pd,
):
    # loss_rates_for does the "nearest *_loss_calibration.json at or before
    # this file's own timestamp" lookup itself (utils.py, fixed 2026-11-23
    # -- its directory fallback used to just grab whichever calibration
    # was newest ANYWHERE in data/, regardless of this file's own date,
    # confirmed wrong on real data), so this notebook doesn't need its own
    # copy of that lookup.
    loss_rates, loss_note = loss_rates_for(DATA_PATH.with_suffix(".json"))
    c2, c4 = loss_rates.get("loop0_hz", 0.0), loss_rates.get("loop1_hz", 0.0)
    c2_err, c4_err = loss_rates.get("loop0_hz_err", 0.0), loss_rates.get("loop1_hz_err", 0.0)
    T_measured = c4 / c2 if c2 else None
    T_measured_std = (T_measured * ((c4_err / c4) ** 2 + (c2_err / c2) ** 2) ** 0.5
                       if c2 and c4 else 0.0)

    # background taken as exactly measured, no error term -- per request
    # 2026-11-23 (kl_rate_mc_error's bg_std already defaults to None/0 for
    # this, so simply not passing it is sufficient).
    bg_by_ch, bg_note = background_rate_by_channel(DATA_PATH.with_suffix(".json"))
    bg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in BINS})
    eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7)
                      for b in BINS})
    print(f"loss calibration: {loss_note}  T_measured={T_measured}  background: {bg_note}")
    return T_measured, T_measured_std, bg, eff


@app.cell(hide_code=True)
def _(BINS, DATA_PATH, drop_count_outliers, pd):
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["herald", "int_time", "input_state"])
    data = data[data["herald"] > 0]
    # only s0 is used for the KL metric, same convention as plot_experimental_data.py
    s0 = data[data["input_state"] == 0].reset_index(drop=True)
    s0["time"] = pd.to_datetime(s0["time"])

    # a single catastrophic detector glitch (one row reading 100-1000x its
    # channel's normal count -- confirmed on real data: one such row alone
    # blew up a file's R_KL error bar 5x) will wreck whichever window it
    # falls into otherwise; strip those out before any windowing/stats, not
    # after -- see drop_count_outliers and this notebook's sibling
    # 2026-11-23 investigation (plot_experimental_data.py's last-3-days
    # series) for how this was found.
    n_before = len(s0)
    s0 = drop_count_outliers(s0, BINS).reset_index(drop=True)
    if len(s0) < n_before:
        print(f"dropped {n_before - len(s0)} outlier row(s) (>15x that channel's median)")
    return (s0,)


@app.cell(hide_code=True)
def _(
    BINS,
    N,
    T_measured,
    T_measured_std,
    bg,
    eff,
    kl_rate_from_output_distribution,
    kl_rate_mc_error,
    loop_pass_k,
    mc_control,
    mo,
    n_samples_control,
    np,
    r_control,
    s0,
):
    mo.stop(T_measured is None, mo.md("**No usable loss calibration (loop0_hz=0) for this file — can't loss-correct, so no KL can be computed.**"))

    r = max(1, r_control.value)
    n_windows = len(s0) // r
    mo.stop(n_windows == 0, mo.md(f"Not enough rows ({len(s0)}) for r={r}."))

    k = np.array([N if b == "coinc_ch7" else loop_pass_k(b, BINS) for b in BINS])
    transmission = T_measured ** k

    times, kls, kl_stds = [], [], []
    _any_unconverged = False
    for i in range(n_windows):
        window = s0.iloc[i * r:(i + 1) * r]
        rates = window[BINS].sum() / window["int_time"].sum()
        corr = ((rates - bg) / eff) / transmission
        frac = (corr.clip(lower=0) / corr.clip(lower=0).sum()).to_numpy()[:N]
        kls.append(kl_rate_from_output_distribution(N, frac))
        times.append(window["time"].iloc[-1])

        if mc_control.value:
            # per-window rate SEM = std(distribution)/sqrt(n) -- the
            # EMPIRICAL spread across this window's own individual-row
            # rates, not a theoretical sqrt(counts) Poisson estimate (that
            # was here before and silently assumed no excess/correlated
            # noise beyond pure shot noise -- the real data in this
            # project keeps turning up reasons not to assume that: residual
            # latch recovery, background exceeding signal, detector
            # glitches). Needs >=2 rows per window to have a std at all.
            window_rates = window[BINS].div(window["int_time"], axis=0)
            if len(window) > 1:
                rate_stds = window_rates.std(ddof=1) / np.sqrt(len(window))
            else:
                rate_stds = window_rates.iloc[0] * 0.0
            _, kl_std, _converged = kl_rate_mc_error(N, rates, rate_stds, T_measured, T_measured_std,
                                                       bg, eff, BINS, n_samples=n_samples_control.value,
                                                       check_convergence=True)
            kl_stds.append(kl_std)
            _any_unconverged = _any_unconverged or not _converged
    if mc_control.value and _any_unconverged:
        print(f"WARNING: at least one window's MC error didn't converge at "
              f"{n_samples_control.value} samples (two independent halves disagreed by >20%) "
              f"-- raise MC samples if you need the error bars to be trustworthy")
    return kl_stds, kls, times


@app.cell(hide_code=True)
def _(kl_stds, kls, mc_control, plt, times):
    fig, ax = plt.subplots(figsize=(9, 4))
    if mc_control.value and kl_stds:
        ax.errorbar(times, kls, yerr=kl_stds, fmt="-o", markersize=3, capsize=2)
    else:
        ax.plot(times, kls, "-o", markersize=3)
    ax.set_xlabel("time")
    # ax.set_ylim(0,0.01)

    ax.set_ylabel("KL rate (bits/step)")
    ax.set_title("KL divergence vs time (s0)")
    fig.autofmt_xdate()
    fig
    return


if __name__ == "__main__":
    app.run()
