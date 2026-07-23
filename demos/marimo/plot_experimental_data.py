import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path

    _DATA_DIR = Path("/home/brendan/Documents/PhD/Year1/EDA/Programs/stochprocsim/data/autoqstochmeasure/data")
    # sort by mtime, not name — old files (measurement_N...) and new
    # date-first files (20260722_..._measurement_N...) don't collate the
    # same way lexicographically, so name-sort silently picks a stale file
    _files = sorted(_DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    file_picker = mo.ui.dropdown(
        options={str(p.relative_to(_DATA_DIR)): p for p in _files},
        value=str(_files[-1].relative_to(_DATA_DIR)),
        label="Measurement file",
    )
    file_picker
    return (file_picker,)


@app.cell(hide_code=True)
def _(file_picker, pd):
    import re

    DATA_PATH = file_picker.value
    N = int(re.search(r"_N(\d+)_", DATA_PATH.name).group(1))  # picks Causal_Models[N]

    # whatever coincidence channels this file actually recorded, in channel
    # order, with the ch7 herald dump (if present) moved to the end
    _cols = pd.read_csv(DATA_PATH, nrows=0).columns
    _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
    BINS = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
    if "coinc_ch7" in _cols:
        BINS = BINS + ["coinc_ch7"]
    return BINS, DATA_PATH, N


@app.cell(hide_code=True)
def _(BINS, DATA_PATH, pd):
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=BINS + ["herald", "int_time", "input_state"])
    data = data[data["herald"] > 0]  # herald blocked / off segments

    # counts per second, grouped by which memory state the machine started in;
    # a single-input file just yields one group and every cell below collapses
    # to the single-distribution case
    _rates = data[BINS].div(data["int_time"], axis=0)
    _rates["input_state"] = data["input_state"]
    _g = _rates.groupby("input_state")
    rates = _g.mean()
    stds = _g.std().div(_g.size() ** 0.5, axis=0)  # standard error of the mean
    print(f"{len(data)} rows kept, input states: {list(rates.index)}")
    return rates, stds


@app.cell(hide_code=True)
def _(BINS, N, np, rates):
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel

    _sim = Simulator(QuantumTransitionModel(Causal_Models[N]))
    p_theory = {}
    for _s in rates.index:
        _p = _sim.get_output_distribution(propagate_outputs=True, start_state=int(_s))
        # _p[i] is the probability of exiting on loop i+1 (channel 2*(i+1));
        # channels beyond the model's N loops (extra recorded channels) get 0
        _loop_p = {2 * (_i + 1): _v for _i, _v in enumerate(_p)}
        p_theory[_s] = np.array([
            1 - sum(_p) if _b == "coinc_ch7" else _loop_p.get(int(_b.removeprefix("coinc_ch")), 0.0)
            for _b in BINS
        ])
        print(f"s{_s}: {np.round(p_theory[_s], 4)}")
    return Causal_Models, QuantumTransitionModel, Simulator, p_theory


@app.cell(hide_code=True)
def _(np, plt):
    def plot_state_grid(frac_by_state, yerr_by_state, p_theory, suptitle):
        _states = sorted(frac_by_state)
        _ncols = min(3, len(_states))
        _nrows = -(-len(_states) // _ncols)
        _fig, _axes = plt.subplots(_nrows, _ncols,
                                   figsize=(4 * _ncols, 3 * _nrows), squeeze=False)
        for _ax in _axes.flat[len(_states):]:
            _ax.set_visible(False)
        for _ax, _s in zip(_axes.flat, _states):
            _f = frac_by_state[_s]
            _x = np.arange(len(_f))
            _ax.bar(_x, _f, yerr=yerr_by_state[_s], capsize=4,
                    width=0.5, color="gold", ecolor="black", label="data")
            _ax.bar(_x, p_theory[_s], width=0.5, fill=False, edgecolor="black",
                    linestyle="dotted", linewidth=1.5, label="theory")
            _ax.set_xticks(_x, _f.index)
            _ax.tick_params(axis="x", labelsize=7)
            _ax.set_title(f"input s{_s}")
        _axes.flat[0].legend()
        _fig.suptitle(suptitle)
        _fig.tight_layout()
        return _fig
    return (plot_state_grid,)


@app.cell(hide_code=True)
def _(p_theory, plot_state_grid, rates, stds):
    # raw fraction per bin vs theory (no loss correction)
    plot_state_grid(
        {_s: rates.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        {_s: stds.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        p_theory,
        "raw fractions vs theory",
    )
    return


@app.cell(hide_code=True)
def _(BINS, DATA_PATH, N, np, p_theory, pd, plot_state_grid, rates, stds):
    # background/noise per channel: read from the noise-calibration json that
    # was in effect for this run (measurement.py points each measurement's
    # sidecar at the calibration file used, by filename, resolved relative to
    # this same data dir) — not fit, so it can't silently absorb real
    # discrepancies. Falls back to zero (no correction) for older
    # measurements taken before calibrate_background() existed.
    import json as _json

    _json_path = DATA_PATH.with_suffix(".json")
    _bg_by_ch = {}
    _cal_note = "none (background=0)"
    if _json_path.exists():
        _cal_ref = _json.loads(_json_path.read_text()).get("noise_calibration")
        if _cal_ref:
            _cal_path = DATA_PATH.parent / _cal_ref["file"]
            if _cal_path.exists():
                _bg_by_ch = _json.loads(_cal_path.read_text()).get("background_rate_hz", {})
                _cal_note = f"{_cal_ref['file']} ({_cal_ref['saved_at']})"
    print(f"noise calibration used: {_cal_note}")
    bg = pd.Series({_b: _bg_by_ch.get(_b.removeprefix("coinc_ch"), 0.0) for _b in BINS})

    # per-loop transmission T fitted over all input states (bin k has
    # traversed k loops, dump counted like the last exit) — real optical
    # loss per round trip, separate from detector background/noise above
    from scipy.optimize import minimize_scalar

    _k = np.array([N if _b == "coinc_ch7" else int(_b.removeprefix("coinc_ch")) // 2 for _b in BINS])
    def _resid(T):
        _tot = 0
        for _s in rates.index:
            _corr = (rates.loc[_s] - bg) / T ** _k
            _tot += np.sum((p_theory[_s] - _corr / _corr.sum()) ** 2)
        return _tot

    T_fit = minimize_scalar(_resid, bounds=(0.05, 1), method="bounded").x
    print(f"transmission fit: T = {T_fit:.3f}  residual = {_resid(T_fit):.4f}")

    frac, yerr = {}, {}
    for _s in rates.index:
        _corr = (rates.loc[_s] - bg) / T_fit ** _k
        frac[_s] = _corr / _corr.sum()
        yerr[_s] = (stds.loc[_s] / T_fit ** _k) / _corr.sum()
    plot_state_grid(frac, yerr, p_theory,
                    f"loss-corrected (background from json, T={T_fit:.2f}) vs theory")
    return (frac,)


@app.cell(hide_code=True)
def _(
    Causal_Models,
    N,
    QuantumTransitionModel,
    Simulator,
    frac,
    np,
    plt,
):
    # divergence-rate-vs-N plot from demo.py, with our measured rate added:
    # build a renewal model from the corrected s0 exit probabilities (the
    # renewal reconstruction is defined from memory state 0), same recipe
    # as the simulated quantum curve
    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.utils import generate_quantum_model

    def _sim_quantum(n):
        _cs = Causal_Models[n]
        _cs.set_U(_cs.U_theo)
        _q = generate_quantum_model(
            np.array(Simulator(QuantumTransitionModel(_cs)).get_output_distribution()))
        _m = get_uniform_renewal(n - 1)
        return eval_diverge(_m.gen_dists(n)[0], _q.gen_dists(n)[0], his_steps=n - 1)

    _NMIN, _NMAX, _C_BUFF = 3, 6, 1
    _x_quant = list(range(_NMIN, _NMAX + 1))
    _x_class = list(range(_NMIN, _NMAX + _C_BUFF))
    _y_quantum = [_sim_quantum(_n) for _n in _x_quant]
    _y_classical = [get_uniform_renewal(_n - 1).classical_bd(4, target_dim=2) for _n in _x_class]
    _qtheo_yerr = [0.002054339006460678, 0.0014301897956533847, 0.0016754397869550568, 0.0015075218255867487]

    # stationary-weighted over every measured input state, not just s0 —
    # same weighting as the raw/fitted KL cell above (sN's weight is 0)
    _pi_exp = get_uniform_renewal(N - 1).steady_state
    _w_exp = np.array([_pi_exp[int(_s)] if int(_s) < N else 0.0 for _s in frac])
    _w_exp = _w_exp / _w_exp.sum()
    _m_exp = get_uniform_renewal(N - 1)

    rate_exp = 0.0
    for _s, _ws in zip(frac, _w_exp):
        if _ws == 0:
            continue
        _q_exp = generate_quantum_model(frac[_s].values[:N])
        rate_exp += _ws * eval_diverge(_m_exp.gen_dists(N)[0], _q_exp.gen_dists(N)[0], his_steps=N - 1)
    print(f"stationary-weighted experimental rate (N={N}): {rate_exp:.4f}")

    _palette = ["#3B5BA5", "#800E13", "#C11B26"]
    _fig, _ax = plt.subplots()
    _ax.plot(_x_class, _y_classical, '-s', label='Classical bound', color=_palette[0], linewidth=1.5)
    _ax.errorbar(_x_quant, _y_quantum, yerr=_qtheo_yerr, fmt='-^', capsize=3,
                 label='Quantum (Target)', color=_palette[1], linewidth=1.5)
    _ax.plot(_x_class, [rate_exp] * len(_x_class), '--o',
             label=f'Experiment N={N}', color=_palette[2], linewidth=1.5)
    # _ax.set_ylim(0.005, 0.06)
    _ax.fill_between(_x_class, _y_classical, _ax.get_ylim()[1], alpha=0.4,
                     color='none', edgecolor=_palette[0], hatch='///')
    _ax.set_xticks(_x_class)
    _ax.set_xlabel('N')
    _ax.set_ylabel('Divergence Rate')
    _ax.set_title('KL Divergence Rate for Process Depth N')
    _ax.legend()
    _ax.grid(True)
    _fig
    return


if __name__ == "__main__":
    app.run()
