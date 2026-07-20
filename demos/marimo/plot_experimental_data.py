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

    _DATA_DIR = Path("/home/brendan/Documents/PhD/Year1/EDA/Programs/stochprocsim/data")
    # includes the pulled autoqstochmeasure clone; timestamped names sort chronologically
    _files = sorted(_DATA_DIR.rglob("measurement_*.csv"), key=lambda p: p.name)
    file_picker = mo.ui.dropdown(
        options={str(p.relative_to(_DATA_DIR)): p for p in _files},
        value=str(_files[-1].relative_to(_DATA_DIR)),
        label="Measurement file",
    )
    file_picker
    return (file_picker,)


@app.cell(hide_code=True)
def _(file_picker):
    import re

    DATA_PATH = file_picker.value
    N = int(re.search(r"_N(\d+)_", DATA_PATH.name).group(1))  # picks Causal_Models[N]

    # coincidences with the herald: N exit bins, then the ch7 dump
    BINS = ["coinc_ch2", "coinc_ch4", "coinc_ch6", "coinc_ch8"][:N] + ["coinc_ch7"]
    assert len(BINS) == N + 1
    return BINS, DATA_PATH, N


@app.cell
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


@app.cell
def _(N, np, rates):
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel

    _sim = Simulator(QuantumTransitionModel(Causal_Models[N]))
    p_theory = {}
    for _s in rates.index:
        _p = _sim.get_output_distribution(propagate_outputs=True, start_state=int(_s))
        p_theory[_s] = np.array(_p + [1 - sum(_p)])  # last bin: remainder to the dump
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


@app.cell
def _(p_theory, plot_state_grid, rates, stds):
    # raw fraction per bin vs theory (no loss correction)
    plot_state_grid(
        {_s: rates.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        {_s: stds.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        p_theory,
        "raw fractions vs theory",
    )
    return


@app.cell
def _(N, np, p_theory, plot_state_grid, rates, stds):
    # one background c and per-loop transmission T fitted jointly over all
    # input states (bin k has traversed k loops, dump counted like the last exit)
    from scipy.optimize import minimize

    _k = np.arange(N + 1)

    def _resid(params):
        _c, _T = params
        _tot = 0
        for _s in rates.index:
            _corr = (rates.loc[_s].values - _c) / _T ** _k
            _tot += np.sum((p_theory[_s] - _corr / _corr.sum()) ** 2)
        return _tot

    c_fit, T_fit = minimize(_resid, x0=[1, 0.4],
                            bounds=[(0, 0.9 * rates.values.min()), (0.05, 1)]).x
    print(f"joint fit: c = {c_fit:.1f}/s  T = {T_fit:.3f}  residual = {_resid([c_fit, T_fit]):.4f}")

    frac, yerr = {}, {}
    for _s in rates.index:
        _corr = (rates.loc[_s] - c_fit) / T_fit ** _k
        frac[_s] = _corr / _corr.sum()
        yerr[_s] = (stds.loc[_s] / T_fit ** _k) / _corr.sum()
    plot_state_grid(frac, yerr, p_theory,
                    f"loss-corrected (c={c_fit:.0f}/s, T={T_fit:.2f}) vs theory")
    return (frac,)


@app.cell
def _(N, frac, np, p_theory, rates):
    # conditional KL divergence vs theory: each input state's divergence weighted
    # by the stationary probability of the machine being in that state, then also
    # as a rate (bits per loop). With one input state this is just its plain KL.
    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge

    # same weighting as kl_divergence.py: stationary probs of s0..s(N-1);
    # the extra prepared state sN is never occupied in steady operation
    _pi = get_uniform_renewal(N - 1).steady_state
    _w = np.array([_pi[int(_s)] if int(_s) < N else 0.0 for _s in rates.index])
    _w = _w / _w.sum()  # renormalise over the states actually measured

    kl_raw = kl_fit = 0.0
    for _s, _ws in zip(rates.index, _w):
        _r = eval_diverge(rates.loc[_s].values / rates.loc[_s].sum(), p_theory[_s])
        _f = eval_diverge(frac[_s].values, p_theory[_s])
        kl_raw += _ws * _r
        kl_fit += _ws * _f
        print(f"s{_s} (w={_ws:.2f}): raw {_r:.4f}  fitted {_f:.4f} bits")
    print(f"stationary-weighted: raw {kl_raw:.4f}  fitted {kl_fit:.4f} bits")
    print(f"rate over {N} loops: raw {kl_raw / N:.4f}  fitted {kl_fit / N:.4f} bits/step")
    return eval_diverge, get_uniform_renewal


@app.cell
def _(
    Causal_Models,
    N,
    QuantumTransitionModel,
    Simulator,
    eval_diverge,
    frac,
    get_uniform_renewal,
    np,
    plt,
):
    # divergence-rate-vs-N plot from demo.py, with our measured rate added:
    # build a renewal model from the corrected s0 exit probabilities (the
    # renewal reconstruction is defined from memory state 0), same recipe
    # as the simulated quantum curve
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

    _p_exp = frac[0].values[:N]
    _q_exp = generate_quantum_model(_p_exp)
    _m_exp = get_uniform_renewal(N - 1)
    rate_exp = eval_diverge(_m_exp.gen_dists(N)[0], _q_exp.gen_dists(N)[0], his_steps=N - 1)
    print(f"experimental rate (N={N}): {rate_exp:.4f}")

    _palette = ["#3B5BA5", "#800E13", "#E63946"]
    _fig, _ax = plt.subplots()
    _ax.plot(_x_class, _y_classical, '-s', label='Classical bound', color=_palette[0], linewidth=1.5)
    _ax.errorbar(_x_quant, _y_quantum, yerr=_qtheo_yerr, fmt='-^', capsize=3,
                 label='Quantum (Target)', color=_palette[1], linewidth=1.5)
    _ax.axhline(rate_exp, color=_palette[2], linestyle=':', linewidth=1.5,
                label=f'Experiment N={N}')
    _ax.set_ylim(0.005, 0.06)
    _ax.fill_between(_x_class, _y_classical, _ax.get_ylim()[1], alpha=0.3,
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
