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
    data = data.dropna(subset=BINS + ["herald", "int_time"])
    data = data[data["herald"] > 0]  # herald blocked / off segments

    # counts per second, so rows with different integration times average cleanly
    _rates = data[BINS].div(data["int_time"], axis=0)
    counts = _rates.mean()
    stds = _rates.std() / len(data) ** 0.5  # standard error of the mean
    print(f"{len(data)} rows kept")
    print(counts)
    return counts, stds


@app.cell
def _(N, np):
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel

    _p = Simulator(QuantumTransitionModel(Causal_Models[N])).get_output_distribution(propagate_outputs=True)
    p_theory = np.array(_p + [1 - sum(_p)])  # last bin: remainder switched to the dump
    print(f"theory p = {np.round(p_theory, 4)}")
    return Causal_Models, QuantumTransitionModel, Simulator, p_theory


@app.cell
def _(counts, np, p_theory, plt, stds):
    # raw fraction per bin vs theory (no loss correction)
    _x = np.arange(len(counts))
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.bar(_x, counts / counts.sum(), yerr=stds / counts.sum(), capsize=5,
            width=0.5, color="gold", ecolor="black", label="data")
    _ax.bar(_x, p_theory, width=0.5, fill=False, edgecolor="black",
            linestyle="dotted", linewidth=1.5, label="theory")
    _ax.set_xticks(_x, counts.index)
    _ax.set_ylabel("Quantity")
    _ax.legend()
    _fig
    return


@app.cell
def _(counts, np, p_theory, plt, stds):
    # fit flat background c and per-loop transmission T (bin k is weighted by T^k,
    # the dump has traversed N loops like the last exit would)
    from scipy.optimize import minimize

    _k = np.arange(len(counts))

    def _resid(params):
        _c, _T = params
        _corr = (counts.values - _c) / (_T ** _k)
        return np.sum((p_theory - _corr / _corr.sum()) ** 2)

    c_fit, T_fit = minimize(_resid, x0=[1, 0.4],
                            bounds=[(0, 0.9 * counts.min()), (0.05, 1)]).x
    print(f"c = {c_fit:.1f}/s  T = {T_fit:.3f}  residual = {_resid([c_fit, T_fit]):.4f}")

    corrected = (counts - c_fit) / T_fit ** _k
    _x = np.arange(len(counts))
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.bar(_x, corrected / corrected.sum(),
            yerr=(stds / T_fit ** _k) / corrected.sum(), capsize=5,
            width=0.5, color="gold", ecolor="black",
            label=f"data (c={c_fit:.0f}, T={T_fit:.2f})")
    _ax.bar(_x, p_theory, width=0.5, fill=False, edgecolor="black",
            linestyle="dotted", linewidth=1.5, label="theory")
    _ax.set_xticks(_x, counts.index)
    _ax.set_ylabel("Quantity")
    _ax.legend()
    _fig
    return (corrected,)


@app.cell
def _(corrected, counts, p_theory):
    # KL divergence (bits over the whole exit distribution) vs theory;
    # scipy normalises inputs so the raw count rates can go in directly
    from stochprocsim.stochprocq.measure import eval_diverge

    print(f"eval_diverge raw:    {eval_diverge(counts.values, p_theory):.4f} bits")
    print(f"eval_diverge fitted: {eval_diverge(corrected.values, p_theory):.4f} bits")
    return (eval_diverge,)


@app.cell
def _(N, corrected, counts, p_theory):
    # KL divergence rate: bits per loop, averaged over the N loops
    from stochprocsim.stochprocq.measure import kl_div

    print(f"kl_div rate raw:    {kl_div(counts.values, p_theory, steps=N):.4f} bits/step")
    print(f"kl_div rate fitted: {kl_div(corrected.values, p_theory, steps=N):.4f} bits/step")
    return


@app.cell
def _(
    Causal_Models,
    N,
    QuantumTransitionModel,
    Simulator,
    corrected,
    eval_diverge,
    np,
    plt,
):
    # divergence-rate-vs-N plot from demo.py, with our measured N=3 rate added:
    # build a renewal model from the corrected exit probabilities, same recipe
    # as the simulated quantum curve
    from stochprocsim.stochprocq import get_uniform_renewal
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

    _p_exp = (corrected / corrected.sum()).values[:N]
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


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
