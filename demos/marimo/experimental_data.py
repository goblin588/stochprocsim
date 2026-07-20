import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return np, pd, plt


@app.cell
def _():
    # DATA_PATH = "/home/brendan/Documents/PhD/Year1/EDA/Programs/stochprocsim/src/stochprocsim/data/2026-07-09-1414_Trace_file_fullpower.txt"
    DATA_PATH = "/home/brendan/Documents/PhD/Year1/EDA/Programs/stochprocsim/src/stochprocsim/data/2026-07-09-2120_Trace_file.txt"
    N = 3  # picks Causal_Models[N]

    HERALD = "Ch3"
    BINS = ["Ch2+Ch3", "Ch3+Ch4", "Ch3+Ch6", "Ch3+Ch7"]  # '1','01','001','0000'(dump)
    assert len(BINS) == N + 1
    return BINS, DATA_PATH, HERALD, N


@app.cell
def _(BINS, DATA_PATH, HERALD, pd):
    # header line holds the names; data rows have extra derived columns at the end,
    # so read only the named ones
    with open(DATA_PATH) as _f:
        _names = _f.readline().strip().split("\t")

    data = pd.read_csv(DATA_PATH, sep="\t", skiprows=1, header=None,
                       names=_names, usecols=range(len(_names)))
    data = data.dropna()
    data = data[data["Duration"] > 0.1]  # partial acquisition rows
    data = data[data[HERALD] > 0]        # herald blocked / off segments

    counts = data[BINS].mean()
    stds = data[BINS].std()
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
    return (p_theory,)


@app.cell
def _():
    # T = 0.1
    # eta_dump = 1
    # eta1=1*(T**0)
    # eta2=eta1*(T**1)
    # eta3 = eta2*(T**2)
    # eta4 = eta3*(T**3)*eta_dump
    # backg1 = 41.3
    # backgdmp = 19.5
    # countsgoblin= np.array([(1/eta1) * ((counts[0])-backg1),  (1/eta2) * (counts[1]-backg1),  (1/eta3) * (counts[2]-backg1),  0*(1/eta4) * (counts[3]-backgdmp)])

    # # raw fraction per bin vs theory (no loss correction)
    # _x = np.arange(len(countsgoblin))
    # _fig, _ax = plt.subplots(figsize=(6, 4))
    # _ax.bar(_x, countsgoblin / countsgoblin.sum(), yerr=stds / countsgoblin.sum(), capsize=5,
    #         width=0.5, color="gold", ecolor="black", label="data")
    # _ax.bar(_x, p_theory, width=0.5, fill=False, edgecolor="black",
    #         linestyle="dotted", linewidth=1.5, label="theory")
    # _ax.set_xticks(_x, range(0,4))
    # _ax.set_ylabel("Quantity")
    # _ax.legend()
    # _fig
    return


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
        _corr = (counts.values - _c) /(_T ** _k)
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
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
