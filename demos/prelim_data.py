import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return np, pd, plt


@app.cell(hide_code=True)
def _(np, pd):
    ## READ DATA 

    _path = "../stochprocsim/src/stochprocsim/Data/2026-06-22-1834_Trace_Brendan_PrelimData_Unnormalised.txt"

    # File has repeated header rows — read all, drop rows where Duration is non-numeric
    _raw = pd.read_csv(_path, sep="\t", header=0, names=["Duration", "Ch1", "Ch2", "Ch1+Ch2", "Ch2+Ch3", "Ch2+Ch6", "Ch2+Ch4", "_8", "_9"])
    _raw = _raw[_raw["Duration"] != "Duration"].reset_index(drop=True)
    _raw = _raw.drop(columns=["_8", "_9"])

    data_raw = _raw.astype(float)


    ## DATA CLEANING

    _count_cols = ["Ch1", "Ch2", "Ch1+Ch2", "Ch2+Ch3", "Ch2+Ch4", "Ch2+Ch6"]

    # Drop 0.1s intervals
    _d = data_raw[data_raw["Duration"] > 0.5].copy()

    # IQR outlier filter on each count channel
    _mask = np.ones(len(_d), dtype=bool)
    for _col in _count_cols:
        _q1, _q3 = _d[_col].quantile(0.25), _d[_col].quantile(0.75)
        _iqr = _q3 - _q1
        _mask &= (_d[_col] >= _q1 - 3 * _iqr) & (_d[_col] <= _q3 + 3 * _iqr)

    data = _d[_mask].reset_index(drop=True)
    print(f"Kept {len(data)} / {len(data_raw)} rows  ({len(data_raw)-len(_d)} short-duration, {len(_d)-len(data)} outliers removed)")

    # COUNTS DICT
    counts = {
        "herald": data["Ch2"].values,
        "signal": data["Ch1"].values,
        "s0":     data["Ch1+Ch2"].values,
        "s1":     data["Ch2+Ch3"].values,
        "s2":     data["Ch2+Ch4"].values,
        "s3":     data["Ch2+Ch6"].values,
    }

    for _k, _v in counts.items():
        print(f"{_k:8s}  mean={_v.mean():.3f}  std={_v.std():.3f}  n={len(_v)}")
    return (counts,)


@app.cell
def _(counts, np, plt):
    _window = 10
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    _keys = ["herald", "signal", "s0", "s1", "s2", "s3"]

    for ax, key in zip(axes.flat, _keys):
        _y = counts[key]
        _ma = np.convolve(_y, np.ones(_window) / _window, mode="valid")
        ax.scatter(np.arange(len(_y)), _y, s=4, alpha=0.3, color="steelblue")
        ax.plot(np.arange(_window - 1, len(_y)), _ma, color="black", linewidth=1.2)
        ax.set_title(key)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Count rate")

    fig.tight_layout()
    fig
    return


@app.cell
def _(np):
    from stochprocsim.Models.Unitaries import U_4, U_4_states

    # P(exit at s_k) = P(tick | s_k) × prod_{j<k} P(survive | s_j)
    _a_prod = 1.0
    p_exit = []
    for _j in range(4):
        _v = U_4 @ U_4_states[_j]
        _a = float(np.linalg.norm(_v[[0, 2]]) ** 2)  # P(survive, path2 NTU)
        _b = float(np.linalg.norm(_v[[1, 3]]) ** 2)  # P(tick,    path1 NTU)
        p_exit.append(_b * _a_prod)
        _a_prod *= _a

    print("Theory P(exit at s_k):", [f"{p:.4f}" for p in p_exit], f"  sum={sum(p_exit):.4f}")
    return (p_exit,)


@app.cell
def _(counts, np, p_exit, plt):
    _T = 0.37
    _keys = ["s0", "s1", "s2", "s3"]
    _means = [counts[k].mean() for k in _keys]
    _stds  = [counts[k].std()  for k in _keys]

    _corrected      = [_means[k] / (_T ** k) for k in range(4)]
    _corrected_stds = [_stds[k]  / (_T ** k) for k in range(4)]

    _total  = sum(_corrected)
    _theory = [p * _total for p in p_exit]

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    _x = np.arange(4)
    ax_bar.bar(_x, _corrected, yerr=_corrected_stds, capsize=5, width=0.5, color="gold", ecolor="black", label="Data (loss corrected)")
    ax_bar.bar(_x, _theory, width=0.5, fill=False, edgecolor="black", linestyle="dotted", linewidth=1.5, label="Theory (N=4)")
    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(_keys)
    ax_bar.set_xlabel("State")
    ax_bar.set_ylabel("Mean count rate")
    ax_bar.set_title(f"Simulator Steadystate Output  (T={_T:.2f})")
    ax_bar.set_ylim(0, max(_corrected) + 3 * max(_corrected_stds))
    ax_bar.legend()
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell
def _(counts, np, p_exit):
    from scipy.optimize import minimize

    _keys = ["s0", "s1", "s2", "s3"]
    _means = np.array([counts[k].mean() for k in _keys])
    _p = np.array(p_exit)

    def _objective(params):
        c, T = params
        tk = np.array([T ** k for k in range(4)])
        corrected = (_means - c) / tk
        fracs = corrected / corrected.sum()
        return float(np.sum((_p - fracs) ** 2))

    result = minimize(
        _objective,
        x0=[5.0, 0.5],
        bounds=[(0, 30), (0.1, 1.0)],
        method="L-BFGS-B",
    )

    c_fit, T_fit = float(result.x[0]), float(result.x[1])
    print(f"Fitted background:    c = {c_fit:.3f} counts")
    print(f"Fitted transmission:  T = {T_fit:.4f}")
    print(f"Residual: {result.fun:.4f}")
    return T_fit, c_fit, result


@app.cell
def _(T_fit, c_fit, counts, np, p_exit, plt, result):
    _keys = ["s0", "s1", "s2", "s3"]
    _means = np.array([counts[k].mean() for k in _keys])
    _stds  = np.array([counts[k].std()  for k in _keys])
    _tk    = np.array([T_fit ** k for k in range(4)])

    _corrected      = (_means - c_fit) / _tk
    _corrected_stds = _stds / _tk
    _theory         = np.array(p_exit) * _corrected.sum()

    fig_fit, ax_fit = plt.subplots(figsize=(6, 4))
    _x = np.arange(4)
    ax_fit.bar(_x, _corrected, yerr=_corrected_stds, capsize=5, width=0.5, color="gold", ecolor="black", label="Data (bg sub, loss corrected)")
    ax_fit.bar(_x, _theory, width=0.5, fill=False, edgecolor="black", linestyle="dotted", linewidth=1.5, label="Theory (N=4)")
    ax_fit.set_xticks(_x)
    ax_fit.set_xticklabels(_keys)
    ax_fit.set_xlabel("State")
    ax_fit.set_ylabel("Mean count rate")
    ax_fit.set_title(f"Fitted: T={T_fit:.3f}, c={c_fit:.1f}, fit={result.fun:.4f}")
    ax_fit.set_ylim(0, max(_corrected) + 3 * max(_corrected_stds))
    ax_fit.legend()
    fig_fit.tight_layout()
    fig_fit
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
