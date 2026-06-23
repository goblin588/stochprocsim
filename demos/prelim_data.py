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


@app.cell
def _():
    DATA_PATH      = "../stochprocsim/src/stochprocsim/Data/2026-06-22-1834_Trace_Brendan_PrelimData_Unnormalised.txt"
    HERALD_CHANNEL = "Ch2"
    SIGNAL_CHANNEL = "Ch1"
    return DATA_PATH, HERALD_CHANNEL, SIGNAL_CHANNEL


@app.cell(hide_code=True)
def _(DATA_PATH, HERALD_CHANNEL, SIGNAL_CHANNEL, np, pd):
    ## READ DATA
    _raw = pd.read_csv(DATA_PATH, sep="\t", header=0)
    # Drop unnamed trailing columns, then repeated header rows, then cast
    _raw = _raw.loc[:, ~_raw.columns.str.startswith("Unnamed")]
    _first_col = _raw.columns[0]
    _raw = _raw[_raw[_first_col].astype(str) != _first_col].reset_index(drop=True)
    _raw = _raw.astype(float)

    ## AUTO-DETECT COINCIDENCE CHANNELS
    _coinc_cols = [c for c in _raw.columns if "+" in c]

    # s0 = signal+herald coincidence (try both orderings)
    _s0_col = f"{SIGNAL_CHANNEL}+{HERALD_CHANNEL}"
    if _s0_col not in _coinc_cols:
        _s0_col = f"{HERALD_CHANNEL}+{SIGNAL_CHANNEL}"

    # Remaining loop channels, sorted by the non-herald channel number
    def _ch_num(col):
        other = next(p for p in col.split("+") if p != HERALD_CHANNEL)
        return int(other.replace("Ch", "").replace("ch", ""))

    _loop_cols = sorted([c for c in _coinc_cols if c != _s0_col], key=_ch_num)
    N = len(_coinc_cols)  # total coincidence channels = memory depth

    print(f"Detected N={N}  |  s0={_s0_col}  |  loop channels={_loop_cols}")

    ## DATA CLEANING
    _count_cols = [HERALD_CHANNEL, SIGNAL_CHANNEL] + _coinc_cols

    # Drop short-duration rows (~0.1 s)
    _d = _raw[_raw["Duration"] > 0.5].copy()

    # IQR outlier filter
    _mask = np.ones(len(_d), dtype=bool)
    for _col in _count_cols:
        _q1, _q3 = _d[_col].quantile(0.25), _d[_col].quantile(0.75)
        _iqr = _q3 - _q1
        _mask &= (_d[_col] >= _q1 - 3 * _iqr) & (_d[_col] <= _q3 + 3 * _iqr)

    data = _d[_mask].reset_index(drop=True)
    print(f"Kept {len(data)} / {len(_raw)} rows  "
          f"({len(_raw)-len(_d)} short-duration, {len(_d)-len(data)} outliers removed)")

    ## COUNTS DICT
    counts = {
        "herald": data[HERALD_CHANNEL].values,
        "signal": data[SIGNAL_CHANNEL].values,
        "s0":     data[_s0_col].values,
        **{f"s{i+1}": data[col].values for i, col in enumerate(_loop_cols)},
    }

    for _k, _v in counts.items():
        print(f"{_k:8s}  mean={_v.mean():.3f}  std={_v.std():.3f}")
    return N, counts, data


@app.cell
def _(N, counts, np, plt):
    _window = 10
    _keys = ["herald", "signal"] + [f"s{k}" for k in range(N)]

    fig, axes = plt.subplots(2, (len(_keys) + 1) // 2, figsize=(12, 7))
    for ax, key in zip(axes.flat, _keys):
        _y = counts[key]
        _ma = np.convolve(_y, np.ones(_window) / _window, mode="valid")
        ax.scatter(np.arange(len(_y)), _y, s=4, alpha=0.3, color="steelblue")
        ax.plot(np.arange(_window - 1, len(_y)), _ma, color="black", linewidth=1.2)
        ax.set_title(key)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Count rate")

    # Hide any unused axes
    for ax in axes.flat[len(_keys):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig
    return axes, fig


@app.cell
def _(N, np):
    from stochprocsim.Models.CausalModels import Causal_Models

    _CS = Causal_Models[N]
    _U  = _CS.U_theo

    # P(exit at s_k) = P(tick | s_k) × prod_{j<k} P(survive | s_j)
    _a_prod = 1.0
    p_exit = []
    for _j in range(N):
        _v = _U @ _CS.states[_j]
        _a = float(np.linalg.norm(_v[[0, 2]]) ** 2)  # P(survive, path2 NTU)
        _b = float(np.linalg.norm(_v[[1, 3]]) ** 2)  # P(tick,    path1 NTU)
        p_exit.append(_b * _a_prod)
        _a_prod *= _a

    print(f"N={N}  Theory P(exit at s_k): {[f'{p:.4f}' for p in p_exit]}  sum={sum(p_exit):.4f}")
    return (p_exit,)


@app.cell
def _(N, counts, np, p_exit, plt):
    _T = 0.37
    _keys = [f"s{k}" for k in range(N)]
    _means = [counts[k].mean() for k in _keys]
    _stds  = [counts[k].std()  for k in _keys]

    _corrected      = [_means[k] / (_T ** k) for k in range(N)]
    _corrected_stds = [_stds[k]  / (_T ** k) for k in range(N)]
    _theory         = [p * sum(_corrected) for p in p_exit]

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    _x = np.arange(N)
    ax_bar.bar(_x, _corrected, yerr=_corrected_stds, capsize=5, width=0.5, color="gold", ecolor="black", label="Data (loss corrected)")
    ax_bar.bar(_x, _theory, width=0.5, fill=False, edgecolor="black", linestyle="dotted", linewidth=1.5, label=f"Theory (N={N})")
    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(_keys)
    ax_bar.set_xlabel("State")
    ax_bar.set_ylabel("Mean count rate")
    ax_bar.set_title(f"Simulator Steadystate Output  (T={_T:.2f})")
    ax_bar.set_ylim(0, max(_corrected) + 3 * max(_corrected_stds))
    ax_bar.legend()
    fig_bar.tight_layout()
    fig_bar
    return ax_bar, fig_bar


@app.cell
def _(N, counts, np, p_exit):
    from scipy.optimize import minimize

    _keys  = [f"s{k}" for k in range(N)]
    _means = np.array([counts[k].mean() for k in _keys])
    _p     = np.array(p_exit)

    def _objective(params):
        c, T = params
        tk        = np.array([T ** k for k in range(N)])
        corrected = (_means - c) / tk
        fracs     = corrected / corrected.sum()
        return float(np.sum((_p - fracs) ** 2))

    _result = minimize(
        _objective,
        x0=[5.0, 0.5],
        bounds=[(0, 30), (0.1, 1.0)],
        method="L-BFGS-B",
    )

    c_fit, T_fit = float(_result.x[0]), float(_result.x[1])
    print(f"Fitted background:    c = {c_fit:.3f} counts")
    print(f"Fitted transmission:  T = {T_fit:.4f}")
    print(f"Residual: {_result.fun:.6f}")
    return T_fit, c_fit


@app.cell
def _(N, T_fit, c_fit, counts, np, p_exit, plt):
    _keys  = [f"s{k}" for k in range(N)]
    _means = np.array([counts[k].mean() for k in _keys])
    _stds  = np.array([counts[k].std()  for k in _keys])
    _tk    = np.array([T_fit ** k for k in range(N)])

    _corrected      = (_means - c_fit) / _tk
    _corrected_stds = _stds / _tk
    _theory         = np.array(p_exit) * _corrected.sum()

    fig_fit, ax_fit = plt.subplots(figsize=(6, 4))
    _x = np.arange(N)
    ax_fit.bar(_x, _corrected, yerr=_corrected_stds, capsize=5, width=0.5, color="gold", ecolor="black", label="Data (bg sub, loss corrected)")
    ax_fit.bar(_x, _theory, width=0.5, fill=False, edgecolor="black", linestyle="dotted", linewidth=1.5, label=f"Theory (N={N})")
    ax_fit.set_xticks(_x)
    ax_fit.set_xticklabels(_keys)
    ax_fit.set_xlabel("State")
    ax_fit.set_ylabel("Mean count rate")
    ax_fit.set_title(f"Fitted: T={T_fit:.3f}, c={c_fit:.1f}")
    ax_fit.set_ylim(0, max(_corrected) + 3 * max(_corrected_stds))
    ax_fit.legend()
    fig_fit.tight_layout()
    fig_fit
    return ax_fit, fig_fit


if __name__ == "__main__":
    app.run()