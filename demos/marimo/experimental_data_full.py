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
    _DATA_DIR = "/home/brendan/Documents/PhD/Year1/EDA/Programs/stochprocsim/src/stochprocsim/data"
    DATA_PATHS = {
        "full power": f"{_DATA_DIR}/2026-07-09-1414_Trace_file_fullpower.txt",
        "half power": f"{_DATA_DIR}/2026-07-09-1407_Trace_file_halfpower.txt",
    }

    # U3 run: herald Ch3; loop outputs Ch2, Ch4, Ch6, Ch8 = strings 1, 01, 001, 0001;
    # Ch7 = dump ("0000", everything remaining is switched out on the 4th pass).
    #
    # Detector architecture: ONE loop-output SNSPD is fanned out electrically to
    # tagger channels 2/4/6/8, so those singles are identical by construction. Which
    # pass a photon exited on is encoded in each coincidence channel's delay relative
    # to the herald (Ch3) — the ch3_chN coincidences ARE the process data. Ch7 is the
    # separate dump detector.
    #
    # Column labels in the 07-09 exports are correct (the 07-08 ones were scrambled),
    # so the mapping is by name. Data rows have trailing unnamed derived columns
    # (ratio, constant) -> ignored.
    NAME_MAP = {
        "Duration": "duration", "Ch2": "ch2_singles", "Ch7": "ch7_singles",
        "Ch2+Ch3": "ch3_ch2", "Ch3+Ch4": "ch3_ch4", "Ch3+Ch6": "ch3_ch6",
        "Ch3+Ch8": "ch3_ch8", "Ch3+Ch7": "ch3_ch7", "Ch3": "ch3_singles",
    }
    DUPLICATE_SINGLES = ["Ch4", "Ch6", "Ch8"]
    STRINGS = {"ch3_ch2": "1", "ch3_ch4": "01", "ch3_ch6": "001",
               "ch3_ch8": "0001", "ch3_ch7": "0000"}
    return DATA_PATHS, DUPLICATE_SINGLES, NAME_MAP, STRINGS


@app.cell(hide_code=True)
def _(DATA_PATHS, DUPLICATE_SINGLES, NAME_MAP, np, pd):
    ## READ + INTEGRITY-CHECK + CLEAN
    def _load(path):
        # Read header separately: data rows have extra trailing derived columns
        with open(path) as _f:
            _names = [c.strip() for c in _f.readline().strip().split("\t")]
        _missing = set(NAME_MAP) - set(_names)
        assert not _missing, f"expected columns missing from header: {_missing}"

        _raw = pd.read_csv(path, sep="\t", skiprows=1, header=None,
                           usecols=range(len(_names)))
        _raw.columns = _names
        # Drop repeated header rows and truncated rows (non-numeric first column)
        _raw = _raw[pd.to_numeric(_raw["Duration"], errors="coerce").notna()]
        _raw = _raw.astype(float).reset_index(drop=True)

        # Ch2/4/6/8 are one SNSPD fanned out to four tagger channels, so their
        # singles must agree; a mismatch means a fan-out/logging problem
        _ok = _raw.dropna()
        for _c in DUPLICATE_SINGLES:
            assert ((_ok[_c] - _ok["Ch2"]).abs() < 0.01 * _ok["Ch2"] + 1).all(), \
                f"{_c} differs from Ch2 but both carry the same fanned-out SNSPD signal"

        _d = _raw[list(NAME_MAP)].rename(columns=NAME_MAP)

        # Drop short-duration rows and any mid-run segment where the herald reads 0
        # (herald blocked / setup being reconfigured — the 07-08 runs had ~20 s of
        # this), then IQR outlier filter (as in prelim_data)
        _d = _d[_d["duration"] > 0.5]
        _n_off = int((_d["ch3_singles"] == 0).sum())
        if _n_off:
            print(f"   dropping {_n_off} rows with Ch3 = 0 (herald off)")
        _d = _d[_d["ch3_singles"] > 0]

        assert _d["ch3_ch8"].mean() < 5, "Ch3+Ch8 should be ~zero for U3"
        assert (_d["ch3_singles"] > _d["ch3_ch2"]).all(), \
            "Ch3 singles must exceed the Ch2+Ch3 coincidence"
        _mask = np.ones(len(_d), dtype=bool)
        for _col in _d.columns.drop("duration"):
            _q1, _q3 = _d[_col].quantile(0.25), _d[_col].quantile(0.75)
            _iqr = _q3 - _q1
            if _iqr > 0:  # zero-IQR channel (ch3_ch8 ~ 0) would reject any count > 0
                _mask &= (_d[_col] >= _q1 - 3 * _iqr) & (_d[_col] <= _q3 + 3 * _iqr)
        return _d[_mask].reset_index(drop=True), len(_raw)

    datasets = {}
    for _label, _path in DATA_PATHS.items():
        _d, _nraw = _load(_path)
        datasets[_label] = _d
        print(f"{_label}: kept {len(_d)}/{_nraw} rows")
        for _k in _d.columns.drop("duration"):
            print(f"   {_k:12s} mean={_d[_k].mean():10.3f}  std={_d[_k].std():.3f}")
    return (datasets,)


@app.cell(hide_code=True)
def _(STRINGS, datasets, mo, np, plt):
    ## COUNT-RATE TRACES
    _window = 10
    _keys = ["ch3_singles", "ch2_singles", "ch7_singles",
             "ch3_ch2", "ch3_ch4", "ch3_ch6", "ch3_ch8", "ch3_ch7"]

    _figs = []
    for _label, _d in datasets.items():
        _fig, _axes = plt.subplots(2, 4, figsize=(13, 6))
        for _ax, _key in zip(_axes.flat, _keys):
            _y = _d[_key].values
            _ma = np.convolve(_y, np.ones(_window) / _window, mode="valid")
            _ax.scatter(np.arange(len(_y)), _y, s=4, alpha=0.3, color="steelblue")
            _ax.plot(np.arange(_window - 1, len(_y)), _ma, color="black", linewidth=1.2)
            _ax.set_title(f"{_key} ({STRINGS[_key]})" if _key in STRINGS else _key)
            _ax.set_xlabel("Sample")
            _ax.set_ylabel("Count rate")
        _fig.suptitle(_label)
        _fig.tight_layout()
        _figs.append(_fig)
    mo.vstack(_figs)
    return


@app.cell
def _(np):
    ## THEORY: U3 output distribution over strings 1, 01, 001, 0000(dump)
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel

    CS = Causal_Models[3]
    _sim = Simulator(QuantumTransitionModel(CS))
    _p3 = _sim.get_output_distribution_exp(propagate_outputs=True)
    # 4th pass switches everything remaining to the Ch7 dump (Ch8 gets nothing)
    p_theory = np.array(_p3 + [1.0 - sum(_p3)])
    print(f"U_theo        p(1,01,001,0000) = {np.round(p_theory, 4)}")

    # Waveplate-implemented unitary barely differs — U_theo is a fair reference
    CS.set_U(CS.U_optics_NTU)
    _p3o = _sim.get_output_distribution_exp(propagate_outputs=True)
    print(f"U_optics_NTU  p(1,01,001,0000) = {np.round(np.array(_p3o + [1 - sum(_p3o)]), 4)}")
    CS.set_U(CS.U_theo)

    # Diagnostic alternative: if the loop keeps the OTHER PBS port ([1,3] instead of
    # [0,2]), the kept state after every pass is the reset state s0 (⟨u24|s0⟩ = 1),
    # so the exit probability is the same every pass -> geometric decay.
    # This shape described the (superseded) 07-08 runs with T ≈ 0.55.
    _v0 = CS.U_theo @ CS.states[0]
    _a0 = float(np.linalg.norm(_v0[[0, 2]]) ** 2)  # exits if ports are swapped
    _b0 = float(np.linalg.norm(_v0[[1, 3]]) ** 2)  # stays if ports are swapped
    p_swapped = np.array([_a0, _b0 * _a0, _b0**2 * _a0, _b0**3])
    print(f"Swapped-port  p(1,01,001,0000) = {np.round(p_swapped, 4)}  (geometric)")
    return p_swapped, p_theory


@app.cell(hide_code=True)
def _(datasets, mo, np, p_swapped, p_theory, plt):
    ## FIT (background c, per-loop transmission T) PER MODEL AND COMPARE
    # Photon exiting bin k (ch3_ch2, ch3_ch4, ch3_ch6, ch3_ch7) has traversed k loops
    # -> weight T^k; the Ch7 dump exits on the 4th pass so it carries T^3, same number
    # of passes as Ch8 would.
    from scipy.optimize import minimize

    _bins = ["ch3_ch2", "ch3_ch4", "ch3_ch6", "ch3_ch7"]
    _tk = lambda T: T ** np.arange(4)

    def _fit(means, p):
        def _obj(params):
            _c, _T = params
            _corr = (means - _c) / _tk(_T)
            return float(np.sum((p - _corr / _corr.sum()) ** 2))
        # background can't exceed the smallest bin (corrected counts must stay > 0)
        _r = minimize(_obj, x0=[1.0, 0.4], bounds=[(0, 0.9 * means.min()), (0.05, 1.0)],
                      method="L-BFGS-B")
        return float(_r.x[0]), float(_r.x[1]), float(_r.fun)

    fits = {}  # (model, dataset) -> (c, T, residual)
    _x = np.arange(4)
    _w = 0.27
    _figs = []
    for _name, _p in [("standard U3", p_theory), ("swapped-port geometric", p_swapped)]:
        _fig, _ax = plt.subplots(figsize=(7, 4))
        for _i, (_label, _color) in enumerate(zip(datasets, ["gold", "steelblue"])):
            _d = datasets[_label]
            _means = np.array([_d[_b].mean() for _b in _bins])
            _stds = np.array([_d[_b].std() for _b in _bins])
            _c, _T, _res = _fit(_means, _p)
            fits[(_name, _label)] = (_c, _T, _res)
            _corr = (_means - _c) / _tk(_T)
            _ax.bar(_x + (_i - 0.5) * _w, _corr / _corr.sum(),
                    yerr=_stds / _tk(_T) / _corr.sum(), width=_w, capsize=4,
                    color=_color, ecolor="black",
                    label=f"{_label}: c={_c:.0f}, T={_T:.2f}, resid={_res:.3f}")
        _ax.bar(_x + _w, _p, width=_w, fill=False, edgecolor="black",
                linestyle="dotted", linewidth=1.5, label=_name)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(["ch3_ch2\n('1')", "ch3_ch4\n('01')",
                             "ch3_ch6\n('001')", "ch3_ch7\n('0000', dump)"])
        _ax.set_ylabel("Probability")
        _ax.set_title(f"Best-fit {_name}: data (bg sub, loss corr) vs model")
        _ax.legend()
        _fig.tight_layout()
        _figs.append(_fig)
    mo.vstack(_figs)
    return (fits,)


@app.cell(hide_code=True)
def _(datasets, np, p_swapped, p_theory, plt):
    ## RAW FRACTIONS (no correction) — the fit above is poor, look at the data as-is
    _bins = ["ch3_ch2", "ch3_ch4", "ch3_ch6", "ch3_ch7"]
    fig_raw, ax_raw = plt.subplots(figsize=(7.5, 4.5))
    _x = np.arange(4)
    _w = 0.2
    for _i, (_label, _color) in enumerate(zip(datasets, ["gold", "steelblue"])):
        _means = np.array([datasets[_label][_b].mean() for _b in _bins])
        ax_raw.bar(_x + (_i - 1.5) * _w, _means / _means.sum(), width=_w,
                   color=_color, label=f"{_label} (raw)")
    ax_raw.bar(_x + 0.5 * _w, p_theory, width=_w, fill=False, edgecolor="black",
               linestyle="dotted", linewidth=1.5, label="Theory U3 (no loss)")
    ax_raw.bar(_x + 1.5 * _w, p_swapped, width=_w, fill=False, edgecolor="dimgray",
               linestyle="dashed", linewidth=1.5, label="Swapped-port geometric (no loss)")
    ax_raw.set_xticks(_x)
    ax_raw.set_xticklabels(["ch3_ch2\n('1')", "ch3_ch4\n('01')",
                            "ch3_ch6\n('001')", "ch3_ch7\n('0000', dump)"])
    ax_raw.set_ylabel("Fraction of herald coincidences")
    ax_raw.set_title("Raw coincidence fractions")
    ax_raw.legend()
    fig_raw.tight_layout()
    fig_raw
    return


@app.cell
def _(datasets, fits, np, p_swapped, p_theory):
    ## CONSISTENCY DIAGNOSTICS
    _bins = ["ch3_ch2", "ch3_ch4", "ch3_ch6", "ch3_ch7"]
    _models = {"standard U3": p_theory, "swapped-port geometric": p_swapped}

    print("Implied per-loop T from each consecutive bin ratio")
    print("(ch3_ch4/ch3_ch2, ch3_ch6/ch3_ch4, ch3_ch7/ch3_ch6 — one T should explain all):")
    for _label, _d in datasets.items():
        _m = np.array([_d[_b].mean() for _b in _bins])
        print(f"  {_label}: consecutive data ratios {np.round(_m[1:] / _m[:-1], 3)}")
        for _name, _p in _models.items():
            _T = (_m[1:] / _m[:-1]) / (_p[1:] / _p[:-1])
            print(f"    {_name:22s}: implied T = {np.round(_T, 3)}")

    print("\nModel comparison, fitted (c, T) and residual:")
    for (_name, _label), (_c, _T, _res) in fits.items():
        print(f"  {_label} / {_name:22s}: c={_c:.2f}  T={_T:.3f}  residual={_res:.4f}")

    print("\nFull -> half power scaling (accidentals would scale as power^2):")
    _full, _half = datasets["full power"], datasets["half power"]
    for _k in ["ch3_singles", "ch3_ch2", "ch3_ch4", "ch3_ch6", "ch3_ch7", "ch3_ch8"]:
        print(f"  {_k:11s} {_half[_k].mean() / _full[_k].mean():.3f}")

    # same SNSPD rate applies to every loop bin regardless of its delay setting
    _acc_loop = _full["ch3_singles"].mean() * _full["ch2_singles"].mean() * 2e-9
    _acc_dump = _full["ch3_singles"].mean() * _full["ch7_singles"].mean() * 2e-9
    print(f"\nAccidentals at ~1 ns window: herald x SNSPD = {_acc_loop:.2f}/s per loop "
          f"bin, herald x Ch7 = {_acc_dump:.2f}/s "
          f"(vs measured ch3_ch7 {_full['ch3_ch7'].mean():.0f}/s)")
    _her_eff = _full["ch3_ch2"].mean() / _full["ch3_singles"].mean()
    _tot = sum(_full[_b].mean() for _b in _bins) / _full["ch3_singles"].mean()
    print(f"Heralding: ch3_ch2/ch3_singles = {_her_eff:.3f}, "
          f"all coincidences/ch3_singles = {_tot:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Notes (2026-07-09, on the fixed 07-09 exports)

    **File format / architecture.** Column labels are correct in these exports (the
    07-08 ones were scrambled) and the loader maps by name. Ch2/4/6/8 singles are
    identical by design — one loop-output SNSPD fanned out to four tagger channels
    (asserted on load); the exit pass lives entirely in each coincidence channel's
    delay relative to the herald, so the ch3_chN coincidences are the process data.
    Trailing unnamed derived columns are ignored.

    **What checks out.**
    - ch3_ch8 ('0001') ≈ 1 count/s — the 4th-pass switch to the dump works.
    - Full- and half-power runs give the same coincidence fractions to <1%, and every
      coincidence channel scales linearly with the herald rate, so the distribution
      shape is not a multi-photon or accidentals artifact (accidentals ≲ 1/s at ns
      windows vs 103/s measured on ch3_ch7).

    **What changed vs the 07-08 runs.** ch3_ch2 / ch3_ch4 fractions are unchanged
    (~0.80 / ~0.13), but ch3_ch6 doubled (0.025 → 0.048) and the ch3_ch7 (dump)
    fraction fell ~2.5× (0.052 → 0.021); the dump excess over U3 theory went from ~13×
    to ~2.8×. Whatever was adjusted between runs mostly moved the late bins.

    **Main finding: the data still does not match the standard U3 mapping, but is well
    described by the swapped-port (geometric) model.**

    - *Standard U3* (loop keeps the [0,2] path, memory states step s0→s1→s2): no single
      per-loop transmission works — the implied T per consecutive bin ratio is
      0.11 / 1.00 / 2.8, and the best fit (residual ≈ 0.07) rails its background at the
      allowed maximum. ch3_ch2 is ~8× over-represented relative to ch3_ch4.
    - *Swapped-port model* (loop keeps the [1,3] path): the tick-branch output state is
      the reset state s0 (⟨u24|s0⟩ = 1), so the photon re-emits with the same
      probability a₀ = 0.677 every pass → geometric p = (0.677, 0.219, 0.071, 0.034)
      before loss. This fits ~30× better (residual ≈ 0.002), and the fitted parameters
      behave like real physics: **T = 0.506 (full) vs 0.499 (half) — power-independent —
      and background c = 84 vs 14 — scaling with power exactly as the herald does
      (0.172 vs 0.175)**. Residual structure: ch3_ch4 sits ~15% below and ch3_ch6 ~50%
      above the model, many σ of statistics, so it is not the full story either. The
      same geometric shape described the 07-08 runs (T ≈ 0.55) apart from their hot dump.

    If the swapped-port reading is right, the loop is re-preparing the reset state every
    pass and the output is a memoryless (geometric) process — the detected bins are not
    sampling the U3 causal-state progression at all. The prime suspect is the port
    convention: the code (`_split_paths` / `QuantumTransitionModel`) assumes indices
    [0,2] ("path2 NTU") stay in the loop; check which physical PBS port actually
    re-enters the delay.

    **Suggested cross-checks:** (1) confirm the loop-return port against the code's
    [0,2]-stays convention; (2) take a full herald–SNSPD start–stop delay histogram —
    one shot shows every pass peak, the coincidence-window alignment, and any flat
    pedestal between peaks; (3) block the loop input and look at the same histogram
    (first-pass leakage would explain the ch3_ch2 excess under the standard reading);
    (4) the fitted flat background c ≈ 84/s per bin (full power) is not accidentals —
    it is ~60× the ~1 ns-window accidental rate and it scales linearly with power
    where accidentals would scale quadratically — so it is herald-correlated light at
    randomized delays (reflections, afterpulsing, or window cross-talk between pass
    peaks); the histogram in (2) would show it directly.
    """
    )
    return


if __name__ == "__main__":
    app.run()
