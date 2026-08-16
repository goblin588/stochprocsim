import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import re

    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel
    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.utils import exit_distribution, generate_quantum_model, rates_by_input_state
    return (
        Causal_Models,
        Path,
        QuantumTransitionModel,
        Simulator,
        exit_distribution,
        generate_quantum_model,
        get_uniform_renewal,
        mo,
        np,
        pd,
        rates_by_input_state,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$R_{KL}[P||Q]=\sum_i{p(S_i)\sum_x{p(x|S_i)\log\frac{p(x|S_i)}{q(x|S_i)}}}$$

    - $S_i$: renewal state (already survived $i$ loops without exiting) — this is the internal state of the renewal-process models themselves, not a separately-measured input state
    - $x$: exit bin (which loop the photon leaves on, or "never" if it survives every remaining loop)
    - $P$ = the exact/target renewal process (`exact_model`): the experiment is trying to emulate this, so it's the reference/true distribution. $Q$ = the process fitted to the measured $p(x|S_0)$ exit fractions (`exp_model`)
    - $p(S_i)$: stationary probability of state $i$ under $P$ (`exact_model.steady_state`)
    - $p(x|S_i)$, $q(x|S_i)$: each model's own exit distribution conditioned on already being in state $i$, derived analytically from that model's per-loop survival probabilities — no separate per-state measurement needed
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    dir_input = mo.ui.text(
        value="/home/brendan/Documents/PhD/Year1/EDA/Programs/stochprocsim/data/autoqstochmeasure/data",
        label="Data directory",
        full_width=True,
    )
    refresh_button = mo.ui.run_button(label="Refresh file list")
    mo.hstack([dir_input, refresh_button], justify="start")
    return dir_input, refresh_button


@app.cell(hide_code=True)
def _(Path, dir_input, mo, re, refresh_button):
    refresh_button  # ponytail: re-scan trigger, value itself unused
    _DATA_DIR = Path(dir_input.value)
    # sort by the timestamp embedded in the filename, not name/mtime -- old
    # files (measurement_N..._YYYYMMDD_HHMMSS) and new date-first files
    # (YYYYMMDD_HHMMSS_measurement_N...) don't collate the same way
    # lexicographically, so plain name-sort can silently pick a stale file
    # (see plot_experimental_data.py, which had this same fix already)
    _files = sorted(_DATA_DIR.glob("*.csv"),
                     key=lambda p: re.search(r"\d{8}_\d{6}", p.name).group())
    mo.stop(not _files, mo.md(f"No CSV files found in `{_DATA_DIR}`"))
    file_picker = mo.ui.dropdown(
        options={str(p.relative_to(_DATA_DIR)): p for p in _files},
        value=str(_files[-1].relative_to(_DATA_DIR)),
        label="Measurement file",
    )
    file_picker
    return (file_picker,)


@app.cell(hide_code=True)
def _(file_picker, re):
    DATA_PATH = file_picker.value
    N = int(re.search(r"_N(\d+)_", DATA_PATH.name).group(1))

    # coincidences with the herald: exit channels are 2,4,6,...,2N, then the ch7 dump
    BINS = [f"coinc_ch{2 * k}" for k in range(1, N + 1)] + ["coinc_ch7"]
    assert len(BINS) == N + 1
    BINS
    return BINS, DATA_PATH, N


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(BINS, DATA_PATH, mo, pd, rates_by_input_state):
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=BINS + ["herald", "int_time", "input_state"])
    data = data[data["herald"] > 0]

    # for tomography files (with a `basis` column), rates_by_input_state
    # combines the conjugate basis pairs into a stand-in single distribution
    # — see its docstring; real state reconstruction is future work
    rates, _stds = rates_by_input_state(data, BINS)
    mo.stop(0 not in rates.index, mo.md(f"`{DATA_PATH.name}` has no input_state == 0 rows"))

    p_x_s0 = (rates.loc[0][BINS] / rates.loc[0][BINS].sum()).tolist()
    print(f"p(x|S0) raw = {[round(v, 4) for v in p_x_s0]}")
    return (rates,)


@app.cell(hide_code=True)
def _(
    BINS,
    Causal_Models,
    DATA_PATH,
    N,
    QuantumTransitionModel,
    Simulator,
    get_uniform_renewal,
    np,
    pd,
    rates,
):
    from scipy.optimize import minimize_scalar

    from stochprocsim.utils import background_rate_by_channel, efficiency_for_channel

    # background/noise per channel: see plot_experimental_data.py for the
    # full explanation — falls back to zero for older measurements taken
    # before either background source existed
    _bg_by_ch, _cal_note = background_rate_by_channel(DATA_PATH.with_suffix(".json"))
    print(f"background used: {_cal_note}")
    bg = pd.Series({_b: _bg_by_ch.get(_b.removeprefix("coinc_ch"), 0.0) for _b in BINS})

    # detector efficiency: see stochprocsim.utils.DETECTOR_EFFICIENCY
    eff = pd.Series({_b: efficiency_for_channel(int(_b.removeprefix("coinc_ch")), dump_ch=7)
                     for _b in BINS})

    # per-loop transmission T fitted against the quantum simulator's S0
    # prediction, not the exact/uniform-renewal target -- the hardware is
    # actually trying to implement the quantum circuit, so that's the shape
    # loss should be calibrated against. The exact model stays the R_KL
    # comparison target below (a separate, deliberate choice).
    _sim = Simulator(QuantumTransitionModel(Causal_Models[N]))
    _p = _sim.get_output_distribution(propagate_outputs=True, start_state=0)
    _loop_p = {2 * (_i + 1): _v for _i, _v in enumerate(_p)}
    _target = np.array([
        1 - sum(_p) if _b == "coinc_ch7" else _loop_p.get(int(_b.removeprefix("coinc_ch")), 0.0)
        for _b in BINS
    ])

    _k = np.array([N if _b == "coinc_ch7" else int(_b.removeprefix("coinc_ch")) // 2 for _b in BINS])

    def _resid(T):
        _corr = ((rates.loc[0] - bg) / eff) / T ** _k
        return np.sum((_target - _corr / _corr.sum()) ** 2)

    T_fit = minimize_scalar(_resid, bounds=(0.05, 1), method="bounded").x
    print(f"transmission fit: T = {T_fit:.3f}  residual = {_resid(T_fit):.4f}")

    _corr = ((rates.loc[0] - bg) / eff) / T_fit ** _k
    frac0 = (_corr / _corr.sum()).tolist()
    print(f"p(x|S0) loss-corrected = {[round(v, 4) for v in frac0]}")
    exact_model = get_uniform_renewal(N - 1)
    return exact_model, frac0


@app.cell(hide_code=True)
def _(N, frac0, generate_quantum_model, np):
    exp_model = generate_quantum_model(np.array(frac0[:N]))
    return (exp_model,)


@app.cell(hide_code=True)
def _(N, exact_model, exit_distribution, exp_model, np):
    # p(S_i): stationary probability of each renewal state under P (the
    # exact/target model) — the experiment is trying to emulate this, so
    # it's the reference distribution, not the fitted one
    p_state = exact_model.steady_state

    R_KL = 0.0
    for _i in range(N):
        _p = exit_distribution(exact_model.probs, start=_i)
        _q = exit_distribution(exp_model.probs, start=_i)

        _cond = sum(_px * np.log2(_px / _qx) for _px, _qx in zip(_p, _q) if _px > 0 and _qx > 0)
        _weighted = p_state[_i] * _cond
        R_KL += _weighted
        print(f"S{_i}: D_KL(p||q) = {_cond:.6f}, "
              f"p(S{_i})={p_state[_i]:.4f} -> {_weighted:.6f}")

    # R_KL above is a total over a full run, not a rate -- eval_diverge (in
    # plot_experimental_data.py) reports bits/elementary-step instead, so to
    # be comparable, divide by the exact model's mean # loops per run
    _mean_loops = sum((_i + 1) * _p for _i, _p in enumerate(exit_distribution(exact_model.probs, start=0)))
    R_KL_rate = R_KL / _mean_loops

    print(f"\nR_KL[P_exact || Q_exp] (total, per run)  = {R_KL:.6f} bits")
    print(f"R_KL[P_exact || Q_exp] (rate, /{_mean_loops:.2f} mean loops) = {R_KL_rate:.6f} bits")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
