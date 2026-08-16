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
    filepath="/home/brendan/Documents/PhD/Year1/EDA/Programs/stochprocsim/data/autoqstochmeasure/data"
    refresh_button = mo.ui.run_button(label="Refresh file list")
    mo.hstack([filepath, refresh_button], justify="start")
    return filepath, refresh_button


@app.cell(hide_code=True)
def _(filepath, mo, refresh_button):
    from pathlib import Path
    import re as _re

    refresh_button  # ponytail: re-scan trigger, value itself unused
    _DATA_DIR = Path(filepath)
    # sort by the timestamp embedded in the filename, not mtime — files can
    # all get the same mtime from a bulk copy/checkout, and old files
    # (measurement_N..._YYYYMMDD_HHMMSS) vs new date-first files
    # (YYYYMMDD_HHMMSS_measurement_N...) don't collate the same way
    # lexicographically either, so plain name-sort also picks a stale file
    _files = sorted(_DATA_DIR.glob("*.csv"),
                     key=lambda p: _re.search(r"\d{8}_\d{6}", p.name).group())
    mo.stop(not _files, mo.md(f"No CSV files found in `{_DATA_DIR}`"))
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
    from stochprocsim.utils import rates_by_input_state

    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=BINS + ["herald", "int_time", "input_state"])
    data = data[data["herald"] > 0]  # herald blocked / off segments

    # counts per second, grouped by which memory state the machine started in
    # (and, for tomography files, combined across polarization basis — see
    # rates_by_input_state); a single-input file just yields one group and
    # every cell below collapses to the single-distribution case
    rates, stds = rates_by_input_state(data, BINS)
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
def _(BINS, N, np, rates):
    # exact/renewal-target exit distribution for every measured state s:
    # renewal state s means "already survived s steps of the same schedule",
    # built by slicing the state-0 model's own probs[s:] and relabeling bins
    # the same way p_theory does (bin i -> channel 2*(i+1), counting steps
    # since the s-injection, not absolute loop number)
    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal

    _probs = _get_uniform_renewal(N - 1).probs
    p_exact = {}
    for _s in rates.index:
        _exit, _a_prod = [], 1.0
        for _p_survive in _probs[int(_s):]:
            _exit.append((1 - _p_survive) * _a_prod)
            _a_prod *= _p_survive
        _exit.append(_a_prod)
        _loop_p = {2 * (_i + 1): _v for _i, _v in enumerate(_exit)}
        p_exact[_s] = np.array([
            0.0 if _b == "coinc_ch7" else _loop_p.get(int(_b.removeprefix("coinc_ch")), 0.0)
            for _b in BINS
        ])
        print(f"exact s{_s}: {np.round(p_exact[_s], 4)}")
    return (p_exact,)


@app.cell(hide_code=True)
def _(np, plt):
    def plot_state_grid(frac_by_state, yerr_by_state, p_theory, suptitle, p_exact=None):
        p_exact = p_exact or {}
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
                    width=0.5, color="gold", ecolor="black", label="experimental")
            _ax.bar(_x, p_theory[_s], width=0.5, fill=False, edgecolor="black",
                    linestyle="dotted", linewidth=1.5, label="theory")
            if _s in p_exact:
                _ax.bar(_x, p_exact[_s], width=0.5, fill=False, edgecolor="red",
                        linestyle="dashed", linewidth=0.8, label="exact model")
            _ax.set_xticks(_x, _f.index)
            _ax.tick_params(axis="x", labelsize=7)
            _ax.set_title(f"input s{_s}")
        _axes.flat[0].legend()
        _fig.suptitle(suptitle)
        _fig.tight_layout()
        return _fig
    return (plot_state_grid,)


@app.cell(hide_code=True)
def _(p_exact, p_theory, plot_state_grid, rates, stds):
    # raw fraction per bin vs theory (no loss correction)
    plot_state_grid(
        {_s: rates.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        {_s: stds.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        p_theory,
        "raw fractions vs theory",
        p_exact=p_exact,
    )

    print(rates.loc[:,:])
    print(p_theory)
    return


@app.cell
def _():
    #REALISTIC LOSS ESIMATION
    # Measurements taken as mean of 200 samples, error is std 

    # Forward Measurements (mW)
    # input_prep_power = 25.18
    # input_prep_power_err = 0.0075
    # out_setup = 15.72
    # out_setup_err = 0.040
    # out_dump = 15.13
    # out_dump_err = 0.209

    # input_setup_out_after_loop = 14.75
    # input_setup_out_after_loop_err = 0.042
    # out_dump_after_looping = 6.45
    # out_dump_after_looping_err = 1.02

    # #Reverse Measurements (mW)
    # reverse_power = 12.78
    # reverse_power_err = 0.98
    # out_loop_path = 7.7
    # out_loop_path_err = 0.652
    # out_input_prep = 10.85
    # out_input_prep_err = 0.865

    # Forward Measurements (mW)
    input_prep_power = 28.97
    input_prep_power_err = 0.02
    out_setup = 18.33
    out_setup_err = 0.018
    out_dump = 17.03
    out_dump_err = 0.207

    input_setup_out_after_loop = 17.67
    input_setup_out_after_loop_err = 0.015
    out_dump_after_looping = 7.422 - 0.259
    out_dump_after_looping_err = 0.380

    #Reverse Measurements (mW) its too different
    # reverse_power = 11.02
    # reverse_power_err = 1.569
    # out_loop_path = 5.17
    # out_loop_path_err = 0.580
    # out_input_prep = 
    # out_input_prep_err = 

    loop_input = 30.48
    loop_input_err = 2
    loop_output = 21.38
    loop_output_err = 0.070

    #Switch path losses
    #Input prep goes to bl in, Loop wh in
    # Setup on wh out, dump bl out
    l_in_bl_out_bl = out_dump/input_prep_power # input to dump
    l_in_bl_out_wh = out_setup/input_prep_power # input to setup
    l_in_wh_out_bl = out_dump_after_looping/input_setup_out_after_loop # loop to dump
    l_in_wh_out_wh = loop_output/loop_input # loop to setup

    print(f'TRANSMISSIONS: \n Input -> Dump: {l_in_bl_out_bl:.2f}', 
    f'\n Input -> U:{l_in_bl_out_wh:.2f}' , 
    f'\n Loop -> Dump:{l_in_wh_out_bl:.2f}', 
    f'\n Loop -> U:{l_in_wh_out_wh:.2f}')

    # print(f'loop wh wh Path: reverse:{l_in_wh_out_wh:.2f} forward: {loop_output/loop_input:.2f}')

    # print(f'bl wh Path: {l_in_bl_out_wh:.2f}|should be greater or equal to: {out_input_prep/reverse_power:.2f}')


    def calc_loss(N, isDump=False):
        # for a run of loops N
        losses = [0] * (N) # initialise a list of length (N-1), + 1 for dump 
        for i in range(len(losses)):
            # print(f'{i}: {l_in_bl_out_wh} * ({l_in_wh_out_wh}**({i}))')
            losses[i] = 1-(l_in_bl_out_wh * (l_in_wh_out_wh**(i)))
        if isDump:
            losses.append(1-(l_in_wh_out_bl*l_in_bl_out_wh * (l_in_wh_out_wh**(N))))
        return losses

    # print(f'loop loss: {1-l_in_wh_out_wh}')

    def reverse_loss(N, counts, bckgnd):
        """ Reverses loss on a bin with counts, at loop i in a process N detected with eff n and background counts bckg"""
        res = [0]*(N+1)
        losses = calc_loss(N, True)
        print(f'Losses: {losses}')
        # reverse loss on loop outputs
        for i in range(len(counts)-1):
            res[i] = ((counts[i] - bckgnd[i])/n_loop)*(1/losses[i])
        res[-1] = ((counts[-1] - bckgnd[-1])/n_dump)*(1/losses[-1])
        return res        
    #Detector efficiencies
    n_dump = 1
    n_loop = 0.9*n_dump

    # print(calc_loss(3, True))
    counts = [100, 50, 20, 8, 5]
    background = [1,1,1,1,1]
    print(f'Reverse loss counts: {reverse_loss(4, counts, background)}')
    return (calc_loss,)


@app.cell(hide_code=True)
def _(
    BINS,
    DATA_PATH,
    N,
    np,
    p_exact,
    p_theory,
    pd,
    plot_state_grid,
    rates,
    stds,
):
    # background/noise per channel: read from the noise-calibration json that
    # was in effect for this run (measurement.py points each measurement's
    # sidecar at the calibration file used, by filename, resolved relative to
    # this same data dir) — not fit, so it can't silently absorb real
    # discrepancies. Falls back to zero (no correction) for older
    # measurements taken before either background source existed.
    from stochprocsim.utils import background_rate_by_channel

    _bg_by_ch, _cal_note = background_rate_by_channel(DATA_PATH.with_suffix(".json"))
    print(f"background used: {_cal_note}")
    bg = pd.Series({_b: _bg_by_ch.get(_b.removeprefix("coinc_ch"), 0.0) for _b in BINS})

    # detector efficiency: 3 physical detectors (herald / dump / one shared,
    # time-multiplexed detector for every loop channel), so loop bins all
    # divide by the same factor and only dump gets its own — see
    # stochprocsim.utils.DETECTOR_EFFICIENCY (placeholder 1.0 = no
    # correction until it's actually calibrated)
    from stochprocsim.utils import efficiency_for_channel

    eff = pd.Series({_b: efficiency_for_channel(int(_b.removeprefix("coinc_ch")), dump_ch=7)
                     for _b in BINS})

    # per-loop transmission T fitted against the S0 input distribution only
    # (bin k has traversed k loops, dump counted like the last exit) — real
    # optical loss per round trip, separate from detector background/noise/
    # efficiency. Other measured states aren't used for the fit.
    from scipy.optimize import minimize_scalar

    _k = np.array([N if _b == "coinc_ch7" else int(_b.removeprefix("coinc_ch")) // 2 for _b in BINS])
    def _resid(T):
        _corr = ((rates.loc[0] - bg) / eff) / T ** _k
        return np.sum((p_theory[0] - _corr / _corr.sum()) ** 2)

    T_fit = minimize_scalar(_resid, bounds=(0.05, 1), method="bounded").x
    print(f"transmission fit: T = {T_fit:.3f}  residual = {_resid(T_fit):.4f}")
    frac, yerr = {}, {}
    for _s in rates.index:
        _corr = ((rates.loc[_s] - bg) / eff) / T_fit ** _k
        frac[_s] = _corr / _corr.sum()
        yerr[_s] = ((stds.loc[_s] / eff) / T_fit ** _k) / _corr.sum()
    print(f"s0 (loss-corrected): {np.round(frac[0].to_numpy(), 4)}")
    plot_state_grid(frac, yerr, p_theory,
                    f"loss-corrected (bckgnd avg= {bg.mean():.2f}, T={T_fit:.2f}) vs theory",
                    p_exact=p_exact)
    return (frac,)


@app.cell(hide_code=True)
def _(
    BINS,
    DATA_PATH,
    N,
    calc_loss,
    np,
    p_exact,
    p_theory,
    pd,
    plot_state_grid,
    rates,
    stds,
):
    # same as the T_fit cell above, but the per-bin correction comes from the
    # measured switch-path transmissions (calc_loss, from the power-meter
    # ratios in the REALISTIC LOSS ESTIMATION cell) instead of a single
    # fitted geometric T. calc_loss already returns a *cumulative*
    # transmission per bin (not per-hop), so no exponent is needed.
    from stochprocsim.utils import background_rate_by_channel

    _bg_by_ch, _cal_note = background_rate_by_channel(DATA_PATH.with_suffix(".json"))
    print(f"background used: {_cal_note}")
    bg = pd.Series({_b: _bg_by_ch.get(_b.removeprefix("coinc_ch"), 0.0) for _b in BINS})

    from stochprocsim.utils import efficiency_for_channel

    eff = pd.Series({_b: efficiency_for_channel(int(_b.removeprefix("coinc_ch")), dump_ch=7)
                     for _b in BINS})

    _max_k = max(int(_b.removeprefix("coinc_ch")) // 2 for _b in BINS if _b != "coinc_ch7")
    _loop_loss = calc_loss(_max_k)              # cumulative loss, bins k=1.._max_k
    _dump_loss = calc_loss(N, isDump=True)[-1]  # cumulative loss through the loop->dump switch
    meas_trans = pd.Series({
        _b: 1 - (_dump_loss if _b == "coinc_ch7" else _loop_loss[int(_b.removeprefix("coinc_ch")) // 2 - 1])
        for _b in BINS
    })
    print(f"measured transmission per bin:\n{meas_trans}")

    frac_loss, yerr_loss = {}, {}
    for _s in rates.index:
        _corr = ((rates.loc[_s] - bg) / eff) / meas_trans
        frac_loss[_s] = _corr / _corr.sum()
        yerr_loss[_s] = ((stds.loc[_s] / eff) / meas_trans) / _corr.sum()
    print(f"s0 (loss-corrected): {np.round(frac_loss[0].to_numpy(), 4)}")
    plot_state_grid(frac_loss, yerr_loss, p_theory,
                    "loss-corrected (measured switch-path transmission) vs theory",
                    p_exact=p_exact)
    return (frac_loss,)


@app.cell(hide_code=True)
def _(N, frac, np):
    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge as _eval_diverge
    from stochprocsim.stochprocq.models.renewal import RenewalProcess as _RenewalProcess

    _model = _get_uniform_renewal(N - 1)
    _k = np.array([N if _b == "coinc_ch7" else int(_b.removeprefix("coinc_ch")) // 2
                    for _b in frac[0].index])

    def _r_kl_from_emit(_q_emit):
        _q_survive_st = np.zeros_like(_q_emit)
        _q_survive_st[0] = _q_emit[0]
        for _i in range(1, len(_q_emit)):
            _q_survive_st[_i] = _q_emit[_i] / np.prod(1 - _q_survive_st[:_i])
        _q_model = _RenewalProcess([1 - _q for _q in _q_survive_st[:-1]])
        _p_dist = _model.gen_dists(N + 1)[0]
        _q_dist = _q_model.gen_dists(N + 1)[0]
        return _eval_diverge(_p_dist, _q_dist, his_steps=N - 1)

    R_KL = _r_kl_from_emit(frac[0].to_numpy()[:N])

    print("Ximing Divergence:", f"R_KL[P_exact || Q_exp] (s0 input) = {R_KL:.6f}")
    return


@app.cell(hide_code=True)
def _(N, frac, p_exact, rates):
    # R_KL[P||Q] = sum_i p(S_i) sum_x p(x|S_i) log(p(x|S_i)/q(x|S_i))
    # P = exact model (p_exact, per prepared state s), Q = experimental data
    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge as _eval_diverge

    _states = sorted(rates.index)
    _pi = _get_uniform_renewal(N - 1).steady_state
    _p_state = {_s: (_pi[int(_s)] if int(_s) < N else 0.0) for _s in _states}
    _norm = sum(_p_state.values())
    _p_state = {_s: _v / _norm for _s, _v in _p_state.items()}

    R_KL_weighted = 0.0
    for _s in _states:
        _p, _q = p_exact[_s][:N], frac[_s].to_numpy()[:N]
        _cond = _eval_diverge(_p, _q, his_steps=None)
        _weighted = _p_state[_s] * _cond
        R_KL_weighted += _weighted
        print(f"S{_s}: D_KL(p||q) = {_cond:.6f} bits, weighted by p(S{_s})={_p_state[_s]:.4f} -> {_weighted:.6f} bits")
    print(f"\nR_KL[P||Q] = {R_KL_weighted:.6f} bits")
    return


@app.cell(hide_code=True)
def _(Causal_Models, QuantumTransitionModel, Simulator, np, plt):
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

    _x_exp = [3, 4, 5, 6]
    rate_exp = [0.020352, 0.031224, 0.025116, 0.036747]
    err_rate_exp = [0.000350, 0.000505, 0.000586, 0.000178]

    # R_KL[P||Q] = sum_i p(S_i) sum_x p(x|S_i) log(p(x|S_i)/q(x|S_i)), from
    # the weighted-sum cell above: P=p_exact (per prepared state), Q=frac
    # (experimental) - _REF files:
    # N3: 20260803_171226_measurement_N3_sall_REF.csv
    # N4: 20260804_092116_measurement_N4_s0_tomo_REF.csv (s0 only)
    # N5: 20260730_171908_measurement_N5_sall_REF.csv
    # N6: 20260731_081434_measurement_N6_sall_REF.csv
    rate_exp_weighted = [0.145167, 0.187207, 0.245449, 0.388857]

    _palette = ["#3B5BA5", "#800E13", "#C11B26", "#4C9A2A"]
    _fig, _ax = plt.subplots()
    _ax.plot(_x_class, _y_classical, '-s', label='Classical bound', color=_palette[0], linewidth=1.5)
    _ax.errorbar(_x_quant, _y_quantum, yerr=_qtheo_yerr, fmt='-^', capsize=3,
                 label='Quantum (Target)', color=_palette[1], linewidth=1.5)
    _ax.errorbar(_x_exp, rate_exp, yerr=err_rate_exp, fmt='p--',
             label='Experiment (eval diverge)', color=_palette[2], markersize=8)
    # _ax.plot(_x_exp, rate_exp_weighted, 'd--',
    #          label='Experiment (weighted sum)', color=_palette[3], markersize=8)
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
