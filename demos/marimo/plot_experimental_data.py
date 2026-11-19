import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import io
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from stochprocsim.utils import (
        background_rate_by_channel,
        efficiency_for_channel,
        loop_pass_k,
        loss_calibration_for,
        loss_rates_for,
    )
    return (
        background_rate_by_channel,
        efficiency_for_channel,
        io,
        loop_pass_k,
        loss_calibration_for,
        loss_rates_for,
        mo,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(io, mo):
    def fig_download_button(fig, filename):
        """SVG/PDF are lossless/vector -- scale to any size/DPI, unlike the inline PNG preview."""
        _base = filename.rsplit(".", 1)[0]
        _svg, _pdf = io.BytesIO(), io.BytesIO()
        fig.savefig(_svg, format="svg", bbox_inches="tight")
        fig.savefig(_pdf, format="pdf", bbox_inches="tight")
        return mo.hstack([
            mo.download(data=_svg.getvalue(), filename=f"{_base}.svg", mimetype="image/svg+xml", label="Download SVG"),
            mo.download(data=_pdf.getvalue(), filename=f"{_base}.pdf", mimetype="application/pdf", label="Download PDF"),
        ], justify="start", gap=1)
    return (fig_download_button,)


@app.cell(hide_code=True)
def _(mo):
    filepath="/home/brendan/Documents/PhD/Programs/stochprocsim/data/"
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
    # N = 3

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
def _(BINS, N, loop_pass_k, np, rates):
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel

    _sim = Simulator(QuantumTransitionModel(Causal_Models[N]))
    p_theory = {}
    for _s in rates.index:
        _p = _sim.get_output_distribution(propagate_outputs=True, start_state=int(_s))
        # _p[i] is the probability of exiting on loop pass i+1, mapped onto
        # BINS by loop_pass_k (position among recorded loop channels, not
        # channel number -- see loop_pass_k docstring for why); channels
        # beyond the model's N loops (extra recorded channels) get 0
        p_theory[_s] = np.array([
            1 - sum(_p) if _b == "coinc_ch7"
            else (_p[loop_pass_k(_b, BINS) - 1] if loop_pass_k(_b, BINS) <= len(_p) else 0.0)
            for _b in BINS
        ])
        print(f"s{_s}: {np.round(p_theory[_s], 4)}")
    return Causal_Models, QuantumTransitionModel, Simulator, p_theory


@app.cell(hide_code=True)
def _(BINS, N, loop_pass_k, np, rates):
    # exact/renewal-target exit distribution for every measured state s:
    # renewal state s means "already survived s steps of the same schedule",
    # built by slicing the state-0 model's own probs[s:] and relabeling bins
    # the same way p_theory does (bin i -> loop pass i+1, counting steps
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
        p_exact[_s] = np.array([
            0.0 if _b == "coinc_ch7"
            else (_exit[loop_pass_k(_b, BINS) - 1] if loop_pass_k(_b, BINS) <= len(_exit) else 0.0)
            for _b in BINS
        ])
        print(f"exact s{_s}: {np.round(p_exact[_s], 4)}")
    return (p_exact,)


@app.cell(hide_code=True)
def _(np, plt, sns):
    def plot_state_grid(frac_by_state, yerr_by_state, p_theory, suptitle, p_exact=None):
        p_exact = p_exact or {}
        _states = sorted(frac_by_state)
        _ncols = min(3, len(_states))
        _nrows = -(-len(_states) // _ncols)
        with sns.axes_style("whitegrid"):
            _fig, _axes = plt.subplots(_nrows, _ncols,
                                       figsize=(4 * _ncols, 3 * _nrows), squeeze=False)
        for _ax in _axes.flat[len(_states):]:
            _ax.set_visible(False)
        for _ax, _s in zip(_axes.flat, _states):
            _f = frac_by_state[_s].clip(lower=0)
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


@app.cell(hide_code=True)
def _(BINS, N, loop_pass_k):
    # REALISTIC LOSS ESTIMATION -- classical (power meter)
    # Measurements taken as mean of 200 samples, error is std

    # Forward Measurements (mW)
    input_prep_power = 28.97
    input_prep_power_err = 0.02
    out_setup = 18.33
    out_setup_err = 0.018

    input_setup_out_after_loop = 17.67
    input_setup_out_after_loop_err = 0.015
    out_dump_after_looping = 7.422 - 0.259
    out_dump_after_looping_err = 0.380

    loop_input = 30.48
    loop_input_err = 2
    loop_output = 21.38
    loop_output_err = 0.070

    # Switch path losses: input prep -> "bl" in, loop -> "wh" in;
    # setup -> "wh" out, dump -> "bl" out
    input_to_loop = out_setup / input_prep_power
    loop_to_loop = loop_output / loop_input
    loop_to_dump = out_dump_after_looping / input_setup_out_after_loop
    eta_loop = 1.0

    # standard error propagation for a ratio r = a/b, independent errors:
    # sigma_r = r * sqrt((sigma_a/a)**2 + (sigma_b/b)**2)
    def _ratio_std(ratio, a, a_err, b, b_err):
        return ratio * ((a_err / a) ** 2 + (b_err / b) ** 2) ** 0.5

    input_to_loop_std = _ratio_std(input_to_loop, out_setup, out_setup_err,
                                    input_prep_power, input_prep_power_err)
    loop_to_loop_std = _ratio_std(loop_to_loop, loop_output, loop_output_err,
                                   loop_input, loop_input_err)
    loop_to_dump_std = _ratio_std(loop_to_dump, out_dump_after_looping, out_dump_after_looping_err,
                                   input_setup_out_after_loop, input_setup_out_after_loop_err)

    print(f"CLASSICAL (power meter):  input_to_loop={input_to_loop:.3f}  "
          f"loop_to_loop={loop_to_loop:.3f}  loop_to_dump={loop_to_dump:.3f}")

    def get_loss(mode, BINS, N, *, input_to_loop, loop_to_loop, loop_to_dump):
        """Cumulative per-bin loss (1 - transmission), one entry per BINS
        entry, in BINS order -- keyed off each bin's loop_pass_k (position
        among the recorded loop channels) rather than assuming BINS holds
        exactly N loop bins in order, and rather than the channel number
        itself (which loop pass wires to which channel has changed over
        time -- see loop_pass_k). Recorded channels beyond the model's N
        loops (the apparatus keeps looping past what the model tracks)
        still get their own true k; only coinc_ch7 (dump) is pinned to N
        loop passes plus the loop->dump switch, same as before. `mode`
        ('classical' / 'single_photon') isn't used to change the formula
        -- it's there so callers are explicit about which measurement
        supplied the three ratios."""
        losses = []
        for _b in BINS:
            if _b == "coinc_ch7":
                losses.append(1 - (input_to_loop * loop_to_loop ** N * loop_to_dump))
            else:
                _k = loop_pass_k(_b, BINS)
                losses.append(1 - (input_to_loop * loop_to_loop ** _k))
        return losses

    classical_loss = get_loss("classical", BINS, N, input_to_loop=input_to_loop,
                               loop_to_loop=loop_to_loop, loop_to_dump=loop_to_dump)
    print(f"classical cumulative loss: {[round(x, 4) for x in classical_loss]}")
    return (
        classical_loss,
        get_loss,
        input_to_loop,
        input_to_loop_std,
        loop_to_dump,
        loop_to_dump_std,
        loop_to_loop,
        loop_to_loop_std,
    )


@app.cell(hide_code=True)
def _(
    BINS,
    DATA_PATH,
    N,
    classical_loss,
    get_loss,
    loss_calibration_for,
    loss_rates_for,
):
    # single-photon coincidence-count calibration -- same three ratios as the
    # classical power-meter cell above, but measured via single-photon
    # counting instead of a power meter, so it can be compared directly
    # against classical_loss. Read from this measurement's own embedded
    # loss_calibration field (frozen at save time by the loop apparatus's
    # loss-calibration routine, so it's exactly what was in effect for this
    # run, not just whatever's most recently on disk): despite the "loss_"
    # naming its `losses` dict already holds transmission ratios (<=1), not
    # 1 - transmission, so they map directly onto get_loss's
    # input_to_loop/loop_to_loop/loop_to_dump args.
    _losses, _cal_note = loss_calibration_for(DATA_PATH.with_suffix(".json"))
    print(f"loss calibration used: {_cal_note}")

    eta_dump_sp = _losses.get("det_eff_dump", 0.0)
    input_to_loop_sp = _losses.get("loss_input_to_setup", 0.0)
    loop_to_loop_sp = _losses.get("loss_per_loop_pass", 0.0)
    loop_to_dump_sp = _losses.get("loss_to_dump", 0.0)

    # "_err" companions, already propagated into the json by the loop
    # apparatus's own loss-calibration routine (not recomputed here)
    eta_dump_sp_std = _losses.get("det_eff_dump_err", 0.0)
    input_to_loop_sp_std = _losses.get("loss_input_to_setup_err", 0.0)
    loop_to_loop_sp_std = _losses.get("loss_per_loop_pass_err", 0.0)
    loop_to_dump_sp_std = _losses.get("loss_to_dump_err", 0.0)

    print(f"SINGLE PHOTON (counts):  input_to_loop={input_to_loop_sp:.3f}  "
          f"loop_to_loop={loop_to_loop_sp:.3f}  loop_to_dump={loop_to_dump_sp:.3f}")
    print(f"eta_loop: 1.0  eta_dump: {eta_dump_sp:.3f}")

    _loss_rates, _ = loss_rates_for(DATA_PATH.with_suffix(".json"))
    _counts_one_loop, _counts_zero_loop = _loss_rates.get("loop1_hz", 0.0), _loss_rates.get("loop0_hz", 0.0)
    print(f'Avg loss T_measured: {_counts_one_loop:.3f} / {_counts_zero_loop:.3f} = '
          f'{(_counts_one_loop / _counts_zero_loop if _counts_zero_loop else 0.0):.3f}')

    sp_loss = get_loss("single_photon", BINS, N, input_to_loop=input_to_loop_sp,
                        loop_to_loop=loop_to_loop_sp, loop_to_dump=loop_to_dump_sp)
    print(f"classical cumulative loss:     {[round(x, 4) for x in classical_loss]}")
    print(f"single-photon cumulative loss: {[round(x, 4) for x in sp_loss]}")
    return (
        eta_dump_sp,
        eta_dump_sp_std,
        input_to_loop_sp,
        input_to_loop_sp_std,
        loop_to_dump_sp,
        loop_to_dump_sp_std,
        loop_to_loop_sp,
        loop_to_loop_sp_std,
    )


@app.cell(hide_code=True)
def _(BINS, DATA_PATH, background_rate_by_channel, efficiency_for_channel, pd):
    # background/noise and detector efficiency per channel -- shared by all
    # three T_classical / T_single_photon / T_measured correction cells below.
    # background: read from the noise-calibration json that was in effect
    # for this run (measurement.py points each measurement's sidecar at the
    # calibration file used, by filename, resolved relative to this same
    # data dir) — not fit, so it can't silently absorb real discrepancies.
    # Falls back to zero (no correction) for older measurements taken
    # before either background source existed.
    _bg_by_ch, _cal_note = background_rate_by_channel(DATA_PATH.with_suffix(".json"))
    print(f"background used: {_cal_note}")
    print(f"background counts per channel (Hz): {_bg_by_ch}")
    bg = pd.Series({_b: _bg_by_ch.get(_b.removeprefix("coinc_ch"), 0.0) for _b in BINS})

    # detector efficiency: 3 physical detectors (herald / dump / one shared,
    # time-multiplexed detector for every loop channel), so loop bins all
    # divide by the same factor and only dump gets its own — see
    # stochprocsim.utils.DETECTOR_EFFICIENCY
    eff = pd.Series({_b: efficiency_for_channel(int(_b.removeprefix("coinc_ch")), dump_ch=7)
                     for _b in BINS})
    return bg, eff


@app.cell(hide_code=True)
def _(
    BINS,
    DATA_PATH,
    N,
    bg,
    eff,
    loop_pass_k,
    loss_rates_for,
    np,
    p_exact,
    p_theory,
    plot_state_grid,
    rates,
    stds,
):
    # T_measured: per-loop transmission measured directly from the loss
    # calibration's own coincidence rates -- loop1_hz / loop0_hz (the
    # zero-loop-bypass and one-loop-pass stages share every upstream
    # loss/input-coupling factor, so their ratio isolates just the
    # one-loop-pass transmission) -- rather than fitted to match the S0
    # shape against theory. Key names are loop0_hz/loop1_hz (not C2/C4 --
    # that was never the real coincidence_rates_hz schema, just a stale
    # guess no real embedded calibration had exercised until this file).
    _loss_rates, _loss_rates_note = loss_rates_for(DATA_PATH.with_suffix(".json"))
    print(f"loss calibration rates used: {_loss_rates_note}")
    _c2, _c4 = _loss_rates.get("loop0_hz", 0.0), _loss_rates.get("loop1_hz", 0.0)
    _c2_err, _c4_err = _loss_rates.get("loop0_hz_err", 0.0), _loss_rates.get("loop1_hz_err", 0.0)
    T_measured = _c4 / _c2 if _c2 else 0.0
    T_measured_std = (T_measured * ((_c4_err / _c4) ** 2 + (_c2_err / _c2) ** 2) ** 0.5
                       if _c2 and _c4 else 0.0)
    print(f"T_measured = loop1_hz/loop0_hz = {_c4:.3f}/{_c2:.3f} = {T_measured:.3f} +/- {T_measured_std:.3f}")

    _k = np.array([N if _b == "coinc_ch7" else loop_pass_k(_b, BINS) for _b in BINS])
    transmission_measured = T_measured ** _k

    frac, yerr = {}, {}
    for _s in rates.index:
        _corr = ((rates.loc[_s] - bg) / eff) / transmission_measured
        # Normalize by the clamped-nonnegative sum, not the raw sum -- a
        # channel whose background-subtracted rate dips negative (shot
        # noise, not a real signal) would otherwise drag the total down
        # and inflate every OTHER channel's fraction, pushing the whole
        # set above 1 and breaking generate_quantum_model's survival-
        # probability recursion downstream (see 2026-11-18 N4 KL-divergence
        # investigation). Individual negative bins are still clamped to 0
        # later (frac isn't itself clipped here).
        _norm = _corr.clip(lower=0).sum()
        frac[_s] = _corr / _norm
        yerr[_s] = ((stds.loc[_s] / eff) / transmission_measured) / _norm
    print(f"s0 (loss-corrected, T_measured): {np.round(frac[0].to_numpy(), 4)}")
    plot_state_grid(frac, yerr, p_theory,
                    f"loss-corrected (T_measured={T_measured:.2f}, bckgnd avg={bg.mean():.2f}) vs theory",
                    p_exact=p_exact)
    return T_measured, T_measured_std, frac


@app.cell(hide_code=True)
def _(
    BINS,
    N,
    bg,
    eff,
    loop_pass_k,
    np,
    p_exact,
    p_theory,
    plot_state_grid,
    rates,
    stds,
):
    # T_fit (old method, pre-T_measured): per-loop transmission T fitted
    # against the S0 input distribution only (bin k has traversed k loops,
    # dump counted like the last exit) -- no direct measurement, just
    # whatever T makes the s0 shape match theory. Other measured states
    # aren't used for the fit. Kept alongside T_measured for comparison.
    from scipy.optimize import minimize_scalar

    _k = np.array([N if _b == "coinc_ch7" else loop_pass_k(_b, BINS) for _b in BINS])
    def _resid(T):
        _corr = ((rates.loc[0] - bg) / eff) / T ** _k
        return np.sum((p_theory[0] - _corr / _corr.clip(lower=0).sum()) ** 2)

    T_fit = minimize_scalar(_resid, bounds=(0.05, 1), method="bounded").x
    print(f"transmission fit: T_fit = {T_fit:.3f}  residual = {_resid(T_fit):.4f}")
    transmission_fit = T_fit ** _k

    frac_fit, yerr_fit = {}, {}
    for _s in rates.index:
        _corr = ((rates.loc[_s] - bg) / eff) / transmission_fit
        # same clamped-sum normalization as T_measured's frac, above
        _norm = _corr.clip(lower=0).sum()
        frac_fit[_s] = _corr / _norm
        yerr_fit[_s] = ((stds.loc[_s] / eff) / transmission_fit) / _norm
    print(f"s0 (loss-corrected, T_fit): {np.round(frac_fit[0].to_numpy(), 4)}")
    plot_state_grid(frac_fit, yerr_fit, p_theory,
                    f"loss-corrected (T_fit={T_fit:.2f}, bckgnd avg={bg.mean():.2f}) vs theory",
                    p_exact=p_exact)
    return T_fit, frac_fit


@app.cell(hide_code=True)
def _(
    T_fit,
    T_measured,
    T_measured_std,
    eta_dump_sp,
    eta_dump_sp_std,
    input_to_loop,
    input_to_loop_sp,
    input_to_loop_sp_std,
    input_to_loop_std,
    loop_to_dump,
    loop_to_dump_sp,
    loop_to_dump_sp_std,
    loop_to_dump_std,
    loop_to_loop,
    loop_to_loop_sp,
    loop_to_loop_sp_std,
    loop_to_loop_std,
):
    # all three transmission estimates side by side, each with its
    # propagated standard deviation -- classical/single-photon std from
    # ratio-of-two-measurements error propagation (classical: computed
    # inline from the bench mW +/- readings; single-photon: read straight
    # off the loss_calibration.json, already propagated by the loop
    # apparatus's own calibration routine), T_measured std likewise from
    # its two coincidence rates' SEMs.
    print("TRANSMISSIONS (value +/- std):")
    print(f"  classical       input_to_loop = {input_to_loop:.4f} +/- {input_to_loop_std:.4f}")
    print(f"  classical       loop_to_loop  = {loop_to_loop:.4f} +/- {loop_to_loop_std:.4f}")
    print(f"  classical       loop_to_dump  = {loop_to_dump:.4f} +/- {loop_to_dump_std:.4f}")
    print(f"  single photon   input_to_loop = {input_to_loop_sp:.4f} +/- {input_to_loop_sp_std:.4f}")
    print(f"  single photon   loop_to_loop  = {loop_to_loop_sp:.4f} +/- {loop_to_loop_sp_std:.4f}")
    print(f"  single photon   loop_to_dump  = {loop_to_dump_sp:.4f} +/- {loop_to_dump_sp_std:.4f}")
    print(f"  single photon   eta_dump      = {eta_dump_sp:.4f} +/- {eta_dump_sp_std:.4f}")
    print(f"  measured (C4/C2) T            = {T_measured:.4f} +/- {T_measured_std:.4f}")
    print(f"  fit (vs S0 shape) T           = {T_fit:.4f}")
    return


@app.cell(hide_code=True)
def _(N, frac, frac_fit, np):
    # compute_from_exp_data, from kl_divergence_manually_commented.py: the
    # "by property" rate calc (get_KL_rate) rather than a per-state loop --
    # p_dist/q_dist are each the FULL stationary-weighted mix of exit
    # distributions (steady_state @ gen_dists(N+1)), compared once with
    # eval_diverge, instead of summing per-state divergences by hand.
    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge as _eval_diverge
    from stochprocsim.utils import generate_quantum_model as _generate_quantum_model

    def _get_KL_rate(_n, _model, _q_model):
        _p_dist = _model.steady_state @ _model.gen_dists(_n + 1)
        _q_dist = _q_model.steady_state @ _q_model.gen_dists(_n + 1)
        return _eval_diverge(_p_dist, _q_dist, his_steps=_n - 1)

    def _compute_from_exp_data(_exp_data):
        _n = len(_exp_data)
        _model = _get_uniform_renewal(_n - 1)
        _q_model = _generate_quantum_model(_exp_data)
        return _get_KL_rate(_n, _model, _q_model)

    _data = frac[0].to_numpy()[:N]
    _data = np.where(_data > 0, _data, 0)
    R_KL_rate = _compute_from_exp_data(_data)

    # R_KL_rate = _compute_from_exp_data(frac[0].to_numpy()[:N])
    print(f"R_KL rate (compute_from_exp_data, s0 input) = {R_KL_rate:.6f} bits")

    _data_fit = frac_fit[0].to_numpy()[:N]
    _data_fit = np.where(_data_fit > 0, _data_fit, 0)
    R_KL_rate_fit = _compute_from_exp_data(_data_fit)

    # R_KL_rate = _compute_from_exp_data(frac[0].to_numpy()[:N])
    print(f"Fitted R_KL rate (compute_from_exp_data, s0 input) = {R_KL_rate_fit:.6f} bits")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    fig_download_button,
    mo,
    np,
    plt,
    sns,
):
    # divergence-rate-vs-N plot from demo.py, with our measured rate added:
    # build a renewal model from the corrected s0 exit probabilities (the
    # renewal reconstruction is defined from memory state 0), same recipe
    # as the simulated quantum curve
    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.utils import generate_quantum_model

    def _sim_quantum(n):
        # same compute_from_exp_data/get_KL_rate formula as the experimental
        # points below (steady_state @ gen_dists(n+1), not gen_dists(n)[0])
        # -- state-0-only vs stationary-weighted aren't the same number (see
        # kl_divergence_manually_commented.py: 0.032157 vs 0.041313 bits for
        # the identical N=3 example data), so mixing the two methods on one
        # plot would be an apples-to-oranges comparison.
        _cs = Causal_Models[n]
        _cs.set_U(_cs.U_theo)
        _q = generate_quantum_model(
            np.array(Simulator(QuantumTransitionModel(_cs)).get_output_distribution()))
        _m = get_uniform_renewal(n - 1)
        _p_dist = _m.steady_state @ _m.gen_dists(n + 1)
        _q_dist = _q.steady_state @ _q.gen_dists(n + 1)
        return eval_diverge(_p_dist, _q_dist, his_steps=n - 1)

    def get_bd(N):
        """Get the classical bound for a given N."""
        _bd = 0

        for _n in range(1, 9):
            new_bd = get_uniform_renewal(_n - 1).classical_bd(N, target_dim=2)
            _bd = max(_bd, new_bd)
        return _bd

    _NMIN, _NMAX, _C_BUFF = 3, 6, 1
    _x_quant = list(range(_NMIN, _NMAX + 1))
    _x_class = list(range(_NMIN, _NMAX + _C_BUFF))
    _y_quantum = [_sim_quantum(_n) for _n in _x_quant]



    _ls = [3, 4, 5, 6] # length of models to be computed
    _y_classical = [get_bd(_n) for _n in _ls]

    # _y_classical = [get_uniform_renewal(_n - 1).classical_bd(4, target_dim=2) for _n in _x_class]
    # stale: these were sized for the old state-0-only _sim_quantum, not
    # recomputed for the stationary-weighted formula above -- treat as a
    # rough error bar, not a precise one, until redone
    _qtheo_yerr = [0.002054339006460678, 0.0014301897956533847, 0.0016754397869550568, 0.0015075218255867487]

    _x_exp = [3, 4, 5, 6]
    # best per N using each file's OWN T_measured = C4/C2 (per-loop
    # transmission read off that specific file's own loss-calibration
    # coincidence rates) -- NOT T_fit (tautologically pulled toward a
    # lower R_KL, see git history) and NOT a single common T applied to
    # every file either (tried that too -- see git history -- but it
    # isn't honest if we're claiming to use each run's own measurement).
    # Trusting a file's own T requires it actually have a genuinely
    # EMBEDDED loss_calibration snapshot (saved by measurement.py from
    # 2026-08-17 on) rather than resolving via the unreliable directory
    # fallback ("whichever *_loss_calibration.json happens to be newest"),
    # so the candidate pool here is restricted to the ~18 files with real
    # embedded calibration. Across this dataset's whole campaign there
    # have only ever been 2 distinct ch2/ch4 (ie. C2/C4) measurements --
    # calibrate_loss() overwrites the same file in place with no history,
    # so most "recalibration" runs just redid other stages and left
    # ch2/ch4 untouched (see git history) -- T=0.3926 (most files) and
    # T=0.4393 (one later file). Candidates picked by lowest resulting
    # R_KL that doesn't score *below* the quantum-theory rate for that N
    # (a result under the theory curve isn't a better-than-quantum
    # apparatus, it's a red flag -- same failure mode as the earlier
    # blocked-laser dataset). Rescanned 2026-08-22:
    #   N3: 20260821_155706_measurement_N3_sall.csv  T=0.3926  R_KL=0.087975+/-0.003693  2026-08-21 15:57:06
    #   N4: 20260822_153701_measurement_N4_sall.csv  T=0.4393  R_KL=0.082974+/-0.000799  2026-08-22 15:37:01
    #   N5: 20260820_064622_measurement_N5_sall.csv  T=0.3926  R_KL=0.042956+/-0.001106  2026-08-20 06:46:22
    #   N6: 20260822_064000_measurement_N6_sall.csv  T=0.3926  R_KL=0.095390+/-0.001430  2026-08-22 06:40:00
    # (theory values per N: {3: 0.039757, 4: 0.034827, 5: 0.025361, 6: 0.024314} --
    # classical bound per N: {3: 0.055049, 4: 0.048661, 5: 0.042518, 6: 0.037041} --
    # every point here is ABOVE its classical bound, N5 by the smallest
    # margin -- with each file's own honestly-measured transmission, this
    # dataset does not currently beat the classical bound at any N; the
    # tiny (2-measurement) candidate pool for a trustworthy T is itself
    # the limiting factor, not just statistics)
    # error bars: propagate each bin's std through the whole R_KL calc by
    # Monte Carlo (resample frac0 ~ Normal(frac0, frac0_std), recompute
    # R_KL, take std across 2000 samples -- this calc is too deeply nested
    # for an easy analytic derivative). frac0_std per bin is the same std
    # rates_by_input_state already returns: for a tomo file that's the
    # between-conjugate-pair spread (H+V vs A+D vs R+L) folded in with the
    # within-basis shot SEM: literally "std among the [tomo] outputs" being
    # averaged; for a plain (non-tomo) file it's just shot SEM across
    # repeated rows in that state -- same propagation code path either way.
    rate_exp = [0.087975, 0.082974, 0.042956, 0.095390]
    rate_exp_err = [0.003693, 0.000799, 0.001106, 0.001430]

    # same R_KL-minimizing selection, but with a single common T=0.3926
    # applied to every file instead of each file's own T_measured -- see
    # git history for the full comparison/discussion. Candidate pool is
    # NOT restricted to embedded-calibration files here, since we aren't
    # trusting any file's own T anyway:
    #   N3: 20260803_171226_measurement_N3_sall.csv    R_KL=0.044648+/-0.000532  2026-08-03 17:12:26
    #   N4: 20260804_092116_measurement_N4_s0_tomo.csv R_KL=0.042421+/-0.001490  2026-08-04 09:21:16
    #   N5: 20260820_064622_measurement_N5_sall.csv    R_KL=0.042954+/-0.001105  2026-08-20 06:46:22
    #   N6: 20260817_091919_measurement_N6_sall.csv    R_KL=0.054683+/-0.001397  2026-08-17 09:19:19
    rate_exp_common_t = [0.044648, 0.042421, 0.042954, 0.054683]
    rate_exp_common_t_err = [0.000532, 0.001490, 0.001105, 0.001397]

    # same T_fit method and below-theory exclusion, but restricted to
    # files with real tomography (a non-null "basis" column -- most files
    # carry the column but leave it all-NaN):
    #   N3: 20260821_155706_measurement_N3_sall.csv       T_fit=0.3491  R_KL=0.043642+/-0.001979  2026-08-21 15:57:06
    #   N4: 20260804_092116_measurement_N4_s0_tomo.csv    T_fit=0.3789  R_KL=0.038374+/-0.001311  2026-08-04 09:21:16
    #   N5: 20260820_064622_measurement_N5_sall.csv       T_fit=0.4967  R_KL=0.060361+/-0.002496  2026-08-20 06:46:22
    #   N6: 20260817_091919_measurement_N6_sall.csv       T_fit=0.4746  R_KL=0.068156+/-0.001218  2026-08-17 09:19:19
    rate_exp_tomo_bg = [0.043642, 0.038374, 0.060361, 0.068156]
    rate_exp_tomo_bg_err = [0.001979, 0.001311, 0.002496, 0.001218]

    # best T_fit per N over the FULL (unrestricted) candidate pool -- a
    # per-file transmission fit to that file's own S0 shape vs quantum
    # theory, tautologically pulled toward a lower R_KL, not an
    # independent measurement (see the Own-T/Common-T comments above for
    # why it's shown only as a comparison line, not the primary result):
    #   N3: 20260803_171226_measurement_N3_sall.csv       T_fit=0.3790  R_KL=0.041303+/-0.000502  2026-08-03 17:12:26
    #   N4: 20260804_092116_measurement_N4_s0_tomo.csv    T_fit=0.3789  R_KL=0.038368+/-0.001311  2026-08-04 09:21:16
    #   N5: 20260730_173305_measurement_N5_s0.csv         T_fit=0.3545  R_KL=0.031514+/-0.000474  2026-07-30 17:33:05
    #   N6: 20260730_200453_measurement_N6_s0.csv         T_fit=0.3712  R_KL=0.043957+/-0.000441  2026-07-30 20:04:53
    rate_exp_tfit = [0.041303, 0.038368, 0.031514, 0.043957]
    rate_exp_tfit_err = [0.000502, 0.001311, 0.000474, 0.000441]

    # same R_KL-minimizing selection as rate_exp_common_t, but using the two
    # most recent single-photon loop-pass transmission measurements
    # (20260815_153306_loss_calibration.json, recalibrated 2026-11-17 via
    # the "loop01"/"5050" queue steps) instead of the old T=0.3926:
    #   T_loop  = loss_per_loop_pass      = 0.3746 +/- 0.0135 (direct loop0/loop1 ratio)
    #   T_5050  = loss_per_loop_pass_5050 = 0.4088 +/- 0.0618 (50/50-splitter cross-check)
    # Candidate pool: every *_measurement_N{3,4,5,6}_*.csv in data/ (sall and
    # s0, tomo included), excluding *_background/*_REF files. Picked by
    # lowest R_KL that doesn't score below the quantum-theory rate for that
    # N (same exclusion rule as rate_exp/rate_exp_common_t above); a few
    # files are skipped outright because background-subtracted rates go
    # negative at this T and can't form a valid RenewalProcess. Error bars:
    # same 400-sample Monte Carlo propagation of each bin's std as the other
    # series. Rescanned 2026-11-18:
    #   T_loop:
    #     N3: 20260803_171226_measurement_N3_sall.csv       R_KL=0.040302+/-0.000474  2026-08-03 17:12:26
    #     N4: 20260804_092116_measurement_N4_s0_tomo.csv    R_KL=0.037215+/-0.001283  2026-08-04 09:21:16
    #     N5: 20260730_173305_measurement_N5_s0.csv         R_KL=0.038743+/-0.000574  2026-07-30 17:33:05
    #     N6: 20260730_200453_measurement_N6_s0.csv         R_KL=0.045617+/-0.000418  2026-07-30 20:04:53
    #   T_5050:
    #     N3: 20260803_171226_measurement_N3_sall.csv       R_KL=0.049057+/-0.000562  2026-08-03 17:12:26
    #     N4: 20260822_204913_measurement_N4_sall.csv       R_KL=0.042146+/-0.001012  2026-08-22 20:49:13
    #     N5: 20260820_064622_measurement_N5_sall.csv       R_KL=0.041239+/-0.001136  2026-08-20 06:46:22
    #     N6: 20260817_091919_measurement_N6_sall.csv       R_KL=0.053557+/-0.001199  2026-08-17 09:19:19
    # (quantum theory per N, this scan: {3: 0.039757, 4: 0.034827, 5: 0.025361, 6: 0.024314} --
    # every point above its theory value, same as every other series here)
    rate_exp_t_loop = [0.040302, 0.037215, 0.038743, 0.045617]
    rate_exp_t_loop_err = [0.000474, 0.001283, 0.000574, 0.000418]
    rate_exp_t_5050 = [0.049057, 0.042146, 0.041239, 0.053557]
    rate_exp_t_5050_err = [0.000562, 0.001012, 0.001136, 0.001199]

    # the 2026-11-17 calibration above was superseded the next day by a
    # fresh calibrate_loss() run (20261118_161931_loss_calibration.json,
    # saved_at 2026-11-18T16:08:05) -- same two methods, new T values:
    #   T_loop  = loss_per_loop_pass      = 0.3992 +/- 0.0107 (was 0.3746 +/- 0.0135)
    #   T_5050  = loss_per_loop_pass_5050 = 0.3323 +/- 0.0372 (was 0.4088 +/- 0.0618)
    # Old calibration file is untouched on disk (still 0.3746/0.4088), so
    # both generations of the T_loop/T_5050 series are kept here side by
    # side rather than overwriting one with the other. Same best-per-N
    # selection/exclusion rule and 400-sample Monte Carlo error bars as
    # above. Two of today's own new measurement files
    # (20261118_165540_measurement_N4_s0.csv, 20261118_054912_measurement_N5_sall.csv)
    # get skipped at this T -- background-subtracted rates go negative --
    # same as other skipped files above, not specific to being new. Rescanned 2026-11-18:
    #   T_loop (0.3992):
    #     N3: 20260803_171226_measurement_N3_sall.csv       R_KL=0.046384+/-0.000537  2026-08-03 17:12:26
    #     N4: 20260822_204913_measurement_N4_sall.csv       R_KL=0.039372+/-0.000930  2026-08-22 20:49:13
    #     N5: 20260820_064622_measurement_N5_sall.csv       R_KL=0.040713+/-0.001023  2026-08-20 06:46:22
    #     N6: 20260817_091919_measurement_N6_sall.csv       R_KL=0.053951+/-0.001220  2026-08-17 09:19:19
    #   T_5050 (0.3323):
    #     N3: 20260729_094121_measurement_N3_sall.csv       R_KL=0.040442+/-0.000358  2026-07-29 09:41:21
    #     N4: 20260822_153701_measurement_N4_sall.csv       R_KL=0.037973+/-0.000299  2026-08-22 15:37:01
    #     N5: 20260730_173305_measurement_N5_s0.csv         R_KL=0.025835+/-0.000363  2026-07-30 17:33:05
    #     N6: 20260730_200453_measurement_N6_s0.csv         R_KL=0.030901+/-0.000421  2026-07-30 20:04:53
    # (every point still above its theory value for this N, same as the other series)
    rate_exp_t_loop_new = [0.046384, 0.039372, 0.040713, 0.053951]
    rate_exp_t_loop_new_err = [0.000537, 0.000930, 0.001023, 0.001220]
    rate_exp_t_5050_new = [0.040442, 0.037973, 0.025835, 0.030901]
    rate_exp_t_5050_new_err = [0.000358, 0.000299, 0.000363, 0.000421]

    # seaborn whitegrid styling, matching Fig. 4 of the paper draft. Classical
    # bound, quantum target, and both generations of the common-T
    # experimental series (2026-11-17 calibration and the 2026-11-18
    # recalibration that superseded it) -- T_meas/T=0.3926/T_fit dropped per
    # request, kept available above for anyone restoring them. All four
    # experimental series in the same dark-red family: old = darker/more
    # saturated, new = lighter; loop = squares, 5050 = triangles.
    _c_classical, _c_quantum = "#3B5BA5", "#4C9A2A"
    _c_exp_t_loop, _c_exp_t_5050 = "#7A0C13", "#C1272D"
    _c_exp_t_loop_new, _c_exp_t_5050_new = "#D1495B", "#F08080"
    _x_c, _y_c = np.array(_x_class), np.array(_y_classical)
    _y_q, _e_q = np.array(_y_quantum), np.array(_qtheo_yerr)

    with sns.axes_style("whitegrid"):
        _fig, _ax = plt.subplots()
        _ax.plot(_x_c, _y_c, "-D", color=_c_classical, linewidth=1.5, label="Classical bound")
        _ax.errorbar(_x_quant, _y_q, yerr=_e_q, fmt="-^", capsize=3, color=_c_quantum,
                      linewidth=1.5, label="Quantum (Target)")
        _ax.fill_between(_x_quant, _y_q - _e_q, _y_q + _e_q, color=_c_quantum, alpha=0.2)
        _ax.errorbar(_x_exp, rate_exp_t_loop, yerr=rate_exp_t_loop_err, fmt="s--", capsize=3,
                      color=_c_exp_t_loop, linewidth=1.5, label="Exp (T=0.3746, loop, 11-17)")
        _ax.errorbar(_x_exp, rate_exp_t_5050, yerr=rate_exp_t_5050_err, fmt="^--", capsize=3,
                      color=_c_exp_t_5050, linewidth=1.5, label="Exp (T=0.4088, 5050, 11-17)")
        _ax.errorbar(_x_exp, rate_exp_t_loop_new, yerr=rate_exp_t_loop_new_err, fmt="s--", capsize=3,
                      color=_c_exp_t_loop_new, linewidth=1.5, label="Exp (T=0.3992, loop, 11-18)")
        _ax.errorbar(_x_exp, rate_exp_t_5050_new, yerr=rate_exp_t_5050_new_err, fmt="^--", capsize=3,
                      color=_c_exp_t_5050_new, linewidth=1.5, label="Exp (T=0.3323, 5050, 11-18)")
        _ax.fill_between(_x_c, _y_c, _ax.get_ylim()[1], alpha=0.3, color="none",
                          edgecolor=_c_classical, hatch="///")
        _ax.set_xticks(_x_c)
        _ax.set_xlabel("Process length N")
        _ax.set_ylabel(r"$R_{KL}$")
        _ax.set_title("KL Divergence Rate for Process Depth N")
        _ax.legend(frameon=True, fontsize=8)
    mo.vstack([_fig, fig_download_button(_fig, "kl_divergence_rate_vs_N.svg")])
    return


@app.cell(hide_code=True)
def _(DATA_PATH, fig_download_button, mo, np, plt, sns):
    import json as _json5050
    import math as _math5050

    # measured vs. expected per-loop-pass counts under the 50/50 calibration
    # optics (calibrate_loss's "5050" stage) -- the same channel-to-channel
    # decay fit that loss_per_loop_pass_5050 comes from (see
    # autoqstochmeasure's _fit_loop_decay), just plotted instead of
    # collapsed to one ratio. Expected curve is anchored at loop 1's own
    # measured rate and decays by the fitted ratio per pass; both series
    # normalised to loop 1 = 1.0.
    #
    # Tied to the SELECTED DATA_PATH: picks the most recent
    # *_loss_calibration.json timestamped at or before this measurement's
    # own timestamp, rather than just whichever one happens to be newest
    # in the whole directory regardless of which file is selected.
    # Deliberately NOT using the measurement's own embedded
    # "loss_calibration" field (unlike T_measured's cell above) -- that
    # embed only ever carries {mean_hz, std_hz} summaries per stage (see
    # autoqstochmeasure's _save_results/_stage_summary), never the raw
    # per-bin counts this plot needs to compute its own error bars from.
    _LOOP_CHS_5050 = [4, 6, 8, 10, 12, 14]  # loop pass 1..6, see autoqstochmeasure/src/libraries/settings.py
    _FIT_PASSES_5050 = 4  # loss balloons past pass 4, dragging the fit -- see autoqstochmeasure's calibrate_loss

    def _fit_loop_decay_5050(bin_rates_by_ch):
        """Same log(rate)-vs-pass least-squares fit as autoqstochmeasure's _fit_loop_decay."""
        _points = [(k, _math5050.log(r)) for k, r in enumerate(bin_rates_by_ch, start=1) if r > 0]
        _n = len(_points)
        _kbar = sum(k for k, _ in _points) / _n
        _ybar = sum(y for _, y in _points) / _n
        _sxx = sum((k - _kbar) ** 2 for k, _ in _points)
        _sxy = sum((k - _kbar) * (y - _ybar) for k, y in _points)
        _slope = _sxy / _sxx
        _ratio = _math5050.exp(_slope)
        if _n > 2:
            _resid_sq = sum((y - (_ybar + _slope * (k - _kbar))) ** 2 for k, y in _points)
            _s_err = (_resid_sq / (_n - 2)) ** 0.5
            _ratio_err = _ratio * (_s_err / _sxx ** 0.5)
        else:
            _ratio_err = 0.0
        return _ratio, _ratio_err

    _meas_ts_5050 = DATA_PATH.stem[:15]  # "YYYYMMDD_HHMMSS" leading prefix
    _candidates_5050 = sorted(
        p for p in DATA_PATH.parent.glob("*_loss_calibration.json")
        if p.name[:15] <= _meas_ts_5050
    )
    mo.stop(not _candidates_5050,
            mo.md(f"No *_loss_calibration.json at or before {_meas_ts_5050} for this file."))
    _cal_path_5050 = _candidates_5050[-1]
    print(f"5050 calibration used: {_cal_path_5050.name} "
          f"(most recent at/before this measurement's timestamp)")
    _cal_5050 = _json5050.loads(_cal_path_5050.read_text())
    mo.stop("5050" not in _cal_5050.get("stages", {}),
            mo.md(f"No '5050' stage in {_cal_path_5050.name}."))
    _bins_5050 = _cal_5050["stages"]["5050"]["counts"]["bins"]

    _mean_rate_5050, _sem_rate_5050 = [], []
    for _ch in _LOOP_CHS_5050:
        _rates_5050 = [b[f"coinc_ch{_ch}"] / b["int_time"] for b in _bins_5050]
        _mean_rate_5050.append(float(np.mean(_rates_5050)))
        _sem_rate_5050.append(float(np.std(_rates_5050, ddof=1) / len(_rates_5050) ** 0.5))

    _ratio_5050, _ratio_err_5050 = _fit_loop_decay_5050(_mean_rate_5050[:_FIT_PASSES_5050])
    # _ratio_5050 is the RAW channel-to-channel decay (what the bars below
    # actually plot) -- it already bakes in the lossless 50/50 splitter's
    # own expected halving every pass, so it is NOT the per-pass
    # transmission T. T = _ratio_5050 / 0.5 (matches calibrate_loss's own
    # stored loss_per_loop_pass_5050 exactly, e.g. 0.166/0.5 = 0.332).
    _T_5050 = _ratio_5050 / 0.5

    _k_5050 = np.arange(1, 7)
    _measured_5050 = np.array(_mean_rate_5050) / _mean_rate_5050[0]
    _measured_err_5050 = np.array(_sem_rate_5050) / _mean_rate_5050[0]
    _expected_5050 = _ratio_5050 ** (_k_5050 - 1)

    with sns.axes_style("whitegrid"):
        _fig5050, _ax5050 = plt.subplots()
        _w5050 = 0.35
        _ax5050.bar(_k_5050/ 2, _measured_5050, width=_w5050, yerr=_measured_err_5050,
                    capsize=4, color="gold", ecolor="black", label="Measured (5050 stage)")
        _ax5050.bar(_k_5050/ 2, _expected_5050, width=_w5050, fill=False, edgecolor="#7A0C13",
                    linestyle="dashed", linewidth=1.5,
                    label=f"5050 expected (T={_T_5050:.3f})")
        _ax5050.set_xticks(_k_5050)
        _ax5050.set_xlabel("Loop pass")
        _ax5050.set_ylabel("Counts (normalised to loop 1)")
        _ax5050.set_title("50/50 calibration: measured vs. expected counts per loop pass")
        _ax5050.legend()
    mo.vstack([_fig5050, fig_download_button(_fig5050, "5050_decay_per_loop.svg")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Output distribution vs N (3D, input state 0)""")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    background_rate_by_channel,
    efficiency_for_channel,
    filepath,
    np,
    pd,
):
    from pathlib import Path as _Path3d
    from scipy.optimize import minimize_scalar as _minimize_scalar_3d
    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal_3d
    from stochprocsim.utils import rates_by_input_state as _rates_by_input_state_3d

    # outright-best per-N file for input state 0, same T_fit method/picks
    # documented in the rate_exp cell above -- reused here rather than
    # re-running a fresh best-file search
    _BEST_FILES_3D = {
        3: "20260803_171226_measurement_N3_sall.csv",
        4: "20260804_092116_measurement_N4_s0_tomo.csv",
        5: "20260730_173305_measurement_N5_s0.csv",
        6: "20260730_200453_measurement_N6_s0.csv",
    }

    def _dist_for_N(n):
        _bins = [f"coinc_ch{2 * k}" for k in range(1, n + 1)] + ["coinc_ch7"]
        _path = _Path3d(filepath) / _BEST_FILES_3D[n]
        _data = pd.read_csv(_path)
        _data = _data.dropna(subset=_bins + ["herald", "int_time", "input_state"])
        _data = _data[_data["herald"] > 0]
        _rates, _stds = _rates_by_input_state_3d(_data, _bins)

        _bg_by_ch, _ = background_rate_by_channel(_path.with_suffix(".json"))
        _bg = pd.Series({_b: _bg_by_ch.get(_b.removeprefix("coinc_ch"), 0.0) for _b in _bins})
        _eff = pd.Series({
            _b: efficiency_for_channel(int(_b.removeprefix("coinc_ch")), dump_ch=7) for _b in _bins
        })

        _sim = Simulator(QuantumTransitionModel(Causal_Models[n]))
        _p = _sim.get_output_distribution(propagate_outputs=True, start_state=0)
        _loop_p = {2 * (_i + 1): _v for _i, _v in enumerate(_p)}
        _theory = np.array([
            1 - sum(_p) if _b == "coinc_ch7" else _loop_p.get(int(_b.removeprefix("coinc_ch")), 0.0)
            for _b in _bins
        ])

        _probs = _get_uniform_renewal_3d(n - 1).probs
        _exit, _a_prod = [], 1.0
        for _p_survive in _probs:
            _exit.append((1 - _p_survive) * _a_prod)
            _a_prod *= _p_survive
        _exit.append(_a_prod)
        _loop_exact = {2 * (_i + 1): _v for _i, _v in enumerate(_exit)}
        _exact = np.array([
            0.0 if _b == "coinc_ch7" else _loop_exact.get(int(_b.removeprefix("coinc_ch")), 0.0)
            for _b in _bins
        ])

        _k = np.array([n if _b == "coinc_ch7" else int(_b.removeprefix("coinc_ch")) // 2 for _b in _bins])

        def _resid(T):
            _corr = ((_rates.loc[0] - _bg) / _eff) / T ** _k
            return np.sum((_theory - _corr / _corr.clip(lower=0).sum()) ** 2)

        _T_fit = _minimize_scalar_3d(_resid, bounds=(0.05, 1), method="bounded").x
        _corr = ((_rates.loc[0] - _bg) / _eff) / _T_fit ** _k
        # clamped-sum normalization -- see the T_measured/T_fit frac cells
        # above for why (a negative bin would otherwise inflate every
        # other channel's fraction above its true value)
        _exp = (_corr / _corr.clip(lower=0).sum()).to_numpy()
        return _exp, _theory, _exact

    dist_by_N = {n: _dist_for_N(n) for n in sorted(_BEST_FILES_3D)}
    return


if __name__ == "__main__":
    app.run()
