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
        block_rate_mean_std,
        drop_count_outliers,
        efficiency_for_channel,
        kl_rate_from_output_distribution,
        kl_rate_mc_error,
        loop_pass_k,
        loss_calibration_for,
        loss_rates_for,
    )
    return (
        background_rate_by_channel,
        block_rate_mean_std,
        drop_count_outliers,
        efficiency_for_channel,
        io,
        kl_rate_from_output_distribution,
        kl_rate_mc_error,
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
def _(BINS, DATA_PATH, drop_count_outliers, pd):
    from stochprocsim.utils import rates_by_input_state

    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=BINS + ["herald", "int_time", "input_state"])
    data = data[data["herald"] > 0]  # herald blocked / off segments

    # a single catastrophic detector glitch (one row reading 100-1000x its
    # channel's normal count) wrecks every downstream number pulled from
    # `rates`/`stds` -- the bar plots, the loss-corrected frac, every R_KL
    # cell below -- not just the aggregate series further down this file.
    # Confirmed on real data: one such row alone blew up a file's R_KL
    # error bar 5x and biased its central value. Applied here, once, right
    # after load, so nothing downstream ever sees it.
    #
    # Grouped by input_state (and basis, for a tomo file): each state
    # routes light differently, so a channel's "normal" rate can differ
    # 4x+ state to state (confirmed on real data) -- a single file-wide
    # median would let a real outlier in a low-baseline state hide below
    # threshold, or flag a genuinely-higher-baseline state's normal rows as
    # outliers against the mix.
    _group_cols = ["input_state", "basis"] if "basis" in data.columns else ["input_state"]
    _n_before = len(data)
    data = data.groupby(_group_cols, group_keys=False, dropna=False).apply(
        lambda g: drop_count_outliers(g, BINS))
    if len(data) < _n_before:
        print(f"dropped {_n_before - len(data)} outlier row(s) (>15x that channel's median, per input state)")

    # counts per second, grouped by which memory state the machine started in
    # (and, for tomography files, combined across polarization basis — see
    # rates_by_input_state); a single-input file just yields one group and
    # every cell below collapses to the single-distribution case
    rates, stds = rates_by_input_state(data, BINS)
    print(f"{len(data)} rows kept, input states: {list(rates.index)}")
    return rates, stds


@app.cell(hide_code=True)
def _(BINS, N, loop_pass_k, np, rates):
    from stochprocsim.libraries.optics_lib import getUtot
    from stochprocsim.models.causal_models import Causal_Models, CausalModel as _CausalModel
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel

    # theory ("theory" bars) now comes from getUtot(angles_NTU) -- the
    # unitary rebuilt from the calibrated waveplate/mirror angles --
    # rather than the frozen U_theo matrix, per 2026-11-24 request.
    # true_theory (new comparison bars, every distribution plot) is the
    # original frozen U_theo matrix's own prediction, kept alongside
    # since the two don't actually agree for this apparatus (confirmed
    # 2026-11-24: getUtot(angles_NTU) != U_theo, even after an NTU/GU
    # convention swap) -- showing both instead of silently picking one.
    _U_getutot = getUtot(Causal_Models[N].angles_NTU)
    _cm_getutot = _CausalModel(U=_U_getutot, states=Causal_Models[N].states,
                               angles_NTU=Causal_Models[N].angles_NTU,
                               angles_GU=Causal_Models[N].angles_GU, name="getutot")
    _sim = Simulator(QuantumTransitionModel(_cm_getutot))
    _sim_true = Simulator(QuantumTransitionModel(Causal_Models[N]))

    def _map_to_bins(_p):
        # _p[i] is the probability of exiting on loop pass i+1, mapped onto
        # BINS by loop_pass_k (position among recorded loop channels, not
        # channel number -- see loop_pass_k docstring for why); channels
        # beyond the model's N loops (extra recorded channels) get 0
        return np.array([
            1 - sum(_p) if _b == "coinc_ch7"
            else (_p[loop_pass_k(_b, BINS) - 1] if loop_pass_k(_b, BINS) <= len(_p) else 0.0)
            for _b in BINS
        ])

    p_theory, true_theory = {}, {}
    for _s in rates.index:
        p_theory[_s] = _map_to_bins(_sim.get_output_distribution(propagate_outputs=True, start_state=int(_s)))
        true_theory[_s] = _map_to_bins(_sim_true.get_output_distribution(propagate_outputs=True, start_state=int(_s)))
        print(f"s{_s} theory (getUtot):     {np.round(p_theory[_s], 4)}")
        print(f"s{_s} true_theory (U_theo): {np.round(true_theory[_s], 4)}")
    return Causal_Models, QuantumTransitionModel, Simulator, p_theory, true_theory


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
    def plot_state_grid(frac_by_state, yerr_by_state, p_theory, suptitle, p_exact=None, true_theory=None, titles=None):
        p_exact = p_exact or {}
        true_theory = true_theory or {}
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
            if _s in true_theory:
                _ax.bar(_x, true_theory[_s], width=0.5, fill=False, edgecolor="#1F9E89",
                        linestyle="dashdot", linewidth=1.2, label="true_theory")
            _ax.set_xticks(_x, _f.index)
            _ax.tick_params(axis="x", labelsize=7)
            _ax.set_title(titles[_s] if titles else f"input s{_s}", fontsize=8)
        _axes.flat[0].legend()
        _fig.suptitle(suptitle)
        _fig.tight_layout()
        return _fig
    return (plot_state_grid,)


@app.cell(hide_code=True)
def _(p_exact, p_theory, plot_state_grid, rates, stds, true_theory):
    # raw fraction per bin vs theory (no loss correction)
    plot_state_grid(
        {_s: rates.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        {_s: stds.loc[_s] / rates.loc[_s].sum() for _s in rates.index},
        p_theory,
        "raw fractions vs theory",
        p_exact=p_exact,
        true_theory=true_theory,
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

    # `.get(key, 0.0)` only falls back when the key is MISSING -- a
    # calibration taken while a stage read all-zero counts (e.g. the
    # detector was latched) has the key present but explicitly `null` in
    # the json, which `.get` happily returns as `None` straight through to
    # the f-string formatting below. `or 0.0` catches that too.
    eta_dump_sp = _losses.get("det_eff_dump") or 0.0
    input_to_loop_sp = _losses.get("loss_input_to_setup") or 0.0
    loop_to_loop_sp = _losses.get("loss_per_loop_pass") or 0.0
    loop_to_dump_sp = _losses.get("loss_to_dump") or 0.0

    # "_err" companions, already propagated into the json by the loop
    # apparatus's own loss-calibration routine (not recomputed here)
    eta_dump_sp_std = _losses.get("det_eff_dump_err") or 0.0
    input_to_loop_sp_std = _losses.get("loss_input_to_setup_err") or 0.0
    loop_to_loop_sp_std = _losses.get("loss_per_loop_pass_err") or 0.0
    loop_to_dump_sp_std = _losses.get("loss_to_dump_err") or 0.0

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
    true_theory,
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
                    p_exact=p_exact, true_theory=true_theory)
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
    true_theory,
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
                    p_exact=p_exact, true_theory=true_theory)
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
def _(mo):
    # single source of truth for "current best"/"second best" per N.
    # Seeded with a 2026-11-23 rescan (same rule the search cell below
    # applies: loss_per_loop_pass [loop01, direct 2-point ratio] trusted
    # over loss_per_loop_pass_5050 [a 6-channel decay fit confirmed wrong
    # on two calibration files this week]; no below-theory floor -- the
    # quantum-theory curve isn't a hard bound a real result can't beat,
    # so ranking is purely by lowest KL, same as the search cell below).
    #
    # Measurement files whose own loss calibration has T_loop < 0.3 were
    # moved to data/archive/ this session (per request -- that low a
    # transmission was judged implausible for this apparatus) and so no
    # longer show up in the glob these picks are scanned from; this
    # knocked out the previous N3/N4/N5 best picks, which had all
    # happened to share the same T=0.239 calibration.
    #
    # Stored as mo.state (not plain dicts) so the "search" button below
    # can update them in place -- every downstream cell (the purple/violet
    # series, the distribution-bar plots) reads get_best_picks()/
    # get_second_best_picks() and recomputes automatically when either
    # changes, same reactivity as before.
    _INITIAL_BEST_PICKS = {
        3: ("20261119_152443_measurement_N3_s0", "loss_per_loop_pass", "loss_per_loop_pass_err"),
        4: ("20261121_162359_measurement_N4_sall", "loss_per_loop_pass", "loss_per_loop_pass_err"),
        5: ("20261122_235532_measurement_N5_sall", "loss_per_loop_pass", "loss_per_loop_pass_err"),
        6: ("20261123_171740_measurement_N6_s0", "loss_per_loop_pass", "loss_per_loop_pass_err"),
    }
    _INITIAL_SECOND_BEST_PICKS = {
        3: ("20261123_165435_measurement_N3_s0", "loss_per_loop_pass", "loss_per_loop_pass_err"),
        4: ("20261119_153736_measurement_N4_s0", "loss_per_loop_pass", "loss_per_loop_pass_err"),
        5: ("20261122_225307_measurement_N5_s0", "loss_per_loop_pass", "loss_per_loop_pass_err"),
        6: ("20261123_020042_measurement_N6_sall", "loss_per_loop_pass", "loss_per_loop_pass_err"),
    }
    get_best_picks, set_best_picks = mo.state(_INITIAL_BEST_PICKS)
    get_second_best_picks, set_second_best_picks = mo.state(_INITIAL_SECOND_BEST_PICKS)
    search_button = mo.ui.run_button(label="Search this range")
    search_window_days = mo.ui.number(value=21, start=7, step=7, label="search window (days)")
    mo.hstack([search_button, search_window_days])
    return (
        get_best_picks,
        get_second_best_picks,
        search_button,
        search_window_days,
        set_best_picks,
        set_second_best_picks,
    )


@app.cell(hide_code=True)
def _(
    DATA_PATH,
    background_rate_by_channel,
    drop_count_outliers,
    efficiency_for_channel,
    kl_rate_from_output_distribution,
    loop_pass_k,
    loss_calibration_for,
    mo,
    np,
    pd,
    search_button,
    search_window_days,
    set_best_picks,
    set_second_best_picks,
):
    # runs only on button click (mo.stop below), not on every edit --
    # bounded to the last `search_window_days` days by filename timestamp
    # for speed (this dataset goes back to 2026-07; scanning all of it on
    # every click would be minutes, not seconds). loop01 only, same reason
    # BEST_PICKS's seed values prefer it -- 5050's multi-pass decay fit
    # isn't reliable enough to trust in an unattended scan. Point-estimate
    # KL only here (no 3000-sample Monte Carlo -- that's what makes this
    # fast enough to run on every click; the MC error still gets computed
    # properly for whichever pick ends up stored, by the cell below).
    #
    # This is a fresh "what's best/second-best IN THIS WINDOW" search
    # every click, not a ratchet that only replaces a stored pick on
    # improvement -- a narrower follow-up search (e.g. widen to check a
    # large range, then narrow back down) must show exactly what's in
    # the narrower range, even if that's worse than a wider search found
    # a moment ago. Tracks best AND second-best per N in a SINGLE O(N)
    # pass over this window's candidate files -- classic top-2
    # incremental update (if a candidate beats best, the old best demotes
    # to second; else if it only beats second, it replaces second), not a
    # sort (O(N log N)) and not a separate second pass (O(2N), same
    # complexity class but wasteful).
    #
    # No below-theory floor -- per request 2026-11-23: the quantum-theory
    # curve isn't a hard bound here, a real result can legitimately beat
    # it, so a candidate is only skipped if the calc itself fails (no
    # usable T, or a degenerate/negative distribution), not based on
    # where its KL lands relative to theory.
    mo.stop(not search_button.value, mo.md("Click **Search this range** above to scan it."))

    _THEORY = {3: 0.039757, 4: 0.034827, 5: 0.025361, 6: 0.024314}

    import datetime as _dt
    import re as _re_search

    def _quick_point_kl(stem, pick_n):
        _json_path = DATA_PATH.parent / f"{stem}.json"
        _csv_path = DATA_PATH.parent / f"{stem}.csv"
        if not _json_path.exists() or not _csv_path.exists():
            return None
        _losses, _ = loss_calibration_for(_json_path)
        _T = _losses.get("loss_per_loop_pass")
        if not _T:
            return None
        _cols = pd.read_csv(_csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        _bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            _bins = _bins + ["coinc_ch7"]
        _data = pd.read_csv(_csv_path)
        _data = _data.dropna(subset=_bins + ["herald", "int_time", "input_state"])
        _data = _data[_data["herald"] > 0]
        _s0 = _data[_data["input_state"] == 0]
        if "basis" in _s0.columns:
            _s0 = _s0[_s0["basis"].isna()]
        if len(_s0) < 10:
            return None
        _s0 = drop_count_outliers(_s0, _bins)
        _rates = _s0[_bins].sum() / _s0["int_time"].sum()
        _bg_by_ch, _ = background_rate_by_channel(_json_path)
        _bg = pd.Series({b: _bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in _bins})
        _eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in _bins})
        _k = np.array([pick_n if b == "coinc_ch7" else loop_pass_k(b, _bins) for b in _bins])
        _corr = ((_rates - _bg) / _eff) / (_T ** _k)
        _total = _corr.clip(lower=0).sum()
        if _total <= 0:
            return None
        _frac = (_corr.clip(lower=0) / _total).to_numpy()[:pick_n]
        return float(kl_rate_from_output_distribution(pick_n, _frac))

    _cutoff = (_dt.datetime.now() - _dt.timedelta(days=search_window_days.value)).strftime("%Y%m%d_%H%M%S")

    # Fresh search EVERY click, not a ratchet against whatever's already
    # stored -- a narrower follow-up search must show the best/second-best
    # actually IN that window, even if it's worse than a wider search found
    # earlier. (Previous version seeded from get_best_picks()/
    # get_second_best_picks() and only replaced on improvement, so
    # widening-then-narrowing the window silently kept showing the wide-
    # window winner -- not what "search this range" should mean.)
    _best = {_n: (None, float("inf")) for _n in _THEORY}
    _second = {_n: (None, float("inf")) for _n in _THEORY}

    for _p in sorted((DATA_PATH.parent).glob("*measurement_N*.csv")):
        _ts_match = _re_search.search(r"\d{8}_\d{6}", _p.name)
        _n_match = _re_search.search(r"_N(\d+)_", _p.name)
        if not _ts_match or not _n_match or _ts_match.group() < _cutoff:
            continue
        _n = int(_n_match.group(1))
        if _n not in _best:
            continue
        _kl = _quick_point_kl(_p.stem, _n)
        if _kl is None:
            continue
        if _kl < _best[_n][1]:
            _second[_n] = _best[_n]
            _best[_n] = (_p.stem, _kl)
        elif _p.stem != _best[_n][0] and _kl < _second[_n][1]:
            _second[_n] = (_p.stem, _kl)

    _updated_best = {_n: (_stem, "loss_per_loop_pass", "loss_per_loop_pass_err")
                      for _n, (_stem, _kl) in _best.items() if _stem}
    _updated_second = {_n: (_stem, "loss_per_loop_pass", "loss_per_loop_pass_err")
                        for _n, (_stem, _kl) in _second.items() if _stem}

    set_best_picks(_updated_best)
    set_second_best_picks(_updated_second)
    for _n in sorted(_THEORY):
        _best_stem, _best_kl = _best[_n]
        _second_stem, _second_kl = _second[_n]
        print(f"N{_n} (last {search_window_days.value}d): "
              f"best={_best_stem or 'none'} ({_best_kl:.6f})  "
              f"second={_second_stem or 'none'} ({_second_kl:.6f})")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    DATA_PATH,
    QuantumTransitionModel,
    Simulator,
    background_rate_by_channel,
    block_rate_mean_std,
    drop_count_outliers,
    efficiency_for_channel,
    get_best_picks,
    get_second_best_picks,
    kl_rate_from_output_distribution,
    kl_rate_mc_error,
    loop_pass_k,
    loss_calibration_for,
    np,
    pd,
):
    BEST_PICKS = get_best_picks()
    SECOND_BEST_PICKS = get_second_best_picks()

    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal_best
    from stochprocsim.libraries.optics_lib import getUtot as _getUtot_best
    from stochprocsim.models.causal_models import CausalModel as _CausalModel_best

    _data_dir = DATA_PATH.parent

    def _best_pick_stats(stem, pick_N, loss_key, err_key):
        json_path = _data_dir / f"{stem}.json"
        csv_path = _data_dir / f"{stem}.csv"
        losses, cal_note = loss_calibration_for(json_path)
        T, T_err = losses[loss_key], losses.get(err_key) or 0.0

        _cols = pd.read_csv(csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            bins = bins + ["coinc_ch7"]
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=bins + ["herald", "int_time", "input_state"])
        data = data[data["herald"] > 0]
        s0 = data[data["input_state"] == 0]
        if "basis" in s0.columns:
            s0 = s0[s0["basis"].isna()]
        s0 = drop_count_outliers(s0, bins)

        rates, stds = block_rate_mean_std(s0, bins, block_size=30)
        # background taken as exactly measured, no error term -- per
        # request 2026-11-23: background_rate_std_by_channel's inline-only
        # SEM was judged not worth modeling (kl_rate_mc_error's bg_std
        # already defaults to None/0 for exactly this case, so simply not
        # passing it here is sufficient).
        bg_by_ch, bg_note = background_rate_by_channel(json_path)
        bg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in bins})
        eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in bins})

        k = np.array([pick_N if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins])
        transmission = T ** k
        corr = ((rates - bg) / eff) / transmission
        norm = corr.clip(lower=0).sum()
        frac = corr.clip(lower=0) / norm
        yerr = ((stds / eff) / transmission) / norm
        point_kl = kl_rate_from_output_distribution(pick_N, frac.to_numpy()[:pick_N])
        mc_mean, mc_std, mc_converged = kl_rate_mc_error(pick_N, rates, stds, T, T_err, bg, eff, bins,
                                                           n_samples=3000, rng=np.random.default_rng(0),
                                                           check_convergence=True)
        if not mc_converged:
            print(f"  WARNING: N{pick_N} MC error estimate may not have converged at 3000 samples "
                  f"(two independent halves disagreed by >20%) -- treat mc_std as approximate")

        # theory bars: getUtot(angles_NTU) (the unitary rebuilt from this
        # N's calibrated waveplate/mirror angles), per 2026-11-24 request
        _U_getutot = _getUtot_best(Causal_Models[pick_N].angles_NTU)
        _cm_getutot = _CausalModel_best(U=_U_getutot, states=Causal_Models[pick_N].states,
                                          angles_NTU=Causal_Models[pick_N].angles_NTU,
                                          angles_GU=Causal_Models[pick_N].angles_GU, name="getutot")
        sim = Simulator(QuantumTransitionModel(_cm_getutot))
        _p = sim.get_output_distribution(propagate_outputs=True, start_state=0)
        p_theory = np.array([1 - sum(_p) if b == "coinc_ch7"
                              else (_p[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p) else 0.0)
                              for b in bins])

        # true_theory bars: the frozen U_theo matrix's own prediction --
        # kept alongside since getUtot(angles_NTU) and U_theo don't
        # actually agree for this apparatus (confirmed 2026-11-24)
        sim_true = Simulator(QuantumTransitionModel(Causal_Models[pick_N]))
        _p_true = sim_true.get_output_distribution(propagate_outputs=True, start_state=0)
        true_theory = np.array([1 - sum(_p_true) if b == "coinc_ch7"
                                 else (_p_true[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p_true) else 0.0)
                                 for b in bins])

        _probs = _get_uniform_renewal_best(pick_N - 1).probs
        _exit, _a_prod = [], 1.0
        for _p_survive in _probs:
            _exit.append((1 - _p_survive) * _a_prod)
            _a_prod *= _p_survive
        _exit.append(_a_prod)
        p_exact = np.array([0.0 if b == "coinc_ch7"
                             else (_exit[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_exit) else 0.0)
                             for b in bins])

        # truncate the DISPLAYED distribution to this N's own loop passes
        # (+ dump) -- the hardware always records all 6 loop channels
        # regardless of which N is active, so e.g. an N=3 pick's ch10/12/14
        # are pure noise/crosstalk, not signal. Showing them anyway is how
        # a near-zero bin with a huge relative SEM (confirmed: ch14 here
        # once had an error bar 15x its own bar height) ends up on the
        # plot -- truncating matches what the KL calc itself already does
        # (kl_rate_from_output_distribution above is only ever given the
        # first pick_N entries).
        _keep = [b == "coinc_ch7" or loop_pass_k(b, bins) <= pick_N for b in bins]
        bins_trunc = [b for b, _m in zip(bins, _keep) if _m]
        frac, yerr = frac[bins_trunc], yerr[bins_trunc]
        p_theory, p_exact, true_theory = p_theory[_keep], p_exact[_keep], true_theory[_keep]

        return {
            "stem": stem, "T": T, "T_err": T_err, "cal_note": cal_note,
            "frac": frac, "yerr": yerr, "bins": bins_trunc,
            "point_kl": point_kl, "mc_mean": mc_mean, "mc_std": mc_std,
            "p_theory": p_theory, "p_exact": p_exact, "true_theory": true_theory,
        }

    best_results = {_n: _best_pick_stats(_stem, _n, _lk, _ek)
                     for _n, (_stem, _lk, _ek) in BEST_PICKS.items()}
    for _n, _r in best_results.items():
        print(f"N{_n} best: {_r['stem']}  T={_r['T']:.4f}+/-{_r['T_err']:.4f} ({_r['cal_note']})  "
              f"point={_r['point_kl']:.6f}  MC={_r['mc_mean']:.6f}+/-{_r['mc_std']:.6f}")

    second_best_results = {_n: _best_pick_stats(_stem, _n, _lk, _ek)
                            for _n, (_stem, _lk, _ek) in SECOND_BEST_PICKS.items()}
    for _n, _r in second_best_results.items():
        print(f"N{_n} second-best: {_r['stem']}  T={_r['T']:.4f}+/-{_r['T_err']:.4f} ({_r['cal_note']})  "
              f"point={_r['point_kl']:.6f}  MC={_r['mc_mean']:.6f}+/-{_r['mc_std']:.6f}")
    return best_results, second_best_results


@app.cell(hide_code=True)
def _(best_results, plot_state_grid):
    # output distribution (loss-corrected frac vs theory) for each N's
    # CURRENT best pick -- reads straight from best_results above, so this
    # redraws automatically whenever BEST_PICKS changes, no manual re-sync.
    _frac_by_n = {_n: _r["frac"] for _n, _r in best_results.items()}
    _yerr_by_n = {_n: _r["yerr"] for _n, _r in best_results.items()}
    _p_theory_by_n = {_n: _r["p_theory"] for _n, _r in best_results.items()}
    _p_exact_by_n = {_n: _r["p_exact"] for _n, _r in best_results.items()}
    _true_theory_by_n = {_n: _r["true_theory"] for _n, _r in best_results.items()}
    _titles = {_n: f"N={_n}  T={_r['T']:.3f}±{_r['T_err']:.3f}\nKL={_r['mc_mean']:.4f}±{_r['mc_std']:.4f}"
               for _n, _r in best_results.items()}
    plot_state_grid(_frac_by_n, _yerr_by_n, _p_theory_by_n,
                     "Current best-rated (last week) output distributions vs theory",
                     p_exact=_p_exact_by_n, true_theory=_true_theory_by_n, titles=_titles)
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    best_results,
    fig_download_button,
    mo,
    np,
    plt,
    second_best_results,
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

    # stale: these were sized for the old state-0-only _sim_quantum, not
    # recomputed for the stationary-weighted formula above -- treat as a
    # rough error bar, not a precise one, until redone
    _qtheo_yerr = [0.002054339006460678, 0.0014301897956533847, 0.0016754397869550568, 0.0015075218255867487]

    # x-axis + data for the kept "T=0.3746, loop, 11-17" series -- every
    # other historical series shared this same [3,4,5,6] x-axis and got
    # removed, this is the one still plotted (see git history for the
    # original per-N file list this came from, e.g.
    # 20260803_171226_measurement_N3_sall.csv -> R_KL=0.040302+/-0.000474).
    _x_exp = [3, 4, 5, 6]
    rate_exp_t_loop = [0.040302, 0.037215, 0.038743, 0.045617]
    rate_exp_t_loop_err = [0.000474, 0.001283, 0.000574, 0.000418]

    _x_exp_recent = sorted(best_results)
    rate_exp_recent = [best_results[_n]["mc_mean"] for _n in _x_exp_recent]
    rate_exp_recent_err = [best_results[_n]["mc_std"] for _n in _x_exp_recent]

    _x_exp_second = sorted(second_best_results)
    rate_exp_second = [second_best_results[_n]["mc_mean"] for _n in _x_exp_second]
    rate_exp_second_err = [second_best_results[_n]["mc_std"] for _n in _x_exp_second]

    _c_classical, _c_quantum = "#3B5BA5", "#4C9A2A"
    _c_exp_t_loop, _c_exp_recent, _c_exp_second = "#7A0C13", "#5C2E8A", "#A77DD1"
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
        _ax.errorbar(_x_exp_recent, rate_exp_recent, yerr=rate_exp_recent_err, fmt="s--", capsize=3,
                      color=_c_exp_recent, linewidth=1.5, label="Exp (current best, last week)")
        _ax.errorbar(_x_exp_second, rate_exp_second, yerr=rate_exp_second_err, fmt="^--", capsize=3,
                      color=_c_exp_second, linewidth=1.5, label="Exp (second-best, last week)")
        _ax.fill_between(_x_c, _y_c, _ax.get_ylim()[1], alpha=0.3, color="none",
                          edgecolor=_c_classical, hatch="///")
        _ax.set_xticks(_x_c)
        # _ax.set_ylim(0, 0.4)
        _ax.set_xlabel("Process length N")
        _ax.set_ylabel(r"$R_{KL}$")
        _ax.set_title("KL Divergence Rate for Process Depth N")
        _ax.legend(frameon=True, fontsize=8)
    mo.vstack([_fig, fig_download_button(_fig, "kl_divergence_rate_vs_N.svg")])
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    DATA_PATH,
    QuantumTransitionModel,
    Simulator,
    background_rate_by_channel,
    block_rate_mean_std,
    drop_count_outliers,
    efficiency_for_channel,
    kl_rate_from_output_distribution,
    kl_rate_mc_error,
    loop_pass_k,
    loss_rates_for,
    np,
    pd,
    plot_state_grid,
):
    # Closest-to-theory_U pick per N using T_measured (loop1_hz/loop0_hz,
    # same ratio as the T_measured cell earlier in this notebook) as the
    # loss correction -- ranked by sum|measured - theory_U| over this N's
    # own loop bins, instead of by lowest KL vs. the uniform-renewal
    # model. These are different yardsticks: a run can match theory_U
    # closely yet have a worse R_KL, since R_KL is never computed against
    # theory_U at all (see 2026-11-24 conversation, e.g.
    # 20261123_213652_measurement_N4_s0 had the SMALLEST theory_U residual
    # of that night's N4 runs but the WORST R_KL). KL is still shown here
    # for context, not as the selection criterion.
    _data_dir = DATA_PATH.parent
    import datetime as _dt_tmeasured
    import re as _re_tmeasured

    def _quick_theory_residual_tmeasured(stem, pick_n):
        _json_path = _data_dir / f"{stem}.json"
        _csv_path = _data_dir / f"{stem}.csv"
        if not _json_path.exists() or not _csv_path.exists():
            return None
        _loss_rates, _ = loss_rates_for(_json_path)
        _c2, _c4 = _loss_rates.get("loop0_hz", 0.0), _loss_rates.get("loop1_hz", 0.0)
        _T = _c4 / _c2 if _c2 else 0.0
        if not _T:
            return None
        _cols = pd.read_csv(_csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        _bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            _bins = _bins + ["coinc_ch7"]
        _data = pd.read_csv(_csv_path)
        _data = _data.dropna(subset=_bins + ["herald", "int_time", "input_state"])
        _data = _data[_data["herald"] > 0]
        _s0 = _data[_data["input_state"] == 0]
        if "basis" in _s0.columns:
            _s0 = _s0[_s0["basis"].isna()]
        if len(_s0) < 10:
            return None
        _s0 = drop_count_outliers(_s0, _bins)
        _rates = _s0[_bins].sum() / _s0["int_time"].sum()
        _bg_by_ch, _ = background_rate_by_channel(_json_path)
        _bg = pd.Series({b: _bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in _bins})
        _eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in _bins})
        _k = np.array([pick_n if b == "coinc_ch7" else loop_pass_k(b, _bins) for b in _bins])
        _corr = ((_rates - _bg) / _eff) / (_T ** _k)
        _total = _corr.clip(lower=0).sum()
        if _total <= 0:
            return None
        _frac = (_corr.clip(lower=0) / _total).to_numpy()

        _sim = Simulator(QuantumTransitionModel(Causal_Models[pick_n]))
        _p = _sim.get_output_distribution(propagate_outputs=True, start_state=0)
        _p_theory = np.array([1 - sum(_p) if b == "coinc_ch7"
                               else (_p[loop_pass_k(b, _bins) - 1] if loop_pass_k(b, _bins) <= len(_p) else 0.0)
                               for b in _bins])
        _keep = [b == "coinc_ch7" or loop_pass_k(b, _bins) <= pick_n for b in _bins]
        return float(np.abs(_frac[_keep] - _p_theory[_keep]).sum())

    _SEARCH_WINDOW_DAYS_TM = 21
    _cutoff_tmeasured = (_dt_tmeasured.datetime.now() - _dt_tmeasured.timedelta(days=_SEARCH_WINDOW_DAYS_TM)).strftime("%Y%m%d_%H%M%S")

    _best_tmeasured = {_n: (None, float("inf")) for _n in (3, 4, 5, 6)}
    for _fp in sorted(_data_dir.glob("*measurement_N*.csv")):
        _ts_match = _re_tmeasured.search(r"\d{8}_\d{6}", _fp.name)
        _n_match = _re_tmeasured.search(r"_N(\d+)_", _fp.name)
        if not _ts_match or not _n_match or _ts_match.group() < _cutoff_tmeasured:
            continue
        _n = int(_n_match.group(1))
        if _n not in _best_tmeasured:
            continue
        _residual = _quick_theory_residual_tmeasured(_fp.stem, _n)
        if _residual is not None and _residual < _best_tmeasured[_n][1]:
            _best_tmeasured[_n] = (_fp.stem, _residual)

    def _theory_match_stats_tmeasured(stem, pick_N):
        json_path = _data_dir / f"{stem}.json"
        csv_path = _data_dir / f"{stem}.csv"
        loss_rates, _ = loss_rates_for(json_path)
        c2, c4 = loss_rates.get("loop0_hz", 0.0), loss_rates.get("loop1_hz", 0.0)
        c2_err, c4_err = loss_rates.get("loop0_hz_err", 0.0), loss_rates.get("loop1_hz_err", 0.0)
        T = c4 / c2 if c2 else 0.0
        T_err = (T * ((c4_err / c4) ** 2 + (c2_err / c2) ** 2) ** 0.5) if c2 and c4 else 0.0

        _cols = pd.read_csv(csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            bins = bins + ["coinc_ch7"]
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=bins + ["herald", "int_time", "input_state"])
        data = data[data["herald"] > 0]
        s0 = data[data["input_state"] == 0]
        if "basis" in s0.columns:
            s0 = s0[s0["basis"].isna()]
        s0 = drop_count_outliers(s0, bins)

        rates, stds = block_rate_mean_std(s0, bins, block_size=30)
        bg_by_ch, _ = background_rate_by_channel(json_path)
        bg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in bins})
        eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in bins})

        k = np.array([pick_N if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins])
        transmission = T ** k
        corr = ((rates - bg) / eff) / transmission
        norm = corr.clip(lower=0).sum()
        frac = corr.clip(lower=0) / norm
        yerr = ((stds / eff) / transmission) / norm
        point_kl = kl_rate_from_output_distribution(pick_N, frac.to_numpy()[:pick_N])
        mc_mean, mc_std, mc_converged = kl_rate_mc_error(pick_N, rates, stds, T, T_err, bg, eff, bins,
                                                           n_samples=300, rng=np.random.default_rng(0),
                                                           check_convergence=True)
        if not mc_converged:
            print(f"  WARNING: N{pick_N} (T_measured) MC error estimate may not have converged at 300 "
                  f"samples (two independent halves disagreed by >20%) -- treat mc_std as approximate")

        # theory bars: getUtot(angles_NTU), per 2026-11-24 request
        from stochprocsim.libraries.optics_lib import getUtot as _getUtot_tm
        from stochprocsim.models.causal_models import CausalModel as _CausalModel_tm
        _U_getutot = _getUtot_tm(Causal_Models[pick_N].angles_NTU)
        _cm_getutot = _CausalModel_tm(U=_U_getutot, states=Causal_Models[pick_N].states,
                                        angles_NTU=Causal_Models[pick_N].angles_NTU,
                                        angles_GU=Causal_Models[pick_N].angles_GU, name="getutot")
        sim = Simulator(QuantumTransitionModel(_cm_getutot))
        _p = sim.get_output_distribution(propagate_outputs=True, start_state=0)
        p_theory = np.array([1 - sum(_p) if b == "coinc_ch7"
                              else (_p[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p) else 0.0)
                              for b in bins])

        # true_theory bars: the frozen U_theo matrix's own prediction
        sim_true = Simulator(QuantumTransitionModel(Causal_Models[pick_N]))
        _p_true = sim_true.get_output_distribution(propagate_outputs=True, start_state=0)
        true_theory = np.array([1 - sum(_p_true) if b == "coinc_ch7"
                                 else (_p_true[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p_true) else 0.0)
                                 for b in bins])

        # exact model (uniform-renewal theory) -- the red dashed bars
        from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal_tm2
        _probs = _get_uniform_renewal_tm2(pick_N - 1).probs
        _exit, _a_prod = [], 1.0
        for _p_survive in _probs:
            _exit.append((1 - _p_survive) * _a_prod)
            _a_prod *= _p_survive
        _exit.append(_a_prod)
        p_exact = np.array([0.0 if b == "coinc_ch7"
                             else (_exit[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_exit) else 0.0)
                             for b in bins])

        _keep = [b == "coinc_ch7" or loop_pass_k(b, bins) <= pick_N for b in bins]
        bins_trunc = [b for b, _m in zip(bins, _keep) if _m]
        frac, yerr = frac[bins_trunc], yerr[bins_trunc]
        p_theory, p_exact, true_theory = p_theory[_keep], p_exact[_keep], true_theory[_keep]
        theory_residual = float(np.abs(frac.to_numpy() - p_theory).sum())

        return {
            "stem": stem, "T": T, "T_err": T_err,
            "frac": frac, "yerr": yerr, "bins": bins_trunc,
            "point_kl": point_kl, "mc_mean": mc_mean, "mc_std": mc_std,
            "p_theory": p_theory, "p_exact": p_exact, "true_theory": true_theory,
            "theory_residual": theory_residual,
        }

    theory_match_tmeasured = {_n: _theory_match_stats_tmeasured(_stem, _n)
                               for _n, (_stem, _score) in _best_tmeasured.items() if _stem}
    for _n, _r in theory_match_tmeasured.items():
        print(f"N{_n} closest-to-theory (T_measured): {_r['stem']}  T={_r['T']:.3f}+/-{_r['T_err']:.3f}  "
              f"residual={_r['theory_residual']:.4f}  KL point={_r['point_kl']:.6f}  "
              f"MC={_r['mc_mean']:.6f}+/-{_r['mc_std']:.6f}")

    _frac_by_n = {_n: _r["frac"] for _n, _r in theory_match_tmeasured.items()}
    _yerr_by_n = {_n: _r["yerr"] for _n, _r in theory_match_tmeasured.items()}
    _p_theory_by_n = {_n: _r["p_theory"] for _n, _r in theory_match_tmeasured.items()}
    _p_exact_by_n = {_n: _r["p_exact"] for _n, _r in theory_match_tmeasured.items()}
    _true_theory_by_n = {_n: _r["true_theory"] for _n, _r in theory_match_tmeasured.items()}
    _titles = {_n: f"N={_n}  T_measured={_r['T']:.3f}±{_r['T_err']:.3f}\nresidual={_r['theory_residual']:.3f}  KL={_r['mc_mean']:.4f}±{_r['mc_std']:.4f}"
               for _n, _r in theory_match_tmeasured.items()}
    plot_state_grid(_frac_by_n, _yerr_by_n, _p_theory_by_n,
                     f"Closest-to-theory-U output distributions, T_measured (last {_SEARCH_WINDOW_DAYS_TM} days)",
                     p_exact=_p_exact_by_n, true_theory=_true_theory_by_n, titles=_titles)
    return (theory_match_tmeasured,)


@app.cell(hide_code=True)
def _(
    Causal_Models,
    DATA_PATH,
    QuantumTransitionModel,
    Simulator,
    background_rate_by_channel,
    block_rate_mean_std,
    drop_count_outliers,
    efficiency_for_channel,
    kl_rate_from_output_distribution,
    kl_rate_mc_error,
    loop_pass_k,
    np,
    pd,
    plot_state_grid,
):
    # Same idea as the T_measured cell above, but the loss correction is
    # T_fit: per-file, fit T by minimizing the squared residual between
    # the corrected s0 frac and theory_U directly (same recipe as this
    # notebook's own T_fit cell earlier, minimize_scalar over bounds
    # (0.05, 1)) rather than reading a measured/calibrated transmission.
    # Because T_fit is already optimized file-by-file to match theory_U
    # as closely as that file's shape allows, its leftover residual tends
    # to run much smaller than T_measured's -- this is expected, not a
    # sign T_fit picks are somehow "better" data; T_fit can paper over a
    # genuinely different per-loop loss with a shape-only fit that
    # T_measured (a real, independent transmission measurement) can't.
    from scipy.optimize import minimize_scalar as _minimize_scalar_tfit

    _data_dir = DATA_PATH.parent
    import datetime as _dt_tfit
    import re as _re_tfit

    def _fit_T_tfit(rates_s0, bg, eff, bins, pick_n, p_theory0):
        _k = np.array([pick_n if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins])
        def _resid(T):
            _corr = ((rates_s0 - bg) / eff) / T ** _k
            return np.sum((p_theory0 - _corr / _corr.clip(lower=0).sum()) ** 2)
        return _minimize_scalar_tfit(_resid, bounds=(0.05, 1), method="bounded").x

    def _quick_theory_residual_tfit(stem, pick_n):
        _json_path = _data_dir / f"{stem}.json"
        _csv_path = _data_dir / f"{stem}.csv"
        if not _json_path.exists() or not _csv_path.exists():
            return None
        _cols = pd.read_csv(_csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        _bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            _bins = _bins + ["coinc_ch7"]
        _data = pd.read_csv(_csv_path)
        _data = _data.dropna(subset=_bins + ["herald", "int_time", "input_state"])
        _data = _data[_data["herald"] > 0]
        _s0 = _data[_data["input_state"] == 0]
        if "basis" in _s0.columns:
            _s0 = _s0[_s0["basis"].isna()]
        if len(_s0) < 10:
            return None
        _s0 = drop_count_outliers(_s0, _bins)
        _rates = _s0[_bins].sum() / _s0["int_time"].sum()
        _bg_by_ch, _ = background_rate_by_channel(_json_path)
        _bg = pd.Series({b: _bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in _bins})
        _eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in _bins})

        _sim = Simulator(QuantumTransitionModel(Causal_Models[pick_n]))
        _p = _sim.get_output_distribution(propagate_outputs=True, start_state=0)
        _p_theory = np.array([1 - sum(_p) if b == "coinc_ch7"
                               else (_p[loop_pass_k(b, _bins) - 1] if loop_pass_k(b, _bins) <= len(_p) else 0.0)
                               for b in _bins])
        _T = _fit_T_tfit(_rates, _bg, _eff, _bins, pick_n, _p_theory)
        _k = np.array([pick_n if b == "coinc_ch7" else loop_pass_k(b, _bins) for b in _bins])
        _corr = ((_rates - _bg) / _eff) / _T ** _k
        _total = _corr.clip(lower=0).sum()
        if _total <= 0:
            return None
        _frac = (_corr.clip(lower=0) / _total).to_numpy()
        _keep = [b == "coinc_ch7" or loop_pass_k(b, _bins) <= pick_n for b in _bins]
        return float(np.abs(_frac[_keep] - _p_theory[_keep]).sum())

    _SEARCH_WINDOW_DAYS_TF = 21
    _cutoff_tfit = (_dt_tfit.datetime.now() - _dt_tfit.timedelta(days=_SEARCH_WINDOW_DAYS_TF)).strftime("%Y%m%d_%H%M%S")

    _best_tfit = {_n: (None, float("inf")) for _n in (3, 4, 5, 6)}
    for _fp in sorted(_data_dir.glob("*measurement_N*.csv")):
        _ts_match = _re_tfit.search(r"\d{8}_\d{6}", _fp.name)
        _n_match = _re_tfit.search(r"_N(\d+)_", _fp.name)
        if not _ts_match or not _n_match or _ts_match.group() < _cutoff_tfit:
            continue
        _n = int(_n_match.group(1))
        if _n not in _best_tfit:
            continue
        _residual = _quick_theory_residual_tfit(_fp.stem, _n)
        if _residual is not None and _residual < _best_tfit[_n][1]:
            _best_tfit[_n] = (_fp.stem, _residual)

    def _theory_match_stats_tfit(stem, pick_N):
        json_path = _data_dir / f"{stem}.json"
        csv_path = _data_dir / f"{stem}.csv"

        _cols = pd.read_csv(csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            bins = bins + ["coinc_ch7"]
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=bins + ["herald", "int_time", "input_state"])
        data = data[data["herald"] > 0]
        s0 = data[data["input_state"] == 0]
        if "basis" in s0.columns:
            s0 = s0[s0["basis"].isna()]
        s0 = drop_count_outliers(s0, bins)

        rates, stds = block_rate_mean_std(s0, bins, block_size=30)
        bg_by_ch, _ = background_rate_by_channel(json_path)
        bg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in bins})
        eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in bins})

        # theory bars (and the fit target for T_fit below): getUtot(angles_NTU),
        # per 2026-11-24 request
        from stochprocsim.libraries.optics_lib import getUtot as _getUtot_tf
        from stochprocsim.models.causal_models import CausalModel as _CausalModel_tf
        _U_getutot = _getUtot_tf(Causal_Models[pick_N].angles_NTU)
        _cm_getutot = _CausalModel_tf(U=_U_getutot, states=Causal_Models[pick_N].states,
                                        angles_NTU=Causal_Models[pick_N].angles_NTU,
                                        angles_GU=Causal_Models[pick_N].angles_GU, name="getutot")
        sim = Simulator(QuantumTransitionModel(_cm_getutot))
        _p = sim.get_output_distribution(propagate_outputs=True, start_state=0)
        p_theory = np.array([1 - sum(_p) if b == "coinc_ch7"
                              else (_p[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p) else 0.0)
                              for b in bins])
        T = _fit_T_tfit(rates, bg, eff, bins, pick_N, p_theory)
        T_err = 0.0  # no uncertainty on a per-file shape fit, unlike a direct measurement

        # true_theory bars: the frozen U_theo matrix's own prediction
        sim_true = Simulator(QuantumTransitionModel(Causal_Models[pick_N]))
        _p_true = sim_true.get_output_distribution(propagate_outputs=True, start_state=0)
        true_theory = np.array([1 - sum(_p_true) if b == "coinc_ch7"
                                 else (_p_true[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p_true) else 0.0)
                                 for b in bins])

        # exact model (uniform-renewal theory) -- the red dashed bars
        from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal_tf2
        _probs = _get_uniform_renewal_tf2(pick_N - 1).probs
        _exit, _a_prod = [], 1.0
        for _p_survive in _probs:
            _exit.append((1 - _p_survive) * _a_prod)
            _a_prod *= _p_survive
        _exit.append(_a_prod)
        p_exact = np.array([0.0 if b == "coinc_ch7"
                             else (_exit[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_exit) else 0.0)
                             for b in bins])

        k = np.array([pick_N if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins])
        transmission = T ** k
        corr = ((rates - bg) / eff) / transmission
        norm = corr.clip(lower=0).sum()
        frac = corr.clip(lower=0) / norm
        yerr = ((stds / eff) / transmission) / norm
        point_kl = kl_rate_from_output_distribution(pick_N, frac.to_numpy()[:pick_N])
        mc_mean, mc_std, mc_converged = kl_rate_mc_error(pick_N, rates, stds, T, T_err, bg, eff, bins,
                                                           n_samples=300, rng=np.random.default_rng(0),
                                                           check_convergence=True)
        if not mc_converged:
            print(f"  WARNING: N{pick_N} (T_fit) MC error estimate may not have converged at 300 "
                  f"samples (two independent halves disagreed by >20%) -- treat mc_std as approximate")

        _keep = [b == "coinc_ch7" or loop_pass_k(b, bins) <= pick_N for b in bins]
        bins_trunc = [b for b, _m in zip(bins, _keep) if _m]
        frac, yerr = frac[bins_trunc], yerr[bins_trunc]
        p_theory, p_exact, true_theory = p_theory[_keep], p_exact[_keep], true_theory[_keep]
        theory_residual = float(np.abs(frac.to_numpy() - p_theory).sum())

        return {
            "stem": stem, "T": T, "T_err": T_err,
            "frac": frac, "yerr": yerr, "bins": bins_trunc,
            "point_kl": point_kl, "mc_mean": mc_mean, "mc_std": mc_std,
            "p_theory": p_theory, "p_exact": p_exact, "true_theory": true_theory,
            "theory_residual": theory_residual,
        }

    theory_match_tfit = {_n: _theory_match_stats_tfit(_stem, _n)
                          for _n, (_stem, _score) in _best_tfit.items() if _stem}
    for _n, _r in theory_match_tfit.items():
        print(f"N{_n} closest-to-theory (T_fit): {_r['stem']}  T_fit={_r['T']:.3f}  "
              f"residual={_r['theory_residual']:.4f}  KL point={_r['point_kl']:.6f}  "
              f"MC={_r['mc_mean']:.6f}+/-{_r['mc_std']:.6f}")

    _frac_by_n = {_n: _r["frac"] for _n, _r in theory_match_tfit.items()}
    _yerr_by_n = {_n: _r["yerr"] for _n, _r in theory_match_tfit.items()}
    _p_theory_by_n = {_n: _r["p_theory"] for _n, _r in theory_match_tfit.items()}
    _p_exact_by_n = {_n: _r["p_exact"] for _n, _r in theory_match_tfit.items()}
    _true_theory_by_n = {_n: _r["true_theory"] for _n, _r in theory_match_tfit.items()}
    _titles = {_n: f"N={_n}  T_fit={_r['T']:.3f}\nresidual={_r['theory_residual']:.3f}  KL={_r['mc_mean']:.4f}±{_r['mc_std']:.4f}"
               for _n, _r in theory_match_tfit.items()}
    plot_state_grid(_frac_by_n, _yerr_by_n, _p_theory_by_n,
                     f"Closest-to-theory-U output distributions, T_fit (last {_SEARCH_WINDOW_DAYS_TF} days)",
                     p_exact=_p_exact_by_n, true_theory=_true_theory_by_n, titles=_titles)
    return (theory_match_tfit,)


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
    theory_match_tfit,
    theory_match_tmeasured,
):
    # same KL-rate-vs-N plot as the "best picks" version earlier in this
    # notebook, but with TWO experimental series -- the T_measured and
    # T_fit closest-to-theory_U picks above -- plotted as separate lines
    # instead of one. Theory curves (quantum target, classical bound) are
    # identical either way, only the plotted measured points differ.
    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal_tm
    from stochprocsim.stochprocq.measure import eval_diverge as _eval_diverge_tm
    from stochprocsim.utils import generate_quantum_model as _generate_quantum_model_tm
    from stochprocsim.libraries.optics_lib import getUtot as _getUtot_tm2
    from stochprocsim.models.causal_models import CausalModel as _CausalModel_tm2

    def _sim_quantum_tm(n):
        _cs = Causal_Models[n]
        _cs.set_U(_cs.U_theo)
        _q = _generate_quantum_model_tm(
            np.array(Simulator(QuantumTransitionModel(_cs)).get_output_distribution()))
        _m = _get_uniform_renewal_tm(n - 1)
        _p_dist = _m.steady_state @ _m.gen_dists(n + 1)
        _q_dist = _q.steady_state @ _q.gen_dists(n + 1)
        return _eval_diverge_tm(_p_dist, _q_dist, his_steps=n - 1)

    def _sim_getutot_tm(n):
        # Same idea as _sim_quantum_tm above (theory vs. the exact/
        # uniform-renewal model, no measured data involved), but for
        # getUtot(angles_NTU) instead of the frozen U_theo matrix -- a
        # local CausalModel, not a mutation of the shared Causal_Models[n]
        # (unlike _sim_quantum_tm's set_U, which is safe since U_theo is
        # that model's own default).
        _cs = Causal_Models[n]
        _U_getutot = _getUtot_tm2(_cs.angles_NTU)
        _cm_getutot = _CausalModel_tm2(U=_U_getutot, states=_cs.states,
                                        angles_NTU=_cs.angles_NTU, angles_GU=_cs.angles_GU,
                                        name="getutot")
        _q = _generate_quantum_model_tm(
            np.array(Simulator(QuantumTransitionModel(_cm_getutot)).get_output_distribution()))
        _m = _get_uniform_renewal_tm(n - 1)
        _p_dist = _m.steady_state @ _m.gen_dists(n + 1)
        _q_dist = _q.steady_state @ _q.gen_dists(n + 1)
        return _eval_diverge_tm(_p_dist, _q_dist, his_steps=n - 1)

    def _get_bd_tm(N):
        _bd = 0
        for _n in range(1, 9):
            new_bd = _get_uniform_renewal_tm(_n - 1).classical_bd(N, target_dim=2)
            _bd = max(_bd, new_bd)
        return _bd

    _NMIN, _NMAX, _C_BUFF = 3, 6, 1
    _x_quant = list(range(_NMIN, _NMAX + 1))
    _x_class = list(range(_NMIN, _NMAX + _C_BUFF))
    _y_quantum = [_sim_quantum_tm(_n) for _n in _x_quant]
    _y_getutot_theory = [_sim_getutot_tm(_n) for _n in _x_quant]

    _ls = [3, 4, 5, 6]
    _y_classical = [_get_bd_tm(_n) for _n in _ls]
    _qtheo_yerr = [0.002054339006460678, 0.0014301897956533847, 0.0016754397869550568, 0.0015075218255867487]

    _x_exp_tmeasured = sorted(theory_match_tmeasured)
    rate_exp_tmeasured = [theory_match_tmeasured[_n]["mc_mean"] for _n in _x_exp_tmeasured]
    rate_exp_tmeasured_err = [theory_match_tmeasured[_n]["mc_std"] for _n in _x_exp_tmeasured]

    _x_exp_tfit = sorted(theory_match_tfit)
    rate_exp_tfit = [theory_match_tfit[_n]["mc_mean"] for _n in _x_exp_tfit]
    rate_exp_tfit_err = [theory_match_tfit[_n]["mc_std"] for _n in _x_exp_tfit]

    _c_classical, _c_quantum = "#3B5BA5", "#4C9A2A"
    _c_exp_tmeasured, _c_exp_tfit = "#C9700E", "#1F9E89"
    _c_getutot_theory = "#8B3A62"
    _x_c, _y_c = np.array(_x_class), np.array(_y_classical)
    _y_q, _e_q = np.array(_y_quantum), np.array(_qtheo_yerr)

    with sns.axes_style("whitegrid"):
        _fig, _ax = plt.subplots()
        _ax.plot(_x_c, _y_c, "-D", color=_c_classical, linewidth=1.5, label="Classical bound")
        _ax.errorbar(_x_quant, _y_q, yerr=_e_q, fmt="-^", capsize=3, color=_c_quantum,
                      linewidth=1.5, label="Quantum (Target)")
        _ax.fill_between(_x_quant, _y_q - _e_q, _y_q + _e_q, color=_c_quantum, alpha=0.2)
        _ax.errorbar(_x_exp_tmeasured, rate_exp_tmeasured, yerr=rate_exp_tmeasured_err, fmt="s--", capsize=3,
                      color=_c_exp_tmeasured, linewidth=1.5, label="Exp (closest to theory_U, T_measured)")
        _ax.errorbar(_x_exp_tfit, rate_exp_tfit, yerr=rate_exp_tfit_err, fmt="^--", capsize=3,
                      color=_c_exp_tfit, linewidth=1.5, label="Exp (closest to theory_U, T_fit)")
        _ax.plot(_x_quant, _y_getutot_theory, "-o", color=_c_getutot_theory,
                 linewidth=1.5, label="getUtot theory vs exact model")
        _ax.fill_between(_x_c, _y_c, _ax.get_ylim()[1], alpha=0.3, color="none",
                          edgecolor=_c_classical, hatch="///")
        _ax.set_xticks(_x_c)
        _ax.set_ylim(0, 0.4)
        _ax.set_xlabel("Process length N")
        _ax.set_ylabel(r"$R_{KL}$")
        _ax.set_title("KL Divergence Rate for Process Depth N (picks ranked by theory_U match)")
        _ax.legend(frameon=True, fontsize=8)
    mo.vstack([_fig, fig_download_button(_fig, "kl_divergence_rate_vs_N_theory_match.svg")])
    return


@app.cell(hide_code=True)
def _(fig_download_button, mo, plt, sns):
    # classical bound vs. N, extended beyond the N=3..6 range used in the
    # experimental-comparison plot above -- same get_bd(N) recipe as that
    # cell: for each N, take the max classical_bd over uniform-renewal
    # models of length 1..8.
    #
    # NOTE: capped at N=15, not N=100. classical_bd(N, ...) calls
    # gen_dists(N, ...) which allocates a DENSE np.zeros((dim,
    # 2**N)) array regardless of the sparse flag -- so cost grows as
    # 2**N. Measured on this machine: N=14 takes ~1.4s per get_bd() call,
    # N=18 ~20s, and the loop below (N=3..16) already totals ~11.5s
    # cumulative; N=100 would need on the order of 2**100 array entries,
    # i.e. not completable. N=15 keeps the whole cell at ~6s.
    from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal

    def _get_bd(N):
        """Get the classical bound for a given N."""
        _bd = 0
        for _n in range(1, 9):
            new_bd = _get_uniform_renewal(_n - 1).classical_bd(N, target_dim=2)
            _bd = max(_bd, new_bd)
        return _bd

    _x_bd = list(range(3, 15))
    _y_bd = [_get_bd(_n) for _n in _x_bd]

    with sns.axes_style("whitegrid"):
        _fig_bd, _ax_bd = plt.subplots()
        _ax_bd.plot(_x_bd, _y_bd, "-", color="#3B5BA5", linewidth=1.5)
        _ax_bd.set_xlabel("Process length N")
        _ax_bd.set_ylabel(r"$R_{KL}$")
        _ax_bd.set_title("Classical Bound vs. N")
    mo.vstack([_fig_bd, fig_download_button(_fig_bd, "classical_bound_vs_N.svg")])
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


@app.cell(hide_code=True)
def _(
    Causal_Models,
    DATA_PATH,
    QuantumTransitionModel,
    Simulator,
    background_rate_by_channel,
    block_rate_mean_std,
    drop_count_outliers,
    efficiency_for_channel,
    get_best_picks,
    kl_rate_from_output_distribution,
    kl_rate_mc_error,
    loop_pass_k,
    loss_calibration_for,
    np,
    pd,
    plot_state_grid,
):
    # Trial: same picks/pipeline as the "best rated" cell earlier in this
    # notebook, but instead of truncating away loop passes beyond this
    # N (pure noise/crosstalk per that cell's own comment) and plotting
    # coinc_ch7 as a bare dump-channel bin, fold passes N..6 AND the dump
    # channel into one "remainder" bin -- the loss-corrected sum of
    # every channel from the Nth loop pass through ch7. This matches
    # theory_U's own dump definition (1 - sum of passes 1..N-1), which
    # already implicitly bundles "pass N onward" into one number rather
    # than predicting each tail pass individually -- this makes the
    # measured side bundle the same way instead of discarding those
    # counts outright (see 2026-11-24 conversation: "why only N+1
    # channels, there should always be the full number").
    BEST_PICKS_REMAINDER = get_best_picks()
    _data_dir = DATA_PATH.parent

    def _best_pick_stats_remainder(stem, pick_N, loss_key, err_key):
        json_path = _data_dir / f"{stem}.json"
        csv_path = _data_dir / f"{stem}.csv"
        losses, cal_note = loss_calibration_for(json_path)
        T, T_err = losses[loss_key], losses.get(err_key) or 0.0

        _cols = pd.read_csv(csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            bins = bins + ["coinc_ch7"]
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=bins + ["herald", "int_time", "input_state"])
        data = data[data["herald"] > 0]
        s0 = data[data["input_state"] == 0]
        if "basis" in s0.columns:
            s0 = s0[s0["basis"].isna()]
        s0 = drop_count_outliers(s0, bins)

        rates, stds = block_rate_mean_std(s0, bins, block_size=30)
        bg_by_ch, bg_note = background_rate_by_channel(json_path)
        bg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in bins})
        eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in bins})

        k = np.array([pick_N if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins])
        transmission = T ** k
        corr = ((rates - bg) / eff) / transmission
        norm = corr.clip(lower=0).sum()
        frac_full = corr.clip(lower=0) / norm
        yerr_full = ((stds / eff) / transmission) / norm
        point_kl = kl_rate_from_output_distribution(pick_N, frac_full.to_numpy()[:pick_N])
        mc_mean, mc_std, mc_converged = kl_rate_mc_error(pick_N, rates, stds, T, T_err, bg, eff, bins,
                                                           n_samples=300, rng=np.random.default_rng(0),
                                                           check_convergence=True)
        if not mc_converged:
            print(f"  WARNING: N{pick_N} MC error estimate may not have converged at 300 samples "
                  f"(two independent halves disagreed by >20%) -- treat mc_std as approximate")

        # theory bars: getUtot(angles_NTU), per 2026-11-24 request
        from stochprocsim.libraries.optics_lib import getUtot as _getUtot_rem
        from stochprocsim.models.causal_models import CausalModel as _CausalModel_rem
        _U_getutot = _getUtot_rem(Causal_Models[pick_N].angles_NTU)
        _cm_getutot = _CausalModel_rem(U=_U_getutot, states=Causal_Models[pick_N].states,
                                         angles_NTU=Causal_Models[pick_N].angles_NTU,
                                         angles_GU=Causal_Models[pick_N].angles_GU, name="getutot")
        sim = Simulator(QuantumTransitionModel(_cm_getutot))
        _p = sim.get_output_distribution(propagate_outputs=True, start_state=0)
        p_theory_full = np.array([1 - sum(_p) if b == "coinc_ch7"
                              else (_p[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p) else 0.0)
                              for b in bins])

        # true_theory bars: the frozen U_theo matrix's own prediction
        sim_true = Simulator(QuantumTransitionModel(Causal_Models[pick_N]))
        _p_true = sim_true.get_output_distribution(propagate_outputs=True, start_state=0)
        true_theory_full = np.array([1 - sum(_p_true) if b == "coinc_ch7"
                                 else (_p_true[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_p_true) else 0.0)
                                 for b in bins])

        from stochprocsim.stochprocq import get_uniform_renewal as _get_uniform_renewal_rem
        _probs = _get_uniform_renewal_rem(pick_N - 1).probs
        _exit, _a_prod = [], 1.0
        for _p_survive in _probs:
            _exit.append((1 - _p_survive) * _a_prod)
            _a_prod *= _p_survive
        _exit.append(_a_prod)
        p_exact_full = np.array([0.0 if b == "coinc_ch7"
                             else (_exit[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(_exit) else 0.0)
                             for b in bins])

        # keep passes 1..N-1 individually; fold pass N, N+1..6 and the
        # ch7 dump into one "remainder" bin for frac/p_theory/p_exact
        # alike (tail errors combined in quadrature, independent shot
        # noise per channel)
        _head = [b for b in bins if b != "coinc_ch7" and loop_pass_k(b, bins) < pick_N]
        _tail = [b for b in bins if b == "coinc_ch7" or loop_pass_k(b, bins) >= pick_N]
        _head_idx = [bins.index(b) for b in _head]
        _tail_idx = [bins.index(b) for b in _tail]

        frac = frac_full[_head].copy()
        frac["remainder"] = frac_full[_tail].sum()
        yerr = yerr_full[_head].copy()
        yerr["remainder"] = float(np.sqrt((yerr_full[_tail] ** 2).sum()))
        p_theory = np.append(p_theory_full[_head_idx], p_theory_full[_tail_idx].sum())
        p_exact = np.append(p_exact_full[_head_idx], p_exact_full[_tail_idx].sum())
        true_theory = np.append(true_theory_full[_head_idx], true_theory_full[_tail_idx].sum())
        bins_trunc = _head + ["remainder"]

        theory_residual = float(np.abs(frac.to_numpy() - p_theory).sum())

        return {
            "stem": stem, "T": T, "T_err": T_err, "cal_note": cal_note,
            "frac": frac, "yerr": yerr, "bins": bins_trunc,
            "point_kl": point_kl, "mc_mean": mc_mean, "mc_std": mc_std,
            "p_theory": p_theory, "p_exact": p_exact, "true_theory": true_theory,
            "theory_residual": theory_residual,
        }

    best_results_remainder = {_n: _best_pick_stats_remainder(_stem, _n, _lk, _ek)
                               for _n, (_stem, _lk, _ek) in BEST_PICKS_REMAINDER.items()}
    for _n, _r in best_results_remainder.items():
        print(f"N{_n} best (remainder trial): {_r['stem']}  T={_r['T']:.4f}+/-{_r['T_err']:.4f} ({_r['cal_note']})  "
              f"point={_r['point_kl']:.6f}  MC={_r['mc_mean']:.6f}+/-{_r['mc_std']:.6f}")

    _frac_by_n = {_n: _r["frac"] for _n, _r in best_results_remainder.items()}
    _yerr_by_n = {_n: _r["yerr"] for _n, _r in best_results_remainder.items()}
    _p_theory_by_n = {_n: _r["p_theory"] for _n, _r in best_results_remainder.items()}
    _p_exact_by_n = {_n: _r["p_exact"] for _n, _r in best_results_remainder.items()}
    _true_theory_by_n = {_n: _r["true_theory"] for _n, _r in best_results_remainder.items()}
    _titles = {_n: f"N={_n}  T={_r['T']:.3f}±{_r['T_err']:.3f}\nKL={_r['mc_mean']:.4f}±{_r['mc_std']:.4f}"
               for _n, _r in best_results_remainder.items()}
    plot_state_grid(_frac_by_n, _yerr_by_n, _p_theory_by_n,
                     "Current best-rated (last week) output distributions vs theory (ch7 -> remainder trial)",
                     p_exact=_p_exact_by_n, true_theory=_true_theory_by_n, titles=_titles)
    return (best_results_remainder,)


@app.cell(hide_code=True)
def _(
    DATA_PATH,
    background_rate_by_channel,
    block_rate_mean_std,
    drop_count_outliers,
    efficiency_for_channel,
    get_best_picks,
    kl_rate_from_output_distribution,
    loop_pass_k,
    loss_calibration_for,
    np,
    pd,
):
    # Does folding the remainder (loop-pass N..6 plus the ch7 dump) into
    # kl_rate_from_output_distribution's own final bin change the KL
    # number, rather than just the plot? The usual call sites truncate
    # with frac.to_numpy()[:N], which keeps only the raw channel-N count
    # and silently throws away any counts that leaked into channels
    # N+1..ch7 -- even though the function's own docstring says that
    # last bin is meant to be a catch-all "everything else", matching
    # theory_U's own dump definition (1 - sum of passes 1..N-1).
    _data_dir = DATA_PATH.parent

    def _kl_naive_vs_folded(stem, pick_N, loss_key, err_key):
        json_path = _data_dir / f"{stem}.json"
        csv_path = _data_dir / f"{stem}.csv"
        losses, _ = loss_calibration_for(json_path)
        T = losses[loss_key]

        _cols = pd.read_csv(csv_path, nrows=0).columns
        _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
        bins = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
        if "coinc_ch7" in _cols:
            bins = bins + ["coinc_ch7"]
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=bins + ["herald", "int_time", "input_state"])
        data = data[data["herald"] > 0]
        s0 = data[data["input_state"] == 0]
        if "basis" in s0.columns:
            s0 = s0[s0["basis"].isna()]
        s0 = drop_count_outliers(s0, bins)

        rates, _ = block_rate_mean_std(s0, bins, block_size=30)
        bg_by_ch, _ = background_rate_by_channel(json_path)
        bg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in bins})
        eff = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in bins})

        k = np.array([pick_N if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins])
        corr = ((rates - bg) / eff) / (T ** k)
        frac_full = corr.clip(lower=0) / corr.clip(lower=0).sum()

        naive = frac_full.to_numpy()[:pick_N]

        _head = [b for b in bins if b != "coinc_ch7" and loop_pass_k(b, bins) < pick_N]
        _tail = [b for b in bins if b == "coinc_ch7" or loop_pass_k(b, bins) >= pick_N]
        folded = np.append(frac_full[_head].to_numpy(), frac_full[_tail].sum())

        return (kl_rate_from_output_distribution(pick_N, naive),
                kl_rate_from_output_distribution(pick_N, folded), naive, folded)

    for _n, (_stem, _lk, _ek) in get_best_picks().items():
        _kl_naive, _kl_folded, _naive, _folded = _kl_naive_vs_folded(_stem, _n, _lk, _ek)
        _delta = _kl_folded - _kl_naive
        print(f"N{_n} {_stem}: naive={_kl_naive:.6f}  folded={_kl_folded:.6f}  "
              f"delta={_delta:+.6f}  ({'better' if _delta < 0 else 'worse' if _delta > 0 else 'same'})")
        print(f"  naive  (tail dropped): {np.round(_naive, 4)}")
        print(f"  folded (tail kept):    {np.round(_folded, 4)}")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    background_rate_by_channel,
    drop_count_outliers,
    efficiency_for_channel,
    get_best_picks,
    loop_pass_k,
    loss_calibration_for,
    np,
    pd,
    plot_state_grid,
):
    from stochprocsim.stochprocq.measure import eval_diverge as _eval_diverge_pg
    from stochprocsim.utils import generate_quantum_model as _generate_quantum_model_pg

    # Playground: the actual getUtot() unitary math (stochprocsim's
    # "theory U" builder, src/stochprocsim/libraries/optics_lib.py),
    # inlined and exposed as plain matrix multiplications instead of a
    # single opaque function call -- edit ANGLES below (or the matrix
    # product itself, e.g. drop/reorder/add a component) and rerun to
    # see how PLAYGROUND_KL moves. Pulls whatever file is currently
    # BEST_PICKS[3] (get_best_picks()), so it always plays with the
    # current best N=3 data, not a hardcoded filename.
    from stochprocsim.libraries.optics_lib import (
        HWP_p1, HWP_p2, QWP_p1, QWP_p2, PBS, PBS_dag, M4,
    )
    from stochprocsim.models.causal_models import CausalModel

    PICK_N = 3

    # ---- EDITABLE: waveplate/mirror angles (degrees) + global phase ----
    # starting point = this N's own calibrated angles (U_{N}_angles_NTU in
    # waveplate_angles.py) -- change any value, or the matrix product
    # below, and rerun this cell.
    ANGLES = dict(Causal_Models[PICK_N].angles_NTU)  # copy, not the shared original
    PIPI = ANGLES["pipi"]

    # ---- the unitary itself, same product as getUtot(), written out ----
    U_playground = (
        HWP_p2(ANGLES["θhf2"]) @ QWP_p2(ANGLES["θqf2"])
        @ HWP_p1(ANGLES["θhf1"]) @ QWP_p1(ANGLES["θqf1"])
        @ PBS_dag
        @ M4(ANGLES["Φm3"], ANGLES["Φm1"])
        @ M4(ANGLES["Φm2"], ANGLES["Φm2"])
        @ M4(ANGLES["Φm1"], ANGLES["Φm3"])
        @ HWP_p2(ANGLES["θh2"]) @ QWP_p2(ANGLES["θq2"])
        @ HWP_p1(ANGLES["θh1"]) @ QWP_p1(ANGLES["θq1"])
        @ PBS
        @ HWP_p2(ANGLES["θhin2"]) @ QWP_p2(ANGLES["θqin2"])
    )
    U_playground = np.exp(-1j * PIPI) * U_playground

    # local CausalModel so this never mutates the shared Causal_Models[N]
    # used everywhere else in the notebook
    cm_playground = CausalModel(
        U=U_playground, states=Causal_Models[PICK_N].states,
        angles_NTU=Causal_Models[PICK_N].angles_NTU,
        angles_GU=Causal_Models[PICK_N].angles_GU, name="playground",
    )

    # ---- pull counts: whatever's currently BEST_PICKS[PICK_N] ----
    _stem, _loss_key, _err_key = get_best_picks()[PICK_N]
    from pathlib import Path as _Path_playground
    _data_dir = _Path_playground("/home/brendan/Documents/PhD/Programs/stochprocsim/data")
    _json_path = _data_dir / f"{_stem}.json"
    _csv_path = _data_dir / f"{_stem}.csv"

    _losses, _ = loss_calibration_for(_json_path)
    T = _losses[_loss_key]

    _cols = pd.read_csv(_csv_path, nrows=0).columns
    _coinc = [c for c in _cols if c.startswith("coinc_ch") and c != "coinc_ch7"]
    bins_pg = sorted(_coinc, key=lambda c: int(c.removeprefix("coinc_ch")))
    if "coinc_ch7" in _cols:
        bins_pg = bins_pg + ["coinc_ch7"]
    data_pg = pd.read_csv(_csv_path)
    data_pg = data_pg.dropna(subset=bins_pg + ["herald", "int_time", "input_state"])
    data_pg = data_pg[data_pg["herald"] > 0]
    s0_pg = data_pg[data_pg["input_state"] == 0]
    if "basis" in s0_pg.columns:
        s0_pg = s0_pg[s0_pg["basis"].isna()]
    s0_pg = drop_count_outliers(s0_pg, bins_pg)
    rates_pg = s0_pg[bins_pg].sum() / s0_pg["int_time"].sum()

    bg_by_ch, _ = background_rate_by_channel(_json_path)
    bg_pg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in bins_pg})
    eff_pg = pd.Series({b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in bins_pg})
    k_pg = np.array([PICK_N if b == "coinc_ch7" else loop_pass_k(b, bins_pg) for b in bins_pg])
    corr_pg = ((rates_pg - bg_pg) / eff_pg) / (T ** k_pg)
    frac_pg = corr_pg.clip(lower=0) / corr_pg.clip(lower=0).sum()

    # ---- theory output distribution from U_playground ----
    sim_pg = Simulator(QuantumTransitionModel(cm_playground))
    _p_pg = sim_pg.get_output_distribution(propagate_outputs=True, start_state=0)
    p_theory_pg = np.array([1 - sum(_p_pg) if b == "coinc_ch7"
                             else (_p_pg[loop_pass_k(b, bins_pg) - 1] if loop_pass_k(b, bins_pg) <= len(_p_pg) else 0.0)
                             for b in bins_pg])

    # ---- true_theory: the frozen U_theo matrix's own prediction, for
    # comparison against whatever ANGLES/matrix edits you're trying ----
    sim_true_pg = Simulator(QuantumTransitionModel(Causal_Models[PICK_N]))
    _p_true_pg = sim_true_pg.get_output_distribution(propagate_outputs=True, start_state=0)
    true_theory_pg = np.array([1 - sum(_p_true_pg) if b == "coinc_ch7"
                                else (_p_true_pg[loop_pass_k(b, bins_pg) - 1] if loop_pass_k(b, bins_pg) <= len(_p_true_pg) else 0.0)
                                for b in bins_pg])

    # KL vs theory_U itself (NOT kl_rate_from_output_distribution, which
    # compares measured data to the uniform-renewal model and is totally
    # insensitive to U -- confirmed while building this cell: it returns
    # the exact same number whether U_playground is the real U_theo or
    # has angles perturbed. This is the one that will actually move when
    # you edit ANGLES/the matrix above.
    theory_model_pg = _generate_quantum_model_pg(p_theory_pg[:PICK_N])
    exp_model_pg = _generate_quantum_model_pg(frac_pg.to_numpy()[:PICK_N])
    _p_dist_pg = theory_model_pg.steady_state @ theory_model_pg.gen_dists(PICK_N + 1)
    _q_dist_pg = exp_model_pg.steady_state @ exp_model_pg.gen_dists(PICK_N + 1)
    PLAYGROUND_KL = _eval_diverge_pg(_p_dist_pg, _q_dist_pg, his_steps=PICK_N - 1)
    print(f"playground N={PICK_N}  file={_stem}  T={T:.4f}")
    print(f"measured:     {np.round(frac_pg.to_numpy(), 4)}")
    print(f"theory_U:     {np.round(p_theory_pg, 4)}")
    print(f"true_theory:  {np.round(true_theory_pg, 4)}")
    print(f"PLAYGROUND_KL (vs theory_U) = {PLAYGROUND_KL:.6f} bits")

    plot_state_grid(
        {0: frac_pg}, {0: frac_pg * 0}, {0: p_theory_pg},
        f"Playground: N={PICK_N} ({_stem}) -- KL={PLAYGROUND_KL:.4f}",
        true_theory={0: true_theory_pg},
        titles={0: f"T={T:.3f}  KL={PLAYGROUND_KL:.4f}"},
    )
    return


if __name__ == "__main__":
    app.run()
