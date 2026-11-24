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

    from stochprocsim.utils import efficiency_for_channel
    return efficiency_for_channel, io, mo, np, pd, plt, sns


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

    refresh_button  # ponytail: re-scan trigger, value itself unused
    _DATA_DIR = Path(filepath)
    # LabVIEW trace filenames are already "YYYY-MM-DD-HHMM_Trace_..." --
    # zero-padded and ISO-ordered, so a plain name sort is already a time
    # sort (unlike the software csv's mixed old/new filename conventions)
    _files = sorted(_DATA_DIR.glob("*Trace*.txt"), key=lambda p: p.name)
    mo.stop(not _files, mo.md(f"No LabVIEW trace .txt files found in `{_DATA_DIR}`"))
    file_picker = mo.ui.dropdown(
        options={str(p.relative_to(_DATA_DIR)): p for p in _files},
        value=str(_files[-1].relative_to(_DATA_DIR)),
        label="LabVIEW trace file",
    )
    file_picker
    return (file_picker,)


@app.cell(hide_code=True)
def _(file_picker, pd):
    import re

    DATA_PATH = file_picker.value
    _m = re.search(r"[uU](\d+)", DATA_PATH.stem)
    mo_msg = f"couldn't find a unitary number (uN) in filename {DATA_PATH.name}"
    assert _m, mo_msg
    N = int(_m.group(1))  # picks Causal_Models[N]

    # LabVIEW's own coincidence columns are "Ch3+ChX" (X = loop/dump
    # channel, Ch3 = herald) -- rename straight to this project's
    # "coinc_chX" convention so every cell from here on (theory, loss
    # correction, plotting) is identical to the software-data notebook
    _cols = pd.read_csv(DATA_PATH, sep="\t", nrows=0).columns
    _coinc = [c for c in _cols if c.startswith("Ch3+Ch") and c != "Ch3+Ch7"]
    BINS = sorted((c.replace("Ch3+Ch", "coinc_ch") for c in _coinc),
                  key=lambda c: int(c.removeprefix("coinc_ch")))
    if "Ch3+Ch7" in _cols:
        BINS = BINS + ["coinc_ch7"]
    return BINS, DATA_PATH, N


@app.cell(hide_code=True)
def _(BINS, DATA_PATH, pd):
    from stochprocsim.utils import rates_by_input_state

    data = pd.read_csv(DATA_PATH, sep="\t")
    _rename = {f"Ch3+{b.removeprefix('coinc_').capitalize()}": b for b in BINS}
    data = data.rename(columns=_rename)
    data = data.dropna(subset=BINS + ["Ch3"])
    data = data[data["Ch3"] > 0]  # blocked-laser / herald-off rows

    # LabVIEW's Ch3+ChX columns are already rates in Hz (not raw counts),
    # so int_time=1 makes rates_by_input_state's count/int_time division a
    # no-op -- it still gives us the mean/SEM-across-rows machinery (and
    # this file is one single input-state run) for free
    data["int_time"] = 1.0
    data["input_state"] = 0

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
def _(filepath, mo):
    from pathlib import Path as _Path_loss_cal

    _cal_files = sorted(_Path_loss_cal(filepath).glob("*_loss_calibration.json"), key=lambda p: p.name)
    mo.stop(not _cal_files, mo.md(f"No *_loss_calibration.json found in `{filepath}`"))
    loss_cal_picker = mo.ui.dropdown(
        options={p.name: p for p in _cal_files},
        value=_cal_files[-1].name,
        label="Loss calibration file",
    )
    loss_cal_picker
    return (loss_cal_picker,)


@app.cell(hide_code=True)
def _(filepath, mo):
    from pathlib import Path as _Path_noise_cal

    _noise_files = sorted(_Path_noise_cal(filepath).glob("*_noise_calibration.json"), key=lambda p: p.name)
    mo.stop(not _noise_files, mo.md(f"No *_noise_calibration.json found in `{filepath}`"))
    noise_cal_picker = mo.ui.dropdown(
        options={p.name: p for p in _noise_files},
        value=_noise_files[-1].name,
        label="Noise calibration file",
    )
    noise_cal_picker
    return (noise_cal_picker,)


@app.cell(hide_code=True)
def _(BINS, efficiency_for_channel, noise_cal_picker, pd):
    # background/noise per channel -- LabVIEW traces carry no embedded
    # sidecar (unlike measurement.py's own .json), so this clone always
    # reads the picked *_noise_calibration.json directly rather than
    # resolving one via background_rate_by_channel's embedded/fallback logic
    import json as _json_bg

    _bg_by_ch = _json_bg.loads(noise_cal_picker.value.read_text()).get("background_rate_hz", {})
    print(f"background used: {noise_cal_picker.value.name}")
    print(f"background counts per channel (Hz): {_bg_by_ch}")
    bg = pd.Series({_b: _bg_by_ch.get(_b.removeprefix("coinc_ch"), 0.0) for _b in BINS})

    # detector efficiency: 3 physical detectors (herald / dump / one shared,
    # time-multiplexed detector for every loop channel) -- see
    # stochprocsim.utils.DETECTOR_EFFICIENCY
    eff = pd.Series({_b: efficiency_for_channel(int(_b.removeprefix("coinc_ch")), dump_ch=7)
                     for _b in BINS})
    return bg, eff


@app.cell(hide_code=True)
def _(loss_cal_picker):
    # loss calibration, read directly from the picked *_loss_calibration.json
    # (no measurement sidecar to resolve an embedded/fallback copy from, so
    # this clone always reads the picked file directly). Current
    # calibrate_loss() only produces two transmission estimates -- the
    # det_eff_dump/loss_input_to_setup/loss_to_dump "single-photon 3-ratio"
    # fields the software-data notebook still reads are from an older,
    # now-dropped calibrate_loss() stage set (fibre-replug stages removed
    # for unattended use) and would just silently read as 0.0 here.
    import json as _json_loss

    _cal = _json_loss.loads(loss_cal_picker.value.read_text())
    _losses = _cal.get("losses", {})
    print(f"loss calibration used: {loss_cal_picker.value.name}")

    T_loop01 = _losses.get("loss_per_loop_pass", 0.0)
    T_loop01_err = _losses.get("loss_per_loop_pass_err", 0.0)
    T_5050 = _losses.get("loss_per_loop_pass_5050", 0.0)
    T_5050_err = _losses.get("loss_per_loop_pass_5050_err", 0.0)
    print(f"T (loop0/loop1 ratio) = {T_loop01:.4f} +/- {T_loop01_err:.4f}")
    print(f"T (50/50 decay fit)   = {T_5050:.4f} +/- {T_5050_err:.4f}")
    return T_5050, T_5050_err, T_loop01, T_loop01_err


@app.cell(hide_code=True)
def _(
    BINS,
    N,
    T_loop01,
    T_loop01_err,
    bg,
    eff,
    np,
    p_exact,
    p_theory,
    plot_state_grid,
    rates,
    stds,
):
    # T_measured: per-loop transmission measured directly from the loss
    # calibration's loop0/loop1 stages (loss_per_loop_pass is already that
    # ratio, with its error already propagated by calibrate_loss itself)
    # -- rather than fitted to match the S0 shape against theory.
    T_measured, T_measured_std = T_loop01, T_loop01_err
    print(f"T_measured (loop0/loop1 ratio) = {T_measured:.4f} +/- {T_measured_std:.4f}")

    _k = np.array([N if _b == "coinc_ch7" else int(_b.removeprefix("coinc_ch")) // 2 for _b in BINS])
    transmission_measured = T_measured ** _k

    frac, yerr = {}, {}
    for _s in rates.index:
        _corr = ((rates.loc[_s] - bg) / eff) / transmission_measured
        frac[_s] = _corr / _corr.sum()
        yerr[_s] = ((stds.loc[_s] / eff) / transmission_measured) / _corr.sum()
    print(f"s0 (loss-corrected, T_measured): {np.round(frac[0].to_numpy(), 4)}")
    plot_state_grid(frac, yerr, p_theory,
                    f"loss-corrected (T_measured={T_measured:.2f}, bckgnd avg={bg.mean():.2f}) vs theory",
                    p_exact=p_exact)
    return T_measured, T_measured_std, frac


@app.cell(hide_code=True)
def _(BINS, N, bg, eff, np, p_exact, p_theory, plot_state_grid, rates, stds):
    # T_fit (old method, pre-T_measured): per-loop transmission T fitted
    # against the S0 input distribution only (bin k has traversed k loops,
    # dump counted like the last exit) -- no direct measurement, just
    # whatever T makes the s0 shape match theory. Other measured states
    # aren't used for the fit. Kept alongside T_measured for comparison.
    from scipy.optimize import minimize_scalar

    _k = np.array([N if _b == "coinc_ch7" else int(_b.removeprefix("coinc_ch")) // 2 for _b in BINS])
    def _resid(T):
        _corr = ((rates.loc[0] - bg) / eff) / T ** _k
        return np.sum((p_theory[0] - _corr / _corr.sum()) ** 2)

    T_fit = minimize_scalar(_resid, bounds=(0.05, 1), method="bounded").x
    print(f"transmission fit: T_fit = {T_fit:.3f}  residual = {_resid(T_fit):.4f}")
    transmission_fit = T_fit ** _k

    frac_fit, yerr_fit = {}, {}
    for _s in rates.index:
        _corr = ((rates.loc[_s] - bg) / eff) / transmission_fit
        frac_fit[_s] = _corr / _corr.sum()
        yerr_fit[_s] = ((stds.loc[_s] / eff) / transmission_fit) / _corr.sum()
    print(f"s0 (loss-corrected, T_fit): {np.round(frac_fit[0].to_numpy(), 4)}")
    plot_state_grid(frac_fit, yerr_fit, p_theory,
                    f"loss-corrected (T_fit={T_fit:.2f}, bckgnd avg={bg.mean():.2f}) vs theory",
                    p_exact=p_exact)
    return (T_fit,)


@app.cell(hide_code=True)
def _(T_5050, T_5050_err, T_fit, T_measured, T_measured_std):
    print("TRANSMISSIONS (value +/- std):")
    print(f"  measured (loop0/loop1) T  = {T_measured:.4f} +/- {T_measured_std:.4f}")
    print(f"  50/50 decay-fit T         = {T_5050:.4f} +/- {T_5050_err:.4f}  (independent cross-check)")
    print(f"  fit (vs S0 shape) T       = {T_fit:.4f}")
    return


@app.cell(hide_code=True)
def _(N, frac, np):
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

    print(f"R_KL rate (compute_from_exp_data, s0 input) = {R_KL_rate:.6f} bits")
    return


if __name__ == "__main__":
    app.run()
