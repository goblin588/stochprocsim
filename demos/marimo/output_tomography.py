import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import io
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import QuantumTomography as qKLib
    from qutip import Bloch
    from scipy.optimize import minimize_scalar

    from stochprocsim.libraries.optics_lib import HWP, QWP
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel
    from stochprocsim.models.unitaries import TRUE_OUTPUT_STATES
    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.utils import (
        background_rate_by_channel,
        efficiency_for_channel,
        exit_distribution,
        generate_quantum_model,
        loop_pass_k,
        rates_by_input_state,
    )
    return (
        Bloch,
        Causal_Models,
        HWP,
        Path,
        QWP,
        QuantumTransitionModel,
        Simulator,
        TRUE_OUTPUT_STATES,
        background_rate_by_channel,
        efficiency_for_channel,
        exit_distribution,
        generate_quantum_model,
        get_uniform_renewal,
        io,
        json,
        loop_pass_k,
        minimize_scalar,
        mo,
        np,
        pd,
        plt,
        qKLib,
        rates_by_input_state,
        sns,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Output-string polarisation tomography vs. theory

    For each process length N, reconstruct the exiting photon's polarisation
    state at each loop-exit channel ("output string") from the best
    available real polarisation-tomography dataset (H/V/A/D/R/L analyzer
    scan), using the Kwiat group's `QuantumTomography` package (maximum
    likelihood single-qubit tomography). Compare against:

    - **theory (model.U)**: `Simulator.get_output_states()`'s per-loop-step
      prediction — the same abstract loop unitary the rest of this repo's
      R_KL/probability calculations already treat as "theory".
    - **true theory** (optional): `TRUE_OUTPUT_STATES` in `unitaries.py` —
      placeholders for now, filled in independently by the group's
      theorists once available.

    Dataset selection is scanned fresh here, not read off the curated lists
    in `plot_experimental_data.py` (which is untouched).
    """
    )
    return


@app.cell(hide_code=True)
def _(Path):
    DATA_DIR = Path("/home/brendan/Documents/PhD/Programs/stochprocsim/data")
    N_VALUES = [3, 4, 5, 6]
    BASES = ["H", "V", "A", "D", "R", "L"]
    MEASUREMENT_STATES = {
        "H": [1, 0], "V": [0, 1],
        "D": [1, 1], "A": [1, -1],
        "R": [1, "1j"], "L": [1, "-1j"],
    }
    return BASES, DATA_DIR, MEASUREMENT_STATES, N_VALUES


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
    mo.md(r"""## Step 1 — find the best real-tomography file per N""")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    DATA_DIR,
    N_VALUES,
    QuantumTransitionModel,
    Simulator,
    background_rate_by_channel,
    efficiency_for_channel,
    exit_distribution,
    generate_quantum_model,
    get_uniform_renewal,
    loop_pass_k,
    minimize_scalar,
    np,
    pd,
    rates_by_input_state,
):
    def _is_real_tomo(df, bases):
        return "basis" in df.columns and set(bases) <= set(df["basis"].dropna().unique())

    def _r_kl_for_file(csv_path, N):
        """T_fit + state-conditional R_KL, matching kl_divergence_experimental.py."""
        data = pd.read_csv(csv_path)
        bins = sorted(
            (c for c in data.columns if c.startswith("coinc_ch") and c != "coinc_ch7"),
            key=lambda c: int(c.removeprefix("coinc_ch")),
        ) + ["coinc_ch7"]
        data = data.dropna(subset=bins + ["herald", "int_time", "input_state"])
        data = data[data["herald"] > 0]
        if data.empty:
            return None
        rates, _stds = rates_by_input_state(data, bins)
        if 0 not in rates.index:
            return None

        bg_by_ch, _ = background_rate_by_channel(csv_path.with_suffix(".json"))
        bg = pd.Series({b: bg_by_ch.get(b.removeprefix("coinc_ch"), 0.0) for b in bins})
        eff = pd.Series({
            b: efficiency_for_channel(int(b.removeprefix("coinc_ch")), dump_ch=7) for b in bins
        })

        sim = Simulator(QuantumTransitionModel(Causal_Models[N]))
        p = sim.get_output_distribution(propagate_outputs=True, start_state=0)
        # bin -> loop pass number is by POSITION among the recorded loop
        # channels (loop_pass_k), not by channel-number arithmetic -- the
        # channel-to-pass wiring shifted on 2026-11-16 (see loop_pass_k
        # docstring); channels beyond the model's N loops get target 0
        target = np.array([
            1 - sum(p) if b == "coinc_ch7"
            else (p[loop_pass_k(b, bins) - 1] if loop_pass_k(b, bins) <= len(p) else 0.0)
            for b in bins
        ])
        k = np.array([
            N if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins
        ])

        def _resid(T):
            corr = ((rates.loc[0] - bg) / eff) / T ** k
            return np.sum((target - corr / corr.sum()) ** 2)

        T_fit = minimize_scalar(_resid, bounds=(0.05, 1), method="bounded").x
        corr = ((rates.loc[0] - bg) / eff) / T_fit ** k
        frac0 = (corr / corr.sum()).tolist()

        exact_model = get_uniform_renewal(N - 1)
        exp_model = generate_quantum_model(np.array(frac0[:N]))
        p_state = exact_model.steady_state
        R_KL = 0.0
        for i in range(N):
            p_i = exit_distribution(exact_model.probs, start=i)
            q_i = exit_distribution(exp_model.probs, start=i)
            cond = sum(px * np.log2(px / qx) for px, qx in zip(p_i, q_i) if px > 0 and qx > 0)
            R_KL += p_state[i] * cond
        mean_loops = sum(
            (i + 1) * pp for i, pp in enumerate(exit_distribution(exact_model.probs, start=0))
        )
        return R_KL / mean_loops, T_fit

    def _theory_r_kl(N):
        sim = Simulator(QuantumTransitionModel(Causal_Models[N]))
        p = sim.get_output_distribution(propagate_outputs=True, start_state=0)
        exact_model = get_uniform_renewal(N - 1)
        exp_model = generate_quantum_model(np.array(p[:N]))
        p_state = exact_model.steady_state
        R_KL = 0.0
        for i in range(N):
            p_i = exit_distribution(exact_model.probs, start=i)
            q_i = exit_distribution(exp_model.probs, start=i)
            cond = sum(px * np.log2(px / qx) for px, qx in zip(p_i, q_i) if px > 0 and qx > 0)
            R_KL += p_state[i] * cond
        mean_loops = sum(
            (i + 1) * pp for i, pp in enumerate(exit_distribution(exact_model.probs, start=0))
        )
        return R_KL / mean_loops

    def find_best_tomo_file(N, bases=("H", "V", "A", "D", "R", "L")):
        """Lowest-R_KL real-tomography file for N, excluding below-theory results."""
        files = sorted(set(DATA_DIR.glob(f"*_N{N}_*.csv")) | set(DATA_DIR.glob(f"*N{N}_*.csv")))
        files = [f for f in files if "REF" not in f.name and "background" not in f.name]

        candidates = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if not _is_real_tomo(df, bases):
                continue
            result = _r_kl_for_file(f, N)
            if result is not None:
                candidates.append((result[0], result[1], f))

        theo = _theory_r_kl(N)
        valid = [c for c in candidates if c[0] >= theo]
        if not valid:
            return None
        rkl, T_fit, f = min(valid, key=lambda c: c[0])
        return {"file": f, "R_KL": rkl, "T_fit": T_fit, "theory_R_KL": theo}

    best_per_N = {N: find_best_tomo_file(N) for N in N_VALUES}
    for _N, _res in best_per_N.items():
        if _res is None:
            print(f"N={_N}: no usable real-tomography file found")
        else:
            print(
                f"N={_N}: {_res['file'].name}  R_KL={_res['R_KL']:.4f}  "
                f"(theory={_res['theory_R_KL']:.4f}, T_fit={_res['T_fit']:.3f})"
            )
    return (best_per_N,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 2 — reconstruct per-output-string density matrices""")
    return


@app.cell(hide_code=True)
def _(BASES, MEASUREMENT_STATES, best_per_N, json, pd, qKLib, tempfile):
    def reconstruct_channel(df, channel, input_state, bases=BASES, measurement_states=MEASUREMENT_STATES):
        """Pool coinc_ch{channel} counts across the 6 analyzer-basis settings
        for one (input_state, channel) and run Kwiat MLE tomography on them."""
        sub = df[df["input_state"] == input_state]
        counts_by_basis = sub.groupby("basis")[f"coinc_ch{channel}"].sum()
        if not set(bases) <= set(counts_by_basis.index):
            return None
        tomo_data = {
            "n_qubits": 1, "n_detectors_per_qubit": 1, "n_measurements_per_qubit": 6,
            "coincidence_window": [0],
            "measurement_states": measurement_states,
            "data": [{"basis": [b], "counts": [int(counts_by_basis[b])]} for b in bases],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(tomo_data, fh)
            path = fh.name
        t = qKLib.Tomography()
        t.import_data(path)
        rho, _intens, _fval = t.run_tomography()
        return rho

    reconstructed = {}  # (N, input_state, channel) -> rho
    for _N, _res in best_per_N.items():
        if _res is None:
            continue
        _data = pd.read_csv(_res["file"])
        _data = _data[_data["herald"] > 0]
        _channels = sorted(
            int(c.removeprefix("coinc_ch")) for c in _data.columns
            if c.startswith("coinc_ch") and c != "coinc_ch7"
        )
        for _s in sorted(_data["input_state"].dropna().unique()):
            for _ch in _channels:
                _rho = reconstruct_channel(_data, _ch, int(_s))
                if _rho is not None:
                    reconstructed[(_N, int(_s), _ch)] = _rho

    print(f"reconstructed {len(reconstructed)} output-string density matrices")
    return (reconstructed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 2b — apply the missing output HWP/QWP

    The real apparatus is *supposed* to have an extra half-wave + quarter-
    wave plate in the output path (calibrated `θhf1`/`θqf1` per N, already
    in `models/waveplate_angles.py` / `Causal_Models[N].angles_NTU` — same
    pair for every channel/input_state at that N) but it was never
    physically installed — it doesn't affect the loop-exit statistics
    (which channel a photon comes out of), only its polarization, so
    theory (`CausalModel.U_theo`, the abstract idealized unitary — not
    `getUtot(angles)`) is unaffected and stays as-is. The reconstructed ρ
    is missing this rotation, so we apply it in post to the measured
    state to get what should have been measured.

    A density matrix transforms under a unitary as ρ → UρU†. Beam-path
    order matches `getUtot`'s own convention (QWP hit first, then HWP):
    `U = HWP(θhf1) @ QWP(θqf1)`, correction `U @ rho @ U.conj().T`.
    """
    )
    return


@app.cell(hide_code=True)
def _(Causal_Models, HWP, N_VALUES, QWP, reconstructed):
    _U_wp_by_N = {
        N: HWP(Causal_Models[N].angles_NTU["θhf1"]) @ QWP(Causal_Models[N].angles_NTU["θqf1"])
        for N in N_VALUES
    }
    reconstructed_corrected = {
        (N, s, ch): _U_wp_by_N[N] @ rho @ _U_wp_by_N[N].conj().T
        for (N, s, ch), rho in reconstructed.items()
    }
    return (reconstructed_corrected,)


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    TRUE_OUTPUT_STATES,
    pd,
    qKLib,
    reconstructed_corrected,
):
    _fidelity_rows = []
    for (_N, _s, _ch), _rho in reconstructed_corrected.items():
        _sim = Simulator(QuantumTransitionModel(Causal_Models[_N]))
        _theory_states = _sim.get_output_states(start_state=_s)
        _channels_for_N = sorted({ch for (n, _s2, ch) in reconstructed_corrected if n == _N})
        _loop_idx = _channels_for_N.index(_ch)
        if _loop_idx >= len(_theory_states):
            continue
        _theory_state, _prob = _theory_states[_loop_idx]
        if _theory_state is None:
            continue
        _fid_U = qKLib.fidelity(_rho, _theory_state)

        _fid_true = None
        _true_states = TRUE_OUTPUT_STATES.get(_N)
        if _true_states and _loop_idx < len(_true_states) and _true_states[_loop_idx] is not None:
            _fid_true = qKLib.fidelity(_rho, _true_states[_loop_idx])

        _fidelity_rows.append({
            "N": _N, "input_state": _s, "channel": _ch, "loop": _loop_idx + 1,
            "fidelity_model_U": _fid_U, "fidelity_true_theory": _fid_true,
        })

    fidelity_df = pd.DataFrame(_fidelity_rows)
    fidelity_df
    return (fidelity_df,)


@app.cell(hide_code=True)
def _(fidelity_df, fig_download_button, mo, np, plt, sns):
    # dataviz-validated 8-hue categorical order (adjacent-pairs safe, which
    # is what matters for grouped bars) -- input_state is the color entity,
    # loop/output-string is the x-axis category within each N's subplot
    _palette = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ]
    _N_values = sorted(fidelity_df["N"].unique())
    with sns.axes_style("whitegrid"):
        _fig, _axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

    for _ax, _N in zip(_axes.flat, _N_values):
        _sub = fidelity_df[fidelity_df["N"] == _N]
        _loops = sorted(_sub["loop"].unique())
        _states = sorted(_sub["input_state"].unique())
        # _width = 0.8 / max(len(_states), 1)

        # for _i, _s in enumerate(_states):
        _i, _s = 0, _states[0]
        _row = _sub[_sub["input_state"] == _s].set_index("loop").reindex(_loops)
        _x = np.array(_loops) #+ (_i - (len(_states) - 1) / 2) #* _width
        _ax.bar(_x, _row["fidelity_model_U"], #width=_width,
                label=f"input_state={_s}", color=_palette[_i % len(_palette)])

        _ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        _ax.set_xticks(_loops)
        _ax.set_xlabel("output string (loop-exit number)")
        _ax.set_ylabel("fidelity vs. model.U theory")
        _ax.set_title(f"N={_N}")
        _ax.set_ylim(0, 1.05)
        _ax.legend(fontsize=8)

    for _ax in _axes.flat[len(_N_values):]:
        _ax.axis("off")

    _fig.suptitle("Reconstructed output-string state fidelity vs. theory")
    _fig.tight_layout()
    mo.vstack([_fig, fig_download_button(_fig, "fidelity_vs_output.svg")])
    return


@app.cell(hide_code=True)
def _(N_VALUES, mo):
    fig4b_N = mo.ui.dropdown(
        options=[str(n) for n in N_VALUES], value="4", label="N"
    )
    fig4b_N
    return (fig4b_N,)


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    fidelity_df,
    fig4b_N,
    fig_download_button,
    mo,
    np,
    plt,
    reconstructed_corrected,
    sns,
):
    _N = int(fig4b_N.value)
    _sub = fidelity_df[fidelity_df["N"] == _N].sort_values(["input_state", "loop"])
    _states = sorted(_sub["input_state"].unique())
    _loops = sorted(_sub["loop"].unique())  # loop k <-> output "0"*(k-1) + "1"

    _ncols = min(4, len(_loops))
    _nrows = -(-len(_loops) // _ncols)
    with sns.axes_style("whitegrid"):
        _fig = plt.figure(figsize=(4 * _ncols, 3 + 3 * _nrows))
        _gs = _fig.add_gridspec(_nrows + 1, _ncols, height_ratios=[1.2] + [1] * _nrows)

        # top: infidelity per output (one group of bars per output, one color
        # per initial state -- this dataset only has real tomography for s0,
        # so that collapses to one bar per output, but the grouping
        # generalizes if more initial states get real tomography scans later)
        _ax_top = _fig.add_subplot(_gs[0, :])
        _width = 0.8 / max(len(_states), 1)
        for _i, _s in enumerate(_states):
            _row = _sub[_sub["input_state"] == _s].set_index("loop").reindex(_loops)
            _x = np.arange(len(_loops)) + (_i - (len(_states) - 1) / 2) * _width
            _ax_top.bar(_x, 1 - _row["fidelity_model_U"], width=_width, label=f"s{int(_s)}")
        _ax_top.set_xticks(range(len(_loops)), [f'"{"0" * (k - 1)}1"' for k in _loops])
        _ax_top.set_xlabel("output")
        _ax_top.set_ylabel("infidelity")
        _ax_top.set_title(f"Infidelity of final memory states per output (N={_N})")
        if len(_states) > 1:
            _ax_top.legend(fontsize=8)

        # bottom: one Re/Im panel per output, for one representative initial state
        _labels = ["Re00", "Re01", "Re10", "Re11", "Im00", "Im01", "Im10", "Im11"]
        _x8 = np.arange(len(_labels))

        def _flatten(_m):
            return np.concatenate([np.real(_m).flatten(), np.imag(_m).flatten()])

        _input_state = int(_states[0])
        _channels = sorted({c for (n, s, c) in reconstructed_corrected if n == _N and s == _input_state})
        _sim = Simulator(QuantumTransitionModel(Causal_Models[_N]))
        _theory_states = _sim.get_output_states(start_state=_input_state)

        for _i, _loop in enumerate(_loops):
            _ax = _fig.add_subplot(_gs[1 + _i // _ncols, _i % _ncols])
            _rho = reconstructed_corrected.get((_N, _input_state, _channels[_loop - 1]))
            _theory_state, _ = _theory_states[_loop - 1]
            _theory_rho = (np.outer(_theory_state, _theory_state.conj())
                           if _theory_state is not None else np.zeros((2, 2)))
            if _rho is not None:
                _ax.bar(_x8, _flatten(_rho), width=0.5, color="gold", edgecolor="black", label="reconstructed")
            _ax.bar(_x8, _flatten(_theory_rho), width=0.5, fill=False, edgecolor="black",
                    linestyle="dotted", linewidth=1.5, label="theory")
            _ax.set_xticks(_x8, _labels, rotation=45, ha="right", fontsize=7)
            _ax.set_ylim(-1, 1)
            _ax.axhline(0, color="gray", linewidth=0.5)
            _ax.set_title(f'output "{"0" * (_loop - 1)}1" (s{_input_state})', fontsize=9)

    _fig.axes[1].legend(fontsize=7)
    _fig.suptitle(f"Fig. 4(b)-style: final memory states vs. theory (N={_N})")
    _fig.tight_layout()
    mo.vstack([_fig, fig_download_button(_fig, f"fig4b_panel_N{_N}.svg")])
    return


@app.cell(hide_code=True)
def _(N_VALUES):
    TOMO_N_VALUES = N_VALUES
    return (TOMO_N_VALUES,)


@app.cell(hide_code=True)
def _(TOMO_N_VALUES, fidelity_df, fig_download_button, mo, plt):
    _sub = fidelity_df[fidelity_df["N"].isin(TOMO_N_VALUES)]
    _max_outputs = int(_sub.groupby("N")["loop"].nunique().max())

    _fig = plt.figure(figsize=(8, 6))
    _ax = _fig.add_subplot(projection="3d")
    _dx, _dy = 0.6, 0.6

    for _yi, _N in enumerate(TOMO_N_VALUES):
        _s0 = sorted(_sub.loc[_sub["N"] == _N, "input_state"].unique())[0]
        _row = _sub[(_sub["N"] == _N) & (_sub["input_state"] == _s0)].sort_values("loop")
        for _, _r in _row.iterrows():
            _xi = int(_r["loop"]) - 1
            _ax.bar3d(_xi - _dx / 2, _yi - _dy / 2, 0, _dx, _dy, 1 - _r["fidelity_model_U"],
                      color="gold", edgecolor="black", linewidth=0.5, shade=True)

    _ax.set_xticks(range(_max_outputs))
    _ax.set_xticklabels([f'{"0" * _k}1' for _k in range(_max_outputs)])
    _ax.set_yticks(range(len(TOMO_N_VALUES)))
    _ax.set_yticklabels([f"N={_n}" for _n in TOMO_N_VALUES])
    _ax.set_zlabel("Infidelity")
    _ax.set_xlabel("Output String")
    _ax.set_title("Infidelity of final memory states")
    mo.vstack([_fig, fig_download_button(_fig, "infidelity_3d.svg")])
    return


@app.cell(hide_code=True)
def _(TOMO_N_VALUES, fidelity_df, fig_download_button, mo, pd, plt, sns):
    # same data as the 3D bar chart above, as the heatmap suggested by Fig. 9
    _sub = fidelity_df[fidelity_df["N"].isin(TOMO_N_VALUES)]
    _max_outputs = int(_sub.groupby("N")["loop"].nunique().max())

    _rows = []
    for _N in TOMO_N_VALUES:
        _s0 = sorted(_sub.loc[_sub["N"] == _N, "input_state"].unique())[0]
        _row = (_sub[(_sub["N"] == _N) & (_sub["input_state"] == _s0)]
                .set_index("loop")["fidelity_model_U"])
        _rows.append(1 - _row.reindex(range(1, _max_outputs + 1)))
    _mat = pd.DataFrame(_rows, index=[f"N={_n}" for _n in TOMO_N_VALUES])
    _mat.columns = [f'{"0" * _k}1' for _k in range(_max_outputs)]

    with sns.axes_style("white"):
        _fig, _ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(_mat, annot=True, fmt=".3f", cmap="viridis", ax=_ax,
                    cbar_kws={"label": "infidelity"}, linewidths=0.5, linecolor="white")
    _ax.set_xlabel("Output string")
    _ax.set_ylabel("")
    _ax.set_title("Infidelity of final memory states (heat map)")
    mo.vstack([_fig, fig_download_button(_fig, "infidelity_heatmap.svg")])
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    fidelity_df,
    np,
    plt,
    reconstructed_corrected,
):
    def plot_memory_states_3d(N, loops=None):
        _loops = sorted(fidelity_df.loc[fidelity_df["N"] == N, "loop"].unique())
        if loops is not None:
            _loops = [l for l in _loops if l in loops]
        _input_state = int(sorted(fidelity_df.loc[fidelity_df["N"] == N, "input_state"].unique())[0])
        _channels = sorted({c for (n, s, c) in reconstructed_corrected if n == N and s == _input_state})
        _sim = Simulator(QuantumTransitionModel(Causal_Models[N]))
        _theory_states = _sim.get_output_states(start_state=_input_state)

        _fig = plt.figure(figsize=(3.2 * len(_loops), 6.5))
        _ket_labels = ["|0⟩", "|1⟩"]
        _cells = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for _ci, _loop in enumerate(_loops):
            _rho = reconstructed_corrected.get((N, _input_state, _channels[_loop - 1]))
            _theory_state, _ = _theory_states[_loop - 1]
            _theory_rho = (np.outer(_theory_state, _theory_state.conj())
                           if _theory_state is not None else np.zeros((2, 2)))
            _recon = _rho if _rho is not None else np.zeros((2, 2), dtype=complex)
        

            for _ri, (_part, _color) in enumerate([(np.real, "#2a78d6"), (np.imag, "#1baf7a")]):
                _ax = _fig.add_subplot(2, len(_loops), _ri * len(_loops) + _ci + 1, projection="3d")
                for (_r, _c) in _cells:
                    _v = _part(_recon)[_r, _c]
                    _t = _part(_theory_rho)[_r, _c]
                    _ax.bar3d(_r, _c - 0.22, 0, 0.5, 0.44, _v,
                              color=_color, edgecolor="black", linewidth=0.5, shade=True)
                    _ax.bar3d(_r - 0.05, _c - 0.27, 0, 0.6, 0.54, _t,
                              color=(1, 1, 1, 0), edgecolor="black", linestyle=":", linewidth=1.0, shade=False)
                _ax.set_xticks([0, 1])
                _ax.set_xticklabels(_ket_labels)
                _ax.set_yticks([0, 1])
                _ax.set_yticklabels(_ket_labels)
                _ax.set_zlim(-1, 1)
                if _ri == 0:
                    _ax.set_title(f'output "{"0" * (_loop - 1)}1"', fontsize=9)
                if _ci == 0:
                    _ax.set_zlabel("Re" if _ri == 0 else "Im")

        _fig.suptitle(f"Reconstructed final memory states, Re/Im (N={N}, s{_input_state})")
        return _fig
    return (plot_memory_states_3d,)


@app.cell
def _(fig_download_button, mo, plot_memory_states_3d):
    _fig = plot_memory_states_3d(3)
    mo.vstack([_fig, fig_download_button(_fig, "memory_states_3d_N3.svg")])
    return


@app.cell
def _(fig_download_button, mo, plot_memory_states_3d):
    _fig = plot_memory_states_3d(4)
    mo.vstack([_fig, fig_download_button(_fig, "memory_states_3d_N4.svg")])
    return


@app.cell
def _(fig_download_button, mo, plot_memory_states_3d):
    _fig = plot_memory_states_3d(5)
    mo.vstack([_fig, fig_download_button(_fig, "memory_states_3d_N5.svg")])
    return


@app.cell
def _(fig_download_button, mo, plot_memory_states_3d):
    _fig = plot_memory_states_3d(6)
    mo.vstack([_fig, fig_download_button(_fig, "memory_states_3d_N6.svg")])
    return


@app.cell
def _(fig_download_button, mo, plot_memory_states_3d):
    _fig = plot_memory_states_3d(6, loops=[1, 6])
    mo.vstack([_fig, fig_download_button(_fig, "memory_states_3d_N6_first_last.svg")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 7b — all output strings per process on Bloch spheres

    One sphere per process length N: every output string ("0"*(k-1)+"1")
    plotted as a solid arrow (reconstructed ρ — arrow length < 1 shows
    mixedness) and a point (theory, always pure) in the same color.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    Bloch,
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    fidelity_df,
    reconstructed_corrected,
):
    _bloch_palette = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ]

    def _bloch_vec_rho(rho):
        return [2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real]

    def _bloch_vec_ket(ket):
        return [
            2 * (ket[0] * ket[1].conjugate()).real,
            -2 * (ket[0] * ket[1].conjugate()).imag,
            (abs(ket[0]) ** 2 - abs(ket[1]) ** 2),
        ]

    def plot_bloch_spheres(N, fig=None, axes=None):
        """All output strings for process length N, reconstructed (arrow)
        vs. theory (point), on a single Bloch sphere."""
        _loops = sorted(fidelity_df.loc[fidelity_df["N"] == N, "loop"].unique())
        _input_state = int(sorted(fidelity_df.loc[fidelity_df["N"] == N, "input_state"].unique())[0])
        _channels = sorted({c for (n, s, c) in reconstructed_corrected if n == N and s == _input_state})
        _theory_states = Simulator(QuantumTransitionModel(Causal_Models[N])).get_output_states(start_state=_input_state)
        _target_vec = _bloch_vec_ket(Causal_Models[N].states[_input_state][[0, 2]])

        b = Bloch(fig=fig, axes=axes)
        for _i, _loop in enumerate(_loops):
            _color = _bloch_palette[_i % len(_bloch_palette)]
            _label = f'"{"0" * (_loop - 1)}1"'
            _rho = reconstructed_corrected.get((N, _input_state, _channels[_loop - 1]))
            _theory_state, _ = _theory_states[_loop - 1]
            if _rho is not None:
                b.add_vectors(_bloch_vec_rho(_rho), colors=[_color])
            if _theory_state is not None:
                b.add_points(_bloch_vec_ket(_theory_state), colors=[_color])
            b.add_annotation(
                [1.25 * c for c in (_bloch_vec_ket(_theory_state) if _theory_state is not None else _bloch_vec_rho(_rho))],
                _label, color=_color, fontsize=9,
            )

        b.make_sphere()
        # qutip's Bloch.plot_vectors/plot_points swap axes for display
        # (plot_x, plot_y, plot_z = vec[1], -vec[0], vec[2]) -- match that
        # here or the dashed line lands in the wrong place vs. the arrows.
        b.axes.plot(
            [0, _target_vec[1]], [0, -_target_vec[0]], [0, _target_vec[2]],
            linestyle="--", color="black", linewidth=1.5, zorder=10, label=f"target s{_input_state}",
        )
        b.axes.legend(loc="upper left", fontsize=7, handlelength=1.5)
        b.axes.set_title(f"N={N}, s{_input_state}", fontsize=11)
        return b
    return (plot_bloch_spheres,)


@app.cell
def _(N_VALUES, fig_download_button, mo, plot_bloch_spheres, plt):
    _fig = plt.figure(figsize=(10, 10))
    for _i, _N in enumerate(N_VALUES):
        _ax = _fig.add_subplot(2, 2, _i + 1, projection="3d")
        plot_bloch_spheres(_N, fig=_fig, axes=_ax)
    _fig.suptitle("Reconstructed vs. theory output-string states")
    _fig.tight_layout()
    mo.vstack([_fig, fig_download_button(_fig, "bloch_spheres.svg")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 8 — print the reconstructed vs. theory density matrices""")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    fidelity_df,
    np,
    mo,
    reconstructed_corrected,
):
    def _fmt(rho):
        return "\n".join(
            "  ".join(f"{v.real:+.3f}{v.imag:+.3f}j" for v in row) for row in rho
        )

    _blocks = []
    for _N in sorted(fidelity_df["N"].unique()):
        _loops = sorted(fidelity_df.loc[fidelity_df["N"] == _N, "loop"].unique())
        _input_state = int(sorted(fidelity_df.loc[fidelity_df["N"] == _N, "input_state"].unique())[0])
        _channels = sorted({c for (n, s, c) in reconstructed_corrected if n == _N and s == _input_state})
        _theory_states = Simulator(QuantumTransitionModel(Causal_Models[_N])).get_output_states(start_state=_input_state)

        for _loop in _loops:
            _rho = reconstructed_corrected.get((_N, _input_state, _channels[_loop - 1]))
            _theory_state, _ = _theory_states[_loop - 1]
            _theory_rho = (np.outer(_theory_state, _theory_state.conj())
                           if _theory_state is not None else np.zeros((2, 2)))
            _recon = _rho if _rho is not None else np.zeros((2, 2), dtype=complex)
            _blocks.append(
                f'### N={_N}, s0, output "{"0" * (_loop - 1)}1" (loop {_loop})\n\n'
                f"**Reconstructed ρ**\n```\n{_fmt(_recon)}\n```\n"
                f"**Theory ρ**\n```\n{_fmt(_theory_rho)}\n```"
            )

    mo.md("\n\n".join(_blocks))
    return


if __name__ == "__main__":
    app.run()
