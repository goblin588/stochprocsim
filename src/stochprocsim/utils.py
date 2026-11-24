import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from stochprocsim.stochprocq.models.renewal import RenewalProcess, get_uniform_renewal
from stochprocsim.stochprocq.measure import eval_diverge

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DETECTOR_EFFICIENCY = {
    'herald': 0.6,
    'dump': 0.68,
    # loop = 1 (reference) -- matches loss_calibration_for's own documented
    # convention ("loop = 1 reference, dump relative to it" via
    # det_eff_dump), which the previous 0.755 silently violated. 0.755 was
    # never actually tied to that calibration mechanism -- det_eff_dump has
    # never been populated in any real calibration file (always defaults
    # to 0.0), so it wasn't "relative to" anything measured, just a free-
    # floating guess. Uniform across all loop channels either way, so this
    # only changes the loop-vs-dump weighting in the normalized
    # distribution, not loop-vs-loop ratios, and doesn't affect T at all
    # (T is a same-detector count ratio, this constant cancels out of it).
    'loop': 1.0,
}


def efficiency_for_channel(channel: int, herald_ch: int = 3, dump_ch: int = 7) -> float:
    """Which physical detector reads `channel`, hence which DETECTOR_EFFICIENCY
    entry applies. herald_ch/dump_ch default to this setup's TRIGG_CH/DUMP_CH."""
    if channel == herald_ch:
        return DETECTOR_EFFICIENCY['herald']
    if channel == dump_ch:
        return DETECTOR_EFFICIENCY['dump']
    return DETECTOR_EFFICIENCY['loop']


def loop_pass_k(bin_name: str, bins: list[str], dump_bin: str = "coinc_ch7") -> int:
    """1-indexed physical loop pass for a coincidence-count bin: its
    position among `bins`' recorded loop channels (ascending), not the
    channel number itself. The channel-to-pass wiring shifted on
    2026-11-16 (ch2 was pass 1 before, ch4 is pass 1 after, and the
    apparatus now always records 6 loop channels regardless of N), but
    the recorded channels stay in physical pass order either way, so
    position is the portable invariant -- channel arithmetic (channel //
    2) is not."""
    return [b for b in bins if b != dump_bin].index(bin_name) + 1


def background_rate_by_channel(json_path: Path) -> tuple[dict, str]:
    """Per-channel background/accidental rate (Hz) for a measurement run,
    keyed by channel number as a string (matches noise_calibration's
    background_rate_hz shape).

    Three sources, in priority order (each a strictly better estimate of
    the run's own background floor than the last):
      1. json_path's own "background" field — the inline reading
         measurement.py takes seconds before that exact run starts.
      2. json_path's "noise_calibration" field, when it's the *entire*
         contents of a *_noise_calibration.json embedded directly (current
         measurement.py behaviour) — background_rate_hz is right there,
         no second file to open.
      3. the older, now-superseded shape where "noise_calibration" was
         just a {file, saved_at} pointer — resolved by filename next to
         json_path, since that's all pre-embedding data has.
    Returns ({}, note) with an explanatory note when none are available.
    """
    if not json_path.exists():
        return {}, "none (background=0)"
    meta = json.loads(json_path.read_text())

    bg_row = meta.get("background")
    if bg_row and bg_row.get("int_time"):
        int_time = bg_row["int_time"]
        by_ch = {k.removeprefix("coinc_ch"): v / int_time
                 for k, v in bg_row.items() if k.startswith("coinc_ch")}
        return by_ch, f"inline (+{meta.get('background_offset_ns', '?')}ns, {int_time:.0f}s)"

    cal = meta.get("noise_calibration")
    if cal and "background_rate_hz" in cal:
        return cal["background_rate_hz"], f"embedded ({cal.get('file', '?')}, {cal.get('saved_at', '?')})"
    if cal and "file" in cal:
        cal_path = json_path.parent / cal["file"]
        if cal_path.exists():
            by_ch = json.loads(cal_path.read_text()).get("background_rate_hz", {})
            return by_ch, f"{cal['file']} ({cal.get('saved_at', '?')})"

    return {}, "none (background=0)"


def background_rate_std_by_channel(json_path: Path) -> tuple[dict, str]:
    """Per-channel background-rate SEM (Hz), companion to
    background_rate_by_channel -- only available for the inline source
    (json_path's own "background" field), since that's the only one that
    keeps the per-bin breakdown (bg_row["bins"]) needed to compute a
    spread; the embedded/pointer noise_calibration sources only ever carry
    a single collapsed background_rate_hz number, so there's nothing to
    derive an error from there -- returns {} (treat as exactly known, the
    previous default) with a note saying why, rather than a fake 0.0 that
    looks like "measured and tiny" instead of "not available".
    """
    if not json_path.exists():
        return {}, "none (no std available)"
    meta = json.loads(json_path.read_text())
    bg_row = meta.get("background")
    bins = bg_row.get("bins") if bg_row else None
    if not bins:
        return {}, "none (no per-bin background data to derive a std from)"
    chs = [k.removeprefix("coinc_ch") for k in bg_row if k.startswith("coinc_ch")]
    by_ch = {}
    for ch in chs:
        bin_rates = [b[f"coinc_ch{ch}"] / b["int_time"] for b in bins if b["int_time"]]
        by_ch[ch] = (np.std(bin_rates, ddof=1) / len(bin_rates) ** 0.5) if len(bin_rates) > 1 else 0.0
    return by_ch, f"inline, {len(bins)} bins"


def _loss_calibration_meta(json_path: Path, *, require: str) -> tuple[dict, str]:
    """The *_loss_calibration.json content in effect for a specific
    measurement — shared lookup behind loss_calibration_for/loss_rates_for.

    Prefers json_path's own "loss_calibration" field — the *entire*
    contents of the *_loss_calibration.json in effect when that
    measurement was saved (current measurement.py behaviour), frozen at
    save time so it can't drift if calibrate_loss() runs again later —
    but only when it actually carries `require`: the embed only ever
    carries "file"/"stages"/"losses", never the standalone file's own
    "coincidence_rates_hz", so a caller that needs rates must fall through
    to the directory lookup below even when an embed is present.
    Falls back to the most recent *_loss_calibration.json AT OR BEFORE
    json_path's own timestamp (parsed from its filename, the same
    YYYYMMDD_HHMMSS pattern used throughout this codebase) — for older
    measurements taken before the embed existed, or whenever the embed
    lacks `require`. This used to just grab whichever *_loss_calibration.json
    was most recent *anywhere in the directory*, with no regard for
    json_path's own date — confirmed wrong on real data (an old
    measurement would silently pick up a brand-new calibration taken days
    later). Still a guess when it's used at all (an embed, when present,
    is exact and preferred above), and if json_path's own timestamp can't
    be parsed or nothing qualifies before it, falls back further to
    whatever's most recent in the directory, clearly labelled as such.
    Returns ({}, note) if nothing is available at all.
    """
    if json_path.exists():
        cal = json.loads(json_path.read_text()).get("loss_calibration")
        if cal and require in cal:
            return cal, f"embedded ({cal.get('file', '?')}, {cal.get('saved_at', '?')})"

    files = sorted(json_path.parent.glob("*_loss_calibration.json"), key=lambda p: p.name)
    if not files:
        return {}, "none (no *_loss_calibration.json found)"

    ts_match = re.search(r"\d{8}_\d{6}", json_path.name)
    if ts_match:
        meas_ts = ts_match.group()
        prior = [f for f in files
                 if (m := re.search(r"\d{8}_\d{6}", f.name)) and m.group() <= meas_ts]
        if prior:
            chosen = prior[-1]
            return json.loads(chosen.read_text()), f"{chosen.name} (nearest prior to this measurement)"

    latest = files[-1]
    return json.loads(latest.read_text()), f"{latest.name} (dir fallback -- no earlier calibration found, not necessarily in effect)"


def loss_calibration_for(json_path: Path) -> tuple[dict, str]:
    """`losses` dict in effect for a specific measurement:
    `det_eff_setup`/`det_eff_dump` (loop = 1 reference, dump relative to it)
    and `loss_input_to_setup`/`loss_per_loop_pass`/`loss_to_dump` — despite
    the "loss_" naming these are transmission ratios (<=1), matching
    get_loss's input_to_loop/loop_to_loop/loop_to_dump args directly, not
    1 - transmission. See _loss_calibration_meta for the embedded-vs-
    fallback lookup this reads from.
    """
    cal, note = _loss_calibration_meta(json_path, require="losses")
    return cal.get("losses", {}), note


def loss_rates_for(json_path: Path) -> tuple[dict, str]:
    """coincidence_rates_hz dict in effect for a specific measurement — the
    raw per-stage rates (loop0_hz = zero-loop-bypass stage, loop1_hz =
    one-loop-pass stage, each with an "_err" SEM companion) that
    loss_calibration_for's losses were computed from. See
    _loss_calibration_meta for the embedded-vs-fallback lookup this reads
    from."""
    cal, note = _loss_calibration_meta(json_path, require="coincidence_rates_hz")
    return cal.get("coincidence_rates_hz", {}), note


def rates_by_input_state(data: pd.DataFrame, bins: list[str],
                          pairs=(("H", "V"), ("A", "D"), ("R", "L"))):
    """Mean count rate and its SEM per input_state, one row per state.

    `data` must have `bins` columns plus `int_time` and `input_state`. If it
    also has a `basis` column (a polarization tomography scan), each
    conjugate pair's rates are summed — H+V, A+D and R+L should each equal
    the same basis-independent total exit rate — and the three pair-sums are
    averaged as a stand-in single distribution. This is not state
    reconstruction, just a placeholder until that's built.

    The returned std combines two things: the within-basis shot noise
    (propagated from each basis's own SEM) and the between-pair spread (SEM
    across the 3 pair-sums) — in practice the between-pair spread dominates
    by roughly an order of magnitude, since H+V/A+D/R+L only agree to a
    couple percent (polarization-dependent loss, waveplate calibration),
    which is far bigger than the per-basis shot noise.
    """
    rates = data[bins].div(data["int_time"], axis=0)
    rates["input_state"] = data["input_state"]

    # every measurement file has a `basis` column, but it's all-NaN for
    # ordinary (non-tomography) runs — only treat it as a tomo scan if some
    # row actually names a basis
    if "basis" not in data.columns or data["basis"].isna().all():
        g = rates.groupby("input_state")
        return g.mean(), g.std().div(g.size() ** 0.5, axis=0)

    rates["basis"] = data["basis"]
    rates = rates.dropna(subset=["basis"])
    g = rates.groupby(["input_state", "basis"])
    mean, sem = g.mean(), g.std().div(g.size() ** 0.5, axis=0)

    pair_sums = [mean.xs(a, level="basis") + mean.xs(b, level="basis") for a, b in pairs]
    combined_rates = sum(pair_sums) / len(pairs)

    within_var = sum(sem.xs(a, level="basis") ** 2 + sem.xs(b, level="basis") ** 2 for a, b in pairs)
    within_sem = within_var ** 0.5 / len(pairs)

    between_std = pd.concat(pair_sums, keys=range(len(pairs))).groupby(level=1).std(ddof=1)
    between_sem = between_std / len(pairs) ** 0.5

    combined_stds = (within_sem ** 2 + between_sem ** 2) ** 0.5
    return combined_rates, combined_stds


def drop_count_outliers(data: pd.DataFrame, bins: list[str], ratio: float = 15.0,
                         floor: float = 50.0) -> pd.DataFrame:
    """Drop rows where any `bins` column reads more than `ratio` x that
    column's own median (and at least `floor`, so a near-zero-median
    channel like the dump bin doesn't flag every nonzero reading as an
    "outlier"). Isolated detector glitches (one row spiking to 1000x+ its
    channel's normal count) blow up a row-level or block-averaged SEM
    alike -- averaging doesn't fix a single catastrophic spike, it just
    spreads the same total deviation over however many blocks it lands in
    (confirmed on real data: block_rate_mean_std barely moved the SEM on a
    file with exactly this kind of spike). This has to run before any mean/
    std is taken, not after.
    """
    med = data[bins].median()
    thresh = {b: max(floor, med[b] * ratio) for b in bins}
    mask = (data[bins] > pd.Series(thresh)).any(axis=1)
    return data[~mask]


def block_rate_mean_std(data: pd.DataFrame, bins: list[str], block_size: int = 30):
    """Per-channel (mean_rate, SEM) computed over block_size-row super-bins
    instead of individual rows -- feeds a Monte Carlo (kl_rate_mc_error)
    that would otherwise see an inflated SEM at 1 row/s resolution on the
    weak, low-count high-pass channels (ch12/ch14-style), where a single
    row's count is often 0-2 and the row-to-row Poisson/discreteness noise
    swamps the real spread. Grouping `block_size` consecutive rows into one
    super-bin (summed counts / summed int_time, same recipe as
    calibrate_loss's own 5s bins elsewhere in this codebase) before taking
    std/sqrt(n_blocks) trades time resolution for a steadier estimate --
    same data, fewer, larger samples. A short trailing remainder (< one
    full block) is dropped rather than padded, same convention as a
    rolling-window average.

    `data` must already be filtered to one input_state/basis combination
    (e.g. s0, same precondition as rates_by_input_state) and in time order.
    Returns (mean: pd.Series, sem: pd.Series) indexed by `bins`.

    Raises ValueError if `data` has fewer than `block_size` rows (zero
    whole blocks) -- `block_means.mean(axis=0)` on an empty array would
    otherwise silently return NaN (with sem=0.0, not NaN, which looks even
    more like a real answer) instead of surfacing that there wasn't
    enough data to compute anything at all.
    """
    n_blocks = len(data) // block_size
    if n_blocks == 0:
        raise ValueError(f"block_rate_mean_std: only {len(data)} rows, need at least "
                          f"block_size={block_size} to form one block")
    block_means = np.empty((n_blocks, len(bins)))
    for i in range(n_blocks):
        block = data.iloc[i * block_size:(i + 1) * block_size]
        block_means[i] = (block[bins].sum() / block["int_time"].sum()).to_numpy()
    mean = pd.Series(block_means.mean(axis=0), index=bins)
    sem = pd.Series(block_means.std(axis=0, ddof=1) / n_blocks ** 0.5, index=bins) if n_blocks > 1 \
        else pd.Series(0.0, index=bins)
    return mean, sem


def kl_divergence(p, q) -> float:
    """KL divergence D(p||q) in bits between two equal-length lists.

    Inputs are normalised to sum to 1, so raw counts/rates are fine, and the
    lists can be any (equal) length — extra bins beyond the theoretical support
    are handled: a bin with p = 0 contributes 0; p > 0 where q = 0 gives inf.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"p and q must be the same length: {p.shape} vs {q.shape}")
    return float(np.sum(rel_entr(p / p.sum(), q / q.sum())) / np.log(2))


_SURVIVAL_CUTOFF = 1e-9  # matches stochprocq.meta.StochModel._DEFAULT_CUTOFF


def generate_quantum_model(output_distribution) -> RenewalProcess:
    """Convert a quantum output distribution to an equivalent RenewalProcess.

    Floors negative bins to 0 -- background-subtracted experimental counts
    can dip slightly below zero on shot noise alone, and an unclamped
    negative q_emit[i] propagates straight through to a negative entry in
    RenewalProcess's transition matrix. A real 0 is left untouched -- this
    only catches values strictly below it.

    q_survive_st (the per-step conditional survival probability, "exits
    now" fraction of whatever's left) is clamped at every step of its own
    recursion too, not just the raw input: a noisy/truncated
    output_distribution that doesn't sum to exactly 1 (e.g. a bin dropped
    for being negative, without renormalizing the rest) can push the
    recursion's running product past 1 partway through, which flips later
    steps negative even though every q_emit[i] going in was already >= 0
    -- same failure, one step downstream.

    That clamp lands in [_SURVIVAL_CUTOFF, 1 - _SURVIVAL_CUTOFF], not
    [0, 1] -- an exact 0 or 1 here means some branch of the resulting
    RenewalProcess is permanently unreachable, and eval_diverge divides by
    that branch's total probability when comparing against a theory model
    that still considers it reachable (0/0 -> nan, poisoning the whole
    divergence sum). A noisy experimental fit landing on a hard boundary
    is a numerical artifact, not a real claim that something is
    impossible, so it's kept a hair off the edge instead. Exact 0/1 are
    still correct for the closed-form theory models (get_uniform_renewal
    etc.) -- this clamp is local to the experimental-data path only.
    """
    q_emit = np.clip(np.asarray(output_distribution, dtype=float), 0, None)
    q_survive_st = np.zeros_like(q_emit)
    q_survive_st[0] = np.clip(q_emit[0], _SURVIVAL_CUTOFF, 1 - _SURVIVAL_CUTOFF)
    for i in range(1, len(q_emit)):
        q_survive_st[i] = np.clip(q_emit[i] / np.prod(1 - q_survive_st[:i]),
                                   _SURVIVAL_CUTOFF, 1 - _SURVIVAL_CUTOFF)
    return RenewalProcess([1 - q for q in q_survive_st[:-1]])


def exit_distribution(probs, start: int = 0) -> np.ndarray:
    """Exit-loop probabilities for a RenewalProcess's `probs` (per-loop
    survival probability), given it has already survived `start` loops
    (renewal state `start`) without exiting.

    Returns an array of length len(probs)+1: index j is P(exit at loop j+1)
    for j >= start, plus a final catch-all entry for surviving every
    remaining loop. Entries before `start` are 0 — the process can't exit
    at a loop it's already passed.
    """
    exit_p = np.zeros(len(probs) + 1)
    survived = 1.0
    for j in range(start, len(probs)):
        exit_p[j] = (1 - probs[j]) * survived
        survived *= probs[j]
    exit_p[len(probs)] = survived
    return exit_p


def kl_rate_from_output_distribution(N: int, output_distribution) -> float:
    """KL-divergence rate (bits/step) between the N-loop uniform renewal
    theory model and an experimental output distribution over exactly N
    bins (loop-pass 1..N-1, plus a final catch-all "everything else"
    bin -- NOT N+1 separate loop-pass-then-dump bins; generate_quantum_model
    builds an N-state RenewalProcess from an N-entry input, matching
    get_uniform_renewal(N-1)'s own N states). Every real caller already
    truncates its (N+1)-or-more-channel array to exactly N entries before
    calling this -- e.g. frac.to_numpy()[:N] -- so matching that length
    here is required, not optional; a wrong-length input gives a silently
    different (wrong) number, not an error. The "_get_KL_rate" calc used
    ad hoc in demos/marimo/plot_experimental_data.py, promoted to a real
    function so it has one definition instead of being re-copy-pasted into
    every notebook/live-measurement caller."""
    theory = get_uniform_renewal(N - 1)
    model = generate_quantum_model(output_distribution)
    p_dist = theory.steady_state @ theory.gen_dists(N + 1)
    q_dist = model.steady_state @ model.gen_dists(N + 1)
    return eval_diverge(p_dist, q_dist, his_steps=N - 1)


def kl_rate_mc_error(N: int, rates, rate_stds, T_measured: float, T_measured_std: float,
                      bg, eff, bins: list[str], bg_std=None, n_samples: int = 200,
                      rng: np.random.Generator | None = None,
                      check_convergence: bool = False):
    """Monte Carlo (mean, std) of kl_rate_from_output_distribution under the
    real sources of uncertainty in the loss-corrected output distribution:
    shot noise in the measured per-bin rates (rates/rate_stds, e.g. from
    rates_by_input_state), uncertainty in the measured per-loop transmission
    T (T_measured/T_measured_std, e.g. from loss_rates_for's loop0_hz/
    loop1_hz SEMs), and -- since 2026-11-23 -- uncertainty in the background
    estimate itself (bg_std, e.g. from background_rate_std_by_channel).
    Background matters most exactly where it's least certain: the loss
    correction divides by T**k, which at high k (late loop passes) can be
    ~1000x smaller than at k=1, so a channel where background is a sizeable
    fraction of the raw rate gets that uncertainty amplified by the same
    ~1000x in the final bar -- omitting it was understating the real error
    on exactly the channels most worth worrying about. bg_std defaults to
    None (no background resampling, the pre-2026-11-23 behaviour) since
    it's genuinely not always available (background_rate_std_by_channel
    returns {} when there's no per-bin breakdown to compute it from).

    Each sample redraws rates/T/background from their already-computed
    Normal(mean, SEM) -- the SEM already *is* the shot-noise/root-n spread,
    so resampling from it reproduces that noise without re-deriving Poisson
    statistics from raw per-bin counts the pipeline doesn't keep -- and
    redoes the full loss-correction + KL-rate calc, matching the method
    designed (but never committed) in plot_experimental_data.py's 2026-11-20
    comments: T is clipped to [0.01, 0.99] per that design, since T outside
    that range is a numerical artifact of the resampling, not a real reading.

    check_convergence=True additionally splits the n_samples draws already
    taken into two independent halves and compares their std()s -- cheap
    (no extra sampling, just re-slicing what was already computed), and
    catches the same failure mode this file's own history already hit
    once: a large T_measured_std can give eval_diverge's output a heavy
    right tail, where std() at "the usual" sample count varies run to run
    by up to 2x and only settles down at 10x more samples (see
    plot_experimental_data.py's 2026-11-20 comment on the 11-19 series,
    which needed 20000 samples instead of 2000 for exactly this reason).
    Returns (mean, std, converged) instead of (mean, std) when enabled --
    converged=False means the two halves disagree by >20%, i.e. std is not
    yet trustworthy at this n_samples and should be rerun with more.
    """
    rng = rng or np.random.default_rng()
    rates = np.asarray(rates, dtype=float)
    rate_stds = np.asarray(rate_stds, dtype=float)
    bg = np.asarray(bg, dtype=float)
    bg_std = np.zeros_like(bg) if bg_std is None else np.asarray(bg_std, dtype=float)
    eff = np.asarray(eff, dtype=float)
    k = np.array([N if b == "coinc_ch7" else loop_pass_k(b, bins) for b in bins])
    kls = np.empty(n_samples)
    for i in range(n_samples):
        T = np.clip(rng.normal(T_measured, T_measured_std), 0.01, 0.99)
        sampled_rates = rng.normal(rates, rate_stds)
        sampled_bg = rng.normal(bg, bg_std)
        corr = ((sampled_rates - sampled_bg) / eff) / (T ** k)
        frac = np.clip(corr, 0, None)
        frac = (frac / frac.sum())[:N]
        kls[i] = kl_rate_from_output_distribution(N, frac)
    mean, std = float(kls.mean()), float(kls.std())
    if not check_convergence:
        return mean, std
    half = n_samples // 2
    std_a, std_b = kls[:half].std(), kls[half:2 * half].std()
    converged = abs(std_a - std_b) <= 0.20 * max(std_a, std_b, 1e-12)
    return mean, std, converged


def _demo():
    import tempfile

    # background_rate_by_channel: inline "background" field takes priority
    # over an embedded/pointer noise_calibration when more than one is present
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cal_path = tmp / "cal.json"
        cal_path.write_text(json.dumps({"background_rate_hz": {"2": 99.0}}))
        run_path = tmp / "run.json"
        run_path.write_text(json.dumps({
            "background": {"coinc_ch2": 20, "coinc_ch4": 10, "int_time": 10.0},
            "background_offset_ns": 20.0,
            "noise_calibration": {"file": "cal.json", "saved_at": "x"},
        }))
        by_ch, note = background_rate_by_channel(run_path)
        assert by_ch == {"2": 2.0, "4": 1.0}
        assert "inline" in note

        # no inline field, noise_calibration is the entire calibration file
        # embedded directly (current measurement.py behaviour) -> read
        # background_rate_hz straight off it, no second file involved
        run_path.write_text(json.dumps({
            "noise_calibration": {"file": "cal.json", "saved_at": "x", "background_rate_hz": {"2": 42.0}},
        }))
        by_ch2, note2 = background_rate_by_channel(run_path)
        assert by_ch2 == {"2": 42.0}
        assert note2.startswith("embedded")

        # older shape: noise_calibration is just a {file, saved_at} pointer
        # -> falls back to resolving that file next to run_path
        run_path.write_text(json.dumps({"noise_calibration": {"file": "cal.json", "saved_at": "x"}}))
        by_ch3, note3 = background_rate_by_channel(run_path)
        assert by_ch3 == {"2": 99.0}
        assert note3.startswith("cal.json")

        # nothing present -> empty, with an explanatory note
        run_path.write_text(json.dumps({}))
        by_ch4, note4 = background_rate_by_channel(run_path)
        assert by_ch4 == {}
        assert "none" in note4
    print("background_rate_by_channel: ok")

    # loss_calibration_for: embedded "loss_calibration" field on the
    # measurement's own json takes priority over a directory-wide fallback
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "20260101_000000_loss_calibration.json").write_text(
            json.dumps({"losses": {"det_eff_dump": 0.5}}))
        run_path = tmp / "run.json"
        run_path.write_text(json.dumps({
            "loss_calibration": {"file": "20260101_000000_loss_calibration.json", "saved_at": "x",
                                  "losses": {"det_eff_dump": 0.9}},
        }))
        losses, note = loss_calibration_for(run_path)
        assert losses == {"det_eff_dump": 0.9}
        assert note.startswith("embedded")

        # no embedded field -> falls back to the most recent file in the
        # same directory (older measurements, taken before the embed existed)
        run_path.write_text(json.dumps({}))
        losses2, note2 = loss_calibration_for(run_path)
        assert losses2 == {"det_eff_dump": 0.5}
        assert "dir fallback" in note2

        # neither present -> empty, with an explanatory note
        (tmp / "20260101_000000_loss_calibration.json").unlink()
        losses3, note3 = loss_calibration_for(run_path)
        assert losses3 == {}
        assert "none" in note3
    print("loss_calibration_for: ok")

    # loss_rates_for: same embedded-first lookup, but pulling
    # coincidence_rates_hz instead of losses
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_path = tmp / "run.json"
        run_path.write_text(json.dumps({
            "loss_calibration": {"file": "x.json", "saved_at": "x",
                                  "losses": {"det_eff_dump": 0.9},
                                  "coincidence_rates_hz": {"loop0_hz": 10.0, "loop1_hz": 4.0}},
        }))
        rates, note = loss_rates_for(run_path)
        assert rates == {"loop0_hz": 10.0, "loop1_hz": 4.0}
        assert note.startswith("embedded")

        # real measurement.py embed shape: carries "losses"/"stages" but
        # never "coincidence_rates_hz" -- loss_calibration_for still uses
        # the embed (it only needs "losses"), but loss_rates_for must fall
        # through to the directory file instead of silently returning {}
        (tmp / "20260101_000000_loss_calibration.json").write_text(json.dumps({
            "coincidence_rates_hz": {"loop0_hz": 99.0, "loop1_hz": 33.0},
        }))
        run_path.write_text(json.dumps({
            "loss_calibration": {"file": "20260101_000000_loss_calibration.json", "saved_at": "x",
                                  "losses": {"det_eff_dump": 0.9}, "stages": {}},
        }))
        losses4, note4 = loss_calibration_for(run_path)
        assert losses4 == {"det_eff_dump": 0.9} and note4.startswith("embedded")
        rates2, note5 = loss_rates_for(run_path)
        assert rates2 == {"loop0_hz": 99.0, "loop1_hz": 33.0}
        assert "dir fallback" in note5
    print("loss_rates_for: ok")

    # loss_rates_for's directory fallback: an OLDER measurement (no embed)
    # must pick the calibration that was actually in effect at ITS OWN
    # time, not whichever calibration is newest in the whole directory --
    # confirmed wrong on real data before this fix (a 2026-11-21 measurement
    # was silently picking up a 2026-11-23 calibration taken two days later)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "20260101_000000_loss_calibration.json").write_text(
            json.dumps({"coincidence_rates_hz": {"loop0_hz": 10.0, "loop1_hz": 1.0}}))
        (tmp / "20260103_000000_loss_calibration.json").write_text(
            json.dumps({"coincidence_rates_hz": {"loop0_hz": 999.0, "loop1_hz": 999.0}}))
        old_run = tmp / "20260102_000000_measurement_N3_sall.json"
        old_run.write_text(json.dumps({}))
        rates3, note6 = loss_rates_for(old_run)
        assert rates3 == {"loop0_hz": 10.0, "loop1_hz": 1.0}, rates3
        assert "nearest prior" in note6 and "20260101" in note6

        # a measurement from BEFORE every calibration file -> no valid prior,
        # falls all the way back to dir-wide latest, clearly labelled as such
        earliest_run = tmp / "20251231_000000_measurement_N3_sall.json"
        earliest_run.write_text(json.dumps({}))
        rates4, note7 = loss_rates_for(earliest_run)
        assert rates4 == {"loop0_hz": 999.0, "loop1_hz": 999.0}
        assert "no earlier calibration found" in note7
    print("loss_rates_for (nearest-prior fallback): ok")

    # ordinary run: no basis column at all -> plain per-state mean/SEM
    plain = pd.DataFrame({
        "input_state": [0, 0, 1, 1],
        "int_time": [1.0, 1.0, 1.0, 1.0],
        "a": [10.0, 20.0, 5.0, 5.0],
    })
    rates, stds = rates_by_input_state(plain, ["a"])
    assert rates.loc[0, "a"] == 15.0 and rates.loc[1, "a"] == 5.0
    assert stds.loc[1, "a"] == 0.0

    # ordinary run with an all-NaN basis column (the normal case for real
    # measurement files) -> same as no basis column at all
    plain["basis"] = np.nan
    rates2, _ = rates_by_input_state(plain, ["a"])
    assert rates2.equals(rates)

    # tomo run: two rows per basis, all bases equal -> pair sums (and hence
    # their average) should reproduce that same total
    tomo = pd.DataFrame({
        "input_state": [0] * 12,
        "int_time": [1.0] * 12,
        "basis": ["H", "H", "V", "V", "A", "A", "D", "D", "R", "R", "L", "L"],
        "a": [3.0, 5.0, 3.0, 5.0, 4.0, 4.0, 4.0, 4.0, 2.0, 6.0, 2.0, 6.0],
    })
    rates3, stds3 = rates_by_input_state(tomo, ["a"])
    assert rates3.loc[0, "a"] == 8.0  # every pair sums to (3+5)/1 each side -> 8
    assert stds3.loc[0, "a"] > 0

    # tomo run where the 3 pair-sums disagree (H+V != A+D != R+L) -- the std
    # must pick up that between-pair spread, not just the within-basis
    # shot noise, since the spread is what actually dominates on real data
    tomo2 = pd.DataFrame({
        "input_state": [0] * 12,
        "int_time": [1.0] * 12,
        "basis": ["H", "H", "V", "V", "A", "A", "D", "D", "R", "R", "L", "L"],
        "a": [9.0, 11.0, 9.0, 11.0, 8.0, 10.0, 8.0, 10.0, 10.0, 12.0, 8.0, 10.0],
    })
    rates4, stds4 = rates_by_input_state(tomo2, ["a"])
    pair_sums = np.array([20.0, 18.0, 20.0])  # H+V, A+D, R+L means
    assert np.isclose(rates4.loc[0, "a"], pair_sums.mean())
    between_sem = pair_sums.std(ddof=1) / np.sqrt(3)
    assert stds4.loc[0, "a"] > between_sem * 0.9  # between-pair spread dominates and is captured
    print("rates_by_input_state: ok")

    # block_rate_mean_std: block-averaged SEM must land close to the naive
    # row-by-row SEM in expectation (same underlying noise, just grouped),
    # but with noticeably less run-to-run scatter -- the whole point of
    # averaging blocks of rows before computing the spread, rather than
    # trusting single rows' Poisson noise directly.
    _rng = np.random.default_rng(0)
    _n_rows = 3000
    _block_df = pd.DataFrame({
        "a": _rng.poisson(5.0, _n_rows).astype(float),
        "int_time": np.ones(_n_rows),
    })
    _naive_mean, _naive_sem = rates_by_input_state(
        pd.DataFrame({**_block_df, "input_state": 0}), ["a"])
    _block_mean, _block_sem = block_rate_mean_std(_block_df, ["a"], block_size=30)
    assert np.isclose(_block_mean["a"], _naive_mean.loc[0, "a"], rtol=0.05)
    assert _block_sem["a"] > 0
    # a block_size of 1 must reduce to exactly the per-row SEM (no grouping)
    _row_mean, _row_sem = block_rate_mean_std(_block_df, ["a"], block_size=1)
    assert np.isclose(_row_sem["a"], _naive_sem.loc[0, "a"], rtol=1e-9)
    # fewer rows than one block must raise, not silently return NaN
    try:
        block_rate_mean_std(_block_df.iloc[:5], ["a"], block_size=30)
        assert False, "expected ValueError for too few rows"
    except ValueError:
        pass
    print("block_rate_mean_std: ok")

    # drop_count_outliers: a single catastrophic spike on one channel must
    # be removed, normal rows and a legitimately-near-zero channel
    # (dump-bin style) must survive untouched
    _outlier_df = pd.DataFrame({
        "coinc_ch4": [7000, 7100, 7050, 6900],
        "coinc_ch7": [0, 1, 0, 0],  # near-zero median -> floor, not ratio, guards it
    })
    _outlier_df.loc[2, "coinc_ch4"] = 500_000  # the spike
    _cleaned = drop_count_outliers(_outlier_df, ["coinc_ch4", "coinc_ch7"])
    assert len(_cleaned) == 3
    assert 500_000 not in _cleaned["coinc_ch4"].to_numpy()
    print("drop_count_outliers: ok")

    # exit_distribution: starting fresh (state 0), a 3-loop process with
    # survival probs [0.5, 0.5] must exit somewhere -> sums to 1, and
    # matches hand-computed per-loop exit probabilities
    dist0 = exit_distribution([0.5, 0.5], start=0)
    assert np.isclose(dist0.sum(), 1.0)
    assert np.allclose(dist0, [0.5, 0.25, 0.25])

    # starting already having survived 1 loop -> can't exit at loop 0 anymore
    dist1 = exit_distribution([0.5, 0.5], start=1)
    assert dist1[0] == 0.0
    assert np.isclose(dist1.sum(), 1.0)
    assert np.allclose(dist1, [0.0, 0.5, 0.5])
    print("exit_distribution: ok")

    # loop_pass_k: position among recorded loop channels, not channel
    # arithmetic -- must agree for both the pre-2026-11-16 (ch2=pass1) and
    # post (ch4=pass1) wiring, since the same bin list shape (ascending
    # loop channels, dump last) occurs under both conventions
    old_bins = ["coinc_ch2", "coinc_ch4", "coinc_ch6", "coinc_ch7"]
    assert [loop_pass_k(b, old_bins) for b in old_bins[:-1]] == [1, 2, 3]
    new_bins = ["coinc_ch4", "coinc_ch6", "coinc_ch8", "coinc_ch10", "coinc_ch12", "coinc_ch14", "coinc_ch7"]
    assert [loop_pass_k(b, new_bins) for b in new_bins[:-1]] == [1, 2, 3, 4, 5, 6]
    print("loop_pass_k: ok")

    # generate_quantum_model: a background-over-subtracted bin going
    # slightly negative must not produce a negative transition-matrix
    # entry (RenewalProcess._check_valid raises ValueError on that), and
    # must land on exactly the same result as a real 0 there -- clamping
    # doesn't perturb an already-valid 0, it only catches what's below it
    model = generate_quantum_model([0.5, -0.02, 0.3, 0.22])
    assert not np.any(model.transitions < 0)
    model_with_zero = generate_quantum_model([0.5, 0.0, 0.3, 0.22])
    assert np.allclose(model.probs, model_with_zero.probs)

    # same failure one step downstream: every bin here is individually
    # >= 0, but they don't sum to 1 (a negative bin got dropped elsewhere
    # without renormalizing the rest, e.g. a 6-channel distribution
    # truncated to N=4 after losing a negative ch12) -- the running
    # survival-probability product overshoots 1 partway through and goes
    # negative from there on, unless q_survive_st itself is also clamped
    real_n4_case = generate_quantum_model([0.32116471, 0.43705113, 0.25965067, 0.03681412])
    assert not np.any(real_n4_case.transitions < 0)
    print("generate_quantum_model: ok")

    # kl_rate_from_output_distribution: the uniform renewal theory model
    # compared against ITSELF must be ~0 (same process, no divergence);
    # a visibly different distribution must come back clearly positive
    N4 = 4
    uniform_theory = get_uniform_renewal(N4 - 1)
    self_dist = exit_distribution(uniform_theory.probs, start=0)
    assert kl_rate_from_output_distribution(N4, self_dist) < 1e-6
    skewed_dist = np.array([0.7, 0.1, 0.1, 0.1])
    assert kl_rate_from_output_distribution(N4, skewed_dist) > 0.05
    # exactly N entries required, NOT N+1 -- a caller passing one extra
    # "dump bin" entry (matching an easy misreading of the docstring
    # before it was corrected) gets a silently different number, not an
    # error, so this is worth locking in explicitly as a real difference
    wrong_length_dist = np.array([0.7, 0.1, 0.1, 0.05, 0.05])  # N4+1 entries
    assert not np.isclose(kl_rate_from_output_distribution(N4, skewed_dist),
                           kl_rate_from_output_distribution(N4, wrong_length_dist))
    print("kl_rate_from_output_distribution: ok")

    # kl_rate_mc_error: tight SEMs (near-exact rates/T) must collapse the MC
    # spread to ~0 and recover close to the point-estimate KL; wide SEMs
    # must blow the spread back up -- std() actually tracks input uncertainty
    _bins = ["coinc_ch4", "coinc_ch6", "coinc_ch8", "coinc_ch7"]
    _rates = np.array([40.0, 12.0, 4.0, 1.0])
    _bg = np.zeros(4)
    _eff = np.ones(4)
    _T, _rng = 0.3, np.random.default_rng(0)
    mean_tight, std_tight = kl_rate_mc_error(
        N4, _rates, _rates * 1e-6, _T, 1e-6, _bg, _eff, _bins, n_samples=200, rng=_rng)
    point_estimate = kl_rate_from_output_distribution(
        N4, np.clip(_rates / _T ** np.array([1, 2, 3, 4]), 0, None) /
        np.clip(_rates / _T ** np.array([1, 2, 3, 4]), 0, None).sum())
    assert std_tight < 1e-3
    assert np.isclose(mean_tight, point_estimate, atol=1e-2)
    _, std_wide = kl_rate_mc_error(
        N4, _rates, _rates * 0.3, _T, 0.1, _bg, _eff, _bins, n_samples=200, rng=_rng)
    assert std_wide > std_tight

    # background uncertainty: with everything else tight, a sizeable
    # background SEM on the weakest (highest-k, most loss-amplified)
    # channel alone must still blow the spread up -- this is the whole
    # point of adding it (2026-11-23: a residual that "looked" certain was
    # silently ignoring exactly this)
    _bg_std = np.array([0.0, 0.0, 0.0, 0.3])  # all on the weak ch7/pass-4 bin
    _, std_bg = kl_rate_mc_error(
        N4, _rates, _rates * 1e-6, _T, 1e-6, _bg, _eff, _bins,
        bg_std=_bg_std, n_samples=200, rng=_rng)
    assert std_bg > std_tight
    print("kl_rate_mc_error: ok")

    # check_convergence: plenty of samples on a tight-SEM case must agree
    # between its two halves (converged=True); forcing the ultra-wide
    # T_measured_std from the earlier "wide" case down to very few samples
    # should trip the other way, letting the check actually catch a real
    # under-sampled case rather than always reporting converged=True
    _, _, _converged_tight = kl_rate_mc_error(
        N4, _rates, _rates * 1e-6, _T, 1e-6, _bg, _eff, _bins,
        n_samples=2000, rng=np.random.default_rng(1), check_convergence=True)
    assert _converged_tight
    _any_unconverged = False
    for _seed in range(10):
        _, _, _c = kl_rate_mc_error(
            N4, _rates, _rates * 0.3, _T, 0.3, _bg, _eff, _bins,
            n_samples=20, rng=np.random.default_rng(_seed), check_convergence=True)
        _any_unconverged = _any_unconverged or not _c
    assert _any_unconverged, "convergence check never fired across 10 seeds at n_samples=20 -- too lax"
    print("kl_rate_mc_error (check_convergence): ok")


if __name__ == "__main__":
    _demo()