import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from stochprocsim.stochprocq.models.renewal import RenewalProcess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_REPO = "https://github.com/goblin588/autoqstochmeasure.git"
_CLONE_DIR = PROJECT_ROOT / "data" / "autoqstochmeasure"


def pull_remote_data():
    """Clone or fast-forward the measurement repo; return its data/ dir."""
    if (_CLONE_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(_CLONE_DIR), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", DATA_REPO, str(_CLONE_DIR)], check=True)
    return _CLONE_DIR / "data"


DETECTOR_EFFICIENCY = {
    'herald': 0.6,
    'dump': 0.68,
    'loop': 0.755,
}


def efficiency_for_channel(channel: int, herald_ch: int = 3, dump_ch: int = 7) -> float:
    """Which physical detector reads `channel`, hence which DETECTOR_EFFICIENCY
    entry applies. herald_ch/dump_ch default to this setup's TRIGG_CH/DUMP_CH."""
    if channel == herald_ch:
        return DETECTOR_EFFICIENCY['herald']
    if channel == dump_ch:
        return DETECTOR_EFFICIENCY['dump']
    return DETECTOR_EFFICIENCY['loop']


def background_rate_by_channel(json_path: Path) -> tuple[dict, str]:
    """Per-channel background/accidental rate (Hz) for a measurement run,
    keyed by channel number as a string (matches noise_calibration's
    background_rate_hz shape).

    Prefers the inline background reading measurement.py now takes at the
    start of every run — json_path's own "background" field, the freshest
    possible since it's measured seconds before that exact run starts —
    over the older noise_calibration pointer (a separate calibrate_background()
    run, possibly stale, resolved by filename next to json_path). Returns
    ({}, note) with an explanatory note when neither is available (data
    from before either existed).
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

    cal_ref = meta.get("noise_calibration")
    if cal_ref:
        cal_path = json_path.parent / cal_ref["file"]
        if cal_path.exists():
            by_ch = json.loads(cal_path.read_text()).get("background_rate_hz", {})
            return by_ch, f"{cal_ref['file']} ({cal_ref['saved_at']})"

    return {}, "none (background=0)"


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


def generate_quantum_model(output_distribution) -> RenewalProcess:
    """Convert a quantum output distribution to an equivalent RenewalProcess."""
    q_emit = np.asarray(output_distribution, dtype=float)
    q_survive_st = np.zeros_like(q_emit)
    q_survive_st[0] = q_emit[0]
    for i in range(1, len(q_emit)):
        q_survive_st[i] = q_emit[i] / np.prod(1 - q_survive_st[:i])
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


def _demo():
    import tempfile

    # background_rate_by_channel: inline "background" field takes priority
    # over a noise_calibration pointer when both are present
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

        # no inline field -> falls back to the noise_calibration pointer
        run_path.write_text(json.dumps({"noise_calibration": {"file": "cal.json", "saved_at": "x"}}))
        by_ch2, note2 = background_rate_by_channel(run_path)
        assert by_ch2 == {"2": 99.0}
        assert note2.startswith("cal.json")

        # neither present -> empty, with an explanatory note
        run_path.write_text(json.dumps({}))
        by_ch3, note3 = background_rate_by_channel(run_path)
        assert by_ch3 == {}
        assert "none" in note3
    print("background_rate_by_channel: ok")

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


if __name__ == "__main__":
    _demo()