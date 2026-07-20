import subprocess
from pathlib import Path

import numpy as np
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