import re

import numpy as np
from scipy.optimize import differential_evolution

from stochprocsim.Libraries.OpticsLib import HWP_p2, QWP_p2
from stochprocsim.Models.CausalModels import Causal_Models

INPUT = np.array([[1], [0], [0], [0]], dtype=complex)
BOUNDS = [(0, 360), (0, 180)]


def to_numpy(state) -> np.ndarray:
    """Convert a state vector to a complex numpy array (handles SymPy or numpy input)."""
    try:
        return np.array(
            [[complex(v) for v in row] for row in state.tolist()],
            dtype=complex,
        )
    except AttributeError:
        return np.asarray(state, dtype=complex)


def fidelity(psi: np.ndarray, target: np.ndarray) -> float:
    """Global-phase-insensitive overlap |⟨target|psi⟩|²."""
    psi = psi.flatten() / np.linalg.norm(psi)
    target = target.flatten() / np.linalg.norm(target)
    return float(np.abs(np.vdot(target, psi)) ** 2)


def build_state(x) -> np.ndarray:
    θ_hwp, θ_qwp = x
    return HWP_p2(θ_hwp) @ QWP_p2(θ_qwp) @ INPUT


def objective(x, target: np.ndarray) -> float:
    return 1.0 - fidelity(build_state(x), target)


def solve_state(target_state, seed=0):
    target = to_numpy(target_state)
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        args=(target,),
        seed=seed,
        polish=True,
        maxiter=200,
        popsize=10,
        tol=1e-9,
    )
    θ_hwp, θ_qwp = result.x
    return {"θhwp": θ_hwp, "θqwp": θ_qwp}, 1.0 - result.fun


def phase_align(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate psi by the global phase that makes ⟨target|psi⟩ real and positive."""
    psi_n = psi.flatten() / np.linalg.norm(psi)
    target_n = target.flatten() / np.linalg.norm(target)
    overlap = np.vdot(target_n, psi_n)
    return psi * np.exp(-1j * np.angle(overlap))


def parse_mathematica(s: str) -> np.ndarray:
    """Parse a Mathematica-format column vector string into a numpy array."""
    s = s.replace(" ", "").replace("I", "j")
    entries = re.findall(r'\{([^{}]*)\}', s)
    values = [complex(e) if e and e != '0' else 0+0j for e in entries]
    return np.array(values, dtype=complex)


def compare_states(psi_a: np.ndarray, psi_b: np.ndarray):
    psi_a = psi_a / np.linalg.norm(psi_a)
    psi_b = psi_b / np.linalg.norm(psi_b)
    fid = abs(np.vdot(psi_a, psi_b)) ** 2
    phase = np.angle(np.vdot(psi_a, psi_b))
    psi_b_aligned = psi_b * np.exp(-1j * phase)
    print(f"Fidelity |<a|b>|²     : {fid:.10f}")
    print(f"Global phase diff     : {np.rad2deg(phase):.4f}°")
    print(f"Distance (post-align) : {np.linalg.norm(psi_a - psi_b_aligned):.2e}")
    print(f"psi_a                 : {np.round(psi_a, 6)}")
    print(f"psi_b (aligned)       : {np.round(psi_b_aligned, 6)}")
    print("Equal up to global phase" if fid > 0.9999 else f"States differ — fidelity {fid:.6f}")


if __name__ == "__main__":
    s0_states = [model.states[0] for model in Causal_Models.values()]

    # Solve all states
    for n, state in enumerate(s0_states):
        angles, F = solve_state(state)
        target_np = to_numpy(state)
        found = build_state([angles["θhwp"], angles["θqwp"]])
        found_aligned = phase_align(found, target_np)
        print(f"── N: {n+3} ──────────────────────────")
        print(f"  θ_HWP = {angles['θhwp']:7.3f}°   θ_QWP = {angles['θqwp']:7.3f}°")
        print(f"  Fidelity : {F:.10f}")
        if F < 0.9999:
            print("  *** WARNING: low fidelity ***")
        print(f"  Found   : {found_aligned.flatten()}")
        print()

    # Compare a specific state against a Mathematica reference
    n = 6
    state = s0_states[n - 3]
    angles, F = solve_state(state)
    target_np = to_numpy(state)
    found = build_state([angles["θhwp"], angles["θqwp"]])
    found_aligned = phase_align(found, target_np)

    mathematica_state = parse_mathematica(
        "{{0.138555 - 0.655664 I}, {0}, {0.693399 - 0.264772 I}, {0}}"
    )
    print(f"\n── N={n} vs Mathematica ──────────────────────────")
    compare_states(found_aligned.flatten(), mathematica_state)
