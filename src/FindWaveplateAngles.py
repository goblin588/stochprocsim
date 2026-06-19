# src/solve_state_angles.py
#%%

import numpy as np
from scipy.optimize import differential_evolution

from stochprocsim.Libraries.OpticsLib import HWP_p2, QWP_p2
from stochprocsim.Models.CausalModels import Causal_Models

INPUT = np.array([[1], [0], [0], [0]], dtype=complex)

BOUNDS = [(0, 360), (0, 180)]


def to_numpy(state) -> np.ndarray:
    """
    Convert a state vector to a complex numpy column vector,
    handling both numpy arrays and SymPy matrices.
    """
    try:
        # SymPy matrix: .tolist() gives a nested list of SymPy expressions;
        # complex() forces each element to a Python complex number.
        return np.array(
            [[complex(v) for v in row] for row in state.tolist()],
            dtype=complex
        )
    except AttributeError:
        # Already a numpy array or similar
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
    target = to_numpy(target_state)  # safe conversion regardless of source type

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
    angles = {"θhwp": θ_hwp, "θqwp": θ_qwp}
    fid = 1.0 - result.fun
    return angles, fid


def phase_align(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Rotate psi by the global phase that minimises the angle to target,
    i.e. make ⟨target|psi⟩ real and positive.
    """
    psi_n = psi.flatten() / np.linalg.norm(psi)
    target_n = target.flatten() / np.linalg.norm(target)
    overlap = np.vdot(target_n, psi_n)   # ⟨target|psi⟩ = r·e^{iφ}
    return psi * np.exp(-1j * np.angle(overlap))


# if __name__ == "__main__":
    # s0_states = [model.states[0] for model in Causal_Models.values()]
    # for n, state in enumerate(s0_states):
    #     angles, F = solve_state(state)
    #     target_np = to_numpy(state)
    #     found     = build_state([angles["θqwp"], angles["θhwp"]])
    #     found_aligned = phase_align(found, target_np)

    #     print(f"── N: {n+3} ──────────────────────────")
    #     print(f"  θ_QWP = {angles['θqwp']:7.3f}°   θ_HWP = {angles['θhwp']:7.3f}°")
    #     print(f"  Fidelity : {F:.10f}")
    #     if F < 0.9999:
    #         print("  *** WARNING: low fidelity — target may not be reachable with this optics chain ***")
    #     print(f"  Target  : {target_np.flatten()}")
    #     print(f"  Found   : {found_aligned.flatten()}")
    #     print()

import numpy as np
import re

def parse_mathematica(s: str) -> np.ndarray:
    """
    Parse a Mathematica column vector
    """
    s = s.replace(" ", "").replace("I", "j")
    entries = re.findall(r'\{([^{}]*)\}', s)
    values = []
    for e in entries:
        e = e.strip()
        values.append(complex(e) if e and e != '0' else 0+0j)
    return np.array(values, dtype=complex)

def compare_states(psi_a: np.ndarray, psi_b: np.ndarray):
    psi_a = psi_a / np.linalg.norm(psi_a)
    psi_b = psi_b / np.linalg.norm(psi_b)

    fidelity = abs(np.vdot(psi_a, psi_b)) ** 2
    overlap  = np.vdot(psi_a, psi_b)
    phase    = np.angle(overlap)
    psi_b_aligned = psi_b * np.exp(-1j * phase)
    distance = np.linalg.norm(psi_a - psi_b_aligned)

    print(f"Fidelity |<a|b>|²     : {fidelity:.10f}")
    print(f"Global phase diff     : {np.rad2deg(phase):.4f}°")
    print(f"Distance (post-align) : {distance:.2e}")
    print(f"psi_a                 : {np.round(psi_a, 6)}")
    print(f"psi_b (aligned)       : {np.round(psi_b_aligned, 6)}")
    print("✅ Equal up to global phase" if fidelity > 0.9999 else f"❌ States differ — fidelity {fidelity:.6f}")

# ── Paste here ────────────────────────────────────────────────────────────────
s0_states = [model.states[0] for model in Causal_Models.values()]
n=3
state = s0_states[n-3]
angles, F = solve_state(state)
target_np = to_numpy(state)
found = build_state([angles["θhwp"], angles["θqwp"]])
found_aligned = phase_align(found, target_np)

print(f"── N: {n} ──────────────────────────")
print(f"  θ_QWP = {angles['θqwp']:7.3f}°   θ_HWP = {angles['θhwp']:7.3f}°")
print(f"  Fidelity : {F:.10f}")
if F < 0.9999:
    print("  *** WARNING: low fidelity — target may not be reachable with this optics chain ***")
# print(f"  Target  : {target_np.flatten()}")
# print(f"  Found   : {found_aligned.flatten()}")

python_state = found_aligned.flatten()

mathematica_state = parse_mathematica(
    "{{0.470057 - 0.29022 I}, {0}, {0.528248 - 0.644804 I}, {0}}"
)

compare_states(python_state, mathematica_state)