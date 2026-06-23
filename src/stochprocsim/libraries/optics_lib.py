"""
Optics component matrices in NTU mode convention.

Mode ordering (NTU): indices [0,1,2,3] = [path1_H, path2_H, path1_V, path2_V]
  - path1 polarisation lives at indices 1, 3
  - path2 polarisation lives at indices 0, 2

GU convention reorders to [1, 3, 0, 2]. Use ntu_to_gu() / gu_to_ntu() to
convert unitaries between the two when interfacing with the experimental setup.
"""

import numpy as np

ROUND_TO = 8

# Permutation that maps NTU indices to GU indices
_GU_PERM = np.array([1, 3, 0, 2])
_NTU_PERM = np.argsort(_GU_PERM)   # inverse: GU → NTU


def ntu_to_gu(U: np.ndarray) -> np.ndarray:
    """Convert a 4×4 unitary from NTU to GU mode ordering."""
    return U[np.ix_(_GU_PERM, _GU_PERM)]


def gu_to_ntu(U: np.ndarray) -> np.ndarray:
    """Convert a 4×4 unitary from GU to NTU mode ordering."""
    return U[np.ix_(_NTU_PERM, _NTU_PERM)]


# ── 2×2 single-path components ────────────────────────────────────────────────

def overlap(psi: np.ndarray, basis: np.ndarray) -> float:
    """Compute |⟨basis|psi⟩|²."""
    return float(np.square(np.abs(np.conj(basis) @ psi)))


def HWP(theta: float) -> np.ndarray:
    theta = np.deg2rad(theta)
    c, s = np.round(np.cos(2*theta), ROUND_TO), np.round(np.sin(2*theta), ROUND_TO)
    return (-1j) * np.array([[c, s], [s, -c]])


def QWP(theta: float) -> np.ndarray:
    theta = np.deg2rad(theta)
    c, s = np.round(np.cos(2*theta), ROUND_TO), np.round(np.sin(2*theta), ROUND_TO)
    return (1/np.sqrt(2)) * np.array([[1 - 1j*c, -1j*s], [-1j*s, 1 + 1j*c]])


def Mirror2(phi: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j*phi)]])


# ── 4×4 two-path components (NTU convention) ──────────────────────────────────

PBS = np.array([
    [1, 0,  0,   0  ],
    [0, 1,  0,   0  ],
    [0, 0,  0,   1j ],
    [0, 0,  1j,  0  ],
], dtype=complex)

PBS_dag = PBS.conj().T


def M4(theta1: float, theta2: float) -> np.ndarray:
    return np.array([
        [1, 0, 0,                    0                  ],
        [0, 1, 0,                    0                  ],
        [0, 0, np.exp(1j * theta2),  0                  ],
        [0, 0, 0,                    np.exp(1j * theta1)],
    ], dtype=complex)


def HWP_p1(theta: float) -> np.ndarray:
    """HWP acting on path-1 polarisation (indices 1,3)."""
    theta = np.deg2rad(theta)
    c, s = np.cos(2*theta), np.sin(2*theta)
    return (-1j) * np.array([
        [1j, 0,  0,  0 ],
        [0,  c,  0,  s ],
        [0,  0,  1j, 0 ],
        [0,  s,  0,  -c],
    ], dtype=complex)


def HWP_p2(theta: float) -> np.ndarray:
    """HWP acting on path-2 polarisation (indices 0,2)."""
    theta = np.deg2rad(theta)
    c, s = np.cos(2*theta), np.sin(2*theta)
    return (-1j) * np.array([
        [c,  0,  s,  0 ],
        [0,  1j, 0,  0 ],
        [s,  0,  -c, 0 ],
        [0,  0,  0,  1j],
    ], dtype=complex)


def QWP_p1(theta: float) -> np.ndarray:
    """QWP acting on path-1 polarisation (indices 1,3)."""
    theta = np.deg2rad(theta)
    c, s = np.cos(2*theta), np.sin(2*theta)
    return (1/np.sqrt(2)) * np.array([
        [np.sqrt(2), 0,          0,          0         ],
        [0,          1 - 1j*c,   0,          -1j*s     ],
        [0,          0,          np.sqrt(2), 0         ],
        [0,          -1j*s,      0,          1 + 1j*c  ],
    ], dtype=complex)


def QWP_p2(theta: float) -> np.ndarray:
    """QWP acting on path-2 polarisation (indices 0,2)."""
    theta = np.deg2rad(theta)
    c, s = np.cos(2*theta), np.sin(2*theta)
    return (1/np.sqrt(2)) * np.array([
        [1 - 1j*c,  0,          -1j*s,      0         ],
        [0,         np.sqrt(2), 0,          0         ],
        [-1j*s,     0,          1 + 1j*c,   0         ],
        [0,         0,          0,          np.sqrt(2)],
    ], dtype=complex)


# ── Full loop unitary ──────────────────────────────────────────────────────────

def getUtot(angles: dict) -> np.ndarray:
    """Build the 4×4 loop unitary from a waveplate angle dictionary (NTU convention)."""
    p = angles['pipi']
    U = (
        HWP_p2(angles['θhf2']) @ QWP_p2(angles['θqf2'])
        @ HWP_p1(angles['θhf1']) @ QWP_p1(angles['θqf1'])
        @ PBS_dag
        @ M4(angles['Φm3'], angles['Φm1'])
        @ M4(angles['Φm2'], angles['Φm2'])
        @ M4(angles['Φm1'], angles['Φm3'])
        @ HWP_p2(angles['θh2']) @ QWP_p2(angles['θq2'])
        @ HWP_p1(angles['θh1']) @ QWP_p1(angles['θq1'])
        @ PBS
        @ HWP_p2(angles['θhin2']) @ QWP_p2(angles['θqin2'])
    )
    return np.exp(-1j * p) * U

    
