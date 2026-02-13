import sympy as sp
import numpy as np
import itertools
import os 

from ..CausalModels import *
from .OpticsLib import *

# Define symbols only for convenience
Φm1, Φm2, Φm3 = sp.symbols('Φm1 Φm2 Φm3')
θh1, θq1, θh2, θq2, θhin2, θqin2, θhf1, θqf1, θhf2, θqf2, pipi = sp.symbols(
    'θh1 θq1 θh2 θq2 θhin2 θqin2 θhf1 θqf1 θhf2 θqf2 pipi'
)

np.random.seed(42)  # reproducibility
angle_error = 0
Nsamples = 5
Model = CS_3

def gaussian_values(mean, sigma, size=3):
    """Draw values from Gaussian distribution."""
    return np.random.normal(mean, sigma, size)


def gen_gauss_optics(angles, samples):
    """Generate Gaussian-distributed optical element parameters."""
    fwhm_degrees = 0
    fwhm_radians = np.deg2rad(fwhm_degrees)
    sigma = fwhm_radians / (2 * np.sqrt(2 * np.log(2)))
    return (
        gaussian_values(angles['θhin2'], sigma, samples),
        gaussian_values(angles['θqin2'], sigma, samples),
        gaussian_values(angles['θh1'], sigma, samples),        
        gaussian_values(angles['θq1'], sigma, samples),
        gaussian_values(angles['θh2'], sigma, samples),
        gaussian_values(angles['θq2'], sigma, samples),
        gaussian_values(angles['θhf2'], sigma, samples),        
        gaussian_values(angles['θqf2'], sigma, samples),
        gaussian_values(angles['θhf1'], sigma, samples),        
        gaussian_values(angles['θqf1'], sigma, samples),
        )


def gen_gauss_unitaries(angles, samples):
    """Yield Gaussian-sampled 4x4 unitaries."""
    (
        θh1_values, θq1_values, θh12_values, θq12_values,
        θhin2_values, θqin2_values, θhf1_values, θqf1_values,
        θhf2_values, θqf2_values
    ) = gen_gauss_optics(angles, samples)

    for values in itertools.product(
        θh1_values, θq1_values, θh12_values, θq12_values,
        θhin2_values, θqin2_values, θhf1_values, θqf1_values,
        θhf2_values, θqf2_values
    ):
        θh1_v, θq1_v, θh2_v, θq2_v, θhin2_v, θqin2_v, θhf1_v, θqf1_v, θhf2_v, θqf2_v = values

        # numeric matrices
        M11_eval = M4(np.pi, 0)
        M31_eval = M4(np.pi, 0)
        M2_eval  = M4(np.pi, np.pi)
        M12_eval = M4(0, np.pi)
        M32_eval = M4(0, np.pi)

        Q1_eval   = QWP_p1(θq1_v)
        Qf1_eval  = QWP_p1(θqf1_v)
        Qin2_eval = QWP_p2(θqin2_v)
        Q2_eval   = QWP_p2(θq2_v)
        Qf2_eval  = QWP_p2(θqf2_v)

        H1_eval   = HWP_p1(θh1_v)
        Hf1_eval  = HWP_p1(θhf1_v)
        Hin2_eval = HWP_p2(θhin2_v)
        H2_eval   = HWP_p2(θh2_v)
        Hf2_eval  = HWP_p2(θhf2_v)

        GlobPhase_eval = np.exp(1j * 3.92699)  

        # full unitary product (numeric)
        Utot = (
            GlobPhase_eval
            * Hf2_eval @ Qf2_eval @ Hf1_eval @ Qf1_eval
            @ PBS.conj().T
            @ M32_eval @ M31_eval @ M2_eval @ M12_eval @ M11_eval
            @ H2_eval @ Q2_eval @ H2_eval @ Q2_eval
            @ PBS
            @ Hin2_eval @ Qin2_eval
        )

        yield np.array(Utot, dtype=np.complex128), {
            "θh1": θh1_v, "θq1": θq1_v,
            "θh2": θh2_v, "θq2": θq2_v,
            "θhin2": θhin2_v, "θqin2": θqin2_v,
            "θhf1": θhf1_v, "θqf1": θqf1_v,
            "θhf2": θhf2_v, "θqf2": θqf2_v
        }

def generate_unitaries_npz(model, samples):
    """Save all unitaries and angles to a single .npz file under /data/."""
    unitaries = []
    angles_list = []

    angles = model.angles.copy()
    directory = "Data"
    os.makedirs(directory, exist_ok=True)  
    filename = os.path.join(directory, f"{model.name}_samples_{samples}.npz")

    # print(f"Saving to {filename}")

    for U, angs in gen_gauss_unitaries(angles, samples):
        unitaries.append(U)
        angles_list.append(list(angs.values()))

    unitaries_array = np.array(unitaries, dtype=np.complex128)
    angles_array = np.array(angles_list, dtype=np.float64)

    np.savez(filename, U=unitaries_array, angles=angles_array)
    print(f"Saved {len(unitaries_array)} unitaries to '{filename}'")


if __name__ == "__main__":
    model = Model
    nsamples = Nsamples
    generate_unitaries_npz(model, samples=nsamples)