import numpy as np
import itertools
import os

from stochprocsim.models.causal_models import CS_3
from stochprocsim.libraries.optics_lib import HWP_p1, HWP_p2, QWP_p1, QWP_p2, PBS, M4

np.random.seed(42)
Nsamples = 5
Model = CS_3


def gaussian_values(mean, sigma, size=3):
    return np.random.normal(mean, sigma, size)


def gen_gauss_optics(angles, samples, fwhm_degrees=1.0):
    """Generate Gaussian-distributed optical element parameters."""
    sigma = np.deg2rad(fwhm_degrees) / (2 * np.sqrt(2 * np.log(2)))
    return (
        gaussian_values(angles['θhin2'], sigma, samples),
        gaussian_values(angles['θqin2'], sigma, samples),
        gaussian_values(angles['θh1'],   sigma, samples),
        gaussian_values(angles['θq1'],   sigma, samples),
        gaussian_values(angles['θh2'],   sigma, samples),
        gaussian_values(angles['θq2'],   sigma, samples),
        gaussian_values(angles['θhf2'],  sigma, samples),
        gaussian_values(angles['θqf2'],  sigma, samples),
        gaussian_values(angles['θhf1'],  sigma, samples),
        gaussian_values(angles['θqf1'],  sigma, samples),
    )


def gen_gauss_unitaries(angles, samples, fwhm_degrees=1.0):
    """Yield Gaussian-sampled 4×4 unitaries."""
    (
        θh1_values, θq1_values, θh12_values, θq12_values,
        θhin2_values, θqin2_values, θhf1_values, θqf1_values,
        θhf2_values, θqf2_values
    ) = gen_gauss_optics(angles, samples, fwhm_degrees)

    for values in itertools.product(
        θh1_values, θq1_values, θh12_values, θq12_values,
        θhin2_values, θqin2_values, θhf1_values, θqf1_values,
        θhf2_values, θqf2_values
    ):
        θh1_v, θq1_v, θh2_v, θq2_v, θhin2_v, θqin2_v, θhf1_v, θqf1_v, θhf2_v, θqf2_v = values

        Utot = (
            np.exp(1j * 3.92699)
            * HWP_p2(θhf2_v) @ QWP_p2(θqf2_v) @ HWP_p1(θhf1_v) @ QWP_p1(θqf1_v)
            @ PBS.conj().T
            @ M4(np.pi, 0) @ M4(np.pi, 0) @ M4(np.pi, np.pi) @ M4(0, np.pi) @ M4(0, np.pi)
            @ HWP_p2(θh2_v) @ QWP_p2(θq2_v) @ HWP_p2(θh2_v) @ QWP_p2(θq2_v)
            @ PBS
            @ HWP_p2(θhin2_v) @ QWP_p2(θqin2_v)
        )

        yield np.array(Utot, dtype=np.complex128), {
            "θh1": θh1_v, "θq1": θq1_v,
            "θh2": θh2_v, "θq2": θq2_v,
            "θhin2": θhin2_v, "θqin2": θqin2_v,
            "θhf1": θhf1_v, "θqf1": θqf1_v,
            "θhf2": θhf2_v, "θqf2": θqf2_v,
        }


def generate_unitaries_npz(model, samples, fwhm_degrees=1.0):
    """Save Gaussian-sampled unitaries and angles to a .npz file."""
    angles = model.angles_NTU.copy()
    directory = "Data"
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{model.name}_samples_{samples}.npz")

    unitaries, angles_list = [], []
    for U, angs in gen_gauss_unitaries(angles, samples, fwhm_degrees):
        unitaries.append(U)
        angles_list.append(list(angs.values()))

    np.savez(filename,
             U=np.array(unitaries, dtype=np.complex128),
             angles=np.array(angles_list, dtype=np.float64))
    print(f"Saved {len(unitaries)} unitaries to '{filename}'")


if __name__ == "__main__":
    generate_unitaries_npz(Model, samples=Nsamples)