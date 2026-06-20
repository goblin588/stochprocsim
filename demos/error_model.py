import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.Models.renewal import RenewalProcess
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.Models.SimulationSampler import Simulator
    from stochprocsim.Models.TransitionModel import QuantumTransitionModel
    from stochprocsim.Models.CausalModels import Causal_Models
    from stochprocsim.Libraries.OpticsLib import getUtot

    def generate_quantum_model(exp_data:np.array) -> RenewalProcess:
        q_emit = exp_data
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))
        return RenewalProcess([1-q for q in q_survive_st[:-1]])
    return (
        Causal_Models,
        QuantumTransitionModel,
        Simulator,
        eval_diverge,
        generate_quantum_model,
        getUtot,
        get_uniform_renewal,
        mo,
        np,
        time,
    )


@app.cell
def _(mo):
    mo.md(f"""## Photon Sampling with Poisson Noise""")
    return


@app.cell
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    eval_diverge,
    generate_quantum_model,
    get_uniform_renewal,
    np,
    time,
):
    def sample_poisson_KL(cm, n):
        q_sim = Simulator(QuantumTransitionModel(cm), nphotons=n)
        N = len(cm)
        kl_divs = []

        sampled_counts = q_sim.sample_counts(print_outputs=False)

        loop_sample_vals = [row[1] for row in sampled_counts]
        res = [val / n for val in loop_sample_vals]

        exact_model = get_uniform_renewal(N - 1)
        p_dist = exact_model.gen_dists(N)[0]
        q_model = generate_quantum_model(res)
        q_dist = q_model.gen_dists(N)[0]

        kl_divs.append(
            eval_diverge(p_dist, q_dist, his_steps=N - 1)
        )

        return np.array(kl_divs)

    def poisson(N, samples):
        cm = Causal_Models[N]
        exact_model = get_uniform_renewal(N - 1)
        p_dist = exact_model.gen_dists(N)[0]

        n = 5000

        all_kl = []

        for _ in range(samples):
            kl_vals = sample_poisson_KL(cm, n)
            all_kl.append(np.mean(kl_vals))

        all_kl = np.array(all_kl)
        print(
            f"N_{N} Avg Divergence: "
            f"{all_kl.mean()} +/- {all_kl.std()}"
        )

    samples = 100

    for _N in range(3, 6+1):
        poisson(_N, samples)
        time.sleep(1)
    return sample_poisson_KL, samples


@app.cell
def _(mo):
    mo.md(r"""### Photon Sampling Gaussian and Poisson""")
    return


@app.cell
def _(
    Causal_Models,
    getUtot,
    get_uniform_renewal,
    np,
    sample_poisson_KL,
    samples,
    time,
):
    def gaussian_values(mean, sigma):
        """Draw values from Gaussian distribution."""
        return np.random.normal(mean, sigma)

    def gen_gauss_optics(angles):
        """Generate Gaussian-distributed optical element parameters."""
        fwhm_degrees = 1
        fwhm_radians = np.deg2rad(fwhm_degrees)
        sigma = fwhm_radians / (2 * np.sqrt(2 * np.log(2)))
        return (
            gaussian_values(np.deg2rad(angles['θhin2']), sigma),
            gaussian_values(np.deg2rad(angles['θqin2']), sigma),
            gaussian_values(np.deg2rad(angles['θh1']), sigma),
            gaussian_values(np.deg2rad(angles['θq1']), sigma),
            gaussian_values(np.deg2rad(angles['θh2']), sigma),
            gaussian_values(np.deg2rad(angles['θq2']), sigma),
            gaussian_values(np.deg2rad(angles['θhf2']), sigma),
            gaussian_values(np.deg2rad(angles['θqf2']), sigma),
            gaussian_values(np.deg2rad(angles['θhf1']), sigma),
            gaussian_values(np.deg2rad(angles['θqf1']), sigma)
            )

    def sample_gauss_U(cm):
        hin2, qin2, h1, q1, h2, q2, hf2, qf2, hf1, qf1 = gen_gauss_optics(cm.angles_NTU)
        angles = {
            "θh1": np.rad2deg(h1),
            "θq1": np.rad2deg(q1),
            "θh2": np.rad2deg(h2),
            "θq2": np.rad2deg(q2),
            "θhin2": np.rad2deg(hin2),
            "θqin2": np.rad2deg(qin2),
            "θhf1": np.rad2deg(hf1),
            "θqf1": np.rad2deg(qf1),
            "θhf2": np.rad2deg(hf2),
            "θqf2": np.rad2deg(qf2),
            "pipi" : np.pi,
            "Φm1": 3.16,
            "Φm2": 3.77,
            "Φm3": 3.74
        }
        return getUtot(angles)


    def gauss_poiss_divergence(N, samples):
        cm = Causal_Models[N]
        exact_model = get_uniform_renewal(N - 1)
        p_dist = exact_model.gen_dists(N)[0]

        n = 5000

        all_kl = []

        for _ in range(samples):
            U = sample_gauss_U(cm)
            cm.set_U(U)
            time.sleep(1)

            kl_vals = sample_poisson_KL(cm, n)
            all_kl.append(np.mean(kl_vals))

        all_kl = np.array(all_kl)
        print(
            f"N_{N} Avg Divergence: "
            f"{all_kl.mean()} +/- {all_kl.std()}"
        )

    for _N in range(3, 6+1):
        gauss_poiss_divergence(_N, samples)
    return


if __name__ == "__main__":
    app.run()
