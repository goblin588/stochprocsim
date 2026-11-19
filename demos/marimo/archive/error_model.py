import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.libraries.optics_lib import getUtot
    from stochprocsim.utils import generate_quantum_model
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
    )


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    eval_diverge,
    generate_quantum_model,
    getUtot,
    get_uniform_renewal,
    np,
):
    NPHOTONS = 5000
    SAMPLES = 100

    def sampled_kl(cm):
        """One Poisson-sampled run -> KL divergence rate vs the exact model."""
        N = len(cm)
        counts = Simulator(
            QuantumTransitionModel(cm), nphotons=NPHOTONS
        ).sample_counts(print_outputs=False)
        q_model = generate_quantum_model([row[1] / NPHOTONS for row in counts])
        exact = get_uniform_renewal(N - 1)
        return eval_diverge(
            exact.gen_dists(N)[0], q_model.gen_dists(N)[0], his_steps=N - 1
        )

    def gauss_U(cm):
        """Loop unitary with each waveplate angle jittered by a 1 deg FWHM Gaussian."""
        sigma = 1 / (2 * np.sqrt(2 * np.log(2)))  # degrees
        return getUtot({
            k: np.random.normal(v, sigma) if k.startswith("θ") else v
            for k, v in cm.angles_NTU.items()
        })

    def report(N, gauss=False):
        cm = Causal_Models[N]
        kls = []
        for _ in range(SAMPLES):
            cm.set_U(gauss_U(cm) if gauss else cm.U_theo)
            kls.append(sampled_kl(cm))
        kls = np.array(kls)
        print(f"N_{N} Avg Divergence: {kls.mean()} +/- {kls.std()}")
    return (report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Photon Sampling with Poisson Noise""")
    return


@app.cell
def _(report):
    for _N in range(3, 6 + 1):
        report(_N)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Photon Sampling with Gaussian Waveplate Noise and Poisson Noise""")
    return


@app.cell
def _(report):
    for _N in range(3, 6 + 1):
        report(_N, gauss=True)
    return


if __name__ == "__main__":
    app.run()
