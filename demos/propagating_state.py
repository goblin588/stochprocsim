import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.Models.renewal import RenewalProcess
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.Models.SimulationSampler import Simulator
    from stochprocsim.Models.TransitionModel import QuantumTransitionModel, ExactTransitionModel
    from stochprocsim.Models.CausalModels import Causal_Models

    def generate_quantum_model(exp_data:np.array) -> RenewalProcess:
        q_emit = exp_data
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))
        return RenewalProcess([1-q for q in q_survive_st[:-1]])
    return (
        Causal_Models,
        ExactTransitionModel,
        QuantumTransitionModel,
        Simulator,
        eval_diverge,
        generate_quantum_model,
        get_uniform_renewal,
        np,
        plt,
    )


@app.cell
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    get_uniform_renewal,
    np,
    plt,
):
    N = 4
    exact_model = get_uniform_renewal(N-1)

    CS = Causal_Models[N]
    CS.set_U(CS.U_theo)

    qq_sim = Simulator(QuantumTransitionModel(CS))
    data = qq_sim.get_output_distribution_exp(propagate_outputs=True, include_loss=True)
    data_theory = qq_sim.get_output_distribution_exp()

    x = np.arange(1, len(data) + 1)

    plt.bar(
        x,
        data,
        color="gold",
        edgecolor="black",
        label="With Noise"
    )

    plt.bar(
        x,
        data_theory,
        facecolor="none",
        edgecolor="black",
        linestyle=":",
        linewidth=2,
        label="Theory (Noiseless Model)"
    )

    plt.xlabel("Output bin")
    plt.ylabel("Probability")
    plt.xticks(x)

    plt.legend()
    plt.show()
    return CS, exact_model


@app.cell
def _(
    CS,
    ExactTransitionModel,
    Simulator,
    eval_diverge,
    exact_model,
    generate_quantum_model,
    np,
):
    Nv = 5
    q_sim = Simulator(ExactTransitionModel(CS))
    q_model = generate_quantum_model(np.array(q_sim.get_output_distribution()))
    p_dist = exact_model.gen_dists(Nv)[0]
    q_dist = q_model.gen_dists(Nv)[0]

    print(q_dist)
    kl_div = eval_diverge(p_dist, q_dist, his_steps=Nv-1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
