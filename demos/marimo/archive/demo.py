import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.models.simulation_sampler import Simulator
    from stochprocsim.models.transition_model import QuantumTransitionModel, ExactTransitionModel
    from stochprocsim.models.causal_models import Causal_Models
    return (
        Causal_Models,
        ExactTransitionModel,
        QuantumTransitionModel,
        Simulator,
        eval_diverge,
        get_uniform_renewal,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(Causal_Models, np):
    def is_unitary(U, tol=1e-7):
        return np.allclose(U @ U.conj().T, np.eye(4), atol=tol)

    for idx, model in Causal_Models.items():
        model.set_U(model.U_theo)
        print(f"Model {idx} unitary: {is_unitary(model.U)}")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    eval_diverge,
    get_uniform_renewal,
    np,
):
    from stochprocsim.utils import generate_quantum_model

    def simulate_quantum_proc(N):
        exact_model = get_uniform_renewal(N-1)
        CS = Causal_Models[N]
        CS.set_U(CS.U_theo)
        q_sim = Simulator(QuantumTransitionModel(CS))
        q_model = generate_quantum_model(np.array(q_sim.get_output_distribution()))
        p_dist = exact_model.gen_dists(N)[0]
        q_dist = q_model.gen_dists(N)[0]
        return eval_diverge(p_dist, q_dist, his_steps=N-1)

    def simulate_quantum_optics_proc(N):
        exact_model = get_uniform_renewal(N-1)
        CS = Causal_Models[N]
        CS.set_U(CS.U_optics_GU)
        q_sim = Simulator(QuantumTransitionModel(CS))
        q_model = generate_quantum_model(np.array(q_sim.get_output_distribution()))
        p_dist = exact_model.gen_dists(N)[0]
        q_dist = q_model.gen_dists(N)[0]
        return eval_diverge(p_dist, q_dist, his_steps=N-1)

    def simulate_classical_proc(N):
        exact_model = get_uniform_renewal(N-1)
        return exact_model.classical_bd(4, target_dim=2)
    return (
        simulate_classical_proc,
        simulate_quantum_optics_proc,
        simulate_quantum_proc,
    )


@app.cell
def _(
    Causal_Models,
    ExactTransitionModel,
    QuantumTransitionModel,
    Simulator,
    get_uniform_renewal,
    np,
):
    N = 3
    m = get_uniform_renewal(N-1)
    cm = Causal_Models[N]

    cm.set_U(cm.U_optics_GU)
    qopt = np.array(Simulator(QuantumTransitionModel(cm)).get_output_distribution())

    cm.set_U(cm.U_theo)
    qtheo = np.array(Simulator(QuantumTransitionModel(cm)).get_output_distribution())

    qexact = np.array(Simulator(ExactTransitionModel(cm)).get_output_distribution())

    print(f'Quantum Optics: {qopt}')
    print(f'Quantum Theory: {qtheo}')
    print(f'Exact: {qexact}')
    return


@app.cell
def _(
    plt,
    simulate_classical_proc,
    simulate_quantum_proc,
):
    NMAX = 6
    NMIN = 3
    C_BUFF = 1

    x_quant = list(range(NMIN, NMAX + 1))
    x_class = list(range(NMIN, NMAX + C_BUFF))

    y_quantum = [simulate_quantum_proc(n) for n in x_quant]
    y_classical = [simulate_classical_proc(n) for n in x_class]

    qtheo_yerr = [0.002054339006460678, 0.0014301897956533847, 0.0016754397869550568, 0.0015075218255867487]

    palette = ["#3B5BA5", "#800E13", "#E63946"]

    plt.plot(x_class, y_classical, '-s', label='Classical bound', color=palette[0], linewidth=1.5)
    plt.errorbar(x_quant, y_quantum, yerr=qtheo_yerr, fmt='-^', capsize=3, label='Quantum (Target)', color=palette[1], linewidth=1.5)

    plt.ylim(0.005, 0.055)

    ymax = plt.gca().get_ylim()[1]
    plt.fill_between(x_class, y_classical, ymax, alpha=0.3, color='none', edgecolor=palette[0], hatch='///')

    plt.xticks(x_class)
    plt.xlabel('N')
    plt.ylabel('Divergence Rate')
    plt.title('KL Divergence Rate for Process Depth N')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
