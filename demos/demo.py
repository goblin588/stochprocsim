import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt 

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.Models.renewal import RenewalProcess
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.Models.SimulationSampler import Simulator
    from stochprocsim.Models.TransitionModel import QuantumTransitionModel, ExactTransitionModel, TransitionModel
    from stochprocsim.Models.CausalModels import Causal_Models
    return (
        Causal_Models,
        ExactTransitionModel,
        QuantumTransitionModel,
        RenewalProcess,
        Simulator,
        eval_diverge,
        get_uniform_renewal,
        np,
        plt,
        sp,
    )


@app.cell(hide_code=True)
def _(Causal_Models, sp):
    def is_unitary(U, tol=1e-7):
        U = sp.Matrix(U)
        prod = U * U.H
        # Check each element numerically
        for i in range(prod.shape[0]):
            for j in range(prod.shape[1]):
                val = complex(prod[i, j])
                if i == j:
                    if abs(val - 1) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
        return True

    for idx, model in Causal_Models.items():
        model.set_U(model.U_theo)
        print(f"Model {idx} unitary: {is_unitary(model.U)}")
    return


@app.cell(hide_code=True)
def _(
    Causal_Models,
    QuantumTransitionModel,
    RenewalProcess,
    Simulator,
    eval_diverge,
    get_uniform_renewal,
    np,
):
    def generate_quantum_model(exp_data:np.array) -> RenewalProcess:
        # Reset probabilities (emitting 1) after k steps starting from state 0 are:
        q_emit = exp_data ## EXPERIEMTNATL DATA GO HERE

        # probability of emitting 0 at each causal state:
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))

        return RenewalProcess([1-q for q in q_survive_st[:-1]]) # the last state always emit 1

    def simulate_quantum_proc(N):
        exact_model = get_uniform_renewal(N-1)

        CS = Causal_Models[N]
        CS.set_U(CS.U_theo)
        q_sim = Simulator(QuantumTransitionModel(CS))
        q_model = generate_quantum_model(np.array(q_sim.get_output_distribution()))
        # print(f'p: {np.array(q_sim.get_quantities())}')
        p_dist = exact_model.gen_dists(N)[0]
        q_dist = q_model.gen_dists(N)[0]
        kl_div = eval_diverge(p_dist, q_dist, his_steps=N-1)
        # print(f'pdist: {p_dist}, \n qdist: {q_dist}')
        # print(f'KL Div rate for {N}: {kl_div}')

        return kl_div

    def simulate_quantum_optics_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        CS = Causal_Models[N]
        CS.set_U(CS.U_optics_GU)
        q_sim = Simulator(QuantumTransitionModel(CS))
        q_model = generate_quantum_model(np.array(q_sim.get_output_distribution()))
        p_dist = exact_model.gen_dists(N)[0]
        q_dist = q_model.gen_dists(N)[0]
        kl_div = eval_diverge(p_dist, q_dist, his_steps=N-1)

        return kl_div

    def simulate_classical_proc(N):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        classical_bd = exact_model.classical_bd(4,target_dim=2)
        return classical_bd
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
    m = get_uniform_renewal(N-1) # 3 causal states
    cm = Causal_Models[N]

    cm.set_U(cm.U_optics_GU)
    # print(cm.U_optics_GU)
    w = np.array(Simulator(QuantumTransitionModel(cm)).get_transition_probabilities())
    qopt = np.array(Simulator(QuantumTransitionModel(cm)).get_output_distribution())

    cm.set_U(cm.U_theo)
    qtheo = np.array(Simulator(QuantumTransitionModel(cm)).get_output_distribution())

    qexact = np.array(Simulator(ExactTransitionModel(cm)).get_output_distribution())

    print(f'Quantum Optics: {qopt}')
    t = 0
    for p in qopt:
        t += p
        print(t)
    print(f'Quantum Theory: {qtheo}')
    print(f'Exact: {qexact}')
    return


@app.cell
def _(
    plt,
    simulate_classical_proc,
    simulate_quantum_optics_proc,
    simulate_quantum_proc,
):
    NMAX = 6
    NMIN = 3
    C_BUFF = 1
    kl_div = []

    x_quant = list(range(NMIN, NMAX + 1))
    x_class = list(range(NMIN, NMAX + C_BUFF))

    y_quantum = [simulate_quantum_proc(n, ) for n in x_quant]
    y_qopt = [simulate_quantum_optics_proc(n) for n in x_quant]
    y_classical = [simulate_classical_proc(n) for n in x_class]

    qtheo_yerr = [0.002054339006460678, 0.0014301897956533847, 0.0016754397869550568, 0.0015075218255867487] # Poisson Err
    qopt_yerr = [0.004107549092911485, 0.0018178942566835653, 0.0020410036273023104, 0.003071878242562731] # Gaussian & Poisson Err # 20 samps


    palette = ["#3B5BA5", "#800E13", "#E63946"]

    plt.plot(x_class, y_classical, '-s', label='Classical bound', color=palette[0], linewidth=1.5)  

    plt.errorbar(x_quant, y_quantum, yerr=qtheo_yerr, fmt='-^', capsize=3, label='Quantum (Target)', color=palette[1], linewidth=1.5)
    # plt.plot(x_quant, y_qopt_GU, '-p', label='Quantum GU (Exp)')
    # plt.errorbar(x_quant, y_qopt, yerr=qopt_yerr, fmt='--o',capsize=3, label='Quantum (Experimental)', color=palette[2], linewidth=1.5) 


    plt.ylim(0.005, 0.055)

    ymax = plt.gca().get_ylim()[1]
    plt.fill_between(x_class, y_classical, ymax, alpha=0.3, color='none', edgecolor=palette[0], hatch='///')

    # plt.plot(x_quant, [0.0841521389913097, 0.0608293450616597, 0.0614688242516527, 0.0559775975679753])

    plt.xticks(x_class)
    plt.xlabel('N')
    plt.ylabel('Divergence Rate')
    # plt.ylim(0,0.1)
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
