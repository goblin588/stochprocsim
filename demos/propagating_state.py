import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
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

    def generate_quantum_model(exp_data:np.array) -> RenewalProcess:
        # Reset probabilities (emitting 1) after k steps starting from state 0 are:
        q_emit = exp_data ## EXPERIEMTNATL DATA GO HERE

        # probability of emitting 0 at each causal state:
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))

        return RenewalProcess([1-q for q in q_survive_st[:-1]]) # the last state always emit 1
    return (
        Causal_Models,
        QuantumTransitionModel,
        Simulator,
        eval_diverge,
        generate_quantum_model,
        get_uniform_renewal,
        np,
        sp,
    )


@app.cell
def _(
    Causal_Models,
    QuantumTransitionModel,
    Simulator,
    get_uniform_renewal,
    sp,
):
    N = 3
    exact_model = get_uniform_renewal(N-1) # 3 causal states

    CS = Causal_Models[N]
    CS.set_U(CS.U_theo)

    s0_p = None

    def transition(v):
        path2 = sp.Matrix([[v[0]],[v[2]]])
        path1 = sp.Matrix([[v[1]],[v[3]]])
        α, β = path2.norm(), path1.norm()
        return path2, α, β

    print(f'Using normal Causal States')
    for i in range(len(CS)):
        s0 = CS.states[i]
        # print(f's{i}: {s0}')
        v = CS.U@s0
        path2, a, b = transition(v)
        print(f"s{i} p0:{(a**2):0.2f} p1:{(b**2):0.2f}")

    print(f'\nPropagating s1')
    s0 = CS.states[0]
    for i in range(len(CS)):
        # print(f's{i}: {s0}')
        v = CS.U@s0
        path2, a, b = transition(v)
        # p11, p12= sp.Matrix([path1[0]]).norm(),sp.Matrix([path1[1]]).norm() 
        print(f"s{i} p0:{(a**2):0.2f} p1:{(b**2):0.2f}")
        s0 = (sp.Matrix([[path2[0]], [0], [path2[1]], [0]]))/a
        # s0.simplify()

    qq_sim = Simulator(QuantumTransitionModel(CS))
    qq_sim.get_output_distribution_propagaged_output_states()
    return CS, N, exact_model


@app.cell
def _(
    CS,
    N,
    QuantumTransitionModel,
    Simulator,
    eval_diverge,
    exact_model,
    generate_quantum_model,
    np,
):

    q_sim = Simulator(QuantumTransitionModel(CS))
    q_model = generate_quantum_model(np.array(q_sim.get_output_distribution()))
    p_dist = exact_model.gen_dists(N)[0]
    q_dist = q_model.gen_dists(N)[0]
    kl_div = eval_diverge(p_dist, q_dist, his_steps=N-1)
    return


if __name__ == "__main__":
    app.run()
