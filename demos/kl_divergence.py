import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt 
    import math 
    import time 

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.Models.renewal import RenewalProcess
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.Models.SimulationSampler import Simulator
    from stochprocsim.Models.TransitionModel import QuantumTransitionModel, ExactTransitionModel, TransitionModel
    from stochprocsim.Models.CausalModels import Causal_Models
    from stochprocsim.Libraries.OpticsLib import getUtot

    def generate_quantum_model(exp_data:np.array) -> RenewalProcess:
        # Reset probabilities (emitting 1) after k steps starting from state 0 are:
        q_emit = exp_data ## EXPERIEMTNATL DATA GO HERE

        # probability of emitting 0 at each causal state:
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))

        return RenewalProcess([1-q for q in q_survive_st[:-1]]) # the last state always emit 1
    return Causal_Models, get_uniform_renewal, mo, np, sp


@app.cell
def _():
    # N = 3
    # m = get_uniform_renewal(N-1) # 3 causal states
    # cm = Causal_Models[N]

    # cm.set_U(cm.U_theo)
    # qtheo = np.array(Simulator(QuantumTransitionModel(cm)).get_output_distribution())
    # qexact = np.array(Simulator(ExactTransitionModel(cm)).get_output_distribution())

    # print(f'Quantum Theory: {qtheo}')
    # print(f'Exact: {qexact}')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    1. For every state $S_i$ and every symbol $x$ calculate $p(x|S_i)$
    2. Compute the conditional KL at each state$$D_{KL} (p(\dot |S_i) ||q(\dot |S_i)) = \sum_xp(x|S_i)log\frac{p(x|S_i)}{q(x|S_i)}$$
    3. Average (expectation) over states under $P$. Using the stationary state distribution $p(S_i), $$R_{KL}[P||Q]=\sum_i{p(S_i)D_{KL}(p(\dot|S_i)))} \\ =\sum_i{p(S_i)\sum_x{p(x|S_i)log\frac{p(x|S_i)}{q(x|S_i)}}}$$
    """
    )
    return


@app.cell
def _(Causal_Models, get_uniform_renewal, np, sp):
    def getOutput(v):
        path2 = sp.Matrix([[v[0]],[v[2]]])
        path1 = sp.Matrix([[v[1]],[v[3]]])
        α, β = path2.norm(), path1.norm()
        αo = α / (α + β)    
        βo = β / (α + β)  
        print(αo + βo)
        return path2, αo, βo

    def exact_model_prob(N, j):
        epsilon = 1e-19
        p_0 = (N-j-1)/(N-j+epsilon)
        p_1 = 1/(N-j+epsilon)
        return p_0, p_1

    for N in range(3,3+1):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        CS = Causal_Models[N]
        CS.set_U(CS.U_theo)
        stat_probs = exact_model.steady_state
        U = CS.U    
        r_kl = []
        for i in range(N):
            str = '1'
            S = CS.states[i]
            d_kl = []
            prod_p = 1
            prod_q = 1
            n = 0
            for j in range(N - i):
                # p 
                v = U@S
                path2, p0, p1 = getOutput(v)
                p = p1**2 * prod_p
                print(f"p({str}|S_{i}): {(p):0.8f}")
                S = (sp.Matrix([[path2[0]], [0], [path2[1]], [0]]))/p0
                n += p
                prod_p *= p0**2

                # q
                q0, q1 = exact_model_prob(N,j)
                q = q1 * prod_q
                # print(f"q({str}|S_{i}): {(q):0.8f}")
                str = '0' + str
                prod_q *= q0

                D_KL = p1**2 * np.log(float(p)/float(q))
                d_kl.append(D_KL)
                # print(f"(dkl|S_{i}): {(D_KL):0.8f}")
            print(f'Sum of parts {n}')
            # print(f'D_KL s_{i} = {sum(d_kl)}')
            p_si = stat_probs[i]
            r_kl.append(p_si*sum(d_kl))
        print(sum(r_kl))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
