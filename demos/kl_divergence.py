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
        αo = α / sp.sqrt(α**2 + β**2)    
        βo = β / sp.sqrt(α**2 + β**2)  
        print(f'path output norm: {αo**2 + βo**2}')
        return path2, αo, βo

    def exact_model_prob(N, j):
        epsilon = 1e-19
        p_0 = (N-j-1)/(N-j+epsilon)
        p_1 = 1/(N-j+epsilon)
        return p_0, p_1

    for N in range(3,6+1):
        exact_model = get_uniform_renewal(N-1) # 3 causal states
        CS = Causal_Models[N]
        CS.set_U(CS.U_theo)
        stat_probs = exact_model.steady_state
        U = CS.U   

        print(stat_probs)

        r_kl = []
        for i in range(N):
            str = '1'
            S = CS.states[i]
            d_kl = []
            prod_p = 1
            prod_q = 1
            n = 0
            for j in range(N - i):
                ### p(x|Si) 
                v = U@S
                print(f'U:{U}')
                print(f'S:{S}')
                print(f'v:{v}')
                path2, p0, p1 = getOutput(v)
                p = p1**2 * prod_p
                print(f"p({str}|S_{i}): {(p):0.8f}")
                S = (sp.Matrix([[path2[0]], [0], [path2[1]], [0]]))/p0
                n += p
                prod_p *= p0**2

                ### q(x|Si)
                q0, q1 = exact_model_prob(N,j)
                q = q1 * prod_q
                # print(f"q({str}|S_{i}): {(q):0.8f}")
                str = '0' + str
                prod_q *= q0

                ### D_KL p(x|Si)*log(p(x|Si)/q(x|Si))
                D_KL = p1**2 * np.log(float(p)/float(q))
                d_kl.append(D_KL)
                # print(f"(dkl|S_{i}): {(D_KL):0.8f}")
            print(f'Sigma p(x|Si): {n}')

            ### p(Si) \Sigma {p(x|Si)*log(p(x|Si)/q(x|Si))}
            p_si = stat_probs[i]
            r_kl.append(p_si*sum(d_kl))
        print(f'\nR_KL[P][Q]: {sum(r_kl)}\n')
    return


@app.cell
def _(sp):
    I = 1j

    U0 = sp.Matrix([
        [0.19220398 + 0.02469635*I, 0.37688051 - 0.26959786*I, 0.67423451 + 0.23455204*I, 0.25630601 - 0.41524233*I], 
        [0.5200337 - 0.0188252*I, 0.40701539 + 0.41239862*I, 0.02624206 + 0.02222067*I, -0.62625619 + 0.0098593*I], 
        [-0.25263155 + 0.10637788*I, -0.25674125 + 0.37673747*I, 0.6970716 - 0.02469635*I, -0.0960142 + 0.47040026*I], 
        [0.7327611 - 0.28210016*I, -0.48786265 + 0.06071876*I, 0.04834923 + 0.0188252*I, 0.34479874 + 0.14214682*I]])

    S0 = sp.Matrix([[0.52238879 + 0.17969575*I], 
                    [0], 
                    [0.833558270000000], 
                    [0]])

    v0 = sp.Matrix([
        [0.562013751729898 + (0.19220398 + 0.02469635*I)*(0.52238879 + 0.17969575*I) + 0.195512792687371*I], 
        [0.0218742861348362 + 0.0185222232434409*I + (0.5200337 - 0.0188252*I)*(0.52238879 + 0.17969575*I)], 
        [0.581049796962132 - 0.0205858467813145*I + (-0.25263155 + 0.10637788*I)*(0.52238879 + 0.17969575*I)], 
        [0.0403019005146321 + (0.52238879 + 0.17969575*I)*(0.7327611 - 0.28210016*I) + 0.015691901144404*I]])


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
