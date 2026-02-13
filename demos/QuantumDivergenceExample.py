import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import sympy as sp

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.Models.renewal import RenewalProcess
    from stochprocsim.stochprocq.measure import eval_diverge
    return RenewalProcess, eval_diverge, get_uniform_renewal, mo, np, sp


@app.cell
def _(get_uniform_renewal):
    # For N = 3
    N = 3
    model = get_uniform_renewal(N-1) # 3 causal states
    print(f'The probability of tick (output 1) at each memory state: {[1-p for p in model.probs] + [1]}')
    print(f'The stationary distribution of the memory states are: {model.steady_state}')
    return N, model


@app.cell
def _(mo):
    mo.md(r"""This is where i become lost. These three numbers '[0.37, 0.32, 0.31]' you said were arbitrary ones which you have chosen. In our meeting, it sounded like this should not influence the divergence rate though its easy to see that they do through varying these numbers.""")
    return


@app.cell
def _(RenewalProcess, eval_diverge, model, np):
    # Define Quantum Model
    q_emit = np.array([0.37, 0.32, 0.31]) 
    # print(sum(q_emit))
    # probability of emitting 0 at each causal state:
    q_survive_st = np.zeros_like(q_emit)
    q_survive_st[0] = q_emit[0]
    for i in range(1, len(q_emit)):
        q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))
        # print(q_survive_st[i])

    # print(f'Inferred probs of emitting 1 at each causal state: {q_survive_st}')
    q_model = RenewalProcess([1-q for q in q_survive_st[:-1]])

    # Calculate distributions and divergence between Quantum and Exact models
    p_dist = model.gen_dists(4)[0]
    q_dist = q_model.gen_dists(4)[0]

    ave_kl_div_rate = eval_diverge(p_dist, q_dist, his_steps=3)
    print(f'Average KL divergence rate between the two models: {ave_kl_div_rate}')
    return ave_kl_div_rate, p_dist


@app.cell
def _(mo):
    mo.md(r"""This is how we get the probabilities for a U of length N. We can put these into the same quantum model code as before to calculate the KL Divergence.""")
    return


@app.cell
def _(N, sp):
    # Unitary 
    U_3 = sp.Matrix([
        [ 0.19220398+0.02469635j, 0.37688051-0.26959786j, 0.67423451+0.23455204j, 0.25630601-0.41524233j],
        [ 0.5200337 -0.0188252j, 0.40701539+0.41239862j, 0.02624206+0.02222067j,-0.62625619+0.0098593j ],
        [-0.25263155+0.10637788j, -0.25674125+0.37673747j, 0.6970716 -0.02469635j, -0.0960142 +0.47040026j],
        [0.7327611 -0.28210016j, -0.48786265+0.06071876j, 0.04834923+0.0188252j, 0.34479874+0.14214682j]
        ])

    # Causal States
    s001 = sp.Matrix([[0.52238879+0.17969575j],[ 0.83355827+0.j]])
    s010 = sp.Matrix([[0.85249472+0.j],[ 0.48584821-0.19288408j]])
    s100 = sp.Matrix([[0.97147785+0.j],[0.21362996-0.10292246j]])
    s000 = sp.Matrix([[ 0.98779056+0.j],[-0.15276737+0.03052753j]])

    U_3_states = [
        sp.Matrix([[s001[0]],[0],[s001[1]],[0]]),
        sp.Matrix([[s010[0]],[0],[s010[1]],[0]]),
        sp.Matrix([[s100[0]],[0],[s100[1]],[0]]),
        sp.Matrix([[s000[0]],[0],[s000[1]],[0]])
    ]

    # Get the proportion of samples which should end up in each of the 3 steps 
    # Get quantity through computing Pr(S_j, 1) * Pr(S_j-1, 0) *...* Pr(S_0, 0)
    a_prod = 1
    res = []
    for j in range(N):
        v = U_3@U_3_states[j]
        v13 = sp.Matrix([[v[0]],[v[2]]])
        v24 = sp.Matrix([[v[1]],[v[3]]])
        a, b = v13.norm()**2, v24.norm()**2
        res.append(b * a_prod)
        print(f"s{j} p1:{(b * a_prod):0.2f}")
        a_prod *= a
    print(sum(res))
    return


@app.cell
def _(RenewalProcess, ave_kl_div_rate, eval_diverge, np, p_dist):
    # Define Quantum Model
    q_emit2 = np.array([0.32, 0.47, 0.18]) 
    # print(sum(q_emit))
    # probability of emitting 0 at each causal state:
    q_survive_st2 = np.zeros_like(q_emit2)
    q_survive_st2[0] = q_emit2[0]
    for k in range(1, len(q_emit2)):
        q_survive_st2[k] = (q_emit2[k] / np.prod(1 - q_survive_st2[:k]))
        # print(q_survive_st[i])

    # print(f'Inferred probs of emitting 1 at each causal state: {q_survive_st}')
    q_model2 = RenewalProcess([1-q for q in q_survive_st2[:-1]])

    # Calculate distributions and divergence between Quantum and Exact models
    q_dist2 = q_model2.gen_dists(4)[0]

    ave_kl_div_rate2 = eval_diverge(p_dist, q_dist2, his_steps=3)
    print(f'Average KL divergence rate between the two models: {ave_kl_div_rate2}, which is much greater than the arbitrarily defined {ave_kl_div_rate}')
    return


@app.cell
def _(mo):
    mo.md(r"""If you think that our probability calculations are incorrect, can you provide us with a code to calculate the quantum divergence.""")
    return


if __name__ == "__main__":
    app.run()
