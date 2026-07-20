import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.models.renewal import RenewalProcess
    from stochprocsim.stochprocq.measure import eval_diverge
    return RenewalProcess, eval_diverge, get_uniform_renewal, mo, np


@app.cell
def _(get_uniform_renewal):
    # For N = 3
    N = 3
    model = get_uniform_renewal(N-1)
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
    q_survive_st = np.zeros_like(q_emit)
    q_survive_st[0] = q_emit[0]
    for i in range(1, len(q_emit)):
        q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))

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
def _(N, np):
    from stochprocsim.models.unitaries import U_3, U_3_states

    a_prod = 1
    res = []
    for j in range(N):
        v = U_3 @ U_3_states[j]
        a = float(np.linalg.norm(v[[0, 2]])**2)
        b = float(np.linalg.norm(v[[1, 3]])**2)
        res.append(b * a_prod)
        print(f"s{j} p1:{(b * a_prod):0.2f}")
        a_prod *= a
    print(sum(res))
    return


@app.cell
def _(RenewalProcess, ave_kl_div_rate, eval_diverge, np, p_dist):
    # Define Quantum Model
    q_emit2 = np.array([0.32, 0.47, 0.18])
    q_survive_st2 = np.zeros_like(q_emit2)
    q_survive_st2[0] = q_emit2[0]
    for k in range(1, len(q_emit2)):
        q_survive_st2[k] = (q_emit2[k] / np.prod(1 - q_survive_st2[:k]))

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
