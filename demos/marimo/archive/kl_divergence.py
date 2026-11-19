import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.models.causal_models import Causal_Models
    from stochprocsim.utils import kl_divergence

    return Causal_Models, get_uniform_renewal, kl_divergence, mo, np


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
def _(Causal_Models, get_uniform_renewal, kl_divergence, np):
    def getOutput(v):
        path2, path1 = v[[0, 2]], v[[1, 3]]
        α = float(np.linalg.norm(path2))
        β = float(np.linalg.norm(path1))
        norm = np.sqrt(α**2 + β**2)
        return path2, α / norm, β / norm

    def exact_model_prob(N, j):
        epsilon = 1e-19
        p_0 = (N-j-1) / (N-j+epsilon)
        p_1 = 1 / (N-j+epsilon)
        return p_0, p_1

    def exit_dists(CS, N, i):
        """p(x|S_i), q(x|S_i) over exit sequences x = 1, 01, 001, ..."""
        S = CS.states[i].copy()
        p, q = [], []
        prod_p = 1.0
        prod_q = 1.0
        for j in range(N - i):
            ### p(x|Si)
            v = CS.U @ S
            path2, p0, p1 = getOutput(v)
            p.append(p1**2 * prod_p)
            S = np.array([path2[0], 0, path2[1], 0], dtype=complex) / p0
            prod_p *= p0**2

            ### q(x|Si)
            q0, q1 = exact_model_prob(N, j)
            q.append(q1 * prod_q)
            prod_q *= q0
        return p, q

    for N in range(3, 6+1):
        CS = Causal_Models[N]
        CS.set_U(CS.U_theo)
        stat_probs = get_uniform_renewal(N-1).steady_state

        r_kl = sum(
            stat_probs[i] * kl_divergence(*exit_dists(CS, N, i))
            for i in range(N)
        )
        print(f'R_KL[P||Q] (N={N}): {r_kl:.6f} bits')
    return


if __name__ == "__main__":
    app.run()
