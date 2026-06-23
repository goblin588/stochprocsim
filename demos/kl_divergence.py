import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.models.causal_models import Causal_Models

    return Causal_Models, get_uniform_renewal, mo, np


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
def _(Causal_Models, get_uniform_renewal, np):
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

    for N in range(3, 6+1):
        exact_model = get_uniform_renewal(N-1)
        CS = Causal_Models[N]
        CS.set_U(CS.U_theo)
        stat_probs = exact_model.steady_state
        U = CS.U

        r_kl = []
        for i in range(N):
            seq = '1'
            S = CS.states[i].copy()
            d_kl = []
            prod_p = 1.0
            prod_q = 1.0
            n = 0.0
            for j in range(N - i):
                ### p(x|Si)
                v = U @ S
                path2, p0, p1 = getOutput(v)
                p = p1**2 * prod_p
                print(f"p({seq}|S_{i}): {p:0.8f}")
                S = np.array([path2[0], 0, path2[1], 0], dtype=complex) / p0
                n += p
                prod_p *= p0**2

                ### q(x|Si)
                q0, q1 = exact_model_prob(N, j)
                q = q1 * prod_q
                seq = '0' + seq
                prod_q *= q0

                ### D_KL = p(x|Si) * log(p(x|Si) / q(x|Si))
                D_KL = p1**2 * np.log(p / q)
                d_kl.append(D_KL)

            p_si = stat_probs[i]
            r_kl.append(p_si * sum(d_kl))
        print(f'\nR_KL[P||Q] (N={N}): {sum(r_kl):.6f}\n')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
