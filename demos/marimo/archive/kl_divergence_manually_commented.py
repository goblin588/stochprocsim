# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.1",
#     "numpy>=2.5.2",
#     "stochprocq",
# ]
#
# [tool.uv.sources]
# stochprocq = { path = "/Users/happy/Documents/projects/StochProcQ" }
# ///

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Example data""")
    return


@app.cell
def _():
    outputs_loss_corrected = [0.3258, 0.4709, 0.1829] #N=3
    # outputs_loss_corrected = [0.22274664, 0.37591663, 0.26085581, 0.10792249] #N=4
    # outputs_loss_corrected = [0.16915964, 0.28724855, 0.26057944, 0.16618311, 0.0787605] #N=5
    # outputs_loss_corrected = [0.13024933, 0.23143282, 0.2397572, 0.1860384, 0.11634931, 0.05941636] #N=6
    N = len(outputs_loss_corrected)

    # outputs_loss_corrected = [0.3258, 0.4709, 0.1829] #N=3
    # outputs_loss_corrected = [0.22274664, 0.37591663, 0.26085581, 0.10792249] #N=4
    # outputs_loss_corrected = [0.16915964, 0.28724855, 0.26057944, 0.16618311, 0.0787605] #N=5
    outputs_loss_corrected = [0.13024933, 0.23143282, 0.2397572, 0.1860384, 0.11634931, 0.05941636] #N=6
    return N, outputs_loss_corrected


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Eval diverge""")
    return


@app.cell
def _(N, np, outputs_loss_corrected):
    from stochprocsim.stochprocq import get_uniform_renewal
    from stochprocsim.stochprocq.measure import eval_diverge
    from stochprocsim.stochprocq.models.renewal import RenewalProcess

    q_emit = outputs_loss_corrected
    q_survive_st = np.zeros_like(q_emit)
    q_survive_st[0] = q_emit[0]
    for i in range(1, len(q_emit)):
        q_survive_st[i] = (q_emit[i] / np.prod(1 - q_survive_st[:i]))

    q_model = RenewalProcess([1-q for q in q_survive_st[:-1]])

    model = get_uniform_renewal(N-1)
    p_dist = model.gen_dists(N+1)[0]
    q_dist = q_model.gen_dists(N+1)[0]
    R_KL_renewal_fit = eval_diverge(p_dist, q_dist, his_steps=N - 1)

    print(f"R_KL[P_exact || Q_fit] (renewal-fit method) = {R_KL_renewal_fit:.6f} bits")
    return RenewalProcess, eval_diverge, get_uniform_renewal, model, q_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### $R_{KL}[P\|Q] = \sum_{i} \pi(S_i) \sum_x p(x\mid S_i)\log\frac{p(x\mid S_i)}{q(x\mid S_i)}$""")
    return


@app.cell
def _(RenewalProcess, get_uniform_renewal, np, outputs_loss_corrected):
    def generate_quantum_model(output_distribution):
        """Convert a quantum output distribution to an equivalent RenewalProcess."""
        q_emit = np.asarray(output_distribution, dtype=float)
        q_survive_st = np.zeros_like(q_emit)
        q_survive_st[0] = q_emit[0]
        for i in range(1, len(q_emit)):
            q_survive_st[i] = q_emit[i] / np.prod(1 - q_survive_st[:i])
        return RenewalProcess([1 - q for q in q_survive_st[:-1]])

    def exit_distribution(probs, start: int = 0) -> np.ndarray:
        """Exit-loop probabilities for a RenewalProcess's `probs` (per-loop
        survival probability), given it has already survived `start` loops
        (renewal state `start`) without exiting.

        Returns an array of length len(probs)+1: index j is P(exit at loop j+1)
        for j >= start, plus a final catch-all entry for surviving every
        remaining loop. Entries before `start` are 0 — the process can't exit
        at a loop it's already passed.
        """
        exit_p = np.zeros(len(probs) + 1)
        survived = 1.0
        for j in range(start, len(probs)):
            exit_p[j] = (1 - probs[j]) * survived
            survived *= probs[j]
        exit_p[len(probs)] = survived
        return exit_p

    def calculate_r_kl(output_distribution):
        _N = len(output_distribution)
        exact_model = get_uniform_renewal(_N - 1)
        exp_model = generate_quantum_model(output_distribution)
        p_state = exact_model.steady_state     # pi(S_j): stationary prob of each state

        R_KL = 0.0
        for _j in range(_N):
            p_distribution = exit_distribution(exact_model.probs, start=_j)
            q_distribution = exit_distribution(exp_model.probs, start=_j)

            # sum p*log2(p/q) for input s_j
            D_KL = 0.0
            for _n in range(_j, _N):
                # D_KL(p(.|Si) || q(.|Si)) = sum_x p(x|Si) log2( p(x|Si) / q(x|Si) )
                px_si = p_distribution[_n]
                qx_si = q_distribution[_n]
                D_KL += px_si * np.log2(px_si / qx_si)

            #pi(S_j) * \sigma D_KL
            R_KL += p_state[_j] * D_KL
        return R_KL

    print(f"\nR_KL[P||Q] = {calculate_r_kl(outputs_loss_corrected):.6f} bits")
    return calculate_r_kl, generate_quantum_model


@app.cell
def _(mo):
    mo.md(r"""### Plot""")
    return


@app.cell(hide_code=True)
def _(calculate_r_kl, get_uniform_renewal):
    import matplotlib.pyplot as plt

    #eval diverge divs
    divs_xim = [0.032157,0.028484,0.021538, 0.021125]


    #calculating all divs with other method
    data = {
        '3': [0.3258, 0.4709, 0.1829], #N=3
        '4': [0.22274664, 0.37591663, 0.26085581, 0.10792249], #N=4
        '5': [0.16915964, 0.28724855, 0.26057944, 0.16618311, 0.0787605], #N=5
        '6': [0.13024933, 0.23143282, 0.2397572, 0.1860384, 0.11634931, 0.05941636] 
    }

    all_r_kl_list_goblin = []
    for __n in data.keys():
        all_r_kl_list_goblin.append(calculate_r_kl(data[__n]))

    _x = list(range(3, 6+1))
    _y_classical = [get_uniform_renewal(_n - 1).classical_bd(4, target_dim=2) for _n in _x]

    _palette = ["#3B5BA5", "#800E13", "#C11B26", "#4C9A2A"]
    _fig, _ax = plt.subplots()
    _ax.plot(_x, _y_classical, '-s', label='Classical bound', linewidth=1.5)
    _ax.plot(_x, divs_xim, 'p--',
             label='Experiment (eval_diverge)', markersize=8)
    _ax.plot(_x, [div for div in all_r_kl_list_goblin], 'd--',
             label='Experiment (My calc)', markersize=8)
    _ax.fill_between(_x, _y_classical, _ax.get_ylim()[1], alpha=0.4,
                     color='none', edgecolor=_palette[0], hatch='///')
    _ax.set_xticks(_x)
    _ax.set_xlabel('N')
    _ax.set_ylabel('Divergence Rate')
    _ax.set_title('KL Divergence Rate for Process Depth N')
    _ax.legend()
    _ax.grid(True)
    _fig
    return data, plt


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Comments

    ### Computing the KL divergence rate

    **XW**: The KL divergence rate should be computed by the stationary distributions as the following:
    """
    )
    return


@app.cell
def _(eval_diverge):
    def get_KL_rate(N, model, q_model):
        _p_dist = model.steady_state @ model.gen_dists(N+1) # Compute the stationary distribution
        _q_dist = q_model.steady_state @ q_model.gen_dists(N+1)
        _result = eval_diverge(_p_dist, _q_dist, his_steps=N - 1)
        return _result
    return (get_KL_rate,)


@app.cell
def _(N, get_KL_rate, model, q_model):
    get_KL_rate(N, model, q_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Do note that, above is not the definition but rather a property of the renewal process. To compute the KL divergence rate by definition, see the following:""")
    return


@app.cell
def _(N, eval_diverge, model, np, q_model):
    _p_dists = model.gen_dists(N+1)
    _q_dists = q_model.gen_dists(N+1)

    _div = []

    for _state, p in enumerate(_p_dists):
        q = _q_dists[_state]
        R_KL_state = eval_diverge(p, q, his_steps=None)
        _div.append(R_KL_state)

    model.steady_state @ np.array(_div) / (N+1) # weight each state by its stationary probability and divide by N+1 to get the divergence **rate**
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Also, note that when compute the KL divergence rate, as long as the history steps is large enough (larger than the markov order), the results should be the same.""")
    return


@app.cell
def _(eval_diverge, model, q_model):
    _p_dist = model.steady_state @ model.gen_dists(10)
    _q_dist = q_model.steady_state @ q_model.gen_dists(10)
    _result = eval_diverge(_p_dist, _q_dist, his_steps=9)

    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Classical Bound

    For each test length, the function `classical_bd` gives **a** lower bound of all classical models with the same number of states. To get a better lower bound, one can try different lengths and take the maximum of all lower bounds.
    """
    )
    return


@app.cell
def _(get_uniform_renewal):
    def get_bd(N):
        """Get the classical bound for a given N."""
        _bd = 0

        for _n in range(1, 9):
            new_bd = get_uniform_renewal(_n - 1).classical_bd(N, target_dim=2)
            _bd = max(_bd, new_bd)
        return _bd
    return (get_bd,)


@app.cell
def _(get_bd):
    _ls = [3, 4, 5, 6] # length of models to be computed

    classical_bounds = [get_bd(_n) for _n in _ls]

    print(classical_bounds)
    return (classical_bounds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In summary, the plot should looks like the following:""")
    return


@app.cell
def _(generate_quantum_model, get_KL_rate, get_uniform_renewal):
    def compute_from_exp_data(exp_data):
        """Compute the KL divergence rate from experimental data."""
        N = len(exp_data)
        model = get_uniform_renewal(N - 1)
        q_model = generate_quantum_model(exp_data)

        _p_dists = model.gen_dists(N+1)
        _q_dists = q_model.gen_dists(N+1)

        return get_KL_rate(N, model, q_model)
    return (compute_from_exp_data,)


@app.cell
def _(classical_bounds, compute_from_exp_data, data, plt):
    _fig, _ax = plt.subplots()
    print(data)
    results = [compute_from_exp_data(dist) for dist in data.values()]

    _ax.plot(data.keys(), results, 'o--', label='Computed from experimental data', markersize=8)
    _ax.plot(data.keys(), classical_bounds, '-s', label='Classical bound', linewidth=1.5)

    _ax.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
