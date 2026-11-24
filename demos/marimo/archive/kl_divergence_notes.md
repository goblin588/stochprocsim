# KL divergence rate: the equation, why `eval_diverge` doesn't fit, and what `kl_divergence_experimental.py` does instead

## 1. The equation, from scratch

For two probability distributions over the same outcomes, KL divergence is

$$D_{KL}[p\|q] = \sum_x p(x)\log\frac{p(x)}{q(x)}$$

For two *processes* (infinite sequences of outputs, not single distributions), you don't
compare one distribution — you compare the distributions of length-$L$ output blocks and
ask how the divergence grows with $L$. The **KL divergence rate** is the per-step limit:

$$R_{KL}[P\|Q] = \lim_{L\to\infty}\frac{1}{L} D_{KL}\big[P(x_1\dots x_L)\,\|\,Q(x_1\dots x_L)\big]$$

For a hidden Markov / causal-state process, this limit has a closed form that doesn't need
$L\to\infty$ at all: once you know the process's internal (causal) states, the future is
conditionally independent of the past given the current state. So the rate reduces to an
average over states of the divergence between their *next-step* (or next-block) output
distributions:

$$R_{KL}[P\|Q] = \sum_{i} \pi(S_i) \sum_x p(x\mid S_i)\log\frac{p(x\mid S_i)}{q(x\mid S_i)}$$

- $S_i$: the causal/memory state (here: which loop the photon was injected into)
- $\pi(S_i)$: stationary probability of being in state $S_i$
- $x$: the next output symbol (here: which channel bin the photon is detected in)
- $p(x\mid S_i)$, $q(x\mid S_i)$: the true and model conditional output distributions

This is the formula printed at the top of `kl_divergence_experimental.py`. Everything below
is about the fact that the two ways of *computing* this same quantity in the codebase
(`eval_diverge` vs. the direct state-conditional sum) are not interchangeable for our setup.

## 2. How `eval_diverge` computes it, and why it doesn't work here

`eval_diverge(p_dist, q_dist, his_steps)` in `stochprocsim/stochprocq/measure.py` implements
the rate formula above, but it gets $p(x\mid S_i)$ a specific way: by literally generating
every possible length-$n$ **binary** output sequence from a single fixed starting state
(`HiddenMarkovModel.gen_dists(length)[0]` — the `[0]` is state 0), reshaping that
$2^n$-length array into $2^{\text{his\_steps}} \times 2^{n-\text{his\_steps}}$, and using the
first `his_steps` bits of history as a stand-in for "which state you're in now":

```python
p_dist_cond = p_dist.reshape((2**his_steps, -1))
...
for p, q in zip(p_dist_cond, q_dist_cond):
    if (p_his := np.sum(p)) > 1e-10:
        diverge += p_his * entropy(p/p_his, q/np.sum(q), base=2)
return diverge / (n - his_steps)
```

This works when two conditions hold:

1. **The output alphabet is binary**, so that "roll the transition tensor forward `length`
   times from state 0" enumerates a full binary tree, and a history-prefix is a valid proxy
   for a causal state.
2. **Both models are literally rooted at the same single starting state (state 0)**, and
   the state you land in after `his_steps` symbols means the *same thing* for $P$ and $Q$.

Neither holds directly for our data:

- Our detector alphabet is categorical (which of $N{+}1$ channels: $N$ loop-exit bins plus
  the ch7 dump), not binary. `generate_quantum_model()` (in `stochprocsim/utils.py`) patches
  this by converting an exit-probability list into an equivalent binary
  survive/emit chain (a `RenewalProcess`) — this is how the "Quantum (Target)" curve in
  `plot_experimental_data.py` gets built from the theory distribution, and it's fine
  *there* because both curves being compared start from a full reset (state 0).
- It breaks for real per-state measurements. Our experiment measures **multiple different
  input states directly** ($S_0, S_1, \dots, S_{N-1}$ — the photon is injected straight into
  loop $s$), not one continuously-running chain that you slice at different points. The
  classical renewal model's state $s$ means "already survived $s$ steps of the *same*
  memoryless chain that started at 0" — a Bayesian update on one trajectory. The quantum
  experiment's `start_state=s` means something physically different: the photon is prepared
  directly at stage $s$ of the interferometer. There's no guarantee — and generally it isn't
  true — that these coincide for $s>0$ (see the comment above `p_exact` in
  `plot_experimental_data.py`: *"the quantum start_state for s>0 isn't that same offset into
  the chain"*). That's exactly why the exact-model reference curve (`p_exact`) is only
  defined for $s=0$ in that file — there's no principled way to build the $s>0$ version via
  `gen_dists`.

So `eval_diverge` is the right formula for comparing two *fully generative* models rooted at
the same reset state (which is what it's used for elsewhere: theory-vs-theory, or the
single-state $s{=}0$ experimental point), but it can't honestly evaluate the other measured
input states without inventing a state-correspondence that isn't physically justified.

## 3. What `kl_divergence_experimental.py` does instead

Instead of forcing the multi-state data through `gen_dists`, it evaluates the state-
conditional sum directly, using each input state's own one-shot categorical measurement:

- $p(x\mid S_i)$: measured fraction of counts in each bin, for shots where the photon was
  injected at state $i$ — just `rates.loc[i] / rates.loc[i].sum()`. No trajectory, no
  binary re-encoding, no `gen_dists`.
- $q(x\mid S_i)$: `Simulator(...).get_output_distribution(start_state=i)` — the quantum
  model's output distribution when *directly initialised* at state $i$. This sidesteps the
  offset problem entirely because it asks the quantum model for state $i$'s distribution
  the same way it was physically prepared, instead of deriving it from unrolling state 0.
- $\pi(S_i)$: the stationary distribution of the *classical/exact* renewal model
  (`get_uniform_renewal(N-1).steady_state`), renormalised over whichever states were
  actually measured in this file.

Then it sums directly:

```python
R_KL = sum(p_state[i] * sum(p_x * log2(p_x/q_x) for x in bins) for i in states)
```

This computes the same rate formula from §1, but needs nothing more than each state's own
categorical exit distribution — so it's well-defined for every measured state, not just
$s=0$, and it doesn't require the classical-renewal-offset assumption that breaks for $s>0$.

## 4. Why the result might look too good / too linear

Two concrete things to check before trusting the current numbers:

**Only two points.** `rate_exp = [0.0354, 0.0316]` for $N=3,4$ in
`plot_experimental_data.py` is two manually pasted values. Two points are *always* exactly
linear — that's not evidence the underlying relationship is linear, it's a property of
having two points. You'd need a third $N$ to say anything about curvature at all.

**`kl_divergence_experimental.py` currently uses raw, uncorrected rates.** Look at its data
cell: `rates = data[BINS].div(data["int_time"], ...)` then straight to
`p_by_state = rates / rates.sum()` — no background subtraction, no detector-efficiency
division, no per-loop transmission fit. Compare that to `plot_experimental_data.py`'s
`frac`, which runs the raw rates through `(rates - bg) / eff / T_fit**k` first. The ideal
quantum theory distribution $q(x\mid S_i)$ has no loss in it at all, so real optical loss
should show up as excess probability mass sitting in the early/low-loop bins relative to
theory (photons that should have looped further are lost and never reach the later bins).
If the raw, uncorrected $R_{KL}$ is coming out this small, either the loss in this apparatus
is small enough not to matter at this $N$, or something in the comparison is absorbing it
without you asking it to — worth rerunning with the same `(bg, eff, T)` correction used
elsewhere and seeing if $R_{KL}$ actually changes.

**No error bars on $R_{KL}$ itself.** `stds` (the SEM per bin, already computed elsewhere in
the codebase) never enters `kl_divergence_experimental.py`'s calculation. Without
propagating that into an uncertainty on $R_{KL}$, "how close to theory" and "how close to
theory relative to measurement noise" are indistinguishable — a number that looks
suspiciously clean might just be a number with an uncommunicated error bar comparable to the
distance from theory.
