# stochprocsim

Simulation and analysis of a quantum photonic stochastic process implemented in a fibre loop.

A single photon enters a unitary gate with a loop. At each round-trip it is either detected ("tick", output 1) or looped and recirculated (output 0), producing a binary stochastic sequence. Four target unitaries (N = 3, 4, 5, 6) correspond to processes of increasing memory depth.

The code computes steady state modelling, error propagation and KL divergence rates between the theoretical quantum process, an exact classical renewal process, and the optics implementation. Also included in this code are the physical angles for the unitary gate.

## Installation

```bash
uv sync
```

## Package structure

```
src/stochprocsim/
├── Libraries/
│   ├── OpticsLib.py                   # HWP, QWP, PBS matrices; getUtot()
│   └── GenerateGaussianUnitaries.py   # Gaussian angle error model
├── Models/
│   ├── Unitaries.py                   # Target 4×4 unitaries and causal states (N=3–6)
│   ├── WaveplateAngles.py             # Optimised HWP/QWP angles per model
│   ├── CausalModels.py                # CausalModel class; per-loop transmission
│   ├── TransitionModel.py             # Quantum and exact transition models
│   └── SimulationSampler.py           # Output distribution and sampling
└── stochprocq/                        # Classical stochastic process library
    ├── hmm.py                         # HiddenMarkovModel, MPS
    ├── measure.py                     # KL divergence, log-fidelity
    └── Models/renewal.py              # RenewalProcess
```

## Demos

All demos are [marimo](https://marimo.io) notebooks. Run with:

```bash
marimo edit demos/<notebook>.py
```

| Notebook | Description |
|---|---|
| `demo.py` | KL divergence rate vs memory depth N |
| `propagating_state.py` | Output distribution with Gaussian angle noise |
| `error_model.py` | Poisson + Gaussian error model |
| `kl_divergence.py` | Per-state KL divergence derivation |
| `prelim_data.py` | Experimental coincidence data analysis — loads timetagger output, detects N automatically, fits per-loop transmission and background |

## Mode convention

All unitaries use the basis: path 2 polarisation at indices [0, 2], path 1 at [1, 3]. 