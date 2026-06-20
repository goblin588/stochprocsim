# stochprocsim — Codebase Reference

## What this project does

Simulates a **quantum photonic stochastic process** implemented in a fibre loop. A single photon enters a loop containing half-wave plates (HWPs), quarter-wave plates (QWPs), and a polarising beam-splitter (PBS). At each round-trip it is either detected ("tick", output 1) or survives ("no tick", output 0), producing a binary stochastic sequence. Four target unitaries (N=3,4,5,6) correspond to processes of increasing memory depth.

The code then computes the **KL divergence rate** between:
- the theoretical quantum process (target unitary)
- the exact (classical) renewal process
- the optics implementation (experimental HWP/QWP angles)

The `stochprocq` subpackage is a self-contained library for classical stochastic process theory (HMM, MPS, renewal processes, divergence measures).

---

## Package layout

```
src/stochprocsim/
├── __init__.py                        # empty
├── Libraries/
│   ├── OpticsLib.py                   # HWP, QWP, PBS, M4 matrices; getUtot(); flipped flag
│   ├── GenerateGaussianUnitaries.py   # Gaussian angle error model → sampled unitaries
│   └── PlotSamples.py                 # BROKEN: script-style, loads files at import
├── Models/
│   ├── Unitaries.py                   # Hardcoded 4×4 unitaries + 2-component causal states
│   ├── WaveplateAngles.py             # HWP/QWP angle dicts for each model (GU and NTU)
│   ├── CausalModels.py                # CausalModel class; module-level CS_3..CS_6 singletons
│   ├── TransitionModel.py             # ExactTransitionModel, QuantumTransitionModel
│   └── SimulationSampler.py           # Simulator class; sampling and output distribution
└── stochprocq/                        # Classical stochastic process library (self-contained)
    ├── __init__.py                    # Exports get_uniform_renewal, get_bio_renewal, HiddenMarkovModel, MPS
    ├── meta.py                        # Abstract base classes: IOProcess, StochModel
    ├── hmm.py                         # HiddenMarkovModel, MPS, IOHiddenMarkovModel, IOMPS
    ├── measure.py                     # kl_div, eval_diverge, eval_log_fid, eval_nll
    ├── _utility.py                    # Combinatorics: generate_partitions, powerset, int2seq
    ├── _backend.py                    # BROKEN: imports torch (not a dep); dead code
    └── Models/
        └── renewal.py                 # RenewalProcess, get_uniform_renewal, get_bio_renewal

src/FindWaveplateAngles.py             # Misplaced script: finds HWP/QWP angles for input state

demos/
├── demo.py                            # Main figure: KL divergence rate vs N (marimo)
├── propagating_state.py               # Output distribution with noise propagation (marimo)
├── error_model.py                     # Poisson + Gaussian error model (marimo)
├── kl_divergence.py                   # Per-state KL divergence derivation (marimo)
└── QuantumDivergenceExample.py        # Introductory example (marimo)
```

---

## Mode conventions

The 4×4 unitary acts on two paths × two polarisations. **NTU is the sole convention** (Phase 3 complete):

- **NTU**: path2 polarisation at indices [0, 2]; path1 at [1, 3]
- **GU**: the permutation [1, 3, 0, 2] applied to rows and columns

The `flipped` global has been removed. `OpticsLib.py` now exports `ntu_to_gu(U)` and `gu_to_ntu(U)` for interfacing with the experimental setup when needed.

---

## Known bugs and problems (as of 2026-06-20)

### Critical
1. **`WaveplateAngles.py` duplicates the GU dicts**: `U_3_angles_GU` through `U_6_angles_GU` are defined **twice** (lines 27–94 and 167–235), identically. The second block silently overwrites the first.

2. **`CausalModel.__getattr__` recursion bug**: `__getattr__` accesses `self._state_map`, but if `_state_map` hasn't been set yet (e.g., an exception fired before line 18 of `__init__`), accessing it calls `__getattr__` again → `RecursionError`. Fix: guard with `if '_state_map' not in self.__dict__: raise AttributeError(name)`.

3. **`Simulator.save()` references non-existent attributes**: `self.model.name`, `self.mrepetitions`, `self.nphotons` don't exist on `Simulator`. Method is broken.

4. **`GenerateGaussianUnitaries.py` wrong import path**: imports `from ..CausalModels import *` but the module lives at `Libraries/`, so the correct path is `from ..Models.CausalModels import *`. Breaks on import.

5. **`_backend.py` imports `torch` unconditionally**: `torch` is not in `pyproject.toml`. Import fails for anyone without it.

### Structural
6. **SymPy for numerical data**: All unitaries and causal states in `Unitaries.py` are `sp.Matrix` objects holding float/complex literals. SymPy matrix ops are 100–1000× slower than numpy for numerical work. `QuantumTransitionModel.get_output_probabilities` is the hot path.

7. **Module-level singletons with side effects**: `CausalModels.py` instantiates `CS_3`–`CS_6` and `Causal_Models` at import time. `PlotSamples.py` loads `.npz` files at import time. Both make testing and reuse painful.

8. **`flipped` global controls function definitions**: In `OpticsLib.py`, the `flipped` branch changes which functions are defined at module load. This means the module cannot switch modes at runtime.

9. **NTU and GU angle dicts are currently identical**: All `U_N_angles_GU` equal `U_N_angles_NTU` (after the duplication bug). If they are supposed to differ, the correct GU values are missing.

10. **`ClassicalTransitionModel` is dead code**: Has only `pass` in `__init__`.

11. **`generate_quantum_model()` duplicated in three demo files**: `demo.py`, `error_model.py`, `propagating_state.py` all define the same function.

12. **`FindWaveplateAngles.py` has script code at module level**: Runs angle-solving on import; `import numpy as np` appears twice; sits in `src/` rather than the package.

13. **`PlotSamples.py` is a script not a module**: Opens files, prints, defines globals at module level. Cannot be safely imported.

14. **`CausalModel.__len__` returns `len(states) - 1`**: Opaque; `CS_3` has 4 states (3 memory + final) so `len(CS_3) == 3`. Document or rename.

15. **`CausalModel.__init__` sets `self.angles_GU` twice**: Line 22 and 23 are identical assignments.

16. **Commented-out code throughout**: Multiple blocks in `SimulationSampler.py`, `OpticsLib.py`, `kl_divergence.py`, `demo.py`, `FindWaveplateAngles.py`.

---

## Proposed redesign (summary)

The core issues are: (a) SymPy used for pure numerical linear algebra, (b) global side-effects make the code hard to test and reuse, (c) duplication in angle dicts and demo utilities, (d) the `flipped` flag is a footgun.

**Phase 1 – Correctness fixes** (no behaviour change):
- Fix `__getattr__` recursion guard in `CausalModel`
- Remove duplicate GU angle dicts
- Fix `GenerateGaussianUnitaries.py` import path
- Fix `Simulator.save()` or delete it
- Remove `_backend.py` torch import / dead code
- Remove duplicate `self.angles_GU` assignment

**Phase 2 – Numerical performance**:
- Convert `Unitaries.py` data to `np.ndarray` (dtype=complex128)
- Rewrite 4-component states as `np.ndarray` of shape `(4,)`
- Update `QuantumTransitionModel` to use numpy throughout
- Profile: this likely gives 100× speedup on the hot path

**Phase 3 – Remove `flipped` / fix mode convention**:
- Commit to NTU mode; delete all `flipped==True` branches
- Rename `HWP_p1/p2`, `QWP_p1/p2` to `HWP_path1/path2` etc.

**Phase 4 – Remove global side-effects**:
- `CausalModels.py`: delete module-level instantiation; expose a factory `build_causal_models(flipped=False)` or just a plain `MODELS` dict built lazily
- `PlotSamples.py`: convert to functions that take a file path argument
- Extract `generate_quantum_model()` into a shared utility (e.g., `stochprocsim/utils.py`)

**Phase 5 – Clean up demos**:
- Move `FindWaveplateAngles.py` to `demos/` or `scripts/`
- Delete `ClassicalTransitionModel`
- Remove commented-out code

---

## Current status (as of 2026-06-20)

### Completed

**Phase 1 – Correctness fixes** ✓
- `CausalModel.__getattr__` recursion guard added
- Duplicate GU angle dicts removed from `WaveplateAngles.py`
- `GenerateGaussianUnitaries.py` import path fixed (`..Models.CausalModels`)
- `Simulator.save()` fixed (was referencing non-existent attributes)
- Duplicate `self.angles_GU` assignment removed from `CausalModel.__init__`
- `ClassicalTransitionModel` dead code deleted
- `PlotSamples.py` converted from script-at-import to proper importable module
- `FindWaveplateAngles.py` script code wrapped in `if __name__ == "__main__"`; duplicate `import numpy as np` removed

**Phase 2 – Numerical performance** ✓
- All `sp.Matrix` data in `Unitaries.py` converted to `np.ndarray(dtype=complex)`
- `QuantumTransitionModel.get_output_probabilities` now uses numpy slicing and `np.linalg.norm`
- All sympy imports removed from package source files
- Verified: max numerical diff vs original = 0.00e+00 for all 4 unitaries and 22 state vectors

**Phase 3 – Mode convention** ✓
- `flipped` global removed from `OpticsLib.py` and `CausalModels.py`
- NTU is the sole backend convention; `ntu_to_gu()` / `gu_to_ntu()` added for experimental interfacing
- `CausalModels.py` no longer has a `flipped` import or conditional instantiation block

**Demo cleanup** ✓
- All demos: removed unused imports (`sympy`, `math`), commented-out blocks, debug print statements
- `demo.py`: `is_unitary` rewritten as `np.allclose` one-liner; removed unused `w`, `y_qopt`, `qopt_yerr`
- `kl_divergence.py`: `getOutput` converted to numpy; `str` variable renamed to `seq`; dead sympy matrix cell removed; imports trimmed to only what's used
- `QuantumDivergenceExample.py`: hardcoded sympy `U_3` / states replaced with `from stochprocsim.Models.Unitaries import U_3, U_3_states`
- `error_model.py`: loop variable renamed `__N` → `_N` (marimo private-variable convention to avoid duplicate-definition error across cells)

### Still outstanding

- **`_backend.py`** still imports `torch` unconditionally (not in deps) — dead code, safe to delete
- **GU angle dicts** (`U_N_angles_GU`) are currently identical to NTU dicts — placeholder values; correct GU-optimised angles needed when available
- **`generate_quantum_model()`** still duplicated across `demo.py`, `error_model.py`, `propagating_state.py` — Phase 4 deferred
- **Module-level singletons** (`CS_3`–`CS_6`, `Causal_Models`) still instantiated at import time — Phase 4 deferred
- **`FindWaveplateAngles.py`** still lives in `src/` rather than `demos/` or `scripts/`
- **`CausalModel.__len__`** returning `len(states) - 1` is still undocumented
