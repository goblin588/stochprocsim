import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from stochprocsim.libraries.optics_lib import H, V, D, A, R, L, HWP, QWP, getUtot, project
    from stochprocsim.models.waveplate_angles import angs
    return A, D, H, HWP, L, QWP, R, V, angs, getUtot, mo, np, project


@app.cell
def _(H, HWP, QWP, mo, np):
    psi_out = QWP(45) @ HWP(22.5) @ H
    mo.md(f"H through HWP(22.5°) then QWP(45°): `{np.round(psi_out, 4)}`")
    return (psi_out,)


@app.cell
def _(angs, getUtot, mo, np):
    psi_in = np.array([1, 0, 0, 0], dtype=complex)
    psi_out4 = getUtot(angs[4]) @ psi_in
    mo.md(f"4×1 input `{psi_in}` through loop unitary (N=4): `{np.round(psi_out4, 4)}`")
    return (psi_out4,)


if __name__ == "__main__":
    app.run()
