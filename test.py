import numpy as np
from tblite.interface import Calculator

def compute_H(numbers, positions):
    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    calc.set("verbosity", 0)
    return calc.singlepoint().get("hamiltonian-matrix")

def finite_diff_H(numbers, positions, h=1e-2):
    """5-point stencil in the same length unit as `positions` (bohr if you pass bohr)."""
    H0 = compute_H(numbers, positions)
    nao = H0.shape[0]
    nat = positions.shape[0]
    dH_num = np.zeros((nao, nao, nat, 3), dtype=float)

    for a in range(nat):
        for ax in range(3):
            pos = positions.copy()

            pos[a, ax] += 2*h; H1 = compute_H(numbers, pos)
            pos[a, ax] -= h;   H2 = compute_H(numbers, pos)
            pos[a, ax] -= 2*h; H3 = compute_H(numbers, pos)
            pos[a, ax] -= h;   H4 = compute_H(numbers, pos)

            dH = (-H1 + 8*H2 - 8*H3 + H4) / (12*h)
            dH_num[:, :, a, ax] = dH
    return dH_num

# --- your snippet ---
from tblite.interface import Calculator
numbers = np.array([3, 1])
positions = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.5]])  # units consistent with h above

calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
calc.set("save-integrals", 1)
calc.set("verbosity", 0)
calc.set("save-hamiltonian-matrix-gradient", 1)
res = calc.singlepoint()

H  = res.get("hamiltonian-matrix")
dH = res.get("hamiltonian-matrix-gradient")  # (nao, nao, nat, 3)

# --- numerical check ---
h = 1e-2  # displacement in same unit as `positions`
dH_num = finite_diff_H(numbers, positions, h=h)

# diagnostics
abs_err = np.abs(dH - dH_num)
rms = np.sqrt(np.mean(abs_err**2))
mx  = abs_err.max()
rel = rms / (np.sqrt(np.mean(dH_num**2)) + 1e-16)

print(f"H shape {H.shape}, dH shape {dH.shape}")
print("H = ")
np.set_printoptions(precision=4, suppress=True)
print(H)

print("\ndH = ")
print(dH)

print("\ndH_num = ")
print(dH_num)
np.set_printoptions()  # Reset print options

print(f"\n\nFinite-diff h = {h}")
print(f"RMS abs error = {rms:.3e}")
print(f"Max abs error = {mx:.3e}")
print(f"RMS relative  = {rel:.3e}")

# optional sanity checks
print("Symmetry check (||dH - dH^T||_F):",
      np.linalg.norm(dH - np.swapaxes(dH, 0, 1)))
