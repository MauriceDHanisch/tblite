import numpy as np
from tblite.interface import Calculator
from tqdm import tqdm

def compute_F(numbers, positions):
    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    calc.set("verbosity", 0)
    return calc.singlepoint().get("fock-matrix")

def finite_diff_F(numbers, positions, h=1e-2):
    """5-point stencil in the same length unit as `positions` (bohr if you pass bohr)."""
    F0 = compute_F(numbers, positions)
    nao = F0.shape[0]
    nat = positions.shape[0]
    dF_num = np.zeros((nao, nao, nat, 3), dtype=float)

    for a in tqdm(range(nat), desc="Displacing atoms"):
        for ax in range(3):
            pos = positions.copy()

            pos[a, ax] += 2*h; F1 = compute_F(numbers, pos)
            pos[a, ax] -= h;   F2 = compute_F(numbers, pos)
            pos[a, ax] -= 2*h; F3 = compute_F(numbers, pos)
            pos[a, ax] -= h;   F4 = compute_F(numbers, pos)

            dF = (-F1 + 8*F2 - 8*F3 + F4) / (12*h)
            dF_num[:, :, a, ax] = dF
    return dF_num

# --- your snippet ---
from tblite.interface import Calculator
numbers = np.array([3, 1])
positions = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.5]])  # units consistent with h above

# numbers = np.array([6, 6, 7, 7, 1, 1, 1, 1, 1, 1, 8, 8,])
# positions = np.array([  
#                 [-3.81469488143921, +0.09993441402912, 0.00000000000000],
#                 [+3.81469488143921, -0.09993441402912, 0.00000000000000],
#                 [-2.66030049324036, -2.15898251533508, 0.00000000000000],
#                 [+2.66030049324036, +2.15898251533508, 0.00000000000000],
#                 [-0.73178529739380, -2.28237795829773, 0.00000000000000],
#                 [-5.89039325714111, -0.02589114569128, 0.00000000000000],
#                 [-3.71254944801331, -3.73605775833130, 0.00000000000000],
#                 [+3.71254944801331, +3.73605775833130, 0.00000000000000],
#                 [+0.73178529739380, +2.28237795829773, 0.00000000000000],
#                 [+5.89039325714111, +0.02589114569128, 0.00000000000000],
#                 [-2.74426102638245, +2.16115570068359, 0.00000000000000],
#                 [+2.74426102638245, -2.16115570068359, 0.00000000000000],
#                 ]) 


calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
calc.set("save-integrals", 1)
calc.set("verbosity", 0)
calc.set("save-fock-matrix-gradient", 1)
res = calc.singlepoint()

F  = res.get("fock-matrix")
dF = res.get("fock-matrix-gradient")  # (nao, nao, nat, 3)

# --- numerical check ---
h = 1e-3  # displacement in same unit as `positions`
dF_num = finite_diff_F(numbers, positions, h=h)

# diagnostics
abs_err = np.abs(dF - dF_num)
rms = np.sqrt(np.mean(abs_err**2))
mx  = abs_err.max()
rel = rms / (np.sqrt(np.mean(dF_num**2)) + 1e-16)

print(f"F shape {F.shape}, dF shape {dF.shape}")
print("F = ")
np.set_printoptions(precision=4, suppress=True)
print(F)

print("\ndF = ")
print(dF)

print("\ndF_num = ")
print(dF_num)
np.set_printoptions()  # Reset print options

print(f"\n\nFinite-diff h = {h}")
print(f"RMS abs error = {rms:.3e}")
print(f"Max abs error = {mx:.3e}")
print(f"RMS relative  = {rel:.3e}")

# optional sanity checks
print("Symmetry check (||dF - dF^T||_F):",
      np.linalg.norm(dF - np.swapaxes(dF, 0, 1)))
