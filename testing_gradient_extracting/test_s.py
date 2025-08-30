import numpy as np
from tblite.interface import Calculator
from tqdm import tqdm

def compute_S(numbers, positions):
    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    calc.set("verbosity", 0)
    return calc.singlepoint().get("overlap-matrix")

def finite_diff_S(numbers, positions, h=1e-2):
    """5-point stencil in the same length unit as `positions` (bohr if you pass bohr)."""
    S0 = compute_S(numbers, positions)
    nao = S0.shape[0]
    nat = positions.shape[0]
    dS_num = np.zeros((nao, nao, nat, 3), dtype=float)

    for a in tqdm(range(nat), desc="Displacing atoms"):
        for ax in range(3):
            pos = positions.copy()

            pos[a, ax] += 2*h; S1 = compute_S(numbers, pos)
            pos[a, ax] -= h;   S2 = compute_S(numbers, pos)
            pos[a, ax] -= 2*h; S3 = compute_S(numbers, pos)
            pos[a, ax] -= h;   S4 = compute_S(numbers, pos)

            dS = (-S1 + 8*S2 - 8*S3 + S4) / (12*h)
            dS_num[:, :, a, ax] = dS
    return dS_num

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
calc.set("save-overlap-matrix-gradient", 1)
res = calc.singlepoint()

S  = res.get("overlap-matrix")
dS = res.get("overlap-matrix-gradient")  # (nao, nao, nat, 3)

# --- numerical check ---
h = 1e-2  # displacement in same unit as `positions`
dS_num = finite_diff_S(numbers, positions, h=h)

# diagnostics
abs_err = np.abs(dS - dS_num)
rms = np.sqrt(np.mean(abs_err**2))
mx  = abs_err.max()
rel = rms / (np.sqrt(np.mean(dS_num**2)) + 1e-16)

print(f"S shape {S.shape}, dS shape {dS.shape}")
print("S = ")
np.set_printoptions(precision=4, suppress=True)
print(S)

print("\ndS = ")
print(dS)

print("\ndS_num = ")
print(dS_num)
np.set_printoptions()  # Reset print options

print(f"\n\nFinite-diff h = {h}")
print(f"RMS abs error = {rms:.3e}")
print(f"Max abs error = {mx:.3e}")
print(f"RMS relative  = {rel:.3e}")

# optional sanity checks
print("Symmetry check (||dS - dS^T||_F):",
      np.linalg.norm(dS - np.swapaxes(dS, 0, 1)))
