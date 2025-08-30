import numpy as np
from tblite.interface import Calculator
from tqdm import tqdm

import torch
import dxtb
from dxtb.typing import DD
from dxtb.config import ConfigCache

def get_jacobian(matrix, pos):
    matrix_jac = torch.zeros(matrix.shape + pos.shape, dtype=matrix.dtype)
    for i, j in tqdm(np.ndindex(matrix.shape), desc="Computing Jacobian"):
        matrix_jac[i, j, :, :] = torch.autograd.grad(matrix[i, j], pos, create_graph=True, retain_graph=True)[0]
    return matrix_jac

def _get_permutation_map_closedshell_convention(element_numbers, calculator="tblite"):
    """
    Permutation map for converting the matrices obtained from `tblite`
    GFN-xTB (closed shell) to the convention for orbnet-equi.

    Input (tblite) convention:
     - H: [1s, 2s]
     - C,N,O,F: [2s, 2px, 2py, 2pz]
    Input (dxtb) convention:
     - H: [1s, 2s]
     - C,N,O,F: [2s, 2pz, 2px, 2py]
    Output (orbnet-equi) convention:
     - H: [1s, 2s] (no change)
     - C,N,O,F: [2s, 2pz, 2py, 2px]
    """
    n_so = 0
    for el in element_numbers:
        if el == 1:
            n_so += 2
        else:
            n_so += 4

    perm_map = np.zeros(n_so, dtype=int)
    idx = 0
    for el in element_numbers:
        if el == 1:
            pmap = [idx, idx + 1]
            perm_map[idx : idx + 2] = pmap
            idx += 2
        else:
            if calculator == "tblite":
                # [s, px, py, pz] -> [s, pz, py, px]
                pmap = [idx, idx + 3, idx + 2, idx + 1]
            elif calculator == "dxtb":
                # [s, pz, px, py] -> [s, pz, py, px]
                pmap = [idx, idx + 1, idx + 3, idx + 2]
            perm_map[idx : idx + 4] = pmap
            idx += 4
    return perm_map

def apply_perm_map(mat, perm_map, pos):
    ndim = mat.ndim
    pos_shape = pos.shape  # Shape of positions tensor, e.g., (n_atoms, 3)
    mat_shape = mat.shape
    if ndim == 1:
        matp = mat[perm_map]
    elif ndim == 2:
        matp = mat[perm_map, :]
        matp = matp[:, perm_map]
    elif ndim >= 3:
        if mat_shape[-2:] == pos_shape:
            matp = mat[perm_map, ...]
            matp = matp[:, perm_map, ...]
        else:
            raise ValueError("The tensor dimensions beyond the first two do not match the positions tensor.")
    return matp


def compute_F(numbers, positions):
    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    calc.set("verbosity", 0)
    calc.set("save-f-gradient", 1)
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
calc.set("save-f-gradient", 1)
res = calc.singlepoint()

F  = res.get("fock-matrix")
dF = res.get("fock-gradient")  # (nao, nao, nat, 3)

# --- numerical check ---
h = 1e-3  # displacement in same unit as `positions`
dF_num = finite_diff_F(numbers, positions, h=h)

# --- dxtb ---
dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}
opts = {"scf_mode": "full"}

nums = torch.tensor(numbers, device=dd["device"], dtype=torch.int32)
pos = torch.tensor(positions, device=dd["device"]).requires_grad_(True)

cache_config = ConfigCache(density=True, coefficients=True, mo_energies=True, fock=True)
calc = dxtb.Calculator(nums, dxtb.GFN1_XTB, **dd, opts=opts)
calc.opts.cache = cache_config
calc.get_energy(pos)



# dxtb F
perm_map_dxtb = _get_permutation_map_closedshell_convention(numbers, "dxtb")
perm_map_tblite = _get_permutation_map_closedshell_convention(numbers, "tblite")

F_dxtb = apply_perm_map(calc.cache["fock"], perm_map_dxtb, positions)
F_tblite = apply_perm_map(F, perm_map_tblite, positions)

# dxtb dH 
dF_dxtb = get_jacobian(F_dxtb, pos)


# Diagnostics
print(f"F_dxtb shape {F_dxtb.shape}, F_tblite shape {F_tblite.shape}")
print(f"max diff F_dxtb - F_tblite: {np.max(np.abs(F_dxtb.detach().numpy() - F_tblite))}\n")

print(f"dF_dxtb shape {dF_dxtb.shape}, dF_tblite shape {apply_perm_map(dF, perm_map_tblite, positions).shape}")
print(f"max diff dF_dxtb - dF_tblite: {np.max(np.abs(dF_dxtb.detach().numpy() - apply_perm_map(dF, perm_map_tblite, positions)))}")
print(f"max diff dF_dxtb - dF_num: {np.max(np.abs(dF_dxtb.detach().numpy() - apply_perm_map(dF_num, perm_map_tblite, positions)))}\n")

abs_err = np.abs(dF - dF_num)
rms = np.sqrt(np.mean(abs_err**2))
mx  = abs_err.max()
rel = rms / (np.sqrt(np.mean(dF_num**2)) + 1e-16)

print(f"F shape {F.shape}, dF shape {dF.shape}")
print(f"\n\nFinite-diff h = {h}")
print(f"RMS abs error = {rms:.3e}")
print(f"Max abs error = {mx:.3e}")
print(f"RMS relative  = {rel:.3e}")

# optional sanity checks
print("Symmetry check (||dF - dF^T||_F):",
      np.linalg.norm(dF - np.swapaxes(dF, 0, 1)))
