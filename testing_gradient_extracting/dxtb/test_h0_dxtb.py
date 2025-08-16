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

    for a in tqdm(range(nat), desc="Displacing atoms"):
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
                      [0.0, 0.0, 5.5]])  # units consistent with h above

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
calc.set("save-hamiltonian-matrix-gradient", 1)
res = calc.singlepoint()

H  = res.get("hamiltonian-matrix")
dH = res.get("hamiltonian-matrix-gradient")  # (nao, nao, nat, 3)

# --- numerical check ---
h = 1e-3  # displacement in same unit as `positions`
dH_num = finite_diff_H(numbers, positions, h=h)

# --- dxtb ---
dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}
opts = {"scf_mode": "full"}

nums = torch.tensor(numbers, device=dd["device"], dtype=torch.int32)
pos = torch.tensor(positions, device=dd["device"]).requires_grad_(True)

cache_config = ConfigCache(density=True, coefficients=True, mo_energies=True)
calc = dxtb.Calculator(nums, dxtb.GFN1_XTB, **dd, opts=opts)
calc.opts.cache = cache_config
calc.integrals.build_overlap(pos)


# dxtb H 
perm_map_dxtb = _get_permutation_map_closedshell_convention(numbers, "dxtb")
perm_map_tblite = _get_permutation_map_closedshell_convention(numbers, "tblite")

H_dxtb = apply_perm_map(calc.integrals.build_hcore(pos), perm_map_dxtb, positions)
H_tblite = apply_perm_map(H, perm_map_tblite, positions)

# dxtb dH 
dH_dxtb = get_jacobian(H_dxtb, pos)


# Diagnostics
print(f"H_dxtb shape {H_dxtb.shape}, H_tblite shape {H_tblite.shape}")
print(f"max diff H_dxtb - H_tblite: {np.max(np.abs(H_dxtb.detach().numpy() - H_tblite))}\n")

print(f"dH_dxtb shape {dH_dxtb.shape}, dH_tblite shape {apply_perm_map(dH, perm_map_tblite, positions).shape}")
print(f"max diff dH_dxtb - dH_tblite: {np.max(np.abs(dH_dxtb.detach().numpy() - apply_perm_map(dH, perm_map_tblite, positions)))}")
print(f"max diff dH_dxtb - dH_num: {np.max(np.abs(dH_dxtb.detach().numpy() - apply_perm_map(dH_num, perm_map_tblite, positions)))}\n")

abs_err = np.abs(dH - dH_num)
rms = np.sqrt(np.mean(abs_err**2))
mx  = abs_err.max()
rel = rms / (np.sqrt(np.mean(dH_num**2)) + 1e-16)

print(f"H shape {H.shape}, dH shape {dH.shape}")
print(f"\n\nFinite-diff h = {h}")
print(f"RMS abs error = {rms:.3e}")
print(f"Max abs error = {mx:.3e}")
print(f"RMS relative  = {rel:.3e}")

# optional sanity checks
print("Symmetry check (||dH - dH^T||_F):",
      np.linalg.norm(dH - np.swapaxes(dH, 0, 1)))
