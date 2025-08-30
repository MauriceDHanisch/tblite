# test_dH_thread_safety.py
import os, numpy as np
from tblite.interface import Calculator

def compute_H(numbers, positions):
    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    calc.set("verbosity", 0)
    return calc.singlepoint().get("hamiltonian-matrix")

def compute_dH(numbers, positions):
    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    calc.set("save-h-gradient", 1)
    calc.set("verbosity", 0)
    res = calc.singlepoint()
    return res.get("hamiltonian-gradient")  # (nao,nao,nat,3)

def finite_diff_H(numbers, positions, h=1e-3):
    H0 = compute_H(numbers, positions)
    nao, nat = H0.shape[0], positions.shape[0]
    dH_num = np.zeros((nao, nao, nat, 3))
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

def metrics(dH, dH_ref):
    abs_err = np.abs(dH - dH_ref)
    rms = np.sqrt(np.mean(abs_err**2))
    mx  = abs_err.max()
    asym = np.linalg.norm(dH - np.swapaxes(dH, 0, 1))  # ||dH - dH^T||_F
    return rms, mx, asym

def run_repeat(numbers, positions, repeats=5, threads="8"):
    os.environ["OMP_NUM_THREADS"] = threads
    dH_fd = finite_diff_H(numbers, positions, h=1e-3)
    vals = []
    for _ in range(repeats):
        dH = compute_dH(numbers, positions)
        vals.append(metrics(dH, dH_fd))
    return np.array(vals)  # shape (repeats, 3) = [rms, max, asym]

if __name__ == "__main__":
    # Simple diatomic; units: Å
    numbers   = np.array([3, 1])  # LiH
    positions = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.5]])

    # 1) Race-stress (multi-thread): old impl => nondeterminism / symmetry break
    multi = run_repeat(numbers, positions, repeats=6, threads="8")
    rms_multi, mx_multi, asym_multi = multi.mean(0), multi.max(0), multi[:,0].std()

    # 2) Control (single-thread): both impls should be deterministic; fixed impl also passes (1)
    single = run_repeat(numbers, positions, repeats=3, threads="1")
    rms_single, mx_single, asym_single = single.mean(0), single.max(0), single[:,0].std()

    print("Multi-thread mean RMS / max / asym, and RMS std across repeats:")
    print(f"RMS={rms_multi[0]:.3e}, MAX={mx_multi[1]:.3e}, ASYM={rms_multi[2]:.3e}, RMS_STD={asym_multi:.3e}")
    print("Single-thread mean RMS / max / asym, and RMS std across repeats:")
    print(f"RMS={rms_single[0]:.3e}, MAX={mx_single[1]:.3e}, ASYM={rms_single[2]:.3e}, RMS_STD={asym_single:.3e}")

    # Tight but realistic tolerances
    tol_rms   = 5e-5   # FD vs analytic
    tol_asym  = 1e-8   # symmetry
    tol_jitter= 1e-8   # run-to-run RMS variability

    # Expectation:
    # - OLD impl + threads>1: either asymmetry or jitter blows past tolerance.
    # - FIXED impl: both multi and single pass.
    def assert_pass(label, arr, jitter):
        rms, mx, asym = arr
        ok = (rms < tol_rms) and (asym < tol_asym) and (jitter < tol_jitter)
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
        return ok

    ok_multi  = assert_pass("Multi-thread", rms_multi, asym_multi)
    ok_single = assert_pass("Single-thread", rms_single, asym_single)

    # Make the test fail clearly if old impl is used
    if not ok_multi and ok_single:
        raise AssertionError("Detected OpenMP race: multi-thread dH is asymmetric or non-deterministic. "
                             "Use the atomics/thread-local fix.")
    elif not (ok_multi and ok_single):
        raise AssertionError("dH gradient check failed. Investigate implementation or tolerances.")
