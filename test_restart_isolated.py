import numpy as np
from tblite.interface import Calculator, Result
from math import isclose

THR = 1.0e-9

def test_calc_restart_compressed_closed_shell():
    """Test compressed restart (qsh, dpat, qmat) for closed-shell system"""
    numbers = np.array([6, 6, 7, 7, 1, 1, 1, 1, 1, 1, 8, 8])
    positions = np.array([
        [-3.81469488143921, +0.09993441402912, 0.00000000000000],
        [+3.81469488143921, -0.09993441402912, 0.00000000000000],
        [-2.66030049324036, -2.15898251533508, 0.00000000000000],
        [+2.66030049324036, +2.15898251533508, 0.00000000000000],
        [-0.73178529739380, -2.28237795829773, 0.00000000000000],
        [-5.89039325714111, -0.02589114569128, 0.00000000000000],
        [-3.71254944801331, -3.73605775833130, 0.00000000000000],
        [+3.71254944801331, +3.73605775833130, 0.00000000000000],
        [+0.73178529739380, +2.28237795829773, 0.00000000000000],
        [+5.89039325714111, +0.02589114569128, 0.00000000000000],
        [-2.74426102638245, +2.16115570068359, 0.00000000000000],
        [+2.74426102638245, -2.16115570068359, 0.00000000000000],
    ])

    calc = Calculator("GFN2-xTB", numbers, positions)

    # Full calculation
    res_full = calc.singlepoint()
    energy_full = res_full.get("energy")

    # Extract compressed restart data
    qsh = res_full.get("shell-charges")
    dpat = res_full.get("atomic-dipoles")
    qmat = res_full.get("atomic-quadrupoles")

    # Restart using compressed data
    log = []
    # calc_restart = Calculator("GFN2-xTB", numbers, positions, logger=log.append)
    calc_restart = Calculator("GFN2-xTB", numbers, positions)

    res_restart = Result()
    res_restart.set("shell-charges-and-moments-guess", (qsh, dpat, qmat))
    res_restart = calc_restart.singlepoint(res_restart)
    energy_restart = res_restart.get("energy")

    # Check that energies are very close
    assert isclose(energy_restart, energy_full, abs_tol=1e-6)

    # # Count iterations from log
    # iterations = 0
    # for line in log:
    #     parts = line.strip().split()
    #     if len(parts) > 0 and parts[0].isdigit():
    #         iterations = max(iterations, int(parts[0]))

    # assert iterations <= 2


def test_calc_restart_compressed_spin_polarized():
    """Test compressed restart (qsh, dpat, qmat) for spin-polarized system"""
    # Use same molecule as closed-shell test
    numbers = np.array([6, 6, 7, 7, 1, 1, 1, 1, 1, 1, 8, 8])
    positions = np.array([
        [-3.81469488143921, +0.09993441402912, 0.00000000000000],
        [+3.81469488143921, -0.09993441402912, 0.00000000000000],
        [-2.66030049324036, -2.15898251533508, 0.00000000000000],
        [+2.66030049324036, +2.15898251533508, 0.00000000000000],
        [-0.73178529739380, -2.28237795829773, 0.00000000000000],
        [-5.89039325714111, -0.02589114569128, 0.00000000000000],
        [-3.71254944801331, -3.73605775833130, 0.00000000000000],
        [+3.71254944801331, +3.73605775833130, 0.00000000000000],
        [+0.73178529739380, +2.28237795829773, 0.00000000000000],
        [+5.89039325714111, +0.02589114569128, 0.00000000000000],
        [-2.74426102638245, +2.16115570068359, 0.00000000000000],
        [+2.74426102638245, -2.16115570068359, 0.00000000000000],
    ])

    # Enable spin polarization (triplet state, uhf=2)
    calc = Calculator("GFN2-xTB", numbers, positions, uhf=2)
    calc.add("spin-polarization", 1.0)

    # Full calculation
    res_full = calc.singlepoint()
    energy_full = res_full.get("energy")

    # Extract compressed restart data (spin-resolved)
    qsh = res_full.get("shell-charges")  # shape (2, nsh)
    dpat = res_full.get("atomic-dipoles")  # shape (2, nat, 3)
    qmat = res_full.get("atomic-quadrupoles")  # shape (2, nat, 6)

    # Validate shapes for spin-polarized case
    assert qsh.ndim == 2 and qsh.shape[0] == 2
    assert dpat.ndim == 3 and dpat.shape[0] == 2 and dpat.shape[2] == 3
    assert qmat.ndim == 3 and qmat.shape[0] == 2 and qmat.shape[2] == 6

    # Restart using compressed spin-resolved data
    log = []
    # calc_restart = Calculator("GFN2-xTB", numbers, positions, uhf=2, logger=log.append)
    calc_restart = Calculator("GFN2-xTB", numbers, positions, uhf=2)
    calc_restart.add("spin-polarization", 1.0)

    res_restart = Result()
    res_restart.set("shell-charges-and-moments-guess", (qsh, dpat, qmat))
    res_restart = calc_restart.singlepoint(res_restart)
    energy_restart = res_restart.get("energy")

    # Check that energies are very close
    assert isclose(energy_restart, energy_full, abs_tol=1e-6)

    # # Count iterations from log
    # iterations = 0
    # for line in log:
    #     parts = line.strip().split()
    #     if len(parts) > 0 and parts[0].isdigit():
    #         iterations = max(iterations, int(parts[0]))
    

    # assert iterations <= 2


if __name__ == "__main__":
    print("Running test_calc_restart_compressed_closed_shell...")
    test_calc_restart_compressed_closed_shell()
    print("Passed!")

    print("Running test_calc_restart_compressed_spin_polarized...")
    test_calc_restart_compressed_spin_polarized()
    print("Passed!")
