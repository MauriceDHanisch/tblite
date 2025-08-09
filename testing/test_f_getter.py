import numpy as np
from tblite.interface import Calculator
import numpy.linalg as LA

numbers = np.array([3, 1])
positions = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.5]])  # units consistent with h above

numbers = np.array([6, 6, 7, 7, 1, 1, 1, 1, 1, 1, 8, 8,])
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


def check_spin(mode):
    calc = Calculator(method="GFN1-xTB", numbers=numbers, positions=positions)
    calc.set("save-integrals", 1)
    calc.set("verbosity", 0)
    if mode == "spin":
        calc.add("spin-polarization", 1.0)
    res = calc.singlepoint()

    F = res.get("fock-matrix")
    S = res.get("overlap-matrix")
    C = res.get("orbital-coefficients")
    E = res.get("orbital-energies")

    if mode == "spin" and F.ndim == 3:
        Fa = F[0]; Fb = F[1]
        Ca = C[0]; Cb = C[1]
        Ea = E[0]; Eb = E[1]
        Fa_rec = S.dot(Ca).dot(np.diag(Ea)).dot(LA.inv(Ca))
        Fb_rec = S.dot(Cb).dot(np.diag(Eb)).dot(LA.inv(Cb))
        print("max|Fa-Fa_rec|=", np.max(np.abs(Fa-Fa_rec)))
        print("||Fa-Fa_rec||=", np.linalg.norm(Fa-Fa_rec))
        print("max|Fb-Fb_rec|=", np.max(np.abs(Fb-Fb_rec)))
        print("||Fb-Fb_rec||=", np.linalg.norm(Fb-Fb_rec))
    else:
        F_rec = S.dot(C).dot(np.diag(E)).dot(LA.inv(C))
        print("F = ", F)
        print("F_reconstructed = ", F_rec)
        print("max|F - F_reconstructed| = ", np.max(np.abs(F - F_rec)))
        print("||F - F_reconstructed|| = ", np.linalg.norm(F - F_rec))

print("-- spin=0 --")
check_spin("nospin")
print("-- spin=1 --")
check_spin("spin")