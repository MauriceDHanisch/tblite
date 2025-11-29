#!/usr/bin/env python
"""Debug script for compressed restart functionality."""

import numpy as np
from tblite.interface import Calculator, Result


# Use CH2O (formaldehyde) - 4 atoms, can be run closed-shell or open-shell
NUMBERS = np.array([6, 8, 1, 1])  # C, O, H, H
POSITIONS = np.array([
    [0.0000,  0.0000,  0.0000],  # C
    [0.0000,  0.0000,  2.2770],  # O
    [0.0000,  1.7820, -1.0260],  # H
    [0.0000, -1.7820, -1.0260],  # H
])  # in Bohr


def test_closed_shell():
    """Test closed-shell formaldehyde."""
    print("=" * 60)
    print("CLOSED-SHELL CH2O (formaldehyde)")
    print("=" * 60)
    
    calc = Calculator('GFN2-xTB', NUMBERS, POSITIONS)
    calc.set('verbosity', 1)
    res = calc.singlepoint()
    print('Full calc energy:', res.get('energy'))

    qsh = res.get('shell-charges')
    dpat = res.get('atomic-dipoles')
    qmat = res.get('atomic-quadrupoles')
    o_en = res.get('orbital-energies')
    print('qsh shape:', qsh.shape)
    print('dpat shape:', dpat.shape)
    print('qmat shape:', qmat.shape)
    print('o_en shape:', o_en.shape)

    print()
    print('--- RESTART ---')
    calc2 = Calculator('GFN2-xTB', NUMBERS, POSITIONS)
    calc2.set('verbosity', 1)
    res2 = Result()
    res2.set('shell-charges-and-moments-guess', (qsh, dpat, qmat))
    res2 = calc2.singlepoint(res2)
    print('Restart energy:', res2.get('energy'))
    print()


def test_spin_polarized():
    """Test spin-polarized formaldehyde (triplet)."""
    print("=" * 60)
    print("SPIN-POLARIZED CH2O (uhf=2, triplet)")
    print("=" * 60)
    
    calc = Calculator('GFN2-xTB', NUMBERS, POSITIONS, uhf=2)
    calc.add('spin-polarization', 1.0)
    calc.set('verbosity', 1)
    res = calc.singlepoint()
    print('Full calc energy:', res.get('energy'))

    qsh = res.get('shell-charges')
    dpat = res.get('atomic-dipoles')
    qmat = res.get('atomic-quadrupoles')
    o_en = res.get('orbital-energies')
    print('qsh shape:', qsh.shape)
    print('dpat shape:', dpat.shape)
    print('qmat shape:', qmat.shape)
    print('o_en shape:', o_en.shape)
    
    # Debug: print qsh values from Python
    print()
    print('[Python DEBUG] qsh from Python:')
    print('[Python DEBUG] qsh[0] (spin 0/alpha):', qsh[0])
    print('[Python DEBUG] qsh[1] (spin 1/beta):', qsh[1])
    print('[Python DEBUG] sum(qsh[0]):', np.sum(qsh[0]))
    print('[Python DEBUG] sum(qsh[1]):', np.sum(qsh[1]))
    print('[Python DEBUG] sum(qsh):', np.sum(qsh))
    print('[Python DEBUG] qsh.ravel():', qsh.ravel())

    print()
    print('--- RESTART ---')
    calc2 = Calculator('GFN2-xTB', NUMBERS, POSITIONS, uhf=2)
    calc2.add('spin-polarization', 1.0)
    calc2.set('verbosity', 1)
    res2 = Result()
    res2.set('shell-charges-and-moments-guess', (qsh, dpat, qmat))
    res2 = calc2.singlepoint(res2)
    print('Restart energy:', res2.get('energy'))
    print()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == 'closed':
            test_closed_shell()
        elif test_name == 'spin':
            test_spin_polarized()
        else:
            print(f"Unknown test: {test_name}")
            print("Available: closed, spin")
    else:
        # Run both tests
        test_closed_shell()
        test_spin_polarized()
