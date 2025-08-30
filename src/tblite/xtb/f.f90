! This file is part of tblite.
! SPDX-Identifier: LGPL-3.0-or-later
!
! tblite is free software: you can redistribute it and/or modify it under
! the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! tblite is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with tblite.  If not, see <https://www.gnu.org/licenses/>.

!> @file tblite/xtb/f.f90
!> Provides the analytical gradient of the AO Fock matrix F = S*C*E*C^T*S

module tblite_xtb_f
   use mctc_env, only : wp
   use tblite_blas, only : gemm
   implicit none

   private
   public :: build_fock_gradient

contains

!> Build the analytical gradient of the AO Fock matrix F = S*C*E*C^T*S
!! Includes response of MO coefficients and eigenvalues via
!! first-order perturbation of the generalized eigenproblem F C = S C E.
subroutine build_fock_gradient(S, dS_dR, dH_dR, C, E, dF_dR)
   implicit none
    real(wp), intent(in) :: S(:, :)
    real(wp), intent(in) :: dS_dR(:, :, :, :)   ! (nao, nao, nat, 3)
    real(wp), intent(in) :: dH_dR(:, :, :, :)   ! (nao, nao, nat, 3) effective Hamiltonian gradient
    real(wp), intent(in) :: C(:, :, :)          ! (nao, nao, nspin)
    real(wp), intent(in) :: E(:, :)             ! (nao, nspin)
    real(wp), intent(out) :: dF_dR(:, :, :, :)

    integer :: ispin, iatom, ic, nao, nspin, nat
    integer :: p, q
    real(wp), allocatable :: A(:, :), CE(:, :), tmp(:, :), left(:, :), right(:, :)
    real(wp), allocatable :: S_mo(:, :), H_mo(:, :), U(:, :), UE(:, :), M(:, :)
    real(wp), allocatable :: dA(:, :), Cspin(:, :), Evec(:)

   nat = size(dS_dR, 3)
   nao = size(S, 1)
   nspin = size(C, 3)
   dF_dR = 0.0_wp

    allocate(A(nao, nao), CE(nao, nao), tmp(nao, nao), left(nao, nao), right(nao, nao))
    allocate(S_mo(nao, nao), H_mo(nao, nao), U(nao, nao), UE(nao, nao), M(nao, nao))
    allocate(dA(nao, nao), Cspin(nao, nao), Evec(nao))

   do ispin = 1, nspin
       ! Spin-specific views
       Cspin(:, :) = C(:, :, ispin)
       Evec(:)     = E(:, ispin)

       ! A = C * diag(E) * C^T
       CE(:, :) = Cspin
       do ic = 1, nao
          CE(:, ic) = CE(:, ic) * Evec(ic)
       end do
       call gemm(CE, Cspin, A, transb='t')

       ! Precompute S*A and A*S for the dS contributions
       call gemm(S, A, left)   ! left = S * A
       call gemm(A, S, right)  ! right = A * S

       do iatom = 1, nat
          do ic = 1, 3
             ! 1) dS terms: dS*A*S + S*A*dS
             call gemm(dS_dR(:, :, iatom, ic), right, tmp)
             dF_dR(:, :, iatom, ic) = dF_dR(:, :, iatom, ic) + tmp
             call gemm(left, dS_dR(:, :, iatom, ic), tmp)
             dF_dR(:, :, iatom, ic) = dF_dR(:, :, iatom, ic) + tmp

             ! 2) MO-response terms: S * (dA) * S
             ! Transform dS and dH to MO basis
             call gemm(Cspin, dS_dR(:, :, iatom, ic), tmp, transa='t')   ! tmp = C^T dS
             call gemm(tmp, Cspin, S_mo)                                 ! S_mo = C^T dS C

             call gemm(Cspin, dH_dR(:, :, iatom, ic), tmp, transa='t')   ! tmp = C^T dH
             call gemm(tmp, Cspin, H_mo)                                 ! H_mo = C^T dH C

             ! Build U (orbital rotation) and dE in MO basis
             U(:, :)  = 0.0_wp
             UE(:, :) = 0.0_wp
             M(:, :)  = 0.0_wp

             do p = 1, nao
                ! Diagonal: dE_p = H_mo(p,p) - E_p * S_mo(p,p)
                M(p, p) = H_mo(p, p) - Evec(p) * S_mo(p, p)
             end do

             do p = 1, nao
                do q = 1, nao
                   if (p == q) cycle
                   U(q, p) = (H_mo(q, p) - Evec(p) * S_mo(q, p)) / (Evec(p) - Evec(q))
                end do
             end do
             ! Ensure normalization gauge on the diagonal
             do p = 1, nao
                U(p, p) = -0.5_wp * S_mo(p, p)
             end do

             ! UE = U * diag(E)
             UE(:, :) = 0.0_wp
             do q = 1, nao
                do p = 1, nao
                   UE(p, q) = U(p, q) * Evec(q)
                end do
             end do

             ! M = UE + UE^T + diag(dE)
             do p = 1, nao
                do q = 1, nao
                   M(p, q) = M(p, q) + UE(p, q) + UE(q, p)
                end do
             end do

             ! dA_AO = C * M * C^T
             call gemm(Cspin, M, tmp)
             call gemm(tmp, Cspin, dA, transb='t')

             ! Add S * dA * S
             call gemm(S, dA, tmp)
             call gemm(tmp, S, dA)
             dF_dR(:, :, iatom, ic) = dF_dR(:, :, iatom, ic) + dA
          end do
       end do
   end do

    deallocate(A, CE, tmp, left, right, S_mo, H_mo, U, UE, M, dA, Cspin, Evec)
end subroutine build_fock_gradient

end module tblite_xtb_f