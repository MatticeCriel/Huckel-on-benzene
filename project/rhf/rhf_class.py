from pyscf import gto, scf
from scipy.linalg import fractional_matrix_power
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RHF:
    def __init__(self, atoms, basis='ccpvdz'):
        self.atoms = atoms
        self.basis = basis
        self.mol = gto.M(atom=self.atoms, basis=self.basis)
        self.S = self.mol.intor('int1e_ovlp')
        self.T = self.mol.intor('int1e_kin')
        self.V = self.mol.intor('int1e_nuc')
        self.H_core = self.T + self.V
        self.eri = self.mol.intor('int2e')
        self.enuc = self.mol.get_enuc()
        self.ndocc = self.mol.nelec[0]
        self.C = None
        self.SCF_E = None

    def rhf(self):
        A = fractional_matrix_power(self.S, -0.5)
        A = np.asarray(A)
        F_p = A @ self.H_core @ A
        epsilon, C_p = np.linalg.eigh(F_p)
        self.C = A @ C_p
        C_occ = self.C[:, :self.ndocc]
        D = np.einsum('ik,kj->ij', C_occ, C_occ.T, optimize=True)

        SCF_E = 0.0
        E_old = 0.0

        MAXITER = 100
        E_conv = 1.0e-6

        for scf_iter in range(1, MAXITER + 1):
            J = np.einsum('pqrs,rs->pq', self.eri, D, optimize=True)
            K = np.einsum('prqs,rs->pq', self.eri, D, optimize=True)

            F = self.H_core + 2 * J - K
            SCF_E = self.enuc + np.sum(D * (self.H_core + F))

            if abs(SCF_E - E_old) < E_conv:
                break
            E_old = SCF_E

            epsilon, C_p = np.linalg.eigh(A @ F @ A)
            self.C = A @ C_p
            C_occ = self.C[:, :self.ndocc]
            D = np.einsum('pi,iq->pq', C_occ, C_occ.T, optimize=True)

            if scf_iter == MAXITER:
                raise Exception("Maximum number of SCF iterations exceeded.")
        self.SCF_E = SCF_E

molecule = RHF('O 0.0 0.0 0.0; H 1.0 0.0 0.0; H 0.0 1.0 0.0')
molecule.rhf()
print(molecule.S)
print(molecule.mol.ao_labels())
print(molecule.C)