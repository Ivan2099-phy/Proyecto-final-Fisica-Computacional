# Pruebas de las implementaciones de Hartree-Fock para átomos y moléculas

import numpy as np
from HF_Solver import *
from basis_sets import *
from integrals import *

def run_atom_H(): # Àtomo de hidrógeno
    R = np.array([0.0, 0.0, 0.0]) # Posición del átomo de hidrógeno

    basis = [STO3G_1s(R)] # Base STO-3G para hidrógeno
    basis = [normal_cont(b) for b in basis] # Reemplazar basics por su versión normalizada

    centers = [R] # Centros del átomo
    Z = [1] # Número atómico en lista para poder usarlo en build_one_electron_matrices

    # Matrices
    S, T, V = build_one_electron_matrices(basis, centers, Z) # solapamiento, cinética y potencial
    Hcore = T + V # Hamiltoniano: ciénica + potencial
    G = build_electron_interact_tensor(basis) # Integrales de repulsión electrónica

    HF = HartreeFockSolver(S, Hcore, G, n_electrons=1, E_nuc=0.0)

    E, eps, C, P = HF.scf_cycle() # Energía, orbitales y matriz de densidad *

    print("\n=== ÁTOMO DE HIDRÓGENO (STO-3G) ===")
    print("E obtenido =", E)
    print("E esperado =", -0.466581, "(ref)")
    print("Error =", abs(E + 0.466581))