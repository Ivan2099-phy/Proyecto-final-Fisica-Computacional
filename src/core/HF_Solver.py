# Solver de Hartree-Fock para átomos y moléculas.
# Ciclo principal de SCF y construcción de matrices Fock.

import numpy as np

# Clase Hartree-Fock
class HartreeFockSolver:
    """Clase para implementar el algortimo de Hartree-Fock."""
 
    def __init__(self, S, H, G, n_electrons, E_nuc=0.0):
            """
            Definimos la clase HartreeFockSolver para sistemas moleculares.

            Parametros:
            -----------
                - basis_set: Conjunto de bases moleculares conocidas.
                - integrals: Integrales necesarias para cálculos de HF.
                - n_electrons: Número de electrones en el sistema.
                - max_iterations: Número máximo de iteraciones SCF.
                - convergence_threshold: Umbral de convergencia para la energía.
            """

            self.S = S
            self.H_core = H
            self.G = G
            self.n_electrons = n_electrons
            self.E_nuc = E_nuc

            eigvals, eigvecs = np.linalg.eigh(S)
            eps_min = 1e-12
            eigvals = np.where(eigvals < eps_min, eps_min, eigvals)
            self.X = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T

            # Historial
            self.energies = []
            self.densities = []
            self.iterations = 0
    
    def build_Fock(self, P):
        """Construye la matriz Fock usando la matriz de densidad e integrales."""
        n = self.H_core.shape[0] # Número de orbitales base
        F = self.H_core.copy() # Iniciar Fock con la matriz H
        for p in range(n):
            for q in range(n):
                G_pq = 0.0 # Inicializar la contribución pq
                for r in range(n):
                    for s in range(n):
                        G_pq += P[r,s] * (self.G[p, q, r, s] - 0.5 * self.G[p, s, r, q]) 
                F[p, q] += G_pq # Añadir la contribución a la matriz de Fock
        return F
    
# Como F = H + Σ P_mv ( <mv|lp> - 0.5*<ml|vp> )
    # Se necesita la matriz de densidad P
    def build_density_matrix_P(self, C):
        """
        Construye la matriz de densidad P a partir de los coeficientes de las orbitales moleculares C.
        P = 2 * Σ (C_mi * C_vi) para orbitales ocupados i
        
        Parameters
        ----------
        C : array (n,n)
            Matriz de coeficientes moleculares. La columna C[:,m] contiene
            el orbital molecular m expresado en la base atómica.

        Returns
        --------
        P : array (n,n)
            Matriz de densidad RHF: P_{mv} = 2 Σ_(orbitales ocupados) C_{mi} C_{vi}
            (si hay número impar de electrones, el último orbital contribuye con 1).
        """
        
        # Inicializar
        n = C.shape[0]
        # Determinar la ocupación de electrones
        n_electrones_Dobles = self.n_electrons // 2
        n_electrones_es_Uno = (self.n_electrons % 2 == 1)
        P = np.zeros((n, n)) # Matriz de densidad

        # Llenar los orbitales doblemente ocupados
        for m in range(n_electrones_Dobles):
            P += 2 * np.outer(C[:, m], C[:, m]) # 2 Σ_Cmi * Cvi

        # Si hay un electrón sin pareja, agregar su contribución
        if n_electrones_es_Uno:
            idx = n_electrones_Dobles # Índice del orbital con el electrón sin pareja
            P += np.outer(C[:, idx], C[:, idx]) # Contribución del electrón sin pareja P = P + 1 * (Cmi * Cvi)
        return P
    
    # Algoritmo principal SCF
    def scf_cycle(self, conv=1e-10, max_iter=100):
        """
        Ejecuta el ciclo SCF (Self-Consistent Field) para resolver Hartree-Fock.

        Args
        ---------
        conv : Tolerancia para energía y densidad.
        max_iter : Iteraciones máximas.

        Returns
        ----------
        E_tot : Energía total convergida.
        eps : array (n,)
            Autovalores de Fock (energías orbitales).
        C : array (n,n)
            Coeficientes moleculares convergidos.
        P : array (n,n)
            Matriz de densidad final.

        Algoritmo
        ----------
        1. Construir un guess inicial diagonalizando H_core en base ortonormal.
        2. Construir densidad inicial P.
        3. Iterar:
             Construir Fock F(P).
             Transformar F -> F' = X^T F X.
             Diagonalizar F'.
             Calcular nuevos C y nueva densidad P
             Calcular energía electrónica y total.
             Verificar convergencia 
        """
        # 1: Construir guess inicial
        n_ocup_guess = self.n_electrons // 2 

        Ft = self.X.T @ self.H_core @ self.X  # F' = X^T H_core X
        eps, C2 = np.linalg.eigh(Ft)  # Diagonalizar F' obtener autovalores y autovectores
        C = self.X @ C2  # Coeficientes moleculares 

        # 2: Construir densidad inicial P
        P = self.build_density_matrix_P(C)
        E_tot_old = 0.0 # Inicializar energía total

        self.energies = []
        self.densities = []
        self.iterations = 0

        #if verbose:
        #    print(f"\nIniciando SCF para {self.n_electrons} electrones en {self.n_basis} bases")
        #    print(f"{'It':>3} {'E_total':>16} {'ΔE':>16} {'ΔP':>16}")
        #    print("-"*60)
    
        # 3: Ciclo SCF
        for iteration in range(max_iter):
            # Construir matriz Fock
            F = self.build_Fock(P)
            # Transformar F -> F' = X^T F X
            Ft = self.X.T @ F @ self.X
            # Diagonalizar F'
            eps, C2 = np.linalg.eigh(Ft)
            C = self.X @ C2  
            # Nueva densidad
            P_new = self.build_density_matrix_P(C)
            dP = np.linalg.norm(P_new - P)  # Cambio en densidad
            P = P_new
            # Energía electrónica
            E_elec = 0.5 * np.sum(P * (self.H_core + F))  
            E_tot = E_elec + self.E_nuc  # Energía total
            dE = E_tot - E_tot_old  # Cambio en energía

            # Guardar historial
            self.energies.append(E_tot)
            self.densities.append(dP)
            self.iterations = iteration

            #if verbose:
            #    print(f"{iteration:3d} {E_tot:16.10f} {dE:16.10f} {dP:16.10f}")
            
            if abs(dE) < conv and dP < conv:
                print(f"Convergencia alcanzada en {iteration+1} iteraciones.")
                return E_tot, eps, C, P
            E_tot_old = E_tot

            # guardar resultados
            self.eps = eps
            self.C = C
            self.P = P
            self.energy = E_tot
            self.F = F

        return E_tot, eps, C, P