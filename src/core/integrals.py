# Integrales necesarios para cálculos de HF
import numpy as np
from scipy.integrate import simpson
from math import pi, sqrt, exp
from scipy.special import erf
from itertools import product

# Función para calcular integrales unidimensionales y tridimensionales con la regla de Simpson
def simpson_1d (f, a, b, n=1000):
    """Calcula la integral de f en el intervalo [a, b] usando la regla de Simpson con n subintervalos."""
    x = np.linspace(a, b, n+1)
    y = f(x)
    return simpson(y, x)

def simpson_3d(f, rmin, rmax, n=100):
    """Calcula la integral triple de f en el cubo definido por [rmin, rmax] en cada dimensión usando la regla de Simpson."""
    x = np.linspace(rmin, rmax, n)
    y = np.linspace(rmin, rmax, n)
    z = np.linspace(rmin, rmax, n)
    # Malla
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    F = f(X, Y, Z)
    # Integra cada dimensión
    integral_x = simpson(F, x, axis=0)
    integral_xy = simpson(integral_x, y, axis=0)
    integral_xyz = simpson(integral_xy, z, axis=0)
    return integral_xyz

# Distancias
def distance2 (A, B):
    A = np.array(A)
    B = np.array(B)
    return float(np.dot(A - B, A - B))

# Bases gaussianas

#Norma de las bases
def gaussian_norm(alpha):
    """Calcula el factor de normalización para una función base gaussiana 1s."""
    N = (2*alpha/pi)**(3/4)
    return N

def gaussian_1s(x,y,z,alpha, Ax, Ay, Az):
    """
    Calcula el valor de una función base gaussiana para 1s en el punto r.Normalizada.
    g(r) = N*exp(-alpha * |r - A|^2)
    N = (2*alpha/pi)^(3/4)
    """
    N = gaussian_norm(alpha)
    return np.exp(-alpha * ((x - Ax)**2 + (y - Ay)**2 + (z - Az)**2)) * N

# Producto de dos gaussianas
def gaussian_product_coef(alpha_i, Ai, alpha_j, Aj):
    """Calcula los coeficientes del producto de dos gaussianas."""
    p = alpha_i + alpha_j
    P = (alpha_i * np.array(Ai) + alpha_j * np.array(Aj)) / p
    return p, P

# F0 función auxiliar para integrales
def Function_f0(t):
    """Función auxiliar F0(t) usada en integrales electrónicas."""
    t = float(t) # Asegura que t es un escalar

    if t > 1e-10:
        return 0.5 * sqrt(pi / t) * erf(sqrt(t))
    else:
        return 1.0  # Aproximación para t cerca de 0

def normal_cont(center_primits):
    """
    Normalizar una función contraída.
    """
    center, primit = center_primits
    S_self = 0.0
    for (ai, ci) in primit:
        for (aj, cj) in primit:
            S_ij = overlap_integral_analytical(ai, center, aj, center)
            S_self += ci * cj * S_ij

    if S_self <= 0:
        return (center, primit)

    norm = S_self**0.5
    primit_norm = [(a, d/norm) for (a, d) in primit]
    return (center, primit_norm)

#============================================================================
# Integrales necesarias para Hartree-Fock
#============================================================================

# Integral de solapamiento S_ij = <chi_i | chi_j>
# Para dos gausianas esta integral se reduce a una forma analítica:
# S_ij = (pi / (alpha_i + alpha_j))^(3/2) * exp(- (alpha_i * alpha_j) / (alpha_i + alpha_j) * |Ai - Aj|^2)
def overlap_integral(alpha_i, Ai, alpha_j, Aj, rmin=-10, rmax=10, n=100):
    """Calcula la integral de solapamiento entre dos funciones base."""
    (Ax, Ay, Az), (Bx, By, Bz) = Ai,Aj  # Coordenadas de los centros de las funciones base
    def integrand(X,Y,Z):
        chi_i = gaussian_1s(X,Y,Z,alpha_i, Ax, Ay, Az)
        chi_j = gaussian_1s(X,Y,Z,alpha_j, Bx, By, Bz)
        return chi_i * chi_j
    return simpson_3d(integrand, rmin=rmin, rmax=rmax, n=n)

def overlap_integral_analytical(alpha_i, Ai, alpha_j, Aj):
    """Calcula la integral de solapamiento entre dos funciones base usando la forma analítica."""
    p = alpha_i + alpha_j
    Rab2 = distance2(Ai, Aj)
    S_ij = (pi / p)**(3/2) * exp(- ((alpha_i * alpha_j) / p ) * Rab2)
    return S_ij

def overlapss(a, A, b, B):
    """
    Calcula el solapamiento entre dos gaussianas primitivas de tipo s.
    """
    return overlap_integral_analytical(a, A, b, B)

# Integral cinética T_ij = <chi_i | -1/2 ∇^2 | chi_j>
# Laplaciano de la función base gaussiana 1s
# De manera analítica la integral cinética queda:
# T_ij = alpha_i * alpha_j / (alpha_i + alpha_j) * (3 - 2 * alpha_i * alpha_j / (alpha_i + alpha_j) * |Ai - Aj|^2) * S_ij
def laplacian_gaussian_1s(alpha_j, Aj, X, Y, Z):
    """
    Calcula el laplaciano de una función base gaussiana 1s.
    Laplaciano(exp(-apha* x**2))  = (4*alpha^2 * |r - A|^2 - 6*alpha) * (-apha* x**2)
    """
    (Ax, Ay, Az) = Aj
    r2 = (X - Ax)**2 + (Y - Ay)**2 + (Z - Az)**2
    gj = gaussian_1s(X,Y,Z,alpha_j, Ax, Ay, Az)
    return (4*alpha_j**2 * r2 - 6*alpha_j) * gj

# Integral cinética
def kinetic_integral(alpha_i, Ai, alpha_j, Aj, rmin=-10, rmax=10, n=100):
    """Calcula la integral cinética entre dos funciones base."""
    (Ax, Ay, Az), (Bx, By, Bz) = Ai,Aj  # Coordenadas de los centros de las funciones base
    def integrand(X,Y,Z):
        chi_i = gaussian_1s(X,Y,Z,alpha_i, Ax, Ay, Az)
        lap_chi_j = laplacian_gaussian_1s(alpha_j, Aj, X, Y, Z)
        return chi_i * (-0.5 * lap_chi_j)
    return simpson_3d(integrand, rmin=rmin, rmax=rmax, n=n)

def kinetic_integral_analytical(alpha_i, Ai, alpha_j, Aj):
    """Calcula la integral cinética entre dos funciones base usando la forma analítica."""
    S_ij = overlap_integral_analytical(alpha_i, Ai, alpha_j, Aj)
    p = alpha_i + alpha_j
    Rab2 = distance2(Ai, Aj)
    T_ij = (alpha_i * alpha_j / p) * (3 - (2 * alpha_i * alpha_j / p) * Rab2) * S_ij
    return T_ij

# Integral de atracción nuclear V_ij = <chi_i | - Z_A / |r - R_A| | chi_j>
# Se puede obtener analíticamente usando la función de error erf:
# V_ij = -2 * pi * Z_A/ (alpha_i + alpha_j) * F0 ( (alpha_i + alpha_j) * |P-R_A|^2 ) * S_ij
# donde F0(t) = 0.5 * sqrt(pi / t) * erf(sqrt(t))  para t > 0
def nuclear_electron_integral(alpha_i, Ai, alpha_j, Aj, ZA, RA, rmin=-10, rmax=10, n=100):
    """Calcula la integral de atracción nuclear entre dos funciones base."""
    (Ax, Ay, Az), (Bx, By, Bz) = Ai,Aj  # Coordenadas de los centros de las funciones base
    (Rx, Ry, Rz) = RA  # Coordenadas del núcleo A
    def integrand(X,Y,Z):
        chi_i = gaussian_1s(X,Y,Z,alpha_i, Ax, Ay, Az)
        chi_j = gaussian_1s(X,Y,Z,alpha_j, Bx, By, Bz)
        rA = np.sqrt((X - Rx)**2 + (Y - Ry)**2 + (Z - Rz)**2)
        return chi_i * (-ZA / rA) * chi_j
    return simpson_3d(integrand, rmin=rmin, rmax=rmax, n=n)

def nuclear_electron_integral_analytical(alpha_i, Ai, alpha_j, Aj, ZA, RA):
    """Calcula la integral de atracción nuclear entre dos funciones base usando la forma analítica."""
    p, P = gaussian_product_coef(alpha_i, Ai, alpha_j, Aj)
    Rab2 = distance2(Ai, Aj)
    K_ab = exp(-alpha_i*alpha_j/p * Rab2)
    RP2 = distance2(P, RA)
    t = p * RP2
    F0 = Function_f0(t)
    V_ij = -2 * pi / p * F0 * K_ab * ZA
    return V_ij

# Integral de repulsión electrónica entre dos pares de funciones base
# Integral doble de: chi_p(r1) chi_q(r1) 1/|r1 - r2| chi_r(r2) chi_s(r2) dr1 dr2
# Se puede obtener analíticamente:
def electron_repulsion_integral_analytical(alpha_p, Ap, alpha_q, Aq, alpha_r, Ar, alpha_s, As):
    """Calcula la integral de repulsión electrónica entre cuatro funciones base usando la forma analítica."""
    p, P = gaussian_product_coef(alpha_p, Ap, alpha_q, Aq)
    q, Q = gaussian_product_coef(alpha_r, Ar, alpha_s, As)

    Rab2 = distance2(Ap, Aq)
    Rcd2 = distance2(Ar, As)
    RP2 = distance2(P, Q)
    K_ab = exp(-alpha_p*alpha_q/p * Rab2)
    K_cd = exp(-alpha_r*alpha_s/q * Rcd2)

    t = (p * q) / (p + q) * RP2
    F0 = Function_f0(t)

    ERI = (2 * pi**(5/2)) / (p * q * sqrt(p + q)) * F0 * K_ab * K_cd
    return float(ERI)  # Asegura que ERI es un escalar

# ============================================================================
# Construción de matrices de integrales
# ============================================================================

# Se construyen los nuevos estados en términos de las bases moleculares conocidas
# psi = Σ c_i * chi_i, con chi_i las funciones base gaussianas 1s
# Así, las matrices de integrales se construyen como:
# S_ij = <psi_i | psi_j> = Σ Σ c_pi * c_qj * <chi_p | chi_q>

def _compute_contribution(prims_mu, prims_nu, A, B, centers=None, Zlist=None):
    """Calcula las contribuciones para S, T y V."""
    valS = valT = valV = 0.0
    
    for (a, ca), (b, cb) in product(prims_mu, prims_nu):
        coeff_product = ca * cb
        valS += coeff_product * overlap_integral_analytical(a, A, b, B)
        valT += coeff_product * kinetic_integral_analytical(a, A, b, B)
        
        if centers is not None and Zlist is not None:
            for C, Z in zip(centers, Zlist):
                valV += coeff_product * float(nuclear_electron_integral_analytical(a, A, b, B, Z, C))
    
    return valS, valT, valV

def build_one_electron_matrices(basis, centers, Zlist):
    """Construye las matrices de solapamiento, energía cinética y potencial nuclear."""
    n = len(basis)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    
    for mu in range(n):
        A, prims_mu = basis[mu]
        for nu in range(mu, n):  # S, T y V son simétricas
            B, prims_nu = basis[nu]
            
            valS, valT, valV = _compute_contribution(prims_mu, prims_nu, A, B, centers, Zlist)
            
            # Asigna valores a ambas mitades de la matriz
            S[mu, nu] = S[nu, mu] = valS
            T[mu, nu] = T[nu, mu] = valT
            V[mu, nu] = V[nu, mu] = valV
    
    return S, T, V

def _eri_contrib(prims_mu, prims_nu, prims_lam, prims_sig, A, B, Cc, D):
    val = 0.0
    for (a, ca) in prims_mu:
        for (b, cb) in prims_nu:
            for (c, cc) in prims_lam:
                for (d, cd) in prims_sig:
                    val += ca * cb * cc * cd * electron_repulsion_integral_analytical(a, A, b, B, c, Cc, d, D)
    return val

def build_electron_interact_tensor(basis):
    n = len(basis)
    eri = np.zeros((n, n, n, n))
    for mu in range(n):
        A, prims_mu = basis[mu]
        for nu in range(n):
            B, prims_nu = basis[nu]
            for lam in range(n):
                Cc, prims_lam = basis[lam]
                for sig in range(n):
                    D, prims_sig = basis[sig]
                    eri[mu, nu, lam, sig] = _eri_contrib(prims_mu, prims_nu, prims_lam, prims_sig, A, B, Cc, D)
    return eri