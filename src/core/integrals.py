# Integrales necesarios para cálculos de HF
import numpy as np
from scipy.integrate import simpson

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

# Bases gaussianas
def gaussian_1s(x,y,z,alpha, Ax, Ay, Az):
    """
    Calcula el valor de una función base gaussiana para 1s en el punto r.
    g(r) = exp(-alpha * |r - A|^2)
    """
    return np.exp(-alpha * ((x - Ax)**2 + (y - Ay)**2 + (z - Az)**2))

# Integrales necesarias para Hartree-Fock

# Integral de solapamiento S_ij = <chi_i | chi_j>
def overlap_integral(alpha_i, Ai, alpha_j, Aj, rmin=-10, rmax=10, n=100):
    """Calcula la integral de solapamiento entre dos funciones base."""
    (Ax, Ay, Az), (Bx, By, Bz) = Ai,Aj  # Coordenadas de los centros de las funciones base
    def integrand(X,Y,Z):
        chi_i = gaussian_1s(X,Y,Z,alpha_i, Ax, Ay, Az)
        chi_j = gaussian_1s(X,Y,Z,alpha_j, Bx, By, Bz)
        return chi_i * chi_j
    return simpson_3d(integrand, rmin=rmin, rmax=rmax, n=n)