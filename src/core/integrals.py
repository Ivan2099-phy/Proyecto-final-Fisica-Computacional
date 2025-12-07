# Integrales necesarios para cálculos de HF
import numpy as np
from scipy.integrate import simpson

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
    integral_xy = simpson(integral_x, y, axis=1)
    integral_xyz = simpson(integral_xy, z, axis=2)
    return integral_xyz
