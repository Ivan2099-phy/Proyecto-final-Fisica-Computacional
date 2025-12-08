# Bases moleculares conocidas

import numpy as np
import math
from sympy import factorial2

#Por ejemplo
alphas = [3.42525091, 0.62391373, 0.16885540]
coeffs = [0.15432897, 0.53532814, 0.44463454]

r = [0, 0, 0] #[x, y, z]
A = [0, 0, 0] #[Ax, Ay, Az]
l = [0, 0, 0] #[lx, ly, lz]

def N_gauss(alpha, l =[0,0,0]):
    """
    Función de normalización.

    Args:
        alpha: Coeficiente.
        l (list): Lista con los valores de lx, ly y lz.
    
    Returns:
        Constante de normalización.

    """
    return (2 * alpha / math.pi) ** 0.75 * (((4*alpha)**(l[0]+l[1]+l[2]))/(factorial2(2*l[0] - 1) * factorial2(2*l[1] - 1) * factorial2(2*l[2] - 1))) ** 0.5

def r_A_2(r, A):
    """
    Devuelve (r−A)^2
    
    Args:
        r (list): Posición del electrón en coordenadas x, y, z.
        A (list): Posición del centro de la gaussiana Ax, Ay, Az
    """
    return (r[0] - A[0])**2 + (r[1] - A[1])**2 + (r[2] - A[2])**2

def gauss_p(r, alpha, A, l):
    """
    Devuelve la gaussiana primitiva.
    
    Args:
        r (list): Posición del electrón.
        alpha: Coeficiente.
        A (list): Posición del centro de la gaussiana.
        l (list): Lista con los valores de lx, ly y lz.

    Returns:
        Gaussiana primitiva evaluada.
    """
    return  N_gauss(alpha, l) * (r[0] - A[0])**l[0] * (r[1] - A[1])**l[1] * (r[2] - A[2])**l[2] * np.exp(-alpha * r_A_2(r, A))

def STO3G_1s(center):
    """
    Construye un orbital 1s en center y devuelve sus 3 gaussianas primitivas normalizadas.
    
    Args:
        center (list): Centro del orbital 1s.
    """
    alphas = np.array([3.42525091, 0.62391373, 0.16885540])
    coeffs  = np.array([0.15432897, 0.53532814, 0.44463454])
    primit = []
    
    for alpha, dp in zip(alphas, coeffs):
        primit.append((float(alpha), float(dp)))
    
    return np.array(center, dtype=float), primit

def normal_cont(center_primits):
    """
    Normalizar una función contraída.
    """
    center, primit = center_primits
    S_self = 0.0
    for (ai, di) in primit:
        Ni = N_gauss(ai)
        for (aj, dj) in primit:
            Nj = N_gauss(aj)
            S_ij = overlapss(ai, center, aj, center)
            S_self += (di * Ni) * (dj * Nj) * S_ij

    if S_self <= 0:
        return (center, primit)

    norm = S_self**0.5
    primit_norm = [(a, d/norm) for (a, d) in primit]
    return (center, primit_norm)

def gauss_cont(r, A, alphas, coeffs, l):
    """
    Devuelve la gaussiana contratada a partir de la primitiva.
    
    Args:
        r (list): Posición del electrón.
        A (list): Posición del centro de la gaussiana.
        alphas (list): Lista con los coeficientes alpha.
        coeffs (list): Lista con los coeficientes de contracción.
        l (list): Lista con los valores de lx, ly y lz.

    Returns:
        gauss_c (list): Gaussiana contratada.
        phi (list): Lista con los valores de los coeficientes alpha, de contracción y de normalización.
    """
    
    phi = []
    gauss_c = 0

    for alpha, dp in zip(alphas, coeffs):
        gauss_c += dp * gauss_p(r, alpha, A, l)
        phi.append((alpha, dp, N_gauss(alpha, l), A))

    return gauss_c, phi