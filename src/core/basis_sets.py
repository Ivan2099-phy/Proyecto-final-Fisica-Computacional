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