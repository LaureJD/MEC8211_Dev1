## MEC8211 - Devoir 1 - Question D
## Auteurs : Wadih Chalhoub, Louis-Charles Girouard, Laure Jalbert-Drouin
## Creation : 09/02/2026
## Modification : 
# -*- coding: utf-8 -*-

"""
MEC8211 - Devoir 1 - Question D
Résolution stationnaire en r (cylindrique) de :
    0 = Deff * [ (1/r) d/dr ( r dC/dr ) ] - S
BC: C(R)=Ce (Dirichlet), dC/dr|_{r=0}=0 (symétrie)
Discrétisation: différences finies 2e ordre (maillage uniforme)

Sorties :
- Figure 1 : profil C(r) numérique vs analytique
- Figure 2 : erreurs L1, L2, Linfty en fonction de N (raffinement)
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Paramètres physiques (modifiez au besoin)
# ----------------------------
Deff = 1.0e-10        # m^2/s
S    = 2.0e-8         # mol/m^3/s (constante pour D)
Ce   = 20.0           # mol/m^3
R    = 0.5            # m  (rayon, D=1 m)

# ----------------------------
# Fonctions utilitaires
# ----------------------------
def solution_analytique(r, Deff, S, Ce, R):
    
    Cr = (S/(4*Deff)) * (r**2 - R**2) + Ce
    
    return Cr

def resout_stationnaire_radial(N, Deff, S, Ce, R):
    """
    Résout le problème stationnaire en r sur N+1 points (i=0..N) avec r_N=R.
    Renvoie r, C_num (taille N+1).
    Discrétisation 2e ordre :
        - centre (i=0) : stencil symétrique équivalant à dC/dr=0
        - intérieur (1..N-1) : (1/r) d/dr(r dC/dr) ≈ C'' + (1/r) C'
        - bord (i=N) : Dirichlet C=Ce
    """
    # Maillage
    N = int(N)
    dr = R / N
    r = np.linspace(0.0, R, N+1)

    # Matrice tridiagonale A et vecteur b pour A*C = b
    A = np.zeros((N+1, N+1), dtype=float)
    b = np.zeros(N+1, dtype=float)

    # --- Noeud centre i=0 (symétrie: dC/dr = 0)
    # Stencil 2e ordre: C''(0) ≈ (2*(C1 - C0))/dr^2
    # De plus, (1/r)C' terme est fini et s'annule avec condition de symétrie.
    # Équation: Deff * [ 2*(C1 - C0)/dr^2 ] - S = 0
    A[0, 0] = -2.0 * Deff / dr**2
    A[0, 1] =  2.0 * Deff / dr**2
    b[0]    =  S

    # --- Noeuds intérieurs i=1..N-1
    for i in range(1, N):
        ri = r[i]
        # Approximation 2e ordre:
        # C'' ≈ (C_{i+1} - 2C_i + C_{i-1})/dr^2
        # (1/r)C' ≈ (1/ri)*(C_{i+1} - C_{i-1})/(2dr)
        aW = Deff * ( 1.0/dr**2 - 1.0/(2.0*ri*dr) )
        aP = -2.0*Deff/dr**2
        aE = Deff * ( 1.0/dr**2 + 1.0/(2.0*ri*dr) )

        A[i, i-1] = aW
        A[i, i]   = aP
        A[i, i+1] = aE
        b[i]      = S  # passe à droite

    # --- Bord i=N : Dirichlet C(R) = Ce
    A[N, N] = 1.0
    b[N]    = Ce

    # Solve
    C = np.linalg.solve(A, b)
    return r, C

