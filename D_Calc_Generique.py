## MEC8211 - Devoir 1 - Question D.a)
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
    Résout le problème stationnaire en r sur N points (i=0..N-1) avec r_N=R.
    Renvoie r, C_num (taille N).
    Discrétisation 2e ordre :
        - centre (i=0) : stencil symétrique équivalant à dC/dr=0
        - intérieur (1..N-1) : (1/r) d/dr(r dC/dr) ≈ C'' + (1/r) C'
        - bord (i=N) : Dirichlet C=Ce
    """
    # Maillage
    N = int(N)
    dr = R / (N-1)
    r = np.linspace(0.0, R, N)

    # Matrice tridiagonale A et vecteur b pour A*C = b
    A = np.zeros((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    # --- Noeud centre i=0 (symétrie: dC/dr = 0)
    # Stencil 2e ordre: C''(0) ≈ (2*(C1 - C0))/dr^2
    # De plus, (1/r)C' terme est fini et s'annule avec condition de symétrie.
    # Équation: Deff * [ 2*(C1 - C0)/dr^2 ] - S = 0
    
    A[0, 0] =  -2.0 * Deff / dr**2
    A[0, 1] =  2.0 * Deff / dr**2
    b[0]    =  S

    # --- Noeuds intérieurs i=1..N-1
    for i in range(1, N-1):
        ri = r[i]
        # Approximation 2e ordre avec terme cylindrique (1/r)*d/dr(r*dC/dr):
        # Schéma non-centré
        
        a1 = Deff * (1.0 / dr**2)                    # Coefficient pour C_{i-1}
        a2 = -2.0 * Deff / dr**2 - Deff / (ri * dr)  # Coefficient pour C_i
        a3 = Deff * (1.0 / dr**2 + 1.0 / (ri * dr))  # Coefficient pour C_{i+1}

        A[i, i-1] = a1
        A[i, i]   = a2
        A[i, i+1] = a3
        b[i]      = S  # passe à droite

    # --- Bord i=N : Dirichlet C(R) = Ce
    # A[0, 0] = A[1, 0]
    # A[0, 1] = A[1, 1]
    # b[0]    = b[1]
    A[N-1, N-1] = 1.0
    b[N-1]    = Ce

    # Solve
    C = np.linalg.solve(A, b)
    return r, C


# ----------------------------
# a) Profils de concentration
# ----------------------------
if __name__ == "__main__":
    N=50
    r_ana=np.linspace(0,R,500)

    r_num,C_numerique=resout_stationnaire_radial(N, Deff, S, Ce, R)
    C_analytique=solution_analytique(r_ana, Deff, S, Ce, R)

    print(C_numerique)
    print(C_analytique)

    #--- Figure 1 : profil
    plt.figure(figsize=(6,4.5))
    plt.plot(r_ana, C_analytique, 'k--', lw=2, label='Analytique')
    plt.plot(r_num, C_numerique, 'o-', ms=4, label='Numérique (N={})'.format(N))
    plt.xlabel('r (m)')
    plt.ylabel('Concentration C (mol/m³)')
    plt.title("Profil stationnaire C(r) – comparaison analytique vs numérique")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.savefig("N=10.png", dpi=300)

 