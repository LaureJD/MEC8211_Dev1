## MEC8211 - Devoir 1 - Question D.a)
## Auteurs : Wadih Chalhoub, Louis-Charles Girouard, Laure Jalbert-Drouin
## Creation : 09/02/2026
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
DEFF = 1.0e-10  # m^2/s
S = 2.0e-8  # mol/m^3/s (constante pour D)
CE = 20.0  # mol/m^3
R = 0.5  # m  (rayon, D=1 m)

# ----------------------------
# Fonctions utilitaires
# ----------------------------
def solution_analytique(r, deff, s, ce, r_val):
    """Calcul de la solution analytique."""
    cr = (s / (4 * deff)) * (r**2 - r_val**2) + ce
    return cr

def resout_stationnaire_radial(n, deff, s, ce, r_max):
    """
    Résout le problème stationnaire en r sur N points (i=0..N-1) avec r_N=R.

    Renvoie r, C_num (taille N).
    Discrétisation 2e ordre :
        - centre (i=0) : stencil symétrique équivalant à dC/dr=0
        - intérieur (1..N-1) : (1/r) d/dr(r dC/dr) ≈ C'' + (1/r) C'
        - bord (i=N) : Dirichlet C=Ce
    """
    # Maillage
    n = int(n)
    dr = r_max / (n - 1)
    r = np.linspace(0.0, r_max, n)

    # Matrice tridiagonale A et vecteur b pour A*C = b
    a_mat = np.zeros((n, n), dtype=float)
    b_vec = np.zeros(n, dtype=float)

    # --- Noeud centre i=0 (symétrie: dC/dr = 0)
    # Stencil 2e ordre: C''(0) ≈ (2*(C1 - C0))/dr^2
    # De plus, (1/r)C' terme est fini et s'annule avec condition de symétrie.
    # Équation: deff * [ 2*(C1 - C0)/dr^2 ] - s = 0
    a_mat[0, 0] = -2.0 * deff / dr**2
    a_mat[0, 1] = 2.0 * deff / dr**2
    b_vec[0] = s

    # --- Noeuds intérieurs i=1..N-1
    for i in range(1, n - 1):
        ri = r[i]
        # Approximation 2e ordre avec terme cylindrique (1/r)*d/dr(r*dC/dr):
        # Schéma non-centré
        a1 = deff * (1.0 / dr**2)  # Coefficient pour C_{i-1}
        a2 = (-2.0 * deff / dr**2 -
              deff / (ri * dr))  # Coefficient pour C_i
        a3 = deff * (1.0 / dr**2 + 1.0 / (ri * dr))  # Coeff C_{i+1}

        a_mat[i, i - 1] = a1
        a_mat[i, i] = a2
        a_mat[i, i + 1] = a3
        b_vec[i] = s  # passe à droite

    # --- Bord i=N : Dirichlet C(R) = Ce
    a_mat[n - 1, n - 1] = 1.0
    b_vec[n - 1] = ce

    # Résolution du système
    c_sol = np.linalg.solve(a_mat, b_vec)
    return r, c_sol


# ----------------------------
# a) Profils de concentration
# ----------------------------
if __name__ == "__main__":
    N = 50
    r_ana = np.linspace(0, R, 500)

    r_num, c_numerique = resout_stationnaire_radial(N, DEFF, S, CE, R)
    c_analytique = solution_analytique(r_ana, DEFF, S, CE, R)

    # --- Figure 1 : profil
    plt.figure(figsize=(6, 4.5))
    plt.plot(r_ana, c_analytique, 'k--', lw=2, label='Analytique')
    plt.plot(r_num, c_numerique, 'o-', ms=4,
             label='Numérique (N={})'.format(N))
    plt.xlabel('r (m)')
    plt.ylabel('Concentration C (mol/m³)')
    plt.title("Profil stationnaire C(r) – comparaison analytique vs numérique")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.savefig("N={}.png".format(N), dpi=300)
    
    