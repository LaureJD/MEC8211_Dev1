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
    
    A[0, 0] =  -2.0 * Deff / dr**2
    A[0, 1] =  2.0 * Deff / dr**2
    b[0]    =  S

    # --- Noeuds intérieurs i=1..N-1
    for i in range(1, N):
        ri = r[i]
        # Approximation 2e ordre avec terme cylindrique (1/r)*d/dr(r*dC/dr):
        # Discrétisation correcte : C'' + (1/r)*C'
        # où les coefficients incluent le facteur r_{i±1/2}/r_i
        
        a1 = Deff * (1.0 / dr**2)                    # C_{i-1}
        a2 = -2.0 * Deff / dr**2 - Deff / (ri * dr)  # C_i
        a3 = Deff * (1.0 / dr**2 + 1.0 / (ri * dr))  # C_{i+1}


        # a3= (Deff/dr**2) * ((dr**2/ri)-1)
        # a2= (- Deff / dr**2)*(dr/ri+2)
        # a1= (Deff/dr**2)
        A[i, i-1] = a1
        A[i, i]   = a2
        A[i, i+1] = a3
        b[i]      = S  # passe à droite

    # --- Bord i=N : Dirichlet C(R) = Ce
    # A[0, 0] = A[1, 0]
    # A[0, 1] = A[1, 1]
    # b[0]    = b[1]
    A[N, N] = 1.0
    b[N]    = Ce

    # Solve
    C = np.linalg.solve(A, b)
    return r, C


def normes_erreur(N, Deff, S, Ce, R):
    r,C_numerique=resout_stationnaire_radial(N, Deff, S, Ce, R)
    C_analytique=solution_analytique(r, Deff, S, Ce, R)
    e = C_numerique - C_analytique
    
    L1 = np.sum(np.abs(e)) / N
    L2 = np.sqrt(np.sum(e**2) / N)
    Linf = np.max(np.abs(e))
    
    return L1, L2, Linf
# ----------------------------
# a) Profils de concentration
# ----------------------------
if __name__ == "__main__":
    N=5
    
    r,C_numerique=resout_stationnaire_radial(N, Deff, S, Ce, R)
    C_analytique=solution_analytique(r, Deff, S, Ce, R)

    #--- Figure 1 : profil
    plt.figure(figsize=(6,4.5))
    plt.plot(r, C_analytique, 'k--', lw=2, label='Analytique')
    plt.plot(r, C_numerique, 'o-', ms=4, label='Numérique (N={})'.format(N))
    plt.xlabel('r (m)')
    plt.ylabel('Concentration C (mol/m³)')
    plt.title("Profil stationnaire C(r) – comparaison analytique vs numérique")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.savefig("N=10.png", dpi=300)

    # --- Figure 2 : 2D cross-section (slice) of the cylinder showing C(x,y)
    # Build a square grid and map radial solution onto it (mask outside circle)
    # nx = 200
    # x = np.linspace(-R, R, nx)
    # y = np.linspace(-R, R, nx)
    # X, Y = np.meshgrid(x, y)
    # Rgrid = np.sqrt(X**2 + Y**2)

    # # prepare 2D concentration array and fill with NaN outside the cylinder
    # C2D = np.full_like(Rgrid, np.nan, dtype=float)
    # inside = Rgrid <= R

    # # interpolate the radial numerical solution onto the grid radii
    # C2D[inside] = np.interp(Rgrid[inside], r, C_numerique)

    # plt.figure(figsize=(6,6))
    # pcm = plt.pcolormesh(X, Y, C2D, shading='auto', cmap='viridis')
    # plt.colorbar(pcm, label='Concentration C (mol/m³)')
    # # add contour lines for clarity
    # plt.contour(X, Y, C2D, levels=8, colors='k', linewidths=0.5)
    # plt.title('Concentration cross-section (slice)')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.gca().set_aspect('equal')
    # plt.tight_layout()
    #plt.show()
    #plt.savefig("N=5.png", dpi=300)
 

# ----------------------------
# Graphique des erreurs
# ----------------------------


# if __name__ == "__main__":
#     Ns=[5, 10, 20, 40, 80]
#     L1, L2, Linf = [], [], []

#     for N in Ns:
#          L1_i, L2_i, Linf_i = normes_erreur(N, Deff, S, Ce, R)
#          L1.append(L1_i)
#          L2.append(L2_i)
#          Linf.append(Linf_i)
    

#     plt.figure(figsize=(6,4.5))
#     plt.loglog(Ns, L1, 'o-', ms=4, label='L1 erreur')
#     plt.loglog(Ns, L2, 's-', ms=4, label='L2 erreur')
#     plt.loglog(Ns, Linf, '^-', ms=4, label='Linf erreur')
#     plt.xlabel('Nombre de noeuds N')
#     plt.ylabel('Norme d\'erreur')
#     plt.title("Erreurs numériques – comparaison des normes")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.show()
#     plt.savefig("Erreurs.png", dpi=300)