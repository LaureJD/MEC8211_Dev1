## MEC8211 - Devoir 1 - Question D.b)
## Auteurs : Laure Jalbert-Drouin
## Creation : 13/02/2026
## Modification:
# -*- coding: utf-8 -*-

"""
MEC8211 - Devoir 1 - Question D.b)
Résolution stationnaire en r (cylindrique) de :
    0 = Deff * [ (1/r) d/dr ( r dC/dr ) ] - S
BC: C(R)=Ce (Dirichlet), dC/dr|_{r=0}=0 (symétrie)
Discrétisation: différences finies 2e ordre (maillage uniforme)

Sorties :
- Figure 1 : erreurs L1, L2, Linfty en fonction de N (raffinement)
"""

import numpy as np
import matplotlib.pyplot as plt
import qd_calc_generique as D_gen

# ----------------------------
# Paramètres physiques (modifiez au besoin)
# ----------------------------
DEFF = 1.0e-10  # m^2/s
S = 2.0e-8  # mol/m^3/s (constante pour D)
CE = 20.0  # mol/m^3
R = 0.5  # m  (rayon, D=1 m)

# ----------------------------
# Erreurs L1, L2, L infini
# ----------------------------

def normes_erreur(n, deff, s, ce, r_max):
    """
    Calcule les erreurs L1, L2, Linf entre la solution numérique et analytique pour N points.
    Renvoie l1, l2, linf (floats).

    """
    r, c_numerique = D_gen.resout_stationnaire_radial(
        n, deff, s, ce, r_max)
    c_analytique = D_gen.solution_analytique(r, deff, s, ce, r_max)
    e = c_numerique - c_analytique

    l1 = np.sum(np.abs(e)) / n
    l2 = np.sqrt(np.sum(e**2) / n)
    linf = np.max(np.abs(e))

    return l1, l2, linf

# ----------------------------
# Données de convergence (Ns -> erreurs)
# ----------------------------
def convergence_data(ns, deff, s, ce, r_max):
    """
    Crée des vecteurs contenant les erreurs L1, L2, Linf pour chaque N de ns.
    Renvoie drs, l1s, l2s, linfs (vecteurs de même taille que ns).

    """
    drs, l1s, l2s, linfs = [], [], [], []
    for n in ns:
        dr = r_max / (n - 1)
        l1, l2, linf = normes_erreur(n, deff, s, ce, r_max)
        drs.append(dr)
        l1s.append(l1)
        l2s.append(l2)
        linfs.append(linf)
    return np.array(drs), np.array(l1s), np.array(l2s), np.array(linfs)

def ordre_conv(drs, errs):
    """
    Calcul l'ordre de convervence p à partir de deux points les plus fins (drs, errs).
    Renvoie p (float) ou np.nan si pas assez de points ou erreurs non positives

    """
    # p entre les deux grilles les plus fines :
    # p = ln(e2/e1)/ln(h2/h1)
    if len(drs) < 2:
        return np.nan
    e1, e2 = errs[-2], errs[-1]
    h1, h2 = drs[-2], drs[-1]
    if e1 <= 0 or e2 <= 0:
        return np.nan
    p = np.log(e2 / e1) / np.log(h2 / h1)
    return p


def convergence_graph(ns, deff, s, ce, r_max):
    """
    Affiche la figure de convergence (erreurs L1, L2, Linf vs dr) 
    et les lignes de référence pour les pentes.
    Renvoie drs, l1s, l2s, linfs (vecteurs de même taille que ns) pour un usage ultérieur.

    """
    drs, l1s, l2s, linfs = convergence_data(ns, deff, s, ce, r_max)
    plt.loglog(drs, l1s, 'o-', label='L1')
    plt.loglog(drs, l2s, 's-', label='L2')
    plt.loglog(drs, linfs, 'd-', label='L∞')

    # Ligne de référence pente 1 et 2
    # (simple normalisation sur le premier point pour visualiser les pentes)
    p1 = ordre_conv(drs, l1s)
    c1 = l1s[-1] / (drs[-1]**p1)
    ref1 = c1 * drs**p1
    p2 = ordre_conv(drs, l2s)
    c2 = l2s[-1] / (drs[-1]**p2)
    ref2 = c2 * drs**p2
    pinf = ordre_conv(drs, linfs)
    cinf = linfs[-1] / (drs[-1]**pinf)
    refinf = cinf * drs**pinf
    plt.loglog(drs, ref1, 'k--', alpha=0.5, label='pente L1')
    plt.loglog(drs, ref2, 'k-.', alpha=0.5, label='pente L2')
    plt.loglog(drs, refinf, 'k:', alpha=0.5, label='pente L∞')

    plt.xlabel('Δr (m)')
    plt.ylabel('Erreur L')
    plt.title('Convergence – erreurs L vs Δr (m)')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.show()

    return drs, l1s, l2s, linfs

def order_report(ns=(10, 20, 40, 80, 160, 320),
                         deff=DEFF, s=S, ce=CE, r_max=R):
    """
    Affiche l'ordre de convergence estimé p pour les erreurs L1, L2, Linf 
    à partir des données de convergence.
    Renvoie un dictionnaire avec les drs, erreurs et p estimés.
    """
    drs, l1s, l2s, linfs = convergence_data(
        ns, deff, s, ce, r_max)

    pl1 = ordre_conv(drs, l1s)
    pl2 = ordre_conv(drs, l2s)
    plinf = ordre_conv(drs, linfs)

    print("[Estimation d'ordre]")
    print(f"  p(L1)  ~ {pl1: .3f}")
    print(f"  p(L2)  ~ {pl2: .3f}")
    print(f"  p(L∞)  ~ {plinf: .3f}")

    return {
        'drs': drs, 'L1': l1s, 'L2': l2s, 'Linf': linfs,
        'p_2grid': {'L1': pl1, 'L2': pl2, 'Linf': plinf},
    }
# ----------------------------
# RÉSULTATS
# ----------------------------
if __name__ == "__main__":
    NS = (10, 20, 30, 40, 80)

    convergence_graph(NS, DEFF, S, CE, R)
    order_report(NS)
