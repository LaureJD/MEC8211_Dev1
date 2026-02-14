## MEC8211 - Devoir 1 - Question D.b)
## Auteurs : Wadih Chalhoub, Louis-Charles Girouard, Laure Jalbert-Drouin
## Creation : 13/02/2026
## Modification : 
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
import D_Calc_Generique as D_gen

# ----------------------------
# Paramètres physiques (modifiez au besoin)
# ----------------------------
Deff = 1.0e-10        # m^2/s
S    = 2.0e-8         # mol/m^3/s (constante pour D)
Ce   = 20.0           # mol/m^3
R    = 0.5            # m  (rayon, D=1 m)

# ----------------------------
# Erreurs L1, L2, L infini
# ----------------------------

def normes_erreur(N, Deff, S, Ce, R):
    r,C_numerique=D_gen.resout_stationnaire_radial(N, Deff, S, Ce, R)
    C_analytique=D_gen.solution_analytique(r, Deff, S, Ce, R)
    e = C_numerique - C_analytique
    
    L1 = np.sum(np.abs(e)) / N
    L2 = np.sqrt(np.sum(e**2) / N)
    Linf = np.max(np.abs(e))
    
    return L1, L2, Linf

# if __name__ == "__main__":
#     Ns=[5, 10, 20, 40, 80, 200]
#     L1, L2, Linf = [], [], []

#     for N in Ns:
#          L1_i, L2_i, Linf_i = D_gen.normes_erreur(N, Deff, S, Ce, R)
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

 
# ----------------------------
# Données de convergence (Ns -> erreurs)
# ----------------------------
def convergence_data(Ns, Deff, S, Ce, R):
    drs, L1s, L2s, LInfs = [], [], [], []
    for N in Ns:
        dr=R/N
        L1, L2, Linf = normes_erreur(N, Deff, S, Ce, R)
        drs.append(dr)
        L1s.append(L1)
        L2s.append(L2)
        LInfs.append(Linf)
    return np.array(drs), np.array(L1s), np.array(L2s), np.array(LInfs)

def convergence_data_log(Ns, Deff, S, Ce, R):
    drs, L1s, L2s, LInfs = [], [], [], []
    for N in Ns:
        dr=R/N
        L1, L2, Linf = normes_erreur(N, Deff, S, Ce, R)
        drs.append(np.log(dr))
        L1s.append(np.log(L1))
        L2s.append(np.log(L2))
        LInfs.append(np.log(Linf))
    return np.array(drs), np.array(L1s), np.array(L2s), np.array(LInfs)

def ordre_conv(drs, errs):
    # p entre les deux grilles les plus fines : p = ln(e2/e1)/ln(h2/h1)
    if len(drs) < 2:
        return np.nan
    e1, e2 = errs[-2], errs[-1]
    h1, h2 = drs[-2], drs[-1]
    if e1 <= 0 or e2 <= 0:
        return np.nan
    p=np.log(e2/e1) / np.log(h2/h1)
    return p
# ----------------------------
# (2) Analyse de convergence - Calcul symbolique (tracé log–log) 
# ----------------------------
def method2_convergence_plot(Ns, Deff, S, Ce, R):
    logdrs, logL1s, logL2s, logLInfs = (convergence_data_log(Ns, Deff, S, Ce, R))
    plt.figure(figsize=(6,4.4))
    # plt.loglog(drs, L1s, 'o-', label='L1')
    # plt.loglog(drs, L2s, 's-', label='L2')
    # plt.loglog(drs, LInfs,'d-', label='L∞')
    plt.plot(logdrs, logL1s, 'o-', label='L1')
    plt.plot(logdrs, logL2s, 's-', label='L2')
    plt.plot(logdrs, logLInfs,'d-', label='L∞')

    # Ligne de référence pente 1 et 2
    # (simple normalisation sur le premier point pour visualiser les pentes)
    # ref1 = L1s[0] / (drs[0] )
    # ref2 = L2s[0] / (drs[0] )
    # refinf = LInfs[0] / (drs[0] )
    drs, L1s, L2s, LInfs = (convergence_data(Ns, Deff, S, Ce, R))
    p1=ordre_conv(drs,L1s)
    ref1= p1*logdrs + np.abs(p1*logdrs[-1] - logL1s[-1])
    p2=ordre_conv(drs,L2s) 
    ref2= p2*logdrs + np.abs(p2*logdrs[-1] - logL2s[-1])
    pInf=ordre_conv(drs,LInfs)
    refInf= pInf*logdrs + np.abs(pInf*logdrs[-1] - logLInfs[-1])
    plt.plot(logdrs, ref1 ,   'k--', alpha=0.5, label='pente L1')
    plt.plot(logdrs, ref2 ,'k-.', alpha=0.5, label='pente L2')
    plt.plot(logdrs, refInf,'k:', alpha=0.5, label='pente Linf')

    #plt.gca().invert_xaxis()
    plt.xlabel('Δr (ln(Δr)')
    plt.ylabel('Erreur ln(L)')
    plt.title(f'Convergence – ln(erreurs) vs ln(Δr) ')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    #plt.tight_layout()
    plt.show()

    return drs, L1s, L2s, LInfs

# ----------------------------
# (3) Estimation de l'ordre de convergence
# ----------------------------



def method3_order_report(Ns=(10,20,40,80,160,320),
                         Deff=Deff, S=S, Ce=Ce, R=R):
    drs, L1s, L2s, LInfs = convergence_data(Ns, Deff, S, Ce, R)

    pL1   = ordre_conv(drs, L1s)
    pL2   = ordre_conv(drs, L2s)
    pLinf = ordre_conv(drs, LInfs)

    print(f"[Estimation d'ordre] Schéma {scheme}")
    print(f"  p(L1)  ~ {pL1: .3f} ")
    print(f"  p(L2)  ~ {pL2: .3f}")
    print(f"  p(L∞)  ~ {pLinf: .3f} ")

    return {
        'drs': drs, 'L1': L1s, 'L2': L2s, 'Linf': LInfs,
        'p_2grid': {'L1': pL1,     'L2': pL2,     'Linf': pLinf},
    }

# ----------------------------
# (4) Test de symétrie au centre
#    - Vérifie dC/dr|_0 ≈ 0 par une FD avant d'ordre 2
#    - Vérifie le résidu de l'équation au centre
# ----------------------------
def method4_symmetry_test(N=80, scheme='C',
                          Deff=Deff, S=S, Ce=Ce, R=R,
                          show_print=True):
    r, Cn = D_gen.resout_stationnaire_radial(N, Deff, S, Ce, R)
    dr = R / N

    # Dérivée au centre (forward 2e ordre) : C'(0) ≈ (-3C0 + 4C1 - C2)/(2Δr)
    if N >= 2:
        dC0 = (-3.0*Cn[0] + 4.0*Cn[1] - Cn[2]) / (2.0*dr)
    else:
        dC0 = (Cn[1] - Cn[0]) / dr  # fallback

    # Résidu au centre selon l'équation discrète utilisée :
    # res0 = Deff * 2*(C1 - C0)/dr^2 - S  -> doit être ~ 0
    res0 = Deff * (2.0 * (Cn[1] - Cn[0]) / dr**2) - S

    # Mesures relatives
    scaleC = max(1.0, np.max(np.abs(Cn)))
    rel_dC0 = np.abs(dC0) / (np.abs(Ce) + 1e-12)
    rel_res = np.abs(res0) / (np.abs(S)  + 1e-30)

    if show_print:
        print(f"[Symétrie] Schéma {scheme}, N={N}")
        print(f"  dC/dr|_0 (approx)  = {dC0: .6e}  (relatif à Ce ≈ {rel_dC0: .3e})")
        print(f"  Résidu centre      = {res0: .6e}  (relatif à S  ≈ {rel_res: .3e})")
        if rel_dC0 < 1e-6 and rel_res < 1e-9:
            print("  -> Test de symétrie : OK (au niveau précision machine).")
        else:
            print("  -> Test de symétrie : vérifier (maillage plus fin ?).")

    return {'dC0': dC0, 'res0': res0, 'rel_dC0': rel_dC0, 'rel_res': rel_res}

# ----------------------------
# EXEMPLES D'UTILISATION (décommente pour lancer)
# ----------------------------
if __name__ == "__main__":
    # --- 1) Erreur sur une grille
    res1 = normes_erreur(80, Deff, S, Ce, R)

    # --- 2) Convergence (tracé)
    Ns = (10, 20, 30, 40, 80 )
    drs, L1s, L2s, LInfs = method2_convergence_plot(Ns, Deff, S, Ce, R)

    # --- 3) Ordre de convergence (impression)
    rep = method3_order_report(Ns, scheme='E')

    # --- 4) Symétrie
    sym = method4_symmetry_test(N=80, scheme='C')