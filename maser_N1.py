"""
================================================================================
N = 1 THREE-LEVEL CAVITY MASER 
Analytical Scully-Lamb Formulation & Full Quantum Master Equation
================================================================================

Level Structure:
    |e>  = index 0  (common upper excited state)
    |gc> = index 1  (cold ground state, upper lasing level)
    |gh> = index 2  (hot ground state,  lower lasing level)

Couplings:
    Cavity mode:    |gc> <-> |gh|, lowering operator sigma_l = |gh><gc|
    Cold reservoir: |e>  <-> |gc|, rates gamma_c*(nc+1) and gamma_c*nc
    Hot reservoir:  |e>  <-> |gh|, rates gamma_h*(nh+1) and gamma_h*nh
    Cavity decay:   Rate kappa, jump operator a

Analytical Scully-Lamb Parameters (Lambda-system):
    Gamma = gamma_h*nh + gamma_c*nc
    Phi   = 3*nh*nc + nh + nc
    Psi   = [gamma_h*(2*nh + 1) + gamma_c*(2*nc + 1)] / (gamma_h*gamma_c)
    A     = 4*g^2 * nh*(nc + 1) / (Gamma * Phi)
    Ab    = 4*g^2 * nc*(nh + 1) / (Gamma * Phi)
    B     = A * 4*g^2 * Psi / (Gamma * Phi)
    G     = A - Ab = 4*g^2 * (nh - nc) / (Gamma * Phi)

Threshold Condition:
    G = kappa  <=>  A - Ab = kappa
================================================================================
"""

import os
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm
from joblib import Parallel, delayed

# =============
# PARAMETERS
# =============

kappa   = 1.0
g       = 14.0 * kappa
gamma_h = 32.0 * kappa
gamma_c = 32.0 * kappa
nc      = 0.05
n_max   = 60                          # Fock-space truncation dimension


hbar_omega_l = 1.0

# Hot-bath occupation grid:
# Dense near turn-on (nh ~ 0.0504) and covering quenching (nh ~ 21.16)
nh_values = np.unique(np.concatenate([
    np.linspace(1e-3, 0.15, 45),
    np.linspace(0.18, 5.0,  35),
    np.linspace(5.5,  24.0, 40)
]))

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)
filename = os.path.join(data_folder, "maser_N1.npz")


# ====================================
# ANALYTICAL SCULLY-LAMB FORMULATION 
# ====================================

def scully_lamb_parameters_lambda(nh, nc, g, kappa, gamma_h, gamma_c):
    """Calculates exact Scully-Lamb coefficients for the Lambda-type system."""
    Gamma = gamma_h * nh + gamma_c * nc
    Phi   = 3.0 * nh * nc + nh + nc
    Psi   = (gamma_h * (2.0 * nh + 1.0) + gamma_c * (2.0 * nc + 1.0)) / (gamma_h * gamma_c)

    # Avoid zero-division at exact zero temperature
    if Gamma <= 1e-15 or Phi <= 1e-15:
        return dict(Gamma=0.0, Phi=0.0, Psi=0.0, A=0.0, Ab=0.0, B=0.0, G=0.0)

    A  = 4.0 * g**2 * nh * (nc + 1.0) / (Gamma * Phi)
    Ab = 4.0 * g**2 * nc * (nh + 1.0) / (Gamma * Phi)
    B  = A * (4.0 * g**2 * Psi) / (Gamma * Phi)
    G  = A - Ab

    return dict(Gamma=Gamma, Phi=Phi, Psi=Psi, A=A, Ab=Ab, B=B, G=G)


def analytic_photon_distribution(nh, nc, g, kappa, gamma_h, gamma_c, nmax):
    """
    Computes P_n using the detailed-balance Scully-Lamb recurrence relation:
        P_n / P_{n-1} = A / [Ab + kappa * (1 + n * B / A)]
    """
    p = scully_lamb_parameters_lambda(nh, nc, g, kappa, gamma_h, gamma_c)
    A, Ab, B = p['A'], p['Ab'], p['B']

    Pn = np.zeros(nmax + 1)
    Pn[0] = 1.0

    if A <= 1e-14:
        return Pn, p

    for n in range(1, nmax + 1):
        denom = Ab + kappa * (1.0 + n * (B / A))
        if denom <= 0:
            break
        Pn[n] = Pn[n - 1] * (A / denom)
        if Pn[n] < 1e-25 * Pn[0]:     # Truncate negligible probability tails
            break

    total = np.sum(Pn)
    if total > 0:
        Pn /= total

    return Pn, p


def analytic_observables(nh, nc, g, kappa, gamma_h, gamma_c, nmax, hbar_w=1.0):
    """Calculates <a†a>, Output Power, and g^(2)(0) analytically from P_n."""
    Pn, p = analytic_photon_distribution(nh, nc, g, kappa, gamma_h, gamma_c, nmax)
    ns = np.arange(len(Pn))

    n_ss = float(np.sum(ns * Pn))
    n2   = float(np.sum(ns * (ns - 1.0) * Pn))   # <a†2 a2>
    Pout = kappa * hbar_w * n_ss
    g2   = (n2 / (n_ss**2)) if n_ss > 1e-12 else np.nan

    return n_ss, Pout, g2, Pn, p


def find_analytical_thresholds(nc, g, kappa, gamma_h, gamma_c):
    """Finds exact roots of G(nh) = kappa."""
    def root_func(nh):
        p = scully_lamb_parameters_lambda(nh, nc, g, kappa, gamma_h, gamma_c)
        return p['G'] - kappa

    # Search in two intervals: [nc + 1e-6, 1.0] and [1.0, 30.0]
    roots = []
    for span in [(nc + 1e-5, 1.0), (1.0, 30.0)]:
        try:
            if root_func(span[0]) * root_func(span[1]) < 0:
                r = brentq(root_func, span[0], span[1])
                roots.append(float(r))
        except ValueError:
            pass
    return roots


# ===================================
#  QUANTUM OPTICAL MASTER EQUATION 
# ===================================

def build_operators(nmax):
    """Constructs tensor operators for the Lambda-maser system."""
    e_at  = qt.basis(3, 0)
    gc_at = qt.basis(3, 1)
    gh_at = qt.basis(3, 2)

    sigma_he = gh_at * e_at.dag()    # |gh><e|: hot bath
    sigma_ce = gc_at * e_at.dag()    # |gc><e|: cold bath
    sigma_l  = gh_at * gc_at.dag()   # |gh><gc|: lasing transition

    I_c  = qt.qeye(nmax + 1)
    I_at = qt.qeye(3)
    a    = qt.destroy(nmax + 1)

    Sigma_he = qt.tensor(sigma_he, I_c)
    Sigma_ce = qt.tensor(sigma_ce, I_c)
    Sigma_l  = qt.tensor(sigma_l,  I_c)
    A_cav    = qt.tensor(I_at, a)

    return Sigma_ce, Sigma_he, Sigma_l, A_cav


def build_liouvillian(nh, nc, g, kappa, gamma_h, gamma_c, nmax):
    """Builds Hamiltonian and collapse operators."""
    Sigma_ce, Sigma_he, Sigma_l, a = build_operators(nmax)

    H = g * (Sigma_l * a.dag() + Sigma_l.dag() * a)

    c_ops = [
        np.sqrt(gamma_h * (nh + 1.0)) * Sigma_he,
        np.sqrt(gamma_h * nh)         * Sigma_he.dag(),
        np.sqrt(gamma_c * (nc + 1.0)) * Sigma_ce,
        np.sqrt(gamma_c * nc)         * Sigma_ce.dag(),
        np.sqrt(kappa)                * a
    ]
    return H, c_ops, a


def numerical_steady_state(nh, nc, g, kappa, gamma_h, gamma_c, nmax):
    """Solves L[rho] = 0 for the steady-state density matrix."""
    H, c_ops, a = build_liouvillian(nh, nc, g, kappa, gamma_h, gamma_c, nmax)
    rho_ss = qt.steadystate(H, c_ops, method='direct')
    return rho_ss, a


def compute_numerical_observables(rho_ss, a, kappa, hbar_w=1.0):
    """Calculates <a†a>, Output Power, and g^(2)(0) from rho_ss."""
    n_op   = a.dag() * a
    n2_op  = a.dag() * a.dag() * a * a

    n_ss   = float(np.real(qt.expect(n_op, rho_ss)))
    n2     = float(np.real(qt.expect(n2_op, rho_ss)))
    Pout   = kappa * hbar_w * n_ss
    g2     = (n2 / (n_ss**2)) if n_ss > 1e-12 else np.nan

    rho_cav = rho_ss.ptrace(1)
    Pn      = np.real(np.diag(rho_cav.full()))
    Pn      = np.maximum(Pn, 0.0)
    Pn     /= np.sum(Pn)

    return n_ss, Pout, g2, Pn


# ===========================
#  WORKER & SANITY CHECKS
# ===========================

def run_point(nh, nc, g, kappa, gamma_h, gamma_c, nmax, hbar_w):
    """Computes both numerical and analytical points for a given nh."""
    # Numerical
    rho_ss, a = numerical_steady_state(nh, nc, g, kappa, gamma_h, gamma_c, nmax)
    n_num, P_num, g2_num, Pn_num = compute_numerical_observables(rho_ss, a, kappa, hbar_w)

    # Analytical
    n_an, P_an, g2_an, Pn_an, pars = analytic_observables(nh, nc, g, kappa, gamma_h, gamma_c, nmax, hbar_w)

    return (nh, n_num, P_num, g2_num, n_an, P_an, g2_an, pars['G'])


def sanity_checks():
    print("=" * 70)
    print("RUNNING SANITY CHECKS (Lambda-Type Convention)")
    print("=" * 70)

    # Analytical threshold verification
    thrs = find_analytical_thresholds(nc, g, kappa, gamma_h, gamma_c)
    print(f"1) Analytical Thresholds (roots of G = kappa):")
    print(f"   Turn-on  threshold: nh* = {thrs[0]:.5f}")
    print(f"   Quenching threshold: nh* = {thrs[1]:.5f}")
    assert np.isclose(thrs[0], 0.0504, atol=1e-3), "Turn-on threshold mismatch!"
    assert np.isclose(thrs[1], 21.16, atol=1e-1), "Quenching threshold mismatch!"

    # Detailed balance population ratio at g = 0
    nh_test = 2.0
    rho_0, _ = numerical_steady_state(nh_test, nc, g=0.0, kappa=kappa,
                                      gamma_h=gamma_h, gamma_c=gamma_c, nmax=5)
    diag_at = np.real(np.diag(rho_0.ptrace(0).full()))
    Pe, Pgc, Pgh = diag_at[0], diag_at[1], diag_at[2]

    Phi_val = 3.0 * nh_test * nc + nh_test + nc
    expected_Pe  = (nh_test * nc) / Phi_val
    expected_Pgc = (nh_test * (nc + 1.0)) / Phi_val
    expected_Pgh = (nc * (nh_test + 1.0)) / Phi_val

    print(f"\n2) g = 0 Atomic Populations check at nh = {nh_test}:")
    print(f"   P(e) : num = {Pe:.5f},  theory = {expected_Pe:.5f}")
    print(f"   P(gc): num = {Pgc:.5f}, theory = {expected_Pgc:.5f}")
    print(f"   P(gh): num = {Pgh:.5f}, theory = {expected_Pgh:.5f}")
    assert np.isclose(Pgc - Pgh, (nh_test - nc) / Phi_val, atol=1e-5), "Inversion mismatch!"
    print("   Populations detailed-balance verified [OK]")

    # Density matrix validity
    tr = float(np.real(rho_0.tr()))
    min_eig = float(np.min(np.real(rho_0.eigenenergies())))
    assert np.isclose(tr, 1.0, atol=1e-6) and min_eig > -1e-6, "Density matrix invalid!"
    print("\n3) Steady-state trace and positivity verified [OK]")
    print("=" * 70 + "\n")


# ==============================
#  SIMULATION & DATA SAVING
# ==============================

if __name__ == "__main__":

    sanity_checks()

    thresholds = find_analytical_thresholds(nc, g, kappa, gamma_h, gamma_c)

    print("Running hot-bath (nh) parameter sweep...")
    results = Parallel(n_jobs=-1)(
        delayed(run_point)(nh, nc, g, kappa, gamma_h, gamma_c, n_max, hbar_omega_l)
        for nh in tqdm(nh_values, desc="Simulating")
    )

    (nh_arr, n_num, P_num, g2_num,
     n_an, P_an, g2_an, G_arr) = map(np.array, zip(*results))

    # Save to file
    np.savez_compressed(
        filename,
        nh_values=nh_arr,
        navg_numerical=n_num,
        navg_analytical=n_an,
        power_numerical=P_num,
        power_analytical=P_an,
        g2_numerical=g2_num,
        g2_analytical=g2_an,
        gain=G_arr,
        thresholds=np.array(thresholds),
        kappa=kappa, g=g, gamma_h=gamma_h, gamma_c=gamma_c, nc=nc
    )
    print(f"Data saved to {filename}")
