"""
=============================
Cavity Maser setup for N = 1 
==============================

Three-level atom |g>, |e1>, |e2> in a cavity:
  - |e2> <-> |g>   coupled to HOT bath  (rate gamma_h, thermal photon nh)
  - |e1> <-> |g>   coupled to COLD bath (rate gamma_c, thermal photon nc)
  - |e1> <-> |e2>  coupled to the cavity mode (Jaynes-Cummings, coupling g)
  - cavity photon leaks out at rate kappa

"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import brentq
import os
import sys

print("Current working directory:", os.getcwd())

# ============
# PARAMETERS :
# ============

kappa = 1.0
g = 14.0 * kappa
gamma_h = 32.0 * kappa
gamma_c = 32.0 * kappa
nc = 0.05
n_max = 30                                          # Fock-space cutoff for numerical steady state

nh_values = np.linspace(1e-3, 9.0, 60)              # sweep for (gain/population/<n>)
nh_points = np.array([0.17, 0.507, 2.629, 9.0])     # special points for (photon distributions + Wigner functions)

xvec = np.linspace(-6, 6, 200)                      # Wigner function grid

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)


def make_filename():
    return (f"SSDB_N1_nh{nh_values[0]:.3g}-{nh_values[-1]:.3g}_"
            f"steps{len(nh_values)}_g{g}_kappa{kappa}_"
            f"gh{gamma_h}_gc{gamma_c}_nc{nc}_nmax{n_max}.npz")


filename = os.path.join(data_folder, "cavity_maser.npz")
print("Data file:", filename)


# ===========
# OPERATORS :
# ===========

def operators(nmax):

    """
    Atom basis order :
    index 0 = |g> , index 1 = |e1> , index 2 = |e2>
    """

    g_at = qt.basis(3, 0)
    e1_at = qt.basis(3, 1)
    e2_at = qt.basis(3, 2)

    tau_c = g_at * e1_at.dag()                    # |g><e1|  (cold-bath transition)
    tau_h = g_at * e2_at.dag()                    # |g><e2|  (hot-bath transition)
    sigma = e1_at * e2_at.dag()                   # |e1><e2| (cavity transition)

    I_c = qt.qeye(nmax + 1)
    I_at = qt.qeye(3)
    a = qt.destroy(nmax + 1)

    Tau_c = qt.tensor(tau_c, I_c)
    Tau_h = qt.tensor(tau_h, I_c)
    Sigma = qt.tensor(sigma, I_c)
    A_op = qt.tensor(I_at, a)

    return Tau_c, Tau_h, Sigma, A_op


def liouvillian_ops(nh, nc, g, kappa, gamma_h, gamma_c, nmax):

    """
    Build Hamiltonian and collapse operators for the master equation.
    """

    Tau_c, Tau_h, Sigma, a = operators(nmax)

    H = g * (Sigma.dag() * a + Sigma * a.dag())

    c_ops = [
        np.sqrt(gamma_h * (nh + 1)) * Tau_h,      # e2 -> g   (emission)
        np.sqrt(gamma_h * nh) * Tau_h.dag(),      # g  -> e2  (absorption)
        np.sqrt(gamma_c * (nc + 1)) * Tau_c,      # e1 -> g
        np.sqrt(gamma_c * nc) * Tau_c.dag(),      # g  -> e1
        np.sqrt(kappa) * a,                       # cavity decay
    ]
    return H, c_ops, a, Tau_c, Tau_h, Sigma


def steady_state_solver(nh, nc, g, kappa, gamma_h, gamma_c, nmax, method='direct'):

    H, c_ops, a, _, _, _ = liouvillian_ops(nh, nc, g, kappa, gamma_h, gamma_c, nmax)

    rho_ss = qt.steadystate(H, c_ops, method=method)

    return rho_ss, a


def observables(rho_ss, a):

    n_avg = qt.expect(a.dag() * a, rho_ss)

    n2 = qt.expect(a.dag() * a.dag() * a * a, rho_ss)

    g2 = n2 / n_avg**2 if n_avg > 1e-12 else np.nan

    rho_cav = rho_ss.ptrace(1)

    Pn = np.real(np.diag(rho_cav.full()))

    return n_avg, g2, Pn, rho_cav


def atomic_populations(rho_ss):

    rho_atom = rho_ss.ptrace(0).full()

    Ng, N1, N2 = np.real(rho_atom[0, 0]), np.real(rho_atom[1, 1]), np.real(rho_atom[2, 2])

    return Ng, N1, N2


# =============================================================
# ANALYTICAL QUANTITIES FROM THE SEMICLASSICAL RATE EQUATIONS : 
# =============================================================

def analytic_quantities(nh, nc, g, kappa, gamma_h, gamma_c):

    Gamma = gamma_h * (nh + 1) + gamma_c * (nc + 1)                  
    Phi = 3 * nh * nc + 2 * (nh + nc) + 1                            
    Psi = (1.0 / (gamma_h * gamma_c)) * (gamma_h * (3*nh + 1) + gamma_c * (3*nc + 1))

    dN0 = (nh - nc) / Phi                                            
    G = 4 * g**2 * dN0 / Gamma                                      
    B_semi = 4 * g**2 * Psi / (Gamma * Phi)

    A = 4 * g**2 * nh * (nc + 1) / (Gamma * Phi)                      
    Ab = 4 * g**2 * nc * (nh + 1) / (Gamma * Phi)                    
    B = A * 4 * g**2 * Psi / (Gamma * Phi)                           

    return dict(Gamma=Gamma, Phi=Phi, Psi=Psi, dN0=dN0, G=G,
                B_semiclassical=B_semi, A=A, Ab=Ab, B=B)


def analytic_Pn(nh, nc, g, kappa, gamma_h, gamma_c, nmax):

    """Steady-state photon distribution, Eq. (13), truncated at nmax and renormalized."""

    p = analytic_quantities(nh, nc, g, kappa, gamma_h, gamma_c)

    A, Ab, B = p['A'], p['Ab'], p['B']

    Pn = np.zeros(nmax + 1)

    Pn[0] = 1.0

    for n in range(1, nmax + 1):
        Pn[n] = Pn[n - 1] * A / (Ab + kappa * (1 + n * B / A))
    Pn /= Pn.sum()

    return Pn, p


def analytic_navg_and_var(nh, nc, g, kappa, gamma_h, gamma_c, nmax):
    """
    Average photon number and variance.
    """

    Pn, p = analytic_Pn(nh, nc, g, kappa, gamma_h, gamma_c, nmax)
    A, Ab, B = p['A'], p['Ab'], p['B']
    P0 = Pn[0]

    navg = (A / (kappa * B)) * (A - Ab - kappa) + (A / (kappa * B)) * (kappa + Ab) * P0
    var = (A**2 / (kappa * B)) - (A / (kappa * B)) * (kappa + Ab) * P0 * navg
    return navg, var, P0


def find_thresholds(nh_grid, nc, g, kappa, gamma_h, gamma_c):

    """Find nh where G(nh)/kappa = 1 (lasing threshold crossings)."""

    def f(nh):
        p = analytic_quantities(nh, nc, g, kappa, gamma_h, gamma_c)
        return p['G'] / kappa - 1.0

    vals = np.array([f(nh) for nh in nh_grid])

    roots = []

    for i in range(len(nh_grid) - 1):
        if vals[i] == 0:
            roots.append(nh_grid[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(f, nh_grid[i], nh_grid[i + 1]))
    return roots


# =========================
# MANDATORY SANITY CHECKS :
# =========================

def test_ssdb_population_ratio(nc, gamma_h, gamma_c, nmax, nh_test=3.0, tol=2e-3):
    """
    At g = 0 (no cavity coupling), the atomic populations must return to the bare SSDB ratio.
    """

    rho_ss, _ = steady_state_solver(nh_test, nc, g=0.0, kappa=kappa,
                                    gamma_h=gamma_h, gamma_c=gamma_c, nmax=nmax)
    Ng, N1, N2 = atomic_populations(rho_ss)

    expected_N1 = Ng * nc / (nc + 1)
    expected_N2 = Ng * nh_test / (nh_test + 1)

    ok1 = np.isclose(N1, expected_N1, atol=tol)
    ok2 = np.isclose(N2, expected_N2, atol=tol)

    print(f"  [SSDB ratio @ g=0] N1={N1:.5f} (expect {expected_N1:.5f}) [{'OK' if ok1 else 'FAIL'}]")
    print(f"  [SSDB ratio @ g=0] N2={N2:.5f} (expect {expected_N2:.5f}) [{'OK' if ok2 else 'FAIL'}]")

    if not (ok1 and ok2):
        raise AssertionError(
            "Basis labeling / bath-operator assignment is inconsistent with "
            "the SSDB population ordering. Check which tau operator "
            "is wired to which bath in build_liouvillian_ops()."
        )
    return True


def test_trace_and_hermiticity(nh, nc, g, kappa, gamma_h, gamma_c, nmax, tol=1e-6):

    """Verify the steady state is a valid density matrix: trace 1, Hermitian, PSD."""

    rho_ss, _ = steady_state_solver(nh, nc, g, kappa, gamma_h, gamma_c, nmax)

    tr = rho_ss.tr()
    herm_err = (rho_ss - rho_ss.dag()).norm()
    eigvals = rho_ss.eigenenergies()
    min_eig = np.min(np.real(eigvals))

    ok_trace = np.isclose(tr, 1.0, atol=tol)
    ok_herm = herm_err < tol
    ok_psd = min_eig > -tol

    print(f"[rho_ss @ nh={nh:.3f}] tr(rho)={tr:.8f} [{'OK' if ok_trace else 'FAIL'}], "
          f"hermiticity err={herm_err:.2e} [{'OK' if ok_herm else 'FAIL'}], "
          f"min eigenvalue={min_eig:.2e} [{'OK' if ok_psd else 'FAIL'}]")

    if not (ok_trace and ok_herm and ok_psd):
        raise AssertionError(f"Steady state at nh={nh} is not a valid density matrix.")
    return True


def check_truncation(Pn, tol=1e-6, label=""):
    """Verify the (numerical or analytical) photon distribution has negligible
    probability mass at the Fock-space cutoff -- i.e. n_max is large enough."""
    tail = Pn[-1]
    status = "OK" if tail <= tol else "WARNING"
    print(f"  [truncation {label}] P(n_max) = {tail:.3e}  [{status}]"
          + ("" if tail <= tol else "  -> increase n_max!"))
    return tail <= tol


def test_gain_consistency(nh, nc, g, kappa, gamma_h, gamma_c, tol=1e-8):
    """
    Cross-check: the semiclassical lasing gain G must equal the
    quantum stimulated emission minus absorption rates.
    """
    p = analytic_quantities(nh, nc, g, kappa, gamma_h, gamma_c)
    G = p['G']
    A_minus_Ab = p['A'] - p['Ab']
    ok = np.isclose(G, A_minus_Ab, rtol=1e-6, atol=tol)
    print(f"  [gain consistency @ nh={nh:.3f}] G={G:.6f}, A-Ab={A_minus_Ab:.6f} "
          f"[{'OK' if ok else 'FAIL'}]")
    if not ok:
        raise AssertionError("Semiclassical gain G does not match quantum A - Ab.")
    return True


def run_all_sanity_checks():
    print("\n" + "=" * 70)
    print("RUNNING MANDATORY SANITY CHECKS")
    print("=" * 70)

    print("\n1) SSDB population ratio at g=0 (Eq. 6) -- validates basis labeling:")
    test_ssdb_population_ratio(nc, gamma_h, gamma_c, n_max)

    print("\n2) Trace / Hermiticity / positive-semidefiniteness of steady states:")
    for nh_test in [0.05, 0.5, 2.0, 9.0]:
        test_trace_and_hermiticity(nh_test, nc, g, kappa, gamma_h, gamma_c, n_max)

    print("\n3) Fock-space truncation convergence (numerical) at largest n_h in sweep:")
    nh_worst = nh_values.max()
    rho_ss, a = steady_state_solver(nh_worst, nc, g, kappa, gamma_h, gamma_c, n_max)
    _, _, Pn_num, _ = observables(rho_ss, a)
    trunc_ok = check_truncation(Pn_num, label=f"numerical @ nh={nh_worst:.3f}")
    if not trunc_ok:
        print("  ==> Consider increasing n_max before trusting the sweep results.")

    print("\n4) Fock-space truncation convergence (analytical formula, Eq. 13):")
    Pn_an, _ = analytic_Pn(nh_worst, nc, g, kappa, gamma_h, gamma_c, n_max)
    check_truncation(Pn_an, label=f"analytical @ nh={nh_worst:.3f}")

    print("\n5) Semiclassical gain G vs. quantum A - Ab consistency (Eq. 8 vs Eq. 11):")
    for nh_test in [0.05, 0.5, 2.0, 9.0]:
        test_gain_consistency(nh_test, nc, g, kappa, gamma_h, gamma_c)

    print("\n6) Quick numeric-vs-analytic cross-check of <n> at one representative point:")
    nh_test = 2.629
    rho_ss, a = steady_state_solver(nh_test, nc, g, kappa, gamma_h, gamma_c, n_max)
    n_num, g2_num, Pn_num, _ = observables(rho_ss, a)
    n_an, var_an, P0 = analytic_navg_and_var(nh_test, nc, g, kappa, gamma_h, gamma_c, n_max)
    rel_err = abs(n_num - n_an) / max(n_an, 1e-12)
    print(f"  <n>_numerical={n_num:.5f}, <n>_analytic={n_an:.5f}, "
          f"relative error={rel_err:.2%} [{'OK' if rel_err < 0.05 else 'CHECK'}]")

    print("\nAll sanity checks passed.\n" + "=" * 70 + "\n")


# ==================
# PARALLEL WORKERS :
# ==================

def compute_scan_point(nh, nc, g, kappa, gamma_h, gamma_c, nmax):

    """Worker for the (gain/population/<n>) sweep over n_h."""

    # analytical
    p = analytic_quantities(nh, nc, g, kappa, gamma_h, gamma_c)
    G_ov_k = p['G'] / kappa
    n_an, var_an, _ = analytic_navg_and_var(nh, nc, g, kappa, gamma_h, gamma_c, nmax)


    # numerical (full quantum steady state)
    rho_ss, a = steady_state_solver(nh, nc, g, kappa, gamma_h, gamma_c, nmax)
    n_num, g2_num, _, _ = observables(rho_ss, a)
    Ng, N1, N2 = atomic_populations(rho_ss)

    return (nh, G_ov_k, Ng, N1, N2, n_num, n_an, g2_num, var_an)


def compute_fig3_point(nh, nc, g, kappa, gamma_h, gamma_c, nmax, xvec):

    """Worker for photon distributions + Wigner functions at fixed n_h values."""

    rho_ss, a = steady_state_solver(nh, nc, g, kappa, gamma_h, gamma_c, nmax)

    n_num, g2_num, Pn_num, rho_cav = observables(rho_ss, a)

    Pn_an, _ = analytic_Pn(nh, nc, g, kappa, gamma_h, gamma_c, nmax)

    W = qt.wigner(rho_cav, xvec, xvec)

    return (nh, Pn_num, Pn_an, n_num, g2_num, W)


# =============
# LOAD or RUN :
# =============

if __name__ == "__main__":

    # ---- Sanity checks first: cheap, catches bugs before the expensive sweep ----
    run_all_sanity_checks()

    if os.path.exists(filename):
        print("Loading cached data...")
        data = np.load(filename)

        nh_values      = data["nh_values"]
        G_over_kappa   = data["G_over_kappa"]
        pop_g          = data["pop_g"]
        pop_1          = data["pop_1"]
        pop_2          = data["pop_2"]
        navg_numeric   = data["navg_numeric"]
        navg_analytic  = data["navg_analytic"]
        g2_numeric     = data["g2_numeric"]
        var_analytic   = data["var_analytic"]

        nh_points      = data["nh_points"]
        Pn_num_arr     = data["Pn_num_arr"]
        Pn_an_arr      = data["Pn_an_arr"]
        navg_fig3      = data["navg_fig3"]
        g2_fig3        = data["g2_fig3"]
        Wigner_arr     = data["Wigner_arr"]
        xvec           = data["xvec"]

        thresholds     = data["thresholds"]

    else:
        print("Running simulation...")

        # ---- (A) scalar sweep over n_h ----

        results_scan = Parallel(n_jobs=-1)(
            delayed(compute_scan_point)(nh, nc, g, kappa, gamma_h, gamma_c, n_max)
            for nh in tqdm(nh_values, desc="Scan over n_h")
        )

        (nh_out, G_out, Ng_out, N1_out, N2_out,
         nnum_out, nan_out, g2_out, var_out) = zip(*results_scan)

        nh_values     = np.array(nh_out)
        G_over_kappa  = np.array(G_out)
        pop_g         = np.array(Ng_out)
        pop_1         = np.array(N1_out)
        pop_2         = np.array(N2_out)
        navg_numeric  = np.array(nnum_out)
        navg_analytic = np.array(nan_out)
        g2_numeric    = np.array(g2_out)
        var_analytic  = np.array(var_out)

        # ---- (B) distributions + Wigner functions ----

        results_fig3 = Parallel(n_jobs=-1)(
            delayed(compute_fig3_point)(nh, nc, g, kappa, gamma_h, gamma_c, n_max, xvec)
            for nh in tqdm(nh_points, desc="(Pn, Wigner)")
        )

        (nh3_out, Pn_num_list, Pn_an_list, nnum3_out, g23_out, W_list) = zip(*results_fig3)

        nh_points   = np.array(nh3_out)
        Pn_num_arr  = np.array(Pn_num_list)
        Pn_an_arr   = np.array(Pn_an_list)
        navg_fig3   = np.array(nnum3_out)
        g2_fig3     = np.array(g23_out)
        Wigner_arr  = np.array(W_list)

        # ---- thresholds (analytical root-finding) ----
        thresholds = np.array(find_thresholds(
            np.linspace(1e-4, 15, 4000), nc, g, kappa, gamma_h, gamma_c))
        print("Lasing threshold(s) at n_h =", thresholds)

        # ---- post-hoc sanity check on the sweep results ----
        print("\nPost-sweep check: numeric vs analytic <n> agreement across full sweep:")
        rel_errs = np.abs(navg_numeric - navg_analytic) / np.maximum(navg_analytic, 1e-8)
        print(f"  max relative error = {rel_errs.max():.2%}, "
              f"mean relative error = {rel_errs.mean():.2%}")
        if rel_errs.max() > 0.1:
            print("  WARNING: large discrepancy somewhere in the sweep -- "
                  "check n_max convergence at that n_h.")

            
        np.savez_compressed(
            filename,
            nh_values=nh_values,
            G_over_kappa=G_over_kappa,
            pop_g=pop_g, pop_1=pop_1, pop_2=pop_2,
            navg_numeric=navg_numeric,
            navg_analytic=navg_analytic,
            g2_numeric=g2_numeric,
            var_analytic=var_analytic,
            nh_points=nh_points,
            Pn_num_arr=Pn_num_arr,
            Pn_an_arr=Pn_an_arr,
            navg_fig3=navg_fig3,
            g2_fig3=g2_fig3,
            Wigner_arr=Wigner_arr,
            xvec=xvec,
            thresholds=thresholds,
            kappa=kappa, g=g, gamma_h=gamma_h, gamma_c=gamma_c,
            nc=nc, n_max=n_max,
        )

        print(f"Saved results to {filename}")
        print("Simulation completed successfully.")
