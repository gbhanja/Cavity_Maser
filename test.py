"""
==============================================================
N = 1 THREE-LEVEL CAVITY MASER
Numerically extracted Scully-Lamb coefficients
==============================================================

Required observables:

    <n>_ss = <a†a>_ss

    P_out = kappa * hbar * omega_l * <n>_ss

    g2(0) = <a†a†aa> / <a†a>^2

    masing threshold

The full QuTiP master equation is treated as the reference model.

Scully-Lamb coefficients A, Ab, B are extracted numerically from
the steady-state photon-number distribution rather than assuming
an adiabatic-elimination expression for B.
"""

import os
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import least_squares, brentq


# ============================================================
# PARAMETERS
# ============================================================

kappa   = 1.0
g       = 14.0 * kappa

gamma_h = 32.0 * kappa
gamma_c = 32.0 * kappa

nc      = 0.05

n_max   = 50

nh_values = np.linspace(1e-3, 9.0, 60)


# ------------------------------------------------------------
# Power units
# ------------------------------------------------------------
#
# If physical power is required, specify omega_l [rad/s].
#
# Otherwise leave:
#
#     hbar_omega_l = 1
#
# and power is reported in units of hbar*omega_l.
#

hbar_omega_l = 1.0


data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

filename = os.path.join(data_folder,"test.npz")


# ============================================================
# OPERATORS
# ============================================================

def operators(nmax):

    # atomic states
    e  = qt.basis(3, 0)
    gc = qt.basis(3, 1)
    gh = qt.basis(3, 2)

    # bath transitions
    sigma_he = gh * e.dag()
    sigma_ce = gc * e.dag()

    # lasing transition
    #
    # gc -> gh + photon
    #
    sigma_l = gh * gc.dag()

    Ic = qt.qeye(nmax + 1)
    Ia = qt.qeye(3)

    a = qt.destroy(nmax + 1)

    return (
        qt.tensor(sigma_ce, Ic),
        qt.tensor(sigma_he, Ic),
        qt.tensor(sigma_l,  Ic),
        qt.tensor(Ia, a)
    )


# ============================================================
# MICROSCOPIC MASTER EQUATION
# ============================================================

def liouvillian_ops(
    nh,
    nc,
    g,
    kappa,
    gamma_h,
    gamma_c,
    nmax
):

    Sigma_ce, Sigma_he, Sigma_l, a = operators(nmax)

    H = g * (
        Sigma_l * a.dag()
        +
        Sigma_l.dag() * a
    )

    c_ops = [

        # hot bath
        np.sqrt(gamma_h * (nh + 1)) * Sigma_he,
        np.sqrt(gamma_h * nh)       * Sigma_he.dag(),

        # cold bath
        np.sqrt(gamma_c * (nc + 1)) * Sigma_ce,
        np.sqrt(gamma_c * nc)       * Sigma_ce.dag(),

        # cavity loss
        np.sqrt(kappa) * a
    ]

    return H, c_ops, a


def steady_state_solver(
    nh,
    nc,
    g,
    kappa,
    gamma_h,
    gamma_c,
    nmax
):

    H, c_ops, a = liouvillian_ops(
        nh,
        nc,
        g,
        kappa,
        gamma_h,
        gamma_c,
        nmax
    )

    rho = qt.steadystate(
        H,
        c_ops,
        method="direct"
    )

    return rho, a


# ============================================================
# EXACT OBSERVABLES
# ============================================================

def exact_observables(rho, a):

    n_op = a.dag() * a

    n = float(
        np.real(
            qt.expect(n_op, rho)
        )
    )

    factorial2 = float(
        np.real(
            qt.expect(
                a.dag() * a.dag() * a * a,
                rho
            )
        )
    )

    if n > 1e-12:
        g2 = factorial2 / n**2
    else:
        g2 = np.nan

    rho_c = rho.ptrace(1)

    Pn = np.real(
        np.diag(rho_c.full())
    )

    Pn = np.maximum(Pn, 0.0)
    Pn /= Pn.sum()

    Pout = (
        kappa
        * hbar_omega_l
        * n
    )

    return n, Pout, g2, Pn


# ============================================================
# ORIGINAL ANALYTICAL A AND Ab
# ============================================================

def candidate_A_Ab(
    nh,
    nc,
    g,
    gamma_h,
    gamma_c
):

    Gamma = (
        gamma_h * (nh + 1)
        +
        gamma_c * (nc + 1)
    )

    Phi = (
        3 * nh * nc
        +
        2 * (nh + nc)
        +
        1
    )

    A = (
        4 * g**2
        * nh
        * (nc + 1)
        /
        (Gamma * Phi)
    )

    Ab = (
        4 * g**2
        * nc
        * (nh + 1)
        /
        (Gamma * Phi)
    )

    return A, Ab


# ============================================================
# NUMERICAL SCULLY-LAMB EXTRACTION
# ============================================================

def fit_scully_lamb(Pn, kappa, A0=None, Ab0=None):
    """
    Extract A, Ab and B from the exact cavity P_n.

    Scully-Lamb recursion:

        P_n / P_(n-1)
        =
        A /
        [Ab + kappa*(1 + n*B/A)]

    Therefore

        1/r_n
        =
        (Ab+kappa)/A
        +
        kappa*B/A^2 * n

    where

        r_n = P_n/P_(n-1).

    Thus 1/r_n should be linear in n.

    This permits a very stable numerical extraction.

    The P_n ratios determine

        C0 = (Ab+kappa)/A
        C1 = kappa*B/A^2

    but do NOT independently determine A and Ab.

    Therefore A-Ab is supplied by the microscopic
    small-signal gain candidate.  The fitted distribution
    then determines B.
    """

    n = np.arange(1, len(Pn))

    valid = (
        (Pn[1:] > 1e-12)
        &
        (Pn[:-1] > 1e-12)
    )

    nfit = n[valid]

    ratio = (
        Pn[1:][valid]
        /
        Pn[:-1][valid]
    )

    y = 1.0 / ratio

    if len(nfit) < 3:
        return np.nan, np.nan

    # linear model:
    #
    # 1/r_n = C0 + C1*n

    coeff = np.polyfit(
        nfit,
        y,
        1
    )

    C1 = coeff[0]
    C0 = coeff[1]

    # coefficient of determination
    yfit = C0 + C1*nfit

    ss_res = np.sum(
        (y-yfit)**2
    )

    ss_tot = np.sum(
        (y-np.mean(y))**2
    )

    R2 = (
        1.0 - ss_res/ss_tot
        if ss_tot > 0
        else 1.0
    )

    return C0, C1, R2


# ============================================================
# CONSTRUCT NUMERICALLY CALIBRATED A, Ab, B
# ============================================================

def numerical_SL_parameters(
    nh,
    nc,
    g,
    kappa,
    gamma_h,
    gamma_c,
    Pn
):

    # --------------------------------------------------------
    # Microscopic candidate small-signal gain
    # --------------------------------------------------------

    A0, Ab0 = candidate_A_Ab(
        nh,
        nc,
        g,
        gamma_h,
        gamma_c
    )

    G0 = A0 - Ab0


    # --------------------------------------------------------
    # Extract distribution coefficients
    # --------------------------------------------------------

    C0, C1, R2 = fit_scully_lamb(
        Pn,
        kappa
    )

    if np.isnan(C0):
        return {
            "A": np.nan,
            "Ab": np.nan,
            "B": np.nan,
            "G": np.nan,
            "R2": np.nan
        }


    # We know
    #
    # C0 = (Ab + kappa)/A
    #
    # and impose
    #
    # A - Ab = G0
    #
    # Hence
    #
    # Ab = A - G0
    #
    # C0 A = A - G0 + kappa
    #
    # A(C0-1) = kappa-G0

    denominator = C0 - 1.0

    if abs(denominator) < 1e-12:

        A = A0
        Ab = Ab0

    else:

        A = (
            kappa - G0
        ) / denominator

        Ab = A - G0


    # --------------------------------------------------------
    # C1 = kappa B/A^2
    # --------------------------------------------------------

    B = (
        C1
        * A**2
        / kappa
    )

    return {
        "A": A,
        "Ab": Ab,
        "B": B,
        "G": A-Ab,
        "R2": R2
    }


# ============================================================
# SCULLY-LAMB DISTRIBUTION
# ============================================================

def scully_lamb_distribution(
    A,
    Ab,
    B,
    kappa,
    nmax
):

    Pn = np.zeros(nmax+1)

    Pn[0] = 1.0

    if (
        not np.isfinite(A)
        or A <= 0
    ):
        return Pn

    for n in range(1, nmax+1):

        denominator = (
            Ab
            +
            kappa * (
                1
                +
                n * B/A
            )
        )

        if denominator <= 0:
            break

        Pn[n] = (
            Pn[n-1]
            * A
            / denominator
        )

    s = Pn.sum()

    if s > 0:
        Pn /= s

    return Pn


# ============================================================
# OBSERVABLES FROM SCULLY-LAMB
# ============================================================

def distribution_observables(Pn):

    n = np.arange(len(Pn))

    navg = np.sum(
        n * Pn
    )

    factorial2 = np.sum(
        n * (n-1) * Pn
    )

    if navg > 1e-12:
        g2 = factorial2 / navg**2
    else:
        g2 = np.nan

    Pout = (
        kappa
        * hbar_omega_l
        * navg
    )

    return navg, Pout, g2


# ============================================================
# WORKER
# ============================================================

def compute_point(nh):

    rho, a = steady_state_solver(
        nh,
        nc,
        g,
        kappa,
        gamma_h,
        gamma_c,
        n_max
    )

    n_exact, P_exact, g2_exact, Pn_exact = (
        exact_observables(
            rho,
            a
        )
    )

    pars = numerical_SL_parameters(
        nh,
        nc,
        g,
        kappa,
        gamma_h,
        gamma_c,
        Pn_exact
    )

    Pn_SL = scully_lamb_distribution(
        pars["A"],
        pars["Ab"],
        pars["B"],
        kappa,
        n_max
    )

    n_SL, P_SL, g2_SL = (
        distribution_observables(Pn_SL)
    )

    return (
        nh,

        n_exact,
        P_exact,
        g2_exact,

        n_SL,
        P_SL,
        g2_SL,

        pars["A"],
        pars["Ab"],
        pars["B"],
        pars["G"],
        pars["R2"]
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\nRunning microscopic master equation...\n")

    results = Parallel(n_jobs=-1)(
        delayed(compute_point)(nh)
        for nh in tqdm(
            nh_values,
            desc="n_h scan"
        )
    )


    (
        nh_values,

        navg_exact,
        power_exact,
        g2_exact,

        navg_SL,
        power_SL,
        g2_SL,

        A_num,
        Ab_num,
        B_num,
        G_num,
        R2_SL

    ) = map(
        np.array,
        zip(*results)
    )


    # ========================================================
    # THRESHOLD
    # ========================================================
    #
    # Scully-Lamb:
    #
    #       A - Ab = kappa
    #

    threshold = np.nan

    threshold_function = (
        G_num - kappa
    )

    for i in range(len(nh_values)-1):

        if (
            np.isfinite(threshold_function[i])
            and
            np.isfinite(threshold_function[i+1])
            and
            threshold_function[i]
            * threshold_function[i+1]
            < 0
        ):

            # linear interpolation is sufficient because
            # coefficients were obtained on a discrete grid

            x1 = nh_values[i]
            x2 = nh_values[i+1]

            y1 = threshold_function[i]
            y2 = threshold_function[i+1]

            threshold = (
                x1
                -
                y1
                * (x2-x1)
                / (y2-y1)
            )

            break


    print("\n======================================")
    print("MASER RESULTS")
    print("======================================")

    print(
        f"Masing threshold n_h* = "
        f"{threshold:.6f}"
    )

    print(
        f"Minimum Scully-Lamb fit R² = "
        f"{np.nanmin(R2_SL):.6f}"
    )


    # ========================================================
    # SAVE ONLY REQUESTED RESULTS + SL PARAMETERS
    # ========================================================

    np.savez_compressed(

        filename,

        nh_values=nh_values,

        navg=navg_exact,

        output_power=power_exact,

        g2=g2_exact,

        threshold=threshold,

        # Numerical Scully-Lamb coefficients
        A=A_num,
        Ab=Ab_num,
        B=B_num,

        # useful validation
        SL_fit_R2=R2_SL
    )


    print(f"\nSaved results to:\n{filename}")