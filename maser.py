# ============================================================
# Stage 1: N = 1 cavity-assisted 3-level Scovil-Schulz-DuBois
# heat engine / single-atom maser
#
# Based on:
#   1. SRCavityMaserv2.pdf, Stage 1, page 6
#   2. Li et al., Phys. Rev. A 96, 063806 (2017)
#
# The code:
#   - builds the 3-level atom + cavity Hilbert space
#   - implements hot-bath and cold-bath dissipators
#   - implements cavity loss
#   - solves the full master equation numerically
#   - computes cavity photon number
#   - computes g^(2)(0)
#   - computes output power
#   - computes Li et al. analytical gain
#   - computes Li et al. analytical photon distribution
#   - compares numerical and analytical results
#   - scans n_h to reveal the two threshold crossings
#
# Install first, if needed:
#     pip install qutip numpy scipy matplotlib
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


# ============================================================
# 1. PARAMETERS
# ============================================================

# We use the same normalized parameters as Fig. 2 of Li et al.
#
# Li et al.:
#     gamma_h = gamma_c = 32*kappa
#     g       = 14*kappa
#     n_c     = 0.05
#
# We set kappa = 1, so all rates are expressed in units of kappa.
# ============================================================

kappa = 1.0

gamma_h = 32.0 * kappa
gamma_c = 32.0 * kappa

g = 14.0 * kappa

n_c = 0.05

# Cavity Fock-space cutoff
# Increase this to 40-60 if the photon distribution approaches
# the cutoff.
N_CAV = 30

# Laser/output transition frequency.
#
# The steady-state dynamics in the resonant interaction picture
# does not require the absolute value of omega_l.
#
# omega_l = 1 gives output power in normalized units:
#     P = hbar * omega_l * kappa * <a^\dagger a>
#
# You can replace omega_l with a physical angular frequency.
omega_l = 1.0

hbar = 1.054571817e-34


# ============================================================
# 2. THREE-LEVEL ATOM
#
# Basis:
#     |g>  = basis(3,0)
#     |e1> = basis(3,1)
#     |e2> = basis(3,2)
#
# Li et al. define
#
#     tau_h^- = |g><e2|
#     tau_c^- = |g><e1|
#     sigma^- = |e1><e2|
#
# and sigma^+ = (sigma^-)^\dagger.
# ============================================================

g_state = qt.basis(3, 0)
e1_state = qt.basis(3, 1)
e2_state = qt.basis(3, 2)

I_atom = qt.qeye(3)


# Atomic lowering operators
tau_h_minus_atom = g_state * e2_state.dag()
tau_c_minus_atom = g_state * e1_state.dag()

sigma_minus_atom = e1_state * e2_state.dag()
sigma_plus_atom = sigma_minus_atom.dag()


# Atomic projectors
P_g_atom = g_state * g_state.dag()
P_e1_atom = e1_state * e1_state.dag()
P_e2_atom = e2_state * e2_state.dag()


# ============================================================
# 3. CAVITY OPERATORS
# ============================================================

a_cav = qt.destroy(N_CAV)
adag_cav = a_cav.dag()

I_cav = qt.qeye(N_CAV)

n_cav = adag_cav * a_cav


# ============================================================
# 4. TENSOR-PRODUCT OPERATORS
#
# Hilbert-space ordering:
#
#       atom x cavity
#
# so every atom operator is tensor(operator, I_cav)
# and every cavity operator is tensor(I_atom, operator).
# ============================================================

tau_h_minus = qt.tensor(tau_h_minus_atom, I_cav)
tau_h_plus = tau_h_minus.dag()

tau_c_minus = qt.tensor(tau_c_minus_atom, I_cav)
tau_c_plus = tau_c_minus.dag()

sigma_minus = qt.tensor(sigma_minus_atom, I_cav)
sigma_plus = sigma_minus.dag()

P_g = qt.tensor(P_g_atom, I_cav)
P_e1 = qt.tensor(P_e1_atom, I_cav)
P_e2 = qt.tensor(P_e2_atom, I_cav)

a = qt.tensor(I_atom, a_cav)
adag = a.dag()

n_op = adag * a


# ============================================================
# 5. HAMILTONIAN
#
# Li et al. work in the resonant interaction picture:
#
#     V = g (sigma^+ a + sigma^- a^\dagger)
#
# Therefore the fast bare atomic/cavity frequencies are removed.
#
# This is exactly the appropriate starting point for Stage 1.
# ============================================================

H_atoms = 0 * qt.tensor(I_atom, I_cav)

H_cav = 0 * qt.tensor(I_atom, I_cav)

H_int = g * (sigma_plus * a + sigma_minus * adag)

H = H_atoms + H_cav + H_int


# ============================================================
# 6. LINDblad COLLAPSE OPERATORS
#
# Li et al. use
#
# L_i[rho] =
#     gamma_i*n_i D[tau_i^+]
#   + gamma_i*(n_i+1) D[tau_i^-]
#
# with
#
# D[c]rho = c rho c^\dagger
#          - 1/2 {c^\dagger c,rho}
#
# Therefore the QuTiP collapse operators are
#
#     sqrt(gamma_i*n_i)       * tau_i^+
#     sqrt(gamma_i*(n_i+1))   * tau_i^-
#
# Cavity:
#
#     sqrt(kappa) * a
#
# IMPORTANT:
# This uses Li et al.'s kappa convention, for which
#
#     da/dt = -(kappa/2) a - i g sigma^-
#
# as in their Eq. (3).
# ============================================================

def build_collapse_operators(n_h):
    """
    Construct the collapse operators for a given hot-bath
    thermal photon number n_h.
    """

    c_ops = []

    # -------------------------
    # Hot bath
    # -------------------------
    # Absorption: |g> -> |e2>
    c_ops.append(
        np.sqrt(gamma_h * n_h) * tau_h_plus
    )

    # Emission: |e2> -> |g>
    c_ops.append(
        np.sqrt(gamma_h * (n_h + 1.0)) * tau_h_minus
    )

    # -------------------------
    # Cold bath
    # -------------------------
    # Absorption: |g> -> |e1>
    c_ops.append(
        np.sqrt(gamma_c * n_c) * tau_c_plus
    )

    # Emission: |e1> -> |g>
    c_ops.append(
        np.sqrt(gamma_c * (n_c + 1.0)) * tau_c_minus
    )

    # -------------------------
    # Cavity loss
    # -------------------------
    c_ops.append(
        np.sqrt(kappa) * a
    )

    return c_ops


# ============================================================
# 7. ANALYTICAL QUANTITIES FROM LI et al.
#
# Li et al. define
#
#   Gamma = gamma_h(n_h+1) + gamma_c(n_c+1)
#
#   Psi =
#       [ gamma_h(3 n_h + 1)
#       + gamma_c(3 n_c + 1) ]
#       / (gamma_h gamma_c)
#
#   Phi = 3 n_h n_c + 2(n_h+n_c) + 1
#
# The linear lasing gain is
#
#   G = 4 g^2/Gamma * (n_h-n_c)/Phi
#
# and the lasing threshold is
#
#   G/kappa = 1
#
# Li et al. report two critical points near
#
#   n_h = 0.187
#   n_h = 8.647
#
# for their Fig. 2 parameters.
# ============================================================

def li_analytical_parameters(n_h):
    """
    Return Gamma, Psi, Phi, A, Abar, B and G
    from Li et al.'s analytical treatment.
    """

    Gamma = (
        gamma_h * (n_h + 1.0)
        + gamma_c * (n_c + 1.0)
    )

    Psi = (
        gamma_h * (3.0 * n_h + 1.0)
        + gamma_c * (3.0 * n_c + 1.0)
    ) / (gamma_h * gamma_c)

    Phi = (
        3.0 * n_h * n_c
        + 2.0 * (n_h + n_c)
        + 1.0
    )

    # Stimulated-emission coefficient
    A = (
        4.0 * g**2 * n_h * (n_c + 1.0)
        / (Gamma * Phi)
    )

    # Stimulated-absorption coefficient
    Abar = (
        4.0 * g**2 * n_c * (n_h + 1.0)
        / (Gamma * Phi)
    )

    # Saturation coefficient
    B = A * (4.0 * g**2 * Psi / Phi)

    # Lasing gain
    G = 4.0 * g**2 * (n_h - n_c) / (Gamma * Phi)

    return Gamma, Psi, Phi, A, Abar, B, G


# ============================================================
# 8. ANALYTICAL PHOTON NUMBER DISTRIBUTION
#
# Li et al. Eq. (13):
#
#   P_n / P_(n-1)
#       = A / [ Abar + kappa * (1 + n B/A) ]
#
# We construct P_n recursively and normalize.
# ============================================================

def analytical_photon_distribution(n_h, Nmax=N_CAV):
    """
    Compute the analytical steady-state photon-number
    distribution P_n from Li et al. Eq. (13).
    """

    _, _, _, A, Abar, B, G = li_analytical_parameters(n_h)

    P = np.zeros(Nmax, dtype=float)

    # Special case: zero hot-bath occupation
    if A <= 0.0:
        P[0] = 1.0
        return P, G

    P[0] = 1.0

    for n in range(1, Nmax):

        ratio = A / (
            Abar
            + kappa * (1.0 + n * B / A)
        )

        P[n] = P[n - 1] * ratio

    # Normalize
    total = np.sum(P)

    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError(
            "Analytical photon distribution could not be normalized."
        )

    P /= total

    return P, G


# ============================================================
# 9. NUMERICAL STEADY STATE
# ============================================================

def solve_steady_state(n_h):
    """
    Solve the full atom+cavity master equation for a
    given hot-bath photon occupation n_h.
    """

    c_ops = build_collapse_operators(n_h)

    # Direct steady-state solution
    rho_ss = qt.steadystate(
        H,
        c_ops,
        method="direct"
    )

    return rho_ss


# ============================================================
# 10. NUMERICAL OBSERVABLES
# ============================================================

def calculate_observables(rho_ss):
    """
    Extract the main Stage-1 observables.
    """

    # Mean cavity photon number
    nbar = qt.expect(n_op, rho_ss)
    nbar = float(np.real(nbar))

    # <a^\dagger a^\dagger a a>
    n2_factorial = qt.expect(
        adag * adag * a * a,
        rho_ss
    )

    n2_factorial = float(np.real(n2_factorial))

    # g^(2)(0)
    if nbar > 1e-14:
        g2 = n2_factorial / (nbar ** 2)
    else:
        g2 = np.nan

    # Atomic populations
    Pg = float(np.real(qt.expect(P_g, rho_ss)))
    Pe1 = float(np.real(qt.expect(P_e1, rho_ss)))
    Pe2 = float(np.real(qt.expect(P_e2, rho_ss)))

    # Population inversion on the masing transition
    inversion = Pe2 - Pe1

    # Output power:
    #
    # P = hbar * omega_l * kappa * <a^\dagger a>
    #
    # If omega_l=1 and hbar=1 this is simply kappa*nbar.
    P_out = hbar * omega_l * kappa * nbar

    return {
        "nbar": nbar,
        "g2": g2,
        "Pg": Pg,
        "Pe1": Pe1,
        "Pe2": Pe2,
        "inversion": inversion,
        "P_out": P_out,
    }


# ============================================================
# 11. SINGLE POINT CALCULATION
# ============================================================

def run_single_point(n_h, plot_distribution=False):
    """
    Run one complete N=1 calculation.
    """

    print("\n" + "=" * 65)
    print(f"Hot bath photon number n_h = {n_h:.6f}")
    print("=" * 65)

    # Numerical steady state
    rho_ss = solve_steady_state(n_h)

    # Numerical observables
    obs = calculate_observables(rho_ss)

    # Analytical gain
    (
        Gamma,
        Psi,
        Phi,
        A,
        Abar,
        B,
        G
    ) = li_analytical_parameters(n_h)

    # Analytical distribution
    P_analytic, _ = analytical_photon_distribution(
        n_h,
        Nmax=N_CAV
    )

    # Numerical reduced cavity density matrix
    rho_cavity = rho_ss.ptrace(1)

    # Diagonal = photon-number probabilities
    P_numeric = np.real(rho_cavity.diag())

    # Numerical normalization
    P_numeric = np.asarray(P_numeric, dtype=float)
    P_numeric /= np.sum(P_numeric)

    # Print results
    print(f"Numerical <n>         = {obs['nbar']:.10f}")
    print(f"Numerical g2(0)       = {obs['g2']:.10f}")
    print(f"Ground population     = {obs['Pg']:.10f}")
    print(f"|e1> population       = {obs['Pe1']:.10f}")
    print(f"|e2> population       = {obs['Pe2']:.10f}")
    print(f"Inversion              = {obs['inversion']:.10f}")

    print()
    print(f"Analytical Gamma      = {Gamma:.10f}")
    print(f"Analytical Phi        = {Phi:.10f}")
    print(f"Analytical Psi        = {Psi:.10f}")
    print(f"A                     = {A:.10f}")
    print(f"Abar                  = {Abar:.10f}")
    print(f"B                     = {B:.10f}")
    print(f"Gain G                 = {G:.10f}")
    print(f"Gain G/kappa           = {G/kappa:.10f}")

    print()
    print(f"Output power           = {obs['P_out']:.6e} W")

    # Check the cavity cutoff
    cutoff_probability = P_numeric[-1]

    if cutoff_probability > 1e-6:
        print()
        print("WARNING:")
        print(
            f"Photon probability at cutoff P({N_CAV-1}) "
            f"= {cutoff_probability:.3e}"
        )
        print(
            "Increase N_CAV to 40, 50, or larger."
        )

    # --------------------------------------------------------
    # Compare numerical and analytical photon distributions
    # --------------------------------------------------------

    if plot_distribution:

        nvals = np.arange(N_CAV)

        plt.figure(figsize=(8, 5))

        width = 0.38

        plt.bar(
            nvals - width / 2,
            P_numeric,
            width=width,
            alpha=0.7,
            label="Numerical master equation"
        )

        plt.bar(
            nvals + width / 2,
            P_analytic,
            width=width,
            alpha=0.7,
            label="Li et al. analytical"
        )

        plt.xlabel("Photon number n")
        plt.ylabel("P(n)")
        plt.title(
            f"Photon distribution, n_h = {n_h}"
        )

        plt.xlim(-0.5, min(N_CAV - 0.5, 15.5))

        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "rho_ss": rho_ss,
        "P_numeric": P_numeric,
        "P_analytic": P_analytic,
        "observables": obs,
        "Gamma": Gamma,
        "Psi": Psi,
        "Phi": Phi,
        "A": A,
        "Abar": Abar,
        "B": B,
        "G": G,
    }


# ============================================================
# 12. TEST THE FOUR POINTS USED IN LI et al. FIG. 3
#
# Li et al. use approximately:
#
#     n_h = 0.17     below threshold
#     n_h = 0.507    above threshold
#     n_h = 2.629    above threshold
#     n_h = 9.0      below the second threshold
#
# Their published critical points are approximately:
#
#     n_h = 0.187
#     n_h = 8.647
#
# ============================================================

benchmark_points = [
    0.17,
    0.507,
    2.629,
    9.0
]

benchmark_results = {}

for nh_value in benchmark_points:

    benchmark_results[nh_value] = run_single_point(
        nh_value,
        plot_distribution=True
    )


# ============================================================
# 13. SCAN HOT-BATH PHOTON NUMBER
#
# This produces:
#
#   (1) numerical cavity photon number
#   (2) analytical lasing gain G/kappa
#   (3) numerical output power
# ============================================================

nh_scan = np.linspace(0.01, 10.0, 120)

nbar_scan = []
power_scan = []
gain_scan = []
g2_scan = []

for nh_value in nh_scan:

    rho_ss = solve_steady_state(nh_value)

    obs = calculate_observables(rho_ss)

    _, _, _, _, _, _, G = li_analytical_parameters(
        nh_value
    )

    nbar_scan.append(obs["nbar"])
    power_scan.append(obs["P_out"])
    gain_scan.append(G / kappa)
    g2_scan.append(obs["g2"])


nbar_scan = np.asarray(nbar_scan)
power_scan = np.asarray(power_scan)
gain_scan = np.asarray(gain_scan)
g2_scan = np.asarray(g2_scan)


# ============================================================
# 14. LOCATE THRESHOLD CROSSINGS NUMERICALLY
#
# Threshold occurs at
#
#       G/kappa = 1
#
# The crossing is based on Li et al.'s analytical gain,
# while the photon number is calculated from the full
# numerical steady-state master equation.
# ============================================================

threshold_crossings = []

for i in range(len(nh_scan) - 1):

    f1 = gain_scan[i] - 1.0
    f2 = gain_scan[i + 1] - 1.0

    if f1 == 0.0:
        threshold_crossings.append(nh_scan[i])

    elif f1 * f2 < 0.0:

        # Linear interpolation
        x1 = nh_scan[i]
        x2 = nh_scan[i + 1]

        y1 = gain_scan[i]
        y2 = gain_scan[i + 1]

        x_threshold = (
            x1
            + (1.0 - y1) * (x2 - x1)
            / (y2 - y1)
        )

        threshold_crossings.append(x_threshold)


print("\n" + "=" * 65)
print("THRESHOLD CHECK")
print("=" * 65)

for j, threshold in enumerate(
        threshold_crossings, start=1):

    print(
        f"Numerical scan crossing {j}: "
        f"n_h ≈ {threshold:.6f}"
    )

print()
print("Li et al. published values:")
print("    first threshold  ≈ 0.187")
print("    second threshold ≈ 8.647")


# ============================================================
# 15. PLOT: CAVITY PHOTON NUMBER
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    nh_scan,
    nbar_scan,
    linewidth=2,
    label="Numerical <a†a>"
)

plt.xlabel("Hot bath photon number n_h")
plt.ylabel("Cavity photon number <n>")
plt.title("N = 1: Cavity photon number")

# Mark Li et al. threshold locations
plt.axvline(
    0.187,
    linestyle="--",
    alpha=0.7,
    label="Li et al. threshold ~0.187"
)

plt.axvline(
    8.647,
    linestyle="--",
    alpha=0.7,
    label="Li et al. threshold ~8.647"
)

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 16. PLOT: ANALYTICAL GAIN
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    nh_scan,
    gain_scan,
    linewidth=2,
    label="Li et al. analytical G/kappa"
)

plt.axhline(
    1.0,
    linestyle="--",
    alpha=0.7,
    label="Threshold: G/kappa = 1"
)

plt.axvline(
    0.187,
    linestyle="--",
    alpha=0.7
)

plt.axvline(
    8.647,
    linestyle="--",
    alpha=0.7
)

plt.xlabel("Hot bath photon number n_h")
plt.ylabel("G / kappa")
plt.title("N = 1: Lasing gain and double threshold")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 17. PLOT: OUTPUT POWER
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    nh_scan,
    power_scan,
    linewidth=2,
    label="Numerical output power"
)

plt.xlabel("Hot bath photon number n_h")
plt.ylabel("Output power (W)")
plt.title("N = 1: Masing output power")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 18. PLOT: g^(2)(0)
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    nh_scan,
    g2_scan,
    linewidth=2
)

plt.axhline(
    1.0,
    linestyle="--",
    alpha=0.7,
    label="g^(2)(0) = 1"
)

plt.xlabel("Hot bath photon number n_h")
plt.ylabel("g^(2)(0)")
plt.title("N = 1: Cavity photon statistics")

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STAGE 1 SUMMARY")
print("=" * 65)

print("System:")
print("    N = 1 three-level emitter + single cavity mode")

print()
print("Parameters:")
print(f"    kappa    = {kappa}")
print(f"    gamma_h  = {gamma_h}")
print(f"    gamma_c  = {gamma_c}")
print(f"    g        = {g}")
print(f"    n_c      = {n_c}")
print(f"    N_CAV    = {N_CAV}")

print()
print("The code has implemented:")
print("    H_atoms")
print("    H_cav")
print("    H_int")
print("    hot-bath dissipator")
print("    cold-bath dissipator")
print("    cavity-loss dissipator")
print("    steady-state master equation")
print("    photon-number distribution")
print("    lasing gain")
print("    output power")
print("    g^(2)(0)")

print()
print("No Gamma_ij or Omega_ij terms are used.")
print("Those are intentionally reserved for Stage 2 (N = 2).")