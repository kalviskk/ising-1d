import numpy as np
import matplotlib.pyplot as plt

filename = "ising1d_fast_adaptive.txt"


# -------------------------------------
# Correlation length (1D Ising, J=1)
# -------------------------------------
def correlation_length(T):
    beta = 1.0 / T
    return -1.0 / np.log(np.tanh(beta))


# -------------------------------------
# Load data
# -------------------------------------
data = np.loadtxt(filename)

# Columns:
# N T repeat C chi domain tau_E tau_M xi_exact
N_col = data[:, 0].astype(int)
T_col = data[:, 1]
domain_col = data[:, 5]

unique_N = sorted(set(N_col))


def exact_ratio(T):
    beta = 1.0 / T
    t = np.tanh(beta)
    xi = -1.0 / np.log(t)
    domain = 2.0 / (1.0 - t)
    return domain / xi


for N in unique_N:

    mask = N_col == N
    T_vals = T_col[mask]
    domain_vals = data[mask, 5]

    unique_T = np.unique(T_vals)

    ratio_vals = []

    for T in unique_T:
        idx = T_vals == T
        domain_mean = np.mean(domain_vals[idx])
        xi = correlation_length(T)
        ratio_vals.append(domain_mean / xi)

    ratio_vals = np.array(ratio_vals)

    # exact curve
    T_dense = np.linspace(min(unique_T), max(unique_T), 300)
    ratio_exact = exact_ratio(T_dense)

    plt.figure()
    plt.plot(unique_T, ratio_vals, "o", label="Simulation")
    plt.plot(T_dense, ratio_exact, "-", label="Exact")
    plt.xlabel("T")
    plt.ylabel("<ℓ> / ξ")
    plt.title(f"<ℓ>/ξ (N={N})")
    plt.legend()
    plt.grid()
    plt.show()
