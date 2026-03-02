# %%
import os
import numpy as np
import matplotlib.pyplot as plt

filename = "ising1d_fast_adaptive.txt"

# Directory to save figures
fig_dir = "ising1d-figures"
os.makedirs(fig_dir, exist_ok=True)


# ----------------------------
# Exact 1D Ising formulas
# ----------------------------
def exact_heat_capacity(T):
    beta = 1.0 / T
    return (beta**2) / (np.cosh(beta) ** 2)


def exact_susceptibility(T):
    beta = 1.0 / T
    return beta * np.exp(2 * beta)


def correlation_length(T):
    beta = 1.0 / T
    return -1.0 / np.log(np.tanh(beta))


def domain_to_xi_ratio(T):
    t = np.tanh(1.0 / T)
    return (2 / (1 - t)) / (-1.0 / np.log(t))


# ----------------------------
# Load data
# ----------------------------
data = np.loadtxt(filename)

# Columns: N T repeat C chi domain tau_E tau_M xi_exact time_sec
N_col = data[:, 0].astype(int)
T_col = data[:, 1]
C_col = data[:, 3]
chi_col = data[:, 4]
domain_col = data[:, 5]

unique_N = sorted(set(N_col))

# ----------------------------
# Plot per N
# ----------------------------
for N in unique_N:

    mask = N_col == N
    T_vals = T_col[mask]
    C_vals = C_col[mask]
    chi_vals = chi_col[mask]
    domain_vals = domain_col[mask]

    unique_T = np.unique(T_vals)

    # Compute median and IQR
    def calc_stats(vals):
        medians = []
        lower = []
        upper = []
        for T in unique_T:
            idx = T_vals == T
            v = vals[idx]
            medians.append(np.median(v))
            lower.append(np.percentile(v, 0))
            upper.append(np.percentile(v, 100))
        return np.array(medians), np.array(lower), np.array(upper)

    C_med, C_lo, C_hi = calc_stats(C_vals)
    chi_med, chi_lo, chi_hi = calc_stats(chi_vals)
    domain_med, domain_lo, domain_hi = calc_stats(domain_vals)

    T_grid = np.linspace(min(unique_T), max(unique_T), 400)

    # ----------------------------
    # Heat Capacity
    # ----------------------------
    plt.figure(dpi=200)
    plt.plot(unique_T, C_med, "-", linewidth=2, label="Median")
    plt.fill_between(unique_T, C_lo, C_hi, color="C0", alpha=0.2, label="IQR")
    plt.plot(T_grid, exact_heat_capacity(T_grid), "--", label="Exact")
    if N == 50000:
        plt.axvline(0.64)
    plt.xlabel("T")
    plt.ylabel("Heat Capacity")
    plt.title(f"1D Ising Heat Capacity (N={N})")
    plt.grid()
    plt.legend()
    # plt.savefig(
    #     os.path.join(fig_dir, f"heat_capacity_N{N}.png"), dpi=200, bbox_inches="tight"
    # )
    plt.show()

    # ----------------------------
    # Susceptibility
    # ----------------------------
    plt.figure(dpi=200)
    plt.plot(unique_T, chi_med, "-", linewidth=2, label="Median")
    plt.fill_between(unique_T, chi_lo, chi_hi, color="C1", alpha=0.2, label="IQR")
    plt.plot(T_grid, exact_susceptibility(T_grid), "--", label="Exact")
    if N == 50000:
        plt.axvline(0.64)
    plt.xlabel("T")
    plt.ylabel("Susceptibility")
    plt.title(f"1D Ising Susceptibility (N={N})")
    plt.grid()
    plt.legend()
    # plt.savefig(
    #     os.path.join(fig_dir, f"susceptibility_N{N}.png"), dpi=200, bbox_inches="tight"
    # )
    plt.show()

    # ----------------------------
    # Domain size vs correlation length
    # ----------------------------
    plt.figure(dpi=200)
    plt.plot(unique_T, domain_med, "-", linewidth=2, label="Median")
    plt.fill_between(unique_T, domain_lo, domain_hi, color="C2", alpha=0.2, label="IQR")
    plt.plot(T_grid, correlation_length(T_grid), "--", label="Correlation length")
    plt.plot(
        T_grid,
        domain_to_xi_ratio(T_grid) * correlation_length(T_grid),
        ":",
        label="Theoretical <ℓ>",
    )
    if N == 50000:
        plt.axvline(0.64)
    plt.xlabel("T")
    plt.ylabel("Size")
    plt.title(f"Domain Size vs Correlation Length (N={N})")
    plt.grid()
    plt.legend()
    # plt.savefig(
    #     os.path.join(fig_dir, f"domain_vs_corr_N{N}.png"), dpi=200, bbox_inches="tight"
    # )
    plt.show()

# %%
# ----------------------------
# Time vs N with proper log-log fit
# ----------------------------
time_col = data[:, -1]

median_times = []
lower_times = []
upper_times = []

for N in unique_N:
    mask = N_col == N
    times = time_col[mask]
    median_times.append(np.median(times))
    lower_times.append(np.percentile(times, 25))
    upper_times.append(np.percentile(times, 75))

median_times = np.array(median_times)
lower_times = np.array(lower_times)
upper_times = np.array(upper_times)
N_arr = np.array(unique_N)

# Take logs
log_N = np.log(N_arr)
log_time = np.log(median_times)

# Linear fit in log-log space
coeffs = np.polyfit(log_N[:-1], log_time[:-1], 1)
alpha = coeffs[0]
a = np.exp(coeffs[1])

print(f"Fitted scaling: t = {a:.3e} * N^{alpha:.3f}")

plt.figure(dpi=200)
plt.errorbar(
    N_arr,
    median_times,
    yerr=[median_times - lower_times, upper_times - median_times],
    fmt="o",
    capsize=5,
    label="Median ± IQR",
)
plt.plot(N_arr, a * N_arr**alpha, "--", label=f"Fit: t ∼ N^{alpha:.2f}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("System Size N (log scale)")
plt.ylabel("Calculation Time [s] (log scale)")
plt.title("Scaling of Calculation Time with N (log-log)")
plt.grid(True, which="both", ls="--")
plt.legend()
# plt.savefig(os.path.join(fig_dir, "time_vs_N.png"), dpi=200, bbox_inches="tight")
plt.show()

# %%
# ----------------------------
# Time vs N at fixed temperatures
# ----------------------------
T_values = np.arange(0.5, 1.01, 0.1)
colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

plt.figure(dpi=200)

for i, T_fixed in enumerate(T_values):
    mask_T = np.isclose(T_col, T_fixed)
    N_vals = N_col[mask_T]
    time_vals = time_col[mask_T]

    if len(N_vals) == 0:
        continue

    median_times = []
    lower_times = []
    upper_times = []
    N_list = []

    for N in unique_N:
        mask_N = N_vals == N
        if np.any(mask_N):
            vals = time_vals[mask_N]
            median_times.append(np.median(vals))
            lower_times.append(np.percentile(vals, 25))
            upper_times.append(np.percentile(vals, 75))
            N_list.append(N)

    N_arr = np.array(N_list)
    median_times = np.array(median_times)
    lower_times = np.array(lower_times)
    upper_times = np.array(upper_times)

    if len(N_arr) < 3:
        continue

    # Exclude last N from fit
    log_N = np.log(N_arr[:-1])
    log_time = np.log(median_times[:-1])
    coeffs = np.polyfit(log_N, log_time, 1)
    alpha = coeffs[0]
    a = np.exp(coeffs[1])
    print(f"T={T_fixed:.1f}: t ≈ {a:.3e} * N^{alpha:.3f}")

    color = colors[i % len(colors)]
    plt.plot(N_arr, median_times, "o", alpha=0.6, color=color)
    plt.plot(
        N_arr,
        a * N_arr**alpha,
        "--",
        color=color,
        label=f"T={T_fixed:.1f}, α={alpha:.2f}",
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("System Size N (log scale)")
plt.ylabel("Calculation Time [s] (log scale)")
plt.title("Scaling of Calculation Time with N at Fixed T")
plt.grid(True, which="both", ls="--")
plt.legend()
# plt.savefig(os.path.join(fig_dir, "time_vs_N_fixedT.png"), dpi=200, bbox_inches="tight")
plt.show()

# %%
# ----------------------------
# Time vs Temperature
# ----------------------------
unique_N = unique_N[:-1]  # exclude last N

for N in unique_N:
    mask = N_col == N
    T_vals = T_col[mask]
    time_vals = time_col[mask]

    unique_T = np.unique(T_vals)

    med = []
    lo = []
    hi = []

    for T in unique_T:
        vals = time_vals[T_vals == T]
        med.append(np.median(vals))
        lo.append(np.percentile(vals, 25))
        hi.append(np.percentile(vals, 75))

    med = np.array(med)
    lo = np.array(lo)
    hi = np.array(hi)

    # log-log fit
    log_T = np.log(unique_T)
    log_med = np.log(med)
    coeffs = np.polyfit(log_T, log_med, 1)
    alpha = coeffs[0]
    a = np.exp(coeffs[1])
    print(f"N={N}: t ≈ {a:.3e} * T^{alpha:.3f}")

    T_grid = np.linspace(min(unique_T), max(unique_T), 400)
    fit_curve = a * T_grid**alpha

    plt.figure(dpi=200)
    plt.plot(unique_T, med, "-", linewidth=2, label="Median")
    plt.fill_between(unique_T, lo, hi, alpha=0.2, label="IQR")
    plt.plot(T_grid, fit_curve, "--", label=f"Fit: T^{alpha:.2f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("T")
    plt.ylabel("Calculation Time [s]")
    plt.title(f"Time vs T (N={N})")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # plt.savefig(
    #     os.path.join(fig_dir, f"time_vs_T_N{N}.png"), dpi=200, bbox_inches="tight"
    # )
    plt.show()
