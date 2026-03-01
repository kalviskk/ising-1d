# %%
import numpy as np
import matplotlib.pyplot as plt

filename = "ising1d_fast_adaptive.txt"


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

# Columns: N T repeat C chi domain tau_E tau_M xi_exact
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
            lower.append(np.percentile(v, 25))
            upper.append(np.percentile(v, 75))
        return np.array(medians), np.array(lower), np.array(upper)

    C_med, C_lo, C_hi = calc_stats(C_vals)
    chi_med, chi_lo, chi_hi = calc_stats(chi_vals)
    domain_med, domain_lo, domain_hi = calc_stats(domain_vals)

    T_grid = np.linspace(min(unique_T), max(unique_T), 400)

    # ----------------------------
    # Heat Capacity
    # ----------------------------
    plt.figure()
    # plt.plot(T_vals, C_vals, "o", alpha=0.3, label="Repeats")
    plt.plot(unique_T, C_med, "-", linewidth=2, label="Median")
    plt.fill_between(unique_T, C_lo, C_hi, color="C0", alpha=0.2, label="IQR")
    plt.plot(T_grid, exact_heat_capacity(T_grid), "--", label="Exact")

    plt.xlabel("T")
    plt.ylabel("Heat Capacity")
    plt.title(f"1D Ising Heat Capacity (N={N})")
    plt.grid()
    plt.legend()
    plt.show()

    # ----------------------------
    # Susceptibility
    # ----------------------------
    plt.figure()
    # plt.plot(T_vals, chi_vals, "o", alpha=0.3, label="Repeats")
    plt.plot(unique_T, chi_med, "-", linewidth=2, label="Median")
    plt.fill_between(unique_T, chi_lo, chi_hi, color="C1", alpha=0.2, label="IQR")
    plt.plot(T_grid, exact_susceptibility(T_grid), "--", label="Exact")

    plt.xlabel("T")
    plt.ylabel("Susceptibility")
    plt.title(f"1D Ising Susceptibility (N={N})")
    plt.grid()
    plt.legend()
    plt.show()

    # ----------------------------
    # Domain size vs correlation length
    # ----------------------------
    plt.figure()
    # plt.plot(T_vals, domain_vals, "o", alpha=0.3, label="Repeats")
    plt.plot(unique_T, domain_med, "-", linewidth=2, label="Median")
    plt.fill_between(unique_T, domain_lo, domain_hi, color="C2", alpha=0.2, label="IQR")
    plt.plot(T_grid, correlation_length(T_grid), "--", label="Correlation length")
    plt.plot(
        T_grid,
        domain_to_xi_ratio(T_grid) * correlation_length(T_grid),
        ":",
        label="Theoretical <ℓ>",
    )

    plt.xlabel("T")
    plt.ylabel("Size")
    plt.title(f"Domain Size vs Correlation Length (N={N})")
    plt.grid()
    plt.legend()
    plt.show()
# %%
# ----------------------------
# Time vs N with proper log-log fit
# ----------------------------

time_col = data[:, -1]  # last column = time_sec

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
alpha = coeffs[0]  # slope = scaling exponent
log_a = coeffs[1]  # intercept
a = np.exp(log_a)

print(f"Fitted scaling: t = {a:.3e} * N^{alpha:.3f}")

# Plot
plt.figure()
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
plt.show()

# ----------------------------
# Time vs Temperature
# ----------------------------
unique_T = np.unique(T_col)
median_times_T = []
lower_times_T = []
upper_times_T = []

for T in unique_T:
    mask = T_col == T
    times = time_col[mask]
    median_times_T.append(np.median(times))
    lower_times_T.append(np.percentile(times, 25))
    upper_times_T.append(np.percentile(times, 75))

median_times_T = np.array(median_times_T)
lower_times_T = np.array(lower_times_T)
upper_times_T = np.array(upper_times_T)
yerr_T = [median_times_T - lower_times_T, upper_times_T - median_times_T]

# Plot
plt.figure()
plt.errorbar(
    unique_T, median_times_T, yerr=yerr_T, fmt="o", capsize=5, label="Median ± IQR"
)
plt.xlabel("Temperature T")
plt.ylabel("Calculation Time [s]")
plt.title("Calculation Time vs Temperature")
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Prepare arrays
time_col = data[:, -1]
T_vals = data[:, 1]
N_vals = data[:, 0].astype(int)

unique_N = np.unique(N_vals)
unique_T = np.unique(T_vals)

# Compute median time for each (N, T)
median_time_grid = np.zeros((len(unique_T), len(unique_N)))

for i, T in enumerate(unique_T):
    for j, N in enumerate(unique_N):
        mask = (N_vals == N) & (T_vals == T)
        if np.any(mask):
            median_time_grid[i, j] = np.median(time_col[mask])
        else:
            median_time_grid[i, j] = np.nan  # missing data

# Plot heatmap
plt.figure(figsize=(8, 6))
im = plt.imshow(
    median_time_grid,
    origin="lower",
    aspect="auto",
    extent=[min(unique_N), max(unique_N), min(unique_T), max(unique_T)],
    cmap="viridis",
    norm=LogNorm(),
)
plt.colorbar(im, label="Median Calculation Time [s]")
plt.xlabel("System Size N")
plt.ylabel("Temperature T")
plt.title("Calculation Time vs N and T (median)")
plt.show()
