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
# Exclude N=20
unique_N = [N for N in unique_N if N != 20]


# ----------------------------
# Helper to compute median + IQR
# ----------------------------
def calc_stats(vals, T_vals, unique_T):
    medians, lower, upper = [], [], []
    for T in unique_T:
        v = vals[T_vals == T]
        medians.append(np.median(v))
        lower.append(np.percentile(v, 25))
        upper.append(np.percentile(v, 75))
    return np.array(medians), np.array(lower), np.array(upper)


# ----------------------------
# Function to plot 4x5 grid
# ----------------------------
# ----------------------------
# Function to plot 4x5 grid with single legend
# ----------------------------
def plot_grid(data_dict, ylabel, title, filename, exact_func=None, ratio_func=None):
    n_plots = len(data_dict)
    ncols = 5
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(20, 12), dpi=200, sharex=False, sharey=False
    )
    axes = axes.flatten()

    # Keep track of plotted handles/labels for global legend
    legend_handles = {}

    for ax, (N, stats) in zip(axes, data_dict.items()):
        T_vals, med, lo, hi = stats
        T_grid = np.linspace(min(T_vals), max(T_vals), 400)

        (h1,) = ax.plot(T_vals, med, "-", linewidth=2)
        h2 = ax.fill_between(T_vals, lo, hi, alpha=0.2)
        if exact_func is not None:
            (h3,) = ax.plot(T_grid, exact_func(T_grid), "--")
        if ratio_func is not None:
            (h4,) = ax.plot(
                T_grid, ratio_func(T_grid) * correlation_length(T_grid), ":"
            )

        # Collect legend entries once
        if "Median" not in legend_handles:
            legend_handles["Median"] = h1
        if "IQR" not in legend_handles:
            legend_handles["IQR"] = h2
        if exact_func is not None and "Exact" not in legend_handles:
            legend_handles["Exact"] = h3
        if ratio_func is not None and "Theoretical <ℓ>" not in legend_handles:
            legend_handles["Theoretical <ℓ>"] = h4

        if N == 50000:
            ax.axvline(0.64)
        ax.set_title(f"N={N}")
        ax.grid(True)

    # Hide any unused subplots
    for ax in axes[n_plots:]:
        ax.axis("off")

    # fig.suptitle(title, fontsize=18)
    fig.text(0.5, 0.04, "T", ha="center", fontsize=14)
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical", fontsize=14)

    # Create a single legend with distinct entries
    fig.legend(legend_handles.values(), legend_handles.keys(), loc="upper right")

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, filename), dpi=500, bbox_inches="tight")
    plt.show()


# ----------------------------
# Prepare data dictionaries
# ----------------------------
C_dict = {}
chi_dict = {}
domain_dict = {}

for N in unique_N:
    mask = N_col == N
    T_vals = np.unique(T_col[mask])

    C_med, C_lo, C_hi = calc_stats(C_col[mask], T_col[mask], T_vals)
    chi_med, chi_lo, chi_hi = calc_stats(chi_col[mask], T_col[mask], T_vals)
    domain_med, domain_lo, domain_hi = calc_stats(domain_col[mask], T_col[mask], T_vals)

    C_dict[N] = (T_vals, C_med, C_lo, C_hi)
    chi_dict[N] = (T_vals, chi_med, chi_lo, chi_hi)
    domain_dict[N] = (T_vals, domain_med, domain_lo, domain_hi)

# ----------------------------
# Plot grouped figures
# ----------------------------
plot_grid(
    C_dict,
    "Heat Capacity",
    "1D Ising Heat Capacity per N",
    "heat_capacity_grid.png",
    exact_func=exact_heat_capacity,
)
plot_grid(
    chi_dict,
    "Susceptibility",
    "1D Ising Susceptibility per N",
    "susceptibility_grid.png",
    exact_func=exact_susceptibility,
)
plot_grid(
    domain_dict,
    "Domain Size",
    "Domain Size vs Correlation Length per N",
    "domain_vs_corr_grid.png",
    ratio_func=domain_to_xi_ratio,
)
