import numpy as np
import os
from numba import njit
import time  # <- add this at the top

# =====================================
# Parameters
# =====================================
N_values = [
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    200,
    300,
    500,
    600,
    700,
    800,
    900,
    1000,
    2000,
    10000,
    20000,
    50000,
]
# N_values = [200]

temperatures = np.linspace(0.5, 1.0, 51)

repeats = 10
thermalization_sweeps = 2000
block_sweeps = 1000
M_required = 100
max_blocks = 3000
measurement_interval = 5

output_file = "ising1d_fast_adaptive.txt"


# =====================================
# Numba Metropolis Sweep
# =====================================
@njit
def metropolis_sweep(spins, beta):
    N = spins.shape[0]
    for _ in range(N):
        i = np.random.randint(0, N)
        left = spins[(i - 1) % N]
        right = spins[(i + 1) % N]
        dE = 2.0 * spins[i] * (left + right)

        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            spins[i] = -spins[i]


@njit
def compute_energy(spins):
    N = spins.shape[0]
    E = 0.0
    for i in range(N):
        E -= spins[i] * spins[(i + 1) % N]
    return E


@njit
def compute_magnetization(spins):
    return np.sum(spins)


@njit
def mean_domain_size(spins):
    N = spins.shape[0]
    walls = 0
    for i in range(N):
        if spins[i] != spins[(i + 1) % N]:
            walls += 1
    if walls == 0:
        return N
    return N / walls


# =====================================
# FFT Autocorrelation
# =====================================
def autocorrelation_time(data):
    x = np.asarray(data, dtype=np.float64)
    n = len(x)
    if n < 10:
        return 1.0

    x -= np.mean(x)
    size = 1 << (2 * n - 1).bit_length()

    f = np.fft.fft(x, size)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]

    tau = 0.5
    for t in range(1, n):
        if acf[t] <= 0:
            break
        tau += acf[t]
        if t > 5.0 * tau:
            break

    return max(tau, 1.0)


# =====================================
# Correlation length (exact 1D)
# =====================================
def correlation_length(T):
    beta = 1.0 / T
    return -1.0 / np.log(np.tanh(beta))


# =====================================
# Restart handling
# =====================================
completed = set()

if os.path.exists(output_file):
    data = np.loadtxt(output_file)
    if data.ndim == 1 and len(data) > 0:
        completed.add((int(data[0]), float(data[1]), int(data[2])))
    elif data.ndim > 1:
        for row in data:
            completed.add((int(row[0]), float(row[1]), int(row[2])))


# =====================================
# Main run
# =====================================
with open(output_file, "a") as f:

    if os.stat(output_file).st_size == 0:
        f.write("# N T repeat C chi domain tau_E tau_M xi_exact time_sec\n")
        f.flush()

    for N in N_values:
        for T in temperatures:
            T = round(float(T), 5)

            for r in range(repeats):

                if (N, T, r) in completed:
                    print(f"Skipping N={N}, T={T}, repeat={r}")
                    continue

                print(f"Running N={N}, T={T}, repeat={r}")

                start_time = time.time()  # start timer

                beta = 1.0 / T
                spins = np.random.choice(np.array([-1, 1]), size=N)

                # ---- Thermalization
                for _ in range(thermalization_sweeps):
                    metropolis_sweep(spins, beta)

                E_series = []
                M_series = []
                domain_series = []

                blocks = 0

                while blocks < max_blocks:

                    for sweep in range(block_sweeps):
                        metropolis_sweep(spins, beta)

                        if sweep % measurement_interval == 0:
                            E_series.append(compute_energy(spins))
                            M_series.append(compute_magnetization(spins))
                            domain_series.append(mean_domain_size(spins))

                    blocks += 1

                    if len(E_series) < 50:
                        continue

                    tau_E = autocorrelation_time(E_series)
                    tau_M = autocorrelation_time(M_series)

                    total_samples = len(E_series)
                    tau_max = max(tau_E, tau_M)

                    if total_samples / tau_max > M_required:
                        break

                E_arr = np.array(E_series)
                M_arr = np.array(M_series)

                C = beta**2 * (np.var(E_arr) / N)
                chi = beta * (np.var(M_arr) / N)
                domain_mean = np.mean(domain_series)
                xi = correlation_length(T)

                end_time = time.time()
                elapsed = end_time - start_time  # elapsed time in seconds

                f.write(
                    f"{N} {T:.5f} {r} "
                    f"{C:.8f} {chi:.8f} {domain_mean:.8f} "
                    f"{tau_E:.4f} {tau_M:.4f} {xi:.8f} {elapsed:.4f}\n"
                )
                f.flush()
                os.fsync(f.fileno())

                print(
                    f"Saved N={N}, T={T}, repeat={r}, "
                    f"samples={len(E_series)}, tau≈{tau_max:.1f}, "
                    f"time={elapsed:.1f}s"
                )

print("Finished.")
