"""
Part B - Automatic Modulation Classification dataset generation.

Problem 5:
Generate noisy I/Q signals for five modulation schemes over
a range of SNR values.
"""

import numpy as np


MODULATIONS = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
]


def make_symbols(mod, n, rng):
    """Generate n unit-average-power symbols for a modulation."""

    if mod == "BPSK":
        return (
            2 * rng.integers(0, 2, n) - 1
        ).astype(complex)

    if mod == "QPSK":
        return np.exp(
            1j * (
                np.pi / 4
                + rng.integers(0, 4, n) * np.pi / 2
            )
        )

    if mod == "8PSK":
        return np.exp(
            1j * (
                rng.integers(0, 8, n) * np.pi / 4
            )
        )

    if mod == "16QAM":
        levels = np.array([-3, -1, 1, 3])

        return (
            rng.choice(levels, n)
            + 1j * rng.choice(levels, n)
        ) / np.sqrt(10)

    if mod == "64QAM":
        levels = np.array(
            [-7, -5, -3, -1, 1, 3, 5, 7]
        )

        return (
            rng.choice(levels, n)
            + 1j * rng.choice(levels, n)
        ) / np.sqrt(42)

    raise ValueError(f"Unknown modulation: {mod}")


def generate_amc_dataset(
    n_per_class_per_snr=100,
    n_symbols=128,
    snrs=range(-10, 21, 2),
    seed=0,
):
    """
    Generate the AMC dataset.

    Returns:
        X      : I/Q signals
                 shape = (N, 2, n_symbols)
        y      : integer modulation labels
        snr_db : SNR value corresponding to each signal
    """

    rng = np.random.default_rng(seed)

    X = []
    y = []
    snr_list = []

    for snr_db in snrs:

        for class_index, mod in enumerate(MODULATIONS):

            for _ in range(n_per_class_per_snr):

                # Generate transmitted symbols
                s = make_symbols(
                    mod,
                    n_symbols,
                    rng
                )

                # Unknown carrier phase
                phase = rng.uniform(
                    0,
                    2 * np.pi
                )

                s = s * np.exp(1j * phase)

                # Signal power
                signal_power = np.mean(
                    np.abs(s) ** 2
                )

                # Noise power from desired SNR
                noise_power = signal_power / (
                    10 ** (snr_db / 10)
                )

                # Complex Gaussian noise
                noise = np.sqrt(
                    noise_power / 2
                ) * (
                    rng.normal(size=n_symbols)
                    + 1j * rng.normal(size=n_symbols)
                )

                # Received signal
                r = s + noise

                # Store I and Q separately
                X.append(
                    np.stack(
                        [r.real, r.imag]
                    )
                )

                y.append(class_index)
                snr_list.append(snr_db)

    return (
        np.array(X),
        np.array(y),
        np.array(snr_list),
    )


if __name__ == "__main__":

    X_iq, y_mod, snr = generate_amc_dataset(
        n_per_class_per_snr=100,
        n_symbols=128,
        seed=0,
    )

    print(
        f"Generated {X_iq.shape[0]} signals"
    )

    print(
        f"Signal shape: {X_iq.shape[1:]}"
    )

    print(
        f"SNR values: {sorted(set(snr))} dB"
    )

    print(
        f"Class balance: {np.bincount(y_mod)}"
    )