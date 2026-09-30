"""
Part B - AMC feature extraction.

Problem 5(c):
Amplitude/phase statistics and higher-order cumulants.
"""

import numpy as np


FEATURE_NAMES_AMC = [
    "std_amp",
    "mean_amp2",
    "std_amp2",
    "kurtosis_amp",
    "skew_amp",
    "abs_C20",
    "abs_C40",
    "abs_C42",
    "C40_norm",
    "C42_norm",
    "std_phase",
    "mean_abs_phase",
]


def extract_amc_features(X_iq_batch):
    """
    Extract 12 features from a batch of I/Q signals.

    Input:
        X_iq_batch:
            shape = (N, 2, n_symbols)

    Output:
        Feature matrix:
            shape = (N, 12)
    """

    features = []

    for iq in X_iq_batch:

        # Reconstruct complex received signal
        r = iq[0] + 1j * iq[1]

        # Normalize to unit average power
        r = r / (
            np.sqrt(
                np.mean(
                    np.abs(r) ** 2
                )
            ) + 1e-12
        )

        # Amplitude
        a = np.abs(r)

        # Higher-order cumulants
        C20 = np.mean(r ** 2)

        C21 = np.mean(
            np.abs(r) ** 2
        )

        C40 = (
            np.mean(r ** 4)
            - 3 * C20 ** 2
        )

        C42 = (
            np.mean(np.abs(r) ** 4)
            - np.abs(C20) ** 2
            - 2 * C21 ** 2
        )

        # Amplitude statistics
        a_mean = np.mean(a)
        a_std = np.std(a) + 1e-12

        # Phase
        phase = np.angle(r)

        features.append([
            np.std(a),

            np.mean(a ** 2),

            np.std(a ** 2),

            np.mean(
                ((a - a_mean) / a_std) ** 4
            ),

            np.mean(
                ((a - a_mean) / a_std) ** 3
            ),

            np.abs(C20),

            np.abs(C40),

            np.abs(C42),

            np.abs(C40) / (
                C21 ** 2 + 1e-12
            ),

            np.abs(C42) / (
                C21 ** 2 + 1e-12
            ),

            np.std(phase),

            np.mean(
                np.abs(phase)
            ),
        ])

    return np.array(features)


if __name__ == "__main__":

    # Import dataset generator
    from dataset import generate_amc_dataset

    X_iq, y_mod, snr = generate_amc_dataset(
        n_per_class_per_snr=100,
        n_symbols=128,
        seed=0,
    )

    F_amc = extract_amc_features(X_iq)

    print(
        f"Feature matrix shape: {F_amc.shape}"
    )

    print(
        f"Number of features: {len(FEATURE_NAMES_AMC)}"
    )

    print(
        "Contains NaN/inf:",
        np.any(
            ~np.isfinite(F_amc)
        )
    )