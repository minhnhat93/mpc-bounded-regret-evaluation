import os
import pickle
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def create_error_graphs():
    ref_fn = "data/offline/ltv/seed-100"
    reference = pickle.load(open(ref_fn, "rb"))
    u_ref = reference.inputs
    episode_length = len(u_ref)

    NOISE_TYPES = ["disturbance", "full"]
    PREDICTION_NOISES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    PREDICTION_HORIZON = list(range(10, 101, 10))
    SEEDS = list(np.arange(200, 205))

    u_matrix = np.zeros((len(NOISE_TYPES), len(PREDICTION_NOISES), len(PREDICTION_HORIZON), len(SEEDS), episode_length, 2))
    for t_idx, t in enumerate(NOISE_TYPES):
        for n_idx, n in enumerate(PREDICTION_NOISES):
            for k_idx, k in enumerate(PREDICTION_HORIZON):
                for s_idx, s in enumerate(SEEDS):
                    run = pickle.load(open(f"./data/online_cost-corrected/ltv/{t},noise-{n},horizon-{k},seed-{s}", "rb"))
                    u_matrix[t_idx, n_idx, k_idx, s_idx] = run.inputs

    u_diff = (u_matrix - u_ref)
    # absolute per-step error
    per_step_error = np.abs(u_diff)
    per_step_error = np.mean(per_step_error, axis=-1)  # average over the 2 input dimension
    per_step_error = np.mean(per_step_error, axis=-2)  # average over the random seed

    def _plot(per_step_error, title, save_fn):
        num_pn, num_phr = per_step_error.shape[0], per_step_error.shape[1]
        fig, axs = plt.subplots(num_pn, num_phr, figsize=(2 * num_pn, 4))
        fig.suptitle(title, fontsize=15)
        for row in range(num_pn):
            for col in range(num_phr):
                steps = list(range(episode_length))
                ax = axs[row, col]
                ax.plot(steps, per_step_error[row, col], color='blue')
                ax.set_xlabel(f"Prediction horizon: {PREDICTION_HORIZON[col]} * T", fontsize=15)
                ax.set_ylabel(f"Noise: {PREDICTION_NOISES[row]}", fontsize=15)

        for ax in axs.flat:
            ax.label_outer()
        plt.tight_layout()
        plt.savefig(save_fn)

    os.makedirs("./figures/noisy_parameters", exist_ok=True)
    _plot(per_step_error[0],
          "Per-step Error for Multiple Prediction Horizon Ratios and Noise Levels: Noise on Disturbance only",
          save_fn="./figures/noisy_parameters/per-step_error_disturbance.png")
    _plot(per_step_error[1],
          "Per-step Error for multiple Prediction Horizon Ratios and Noise Levels: Noise on All Parameters",
          save_fn="./figures/noisy_parameters/per-step_error_table_all.png")

    print("Finished")

if __name__ == '__main__':
    create_error_graphs()
