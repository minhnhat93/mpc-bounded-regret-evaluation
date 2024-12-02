import pickle
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize



def plot_matrix_heatmap(matrix, row_idxs, column_idxs, row_label, column_label,
                        figsize=(10, 6), title=None, format="4.1f", save_fn=None, cbar=False, cmap="icefire"):
    df = pd.DataFrame(matrix, index=row_idxs,
                      columns=column_idxs)
    fig = plt.figure(figsize=figsize)
    #annot = df.apply(lambda x: f"{x:.2f}" if x < 1000 else f"{x:.2g}")
    s = sn.heatmap(df, annot=True, fmt=format, cbar=cbar, cmap=cmap, linewidth=.5)
    s.set_xlabel(column_label, fontsize=15)
    s.set_ylabel(row_label, fontsize=15)
    if title:
        fig.suptitle(title, fontsize=15)
    if save_fn:
        plt.savefig(save_fn)
    plt.tight_layout()
    return s

def create_dynamic_regret_heatmap_noisy_prediction():
    ref_fn = "data.backup/offline/ltv/seed-100"
    reference = pickle.load(open(ref_fn, "rb"))
    ref_cost = sum(reference.step_costs)

    NOISE_TYPES = ["disturbance", "full"]
    PREDICTION_NOISES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    PREDICTION_HORIZON = list(range(10, 101, 10))
    SEEDS = list(np.arange(200, 205))

    cost_matrix = np.zeros((len(NOISE_TYPES), len(PREDICTION_NOISES), len(PREDICTION_HORIZON), len(SEEDS)))
    for t_idx, t in enumerate(NOISE_TYPES):
        for n_idx, n in enumerate(PREDICTION_NOISES):
            for k_idx, k in enumerate(PREDICTION_HORIZON):
                for s_idx, s in enumerate(SEEDS):
                    run = pickle.load(open(f"./data.backup/online_cost-corrected/ltv/{t},noise-{n},horizon-{k},seed-{s}", "rb"))
                    cost_matrix[t_idx, n_idx, k_idx, s_idx] = sum(run.step_costs)

    regret_matrix = cost_matrix - ref_cost
    regret_mean = np.mean(regret_matrix, axis=-1)
    regret_std = np.std(regret_matrix, axis=-1)

    row_idxs = [e for e in PREDICTION_NOISES]
    col_idxs = [e for e in PREDICTION_HORIZON]

    plot_matrix_heatmap(regret_mean[0], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on Disturbance Only: Mean Cost", save_fn="./figures/cost_mean_disturbance_only.png")
    plot_matrix_heatmap(regret_mean[1], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on All Parameters: Mean Cost", save_fn="./figures/cost_mean_full.png")
    plot_matrix_heatmap(regret_std[0], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on Disturbance Only: Standard Deviation",
                        save_fn="./figures/cost_std_disturbance_only.png")
    plot_matrix_heatmap(regret_std[1], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on All Parameters: Standard Deviation",
                        save_fn="./figures/cost_std_full.png")
    print("Finished")

def create_dynamic_regret_heatmap_nn_prediction():
    ref_fn = "data.backup/offline/ltv/seed-100"
    reference = pickle.load(open(ref_fn, "rb"))
    ref_cost = sum(reference.step_costs)

    NOISE_TYPES = ["disturbance", "full"]
    PREDICTION_NOISES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    PREDICTION_HORIZON = list(range(10, 101, 10))
    SEEDS = list(np.arange(200, 205))

    cost_matrix = np.zeros((len(NOISE_TYPES), len(PREDICTION_NOISES), len(PREDICTION_HORIZON), len(SEEDS)))
    for t_idx, t in enumerate(NOISE_TYPES):
        for n_idx, n in enumerate(PREDICTION_NOISES):
            for k_idx, k in enumerate(PREDICTION_HORIZON):
                for s_idx, s in enumerate(SEEDS):
                    run = pickle.load(open(f"./data.backup/online_cost-corrected/ltv/{t},noise-{n},horizon-{k},seed-{s}", "rb"))
                    cost_matrix[t_idx, n_idx, k_idx, s_idx] = sum(run.step_costs)

    regret_matrix = cost_matrix - ref_cost
    regret_mean = np.mean(regret_matrix, axis=-1)
    regret_std = np.std(regret_matrix, axis=-1)

    row_idxs = [e for e in PREDICTION_NOISES]
    col_idxs = [e for e in PREDICTION_HORIZON]

    plot_matrix_heatmap(regret_mean[0], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on Disturbance Only: Mean Dynamic Regret", save_fn="./figures/cost_mean_disturbance_only.png")
    plot_matrix_heatmap(regret_mean[1], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on All Parameters: Mean Dynamic Regret", save_fn="./figures/cost_mean_full.png")
    plot_matrix_heatmap(regret_std[0], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on Disturbance Only: Dynamic Regret Standard Deviation",
                        save_fn="./figures/cost_std_disturbance_only.png")
    plot_matrix_heatmap(regret_std[1], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        title="Noisy Prediction on All Parameters: Dynamic Regret Standard Deviation",
                        save_fn="./figures/cost_std_full.png")
    print("Finished")

if __name__ == '__main__':
    create_dynamic_regret_heatmap_noisy_prediction()