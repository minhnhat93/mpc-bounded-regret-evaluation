import os
import pickle
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize



def plot_matrix_heatmap(matrix, row_idxs, column_idxs, row_label, column_label,
                        figsize=(12, 7), title=None, format="4.4f", save_fn=None, cbar=False, cmap=None,
                        log_plot=True):
    df = pd.DataFrame(matrix, index=row_idxs,
                      columns=column_idxs)
    fig = plt.figure(figsize=figsize)
    #annot = df.apply(lambda x: f"{x:.2f}" if x < 1000 else f"{x:.2g}")
    if log_plot:
        s = sn.heatmap(df, annot=True, fmt=format, cbar=cbar, cmap=cmap, linewidth=.5, norm=LogNorm())
    else:
        s = sn.heatmap(df, annot=True, fmt=format, cbar=cbar, cmap=cmap, linewidth=.5)
    s.set_xlabel(column_label, fontsize=15)
    s.set_ylabel(row_label, fontsize=15)
    if title:
        fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    if save_fn:
        plt.savefig(save_fn)
    plt.close()

def create_dynamic_regret_heatmap_noisy_prediction():
    ref_fn = "data/offline/ltv/seed-100"
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
                    run = pickle.load(open(f"./data/online_cost-corrected/ltv/{t},noise-{n},horizon-{k},seed-{s}", "rb"))
                    cost_matrix[t_idx, n_idx, k_idx, s_idx] = sum(run.step_costs)

    regret_matrix = cost_matrix - ref_cost
    regret_mean = np.mean(regret_matrix, axis=-1)
    regret_std = np.std(regret_matrix, axis=-1)
    regret_std[:, 0] = 1e-8

    row_idxs = [e for e in PREDICTION_NOISES]
    col_idxs = [e for e in PREDICTION_HORIZON]

    os.makedirs("./figures/noisy_parameters", exist_ok=True)
    plot_matrix_heatmap(regret_mean[0], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon", format="4.4f", log_plot=True,
                        title="Noises on Disturbance only: Mean Dynamic Regret", save_fn="./figures/noisy_parameters/regret_mean_disturbance_only.png")
    plot_matrix_heatmap(regret_mean[1], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon",
                        format="4.4g", log_plot=True,
                        title="Noises on All Parameters: Mean Dynamic Regret", save_fn="./figures/noisy_parameters/regret_mean_full.png")
    plot_matrix_heatmap(regret_std[0], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon", format="4.4f",
                        title="Noises on Disturbance only: Standard Deviation of Dynamic Regret",
                        save_fn="./figures/noisy_parameters/regret_std_disturbance_only.png")
    plot_matrix_heatmap(regret_std[1], row_idxs, col_idxs, "Noise Strength", "Prediction Horizon", format="4.4g",
                        title="Noises on All Parameters: Standard Deviation of Dynamic Regret",
                        save_fn="./figures/noisy_parameters/regret_std_full.png")
    print("Finished")

def plot_nn_eval_data():
    eval_data = pickle.load(open("data/neural_networks/run_results_cost-corrected/10/evaluation_data.pkl", "rb"))
    EVALUATION_ITERATIONS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
    mse = np.asarray([e['MSE'].detach().cpu().item() for e in eval_data])
    mae = np.asarray([e['MAE'].detach().cpu().item() for e in eval_data])
    mse_plot = mse[[e-1 for e in EVALUATION_ITERATIONS[1:]]]
    mae_plot = mae[[e-1 for e in EVALUATION_ITERATIONS[1:]]]

    sn.set_style('whitegrid')
    fig = plt.figure(figsize=(12,4))
    fig.suptitle("Error during Neural Network training", fontsize=15)
    s = sn.lineplot(mae_plot)
    ax = plt.gca()
    labels = [str(e) for e in EVALUATION_ITERATIONS[1:]]
    ax.set_xticks(range(len(EVALUATION_ITERATIONS) - 1))
    ax.set_xticklabels(labels)
    s.set(yscale="log")
    s.set_xlabel("Training steps", fontsize=15)
    s.set_ylabel("Mean Absolute Error", fontsize=15)
    plt.tight_layout()
    plt.savefig("./figures/neural_networks/nn_mae.png")
    plt.close()

def create_dynamic_regret_heatmap_nn_prediction():
    ref_fn = "data/offline/ltv/seed-100"
    reference = pickle.load(open(ref_fn, "rb"))
    ref_cost = sum(reference.step_costs)

    PREDICTION_HORIZON = list(range(10, 101, 10))
    EVALUATION_ITERATIONS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]

    cost_matrix = np.zeros((len(EVALUATION_ITERATIONS), len(PREDICTION_HORIZON)))
    for i_idx, i in enumerate(EVALUATION_ITERATIONS):
        for k_idx, k in enumerate(PREDICTION_HORIZON):
            run = pickle.load(open(f"./data/neural_networks/run_results_cost-corrected/{k}/{i}", "rb"))
            cost_matrix[i_idx, k_idx] = sum(run.step_costs)

    regret_matrix = cost_matrix - ref_cost

    row_idxs = list(reversed([e for e in EVALUATION_ITERATIONS]))
    col_idxs = [e for e in PREDICTION_HORIZON]
    os.makedirs("./figures/neural_networks", exist_ok=True)
    plot_matrix_heatmap(regret_matrix[::-1], row_idxs, col_idxs, "Number of training steps", "Prediction Horizon",
                        title="Dynamic Regret for Neural Network Prediction Model vs Training Steps",
                        format="4.4g", save_fn="./figures/neural_networks/regret.png")
    print("Finished")

def create_dynamic_regret_curves():
    EPISODE_LENGTHS = np.arange(20, 201, 20)
    PREDICTION_NOISES = [0.0, 0.1, 0.5, 1.0]
    PREDICTION_HORIZON_RATIOS = [0.1, 0.5, 1.0]
    def _plot(regret, title, save_fn):
        num_pn, num_phr = regret.shape[0], regret.shape[1]
        fig, axs = plt.subplots(num_pn, num_phr, figsize=(4 * num_pn, 4 * num_phr))
        fig.suptitle(title, fontsize=15)
        for row in range(num_pn):
            for col in range(num_phr):
                episode_lengths = EPISODE_LENGTHS[:len(regret[row, col])]
                ax = axs[row, col]
                ax.plot(episode_lengths, regret[row, col], color='blue')
                ax.plot(episode_lengths, episode_lengths, color='red')
                ax.set_xlabel(f"Prediction horizon: {PREDICTION_HORIZON_RATIOS[col]} * T", fontsize=15)
                ax.set_ylabel(f"Noise: {PREDICTION_NOISES[row]}", fontsize=15)

        for ax in axs.flat:
            ax.label_outer()

        plt.tight_layout()

        plt.savefig(save_fn)

    # Note: in this function all the cost have been corrected during run time, so no need to correct anymore.
    os.makedirs("./figures/noisy_parameters", exist_ok=True)
    regret_matrix = pickle.load(open("data/test_dynamic_regret_curve/final_result", "rb"))
    _plot(regret_matrix[0],
          "Dynamic Regret Curve for multiple Prediction Horizon Ratios and Noise Levels: Noise on Disturbance only",
          save_fn="./figures/noisy_parameters/dynamic_regret_curve_table_disturbance.png")
    _plot(regret_matrix[1],
          "Dynamic Regret Curve for multiple Prediction Horizon Ratios and Noise Levels: Noise on All Parameters",
          save_fn="./figures/noisy_parameters/dynamic_regret_curve_table_all.png")


if __name__ == '__main__':
    create_dynamic_regret_curves()
    create_dynamic_regret_heatmap_noisy_prediction()
    create_dynamic_regret_heatmap_nn_prediction()
    plot_nn_eval_data()
