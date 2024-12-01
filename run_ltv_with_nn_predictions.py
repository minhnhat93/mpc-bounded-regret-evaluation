import numpy.random
import torch.random

from utilities import plot_results
from nn_prediction import LTVSystemWithNeuralNetPrediction, NNPredTrainer, PredictionNetwork
import numpy as np
from mpc import MPCResult
import pickle
from ltv_system import LTVSystemWithParameterNoise, NoisyLTVPrediction, NoisyDisturbanceLTVPrediction
import timeit
import os


from run_ltv_with_param_noise import run_online_mpc


def a(noise_type: str, prediction_noise: float, prediction_horizon: int,
                   random_seeds: list, reference_offline_run_file: str):
    # THESE ARE ADDITIONAL PARAMETERS (hard coded for now)
    initial_state = np.array([0.0, 0.0])
    episode_length = 100
    disturbance_strength = 0.2
    dt = 0.1

    # this is the chosen offline run reference parameters to perform experiments
    offline_run = pickle.load(open(reference_offline_run_file, "rb"))
    reference_parameters = offline_run.opt_datas.parameters

    for seed in random_seeds:
        if noise_type == "disturbance":
            add_noise_func = NoisyDisturbanceLTVPrediction(noise_scale=prediction_noise, rng_seed=seed)
        elif noise_type == "full":
            add_noise_func = NoisyLTVPrediction(noise_scale=prediction_noise, rng_seed=seed)
        system = LTVSystemWithParameterNoise(
            reference_parameters, add_noise_func,
            dt=dt, episode_length=episode_length, disturbance_strength=disturbance_strength,
        )
        results = run_online_mpc(system, initial_state, episode_length, prediction_horizon)
        # plot_results(results)
        pickle.dump(results, open(f"./data/online/ltv/{noise_type},noise-{prediction_noise},horizon-{prediction_horizon},seed-{seed}", "wb"))


def train_nn_predictor_and_evaluate_online_mpc(data_fn, model_directory,
                                               prediction_horizon, evaluation_iterations: list[int], general_seed: int):
    os.makedirs(model_directory, exist_ok=True)

    numpy.random.seed(general_seed)
    torch_seed = np.random.randint(low=1)
    torch.random.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    initial_state = np.array([0.0, 0.0])
    episode_length = 100
    disturbance_strength = 0.2
    dt = 0.1

    prediction_nn = PredictionNetwork()
    system_seed = np.random.randint(low=1)
    system = LTVSystemWithNeuralNetPrediction(
        prediction_nn, dt=dt, episode_length=episode_length, disturbance_strength=disturbance_strength,
        rng_seed=system_seed
    )

    trainer = NNPredTrainer(data_fn, dt)
    last_iteration = 0
    mpc_results = []
    for j in evaluation_iterations:
        num_train_iteration = j - last_iteration
        print(f"Current model has been trained for {last_iteration} steps. Training model for {num_train_iteration} steps.")
        trainer.train(j - last_iteration, verbose=j)

        print(f"Running online MPC using the current prediction model.")
        results = run_online_mpc(system, initial_state, episode_length, prediction_horizon)
        print(f"Finished. Total realized cost: {sum(results.step_costs)}")

        torch_file_path = os.path.join(model_directory, f"{j}.pt")
        torch.save(prediction_nn.state_dict(), torch_file_path)
        print(f"Model saved at {torch_file_path}")

        mpc_results.append(results)

    return dict(
        evaluation_iterations=evaluation_iterations, mpc_results=mpc_results, nn_eval_data=trainer.eval_data,
    )

# Example usage
if __name__ == "__main__":
    os.makedirs("./data/neural_networks/run_results/", exist_ok=True)
    general_seed = 200
    data_file = "data/offline/ltv/seed-100"
    PREDICTION_HORIZON = list(range(10, 101, 10))
    EVALUATION_ITERATIONS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                             10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    offline_run = "data/offline/ltv/seed-100"
    print(f"Running Neural Network predictions. Reference offline run file is {offline_run}")
    print(f"Prediction horizons: {PREDICTION_HORIZON}. Evaluation iterations: {EVALUATION_ITERATIONS}.")
    for prediction_horizon in PREDICTION_HORIZON:
        print(f"Running: prediction_horizon={prediction_horizon}.")
        start = timeit.default_timer()
        results = train_nn_predictor_and_evaluate_online_mpc(
            data_file, os.path.join(f"./data/neural_networks/saved_models/{prediction_horizon}/"), prediction_horizon, EVALUATION_ITERATIONS, general_seed)
        stop = timeit.default_timer()
        print(f"Time taken: {stop-start:.2f} secs.")
        pkl_file = f"./data/neural_networks/run_results/{prediction_horizon}.pkl"
        pickle.dump(results, open(pkl_file, "wb"))
        print(f"Result written to {pkl_file}.")
