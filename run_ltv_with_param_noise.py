from utilities import plot_results
from ltv_system import LTVSystem
import numpy as np
from mpc import MPCResult, run_online_mpc
import pickle
from ltv_system import LTVSystemWithParameterNoise, NoisyLTVPrediction, NoisyDisturbanceLTVPrediction
import timeit
import os


def run_online_ltv(noise_type: str, prediction_noise: float, prediction_horizon: int,
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
        print(f"Seed={seed}. Finished. Total nominal (NOT TRUE) cost: {sum(results.step_costs)}")
        # plot_results(results)
        pickle.dump(results, open(f"./data/online/ltv/{noise_type},noise-{prediction_noise},horizon-{prediction_horizon},seed-{seed}", "wb"))


def run_full_grid_online_mpc_ltv():
    os.makedirs("./data/online/ltv", exist_ok=True)
    NOISE_TYPES = ["disturbance", "full"]
    PREDICTION_NOISES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    PREDICTION_HORIZON = list(range(10, 101, 10))
    SEEDS = list(np.arange(200, 205))
    offline_run = "data/offline/ltv/seed-100"
    print(f"Running full grid for LTV system. Reference offline run file is {offline_run}")
    print(f"Noise types: {NOISE_TYPES}. Prediction noises: {PREDICTION_NOISES}. Prediction horizons: {PREDICTION_HORIZON}. Random seeds: {SEEDS}")
    for noise_type in NOISE_TYPES:
        for prediction_noise in PREDICTION_NOISES:
            for prediction_horizon in PREDICTION_HORIZON:
                print(f"Running: noise_type={noise_type}, prediction_noise={prediction_noise}, prediction_horizon={prediction_horizon}")
                start = timeit.default_timer()
                run_online_ltv(noise_type, prediction_noise, prediction_horizon, SEEDS, offline_run)
                stop = timeit.default_timer()
                print(f"Time taken: {stop-start:.2f} secs.")


# Example usage
if __name__ == "__main__":
    run_full_grid_online_mpc_ltv()