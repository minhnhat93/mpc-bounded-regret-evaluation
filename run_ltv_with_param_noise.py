from utilities import plot_results
from ltv_system import LTVSystem
import numpy as np
from mpc import MPCResult
import pickle
from ltv_system import LTVSystemWithParameterNoise, NoisyLTVPrediction, NoisyDisturbanceLTVPrediction
import timeit
import os


def run_offline_mpc(system, initial_state: np.ndarray, episode_length) -> MPCResult:
    """Run MPC simulation and return results"""
    # Solve MPC problem
    opt, opt_data = system.get_mpc_optimization(initial_state, 0, episode_length)
    objectives = np.cumsum(opt_data.step_costs[::-1])[::-1]
    return MPCResult(opt_data.states, opt_data.inputs, opt_data.step_costs, objectives, opt_data)


def collect_ltv_trajectories():
    os.makedirs("./data/offline/ltv", exist_ok=True)
    initial_state = np.array([0.0, 0.0])
    episode_length = 100
    disturbance_strength = 0.2
    dt = 0.1

    for seed in np.arange(100, 105):
        system = LTVSystem(dt=dt, episode_length=episode_length, disturbance_strength=disturbance_strength,
                           rng_seed=seed)
        results = run_offline_mpc(system, initial_state, episode_length)
        print(f"Total cost: {sum(results.step_costs)}")
        plot_results(results)
        pickle.dump(results, open(f"./data/offline/ltv/seed-{seed}", "wb"))

def run_online_mpc(system, initial_state: np.ndarray, episode_length: int, prediction_horizon: int) -> MPCResult:
    """Run MPC simulation and return results"""
    states = np.zeros((episode_length + 1, 2))
    inputs = np.zeros((episode_length, 2))
    costs = np.zeros(episode_length + 1)
    objectives = np.zeros(episode_length)
    opt_datas = []

    states[0] = initial_state

    for t in range(episode_length):
        # Solve MPC problem
        end_step = min(t + prediction_horizon, episode_length)
        num_steps = end_step - t
        opt, opt_data = system.get_mpc_optimization(states[t], t, num_steps)
        opt_datas.append(opt_data)
        objectives[t] = opt.value

        opt_inputs = opt_data.inputs
        inputs[t] = opt_inputs[0]
        costs[t] = opt_data.step_costs[0]
        states[t+1] = opt_data.states[1]

    costs[episode_length] = opt_data.step_costs[-1]

    return MPCResult(states, inputs, costs, objectives, opt_datas)


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
        print(f"Finished. Total realized cost: {sum(results.step_costs)}")
        # plot_results(results)
        pickle.dump(results, open(f"./data/online/ltv/{noise_type},noise-{prediction_noise},horizon-{prediction_horizon},seed-{seed}", "wb"))


def run_full_grid_online_mpc_ltv():
    os.makedirs("./data/online/ltv", exist_ok=True)
    NOISE_TYPES = ["disturbance", "full"]
    PREDICTION_NOISES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    PREDICTION_HORIZON = list(range(10, 101, 10))
    # NOISE_TYPES = ["full"]
    # PREDICTION_NOISES = [0.5, 1.0]
    # PREDICTION_HORIZON = list(range(10, 101, 10))
    # NOISE_TYPES = ["disturbance", "full"]
    # PREDICTION_NOISES = [0.02, 0.05]
    # PREDICTION_HORIZON = list(range(10, 101, 10))
    SEEDS = list(np.arange(200, 205))
    offline_run = "data/offline/ltv/seed-100"
    print(f"Running full grid for LTV system. Reference offline run file is {offline_run}")
    print(f"Noise types: {NOISE_TYPES}. Prediction noises: {PREDICTION_NOISES}. Prediction horizons: {PREDICTION_HORIZON}. Random seeds: {SEEDS}")
    for noise_type in NOISE_TYPES:
        for prediction_noise in PREDICTION_NOISES:
            for prediction_horizon in PREDICTION_HORIZON:
                print(f"Running: noise_type={noise_type}, prediction_noise={prediction_noise}, prediction_horizon={prediction_horizon}", end=". ")
                start = timeit.default_timer()
                run_online_ltv(noise_type, prediction_noise, prediction_horizon, SEEDS, offline_run)
                stop = timeit.default_timer()
                print(f"Time taken: {stop-start:.2f} secs.")


# Example usage
if __name__ == "__main__":
    collect_ltv_trajectories()
    run_full_grid_online_mpc_ltv()