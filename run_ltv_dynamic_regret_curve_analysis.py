import pickle
import timeit
import os
import numpy as np
from mpc import run_online_mpc
import pickle
from ltv_system import LTVSystemWithParameterNoise, NoisyLTVPrediction, NoisyDisturbanceLTVPrediction, LTVSystem
from run_cost_correction import correct_step_costs
from run_ltv_ground_truth_parameters import run_offline_mpc
import math


def run_noisy_parameters_multiple_episode_length():
    out_dir = "./data/test_dynamic_regret_curve"
    os.makedirs(out_dir, exist_ok=True)
    NOISE_TYPES = ["disturbance", "full"]
    EPISODE_LENGTHS = np.arange(20, 201, 20)
    PREDICTION_NOISES = [0.0, 0.1, 0.5, 1.0]
    PREDICTION_HORIZON_RATIOS = [0.1, 0.5, 1.0]
    noise_seed = 200
    system_seed = 100
    initial_state = np.array([0.0, 0.0])
    disturbance_strength = 0.2
    dt = 0.1

    regret_matrix = np.zeros((len(NOISE_TYPES), len(PREDICTION_NOISES), len(PREDICTION_HORIZON_RATIOS), len(EPISODE_LENGTHS)))
    for el_id, episode_length in enumerate(EPISODE_LENGTHS):
        # Run offline MPC to get reference
        print(f"====EPISODE LENGTH {episode_length}====")
        system = LTVSystem(dt=dt, episode_length=episode_length, disturbance_strength=disturbance_strength,
                           rng_seed=system_seed)
        reference_run = run_offline_mpc(system, initial_state, episode_length)
        reference_parameters = reference_run.opt_datas.parameters
        print(f"Offline MPC total cost: {sum(reference_run.step_costs)}")
        pickle.dump(reference_run, open(os.path.join(out_dir, f"reference_run-{episode_length}"), "wb"))

        for nt_id, noise_type in enumerate(NOISE_TYPES):
            for pn_id, prediction_noise in enumerate(PREDICTION_NOISES):
                for phr_id, prediction_horizon_ratio in enumerate(PREDICTION_HORIZON_RATIOS):
                    if noise_type == "disturbance":
                        add_noise_func = NoisyDisturbanceLTVPrediction(noise_scale=prediction_noise, rng_seed=noise_seed)
                    elif noise_type == "full":
                        add_noise_func = NoisyLTVPrediction(noise_scale=prediction_noise, rng_seed=noise_seed)
                    system = LTVSystemWithParameterNoise(
                        reference_parameters, add_noise_func,
                        dt=dt, episode_length=episode_length, disturbance_strength=disturbance_strength,
                    )
                    prediction_horizon = int(prediction_horizon_ratio * episode_length)
                    result_fn = (f"{noise_type},episode_length={episode_length},"
                                 f"prediction_horizon={prediction_horizon},noise={prediction_noise}")
                    print(f"Running configuration: {result_fn}", end=".")
                    start = timeit.default_timer()
                    results = run_online_mpc(system, initial_state, episode_length, prediction_horizon)
                    stop = timeit.default_timer()
                    results = correct_step_costs(reference_run, results)
                    regret = sum(results.step_costs) - sum(reference_run.step_costs)
                    print(f"Time taken: {stop - start:.2f} secs. Total cost: {sum(results.step_costs)}. "
                          f"Regret: {regret}")
                    pickle.dump(results, open(os.path.join(out_dir, result_fn), "wb"))
                    regret_matrix[nt_id, pn_id, phr_id, el_id] = regret
    final_result_fn = os.path.join(out_dir, "final_result")
    print(f"Saving dynamic regret result to {final_result_fn}.")
    pickle.dump(regret_matrix, open(final_result_fn, "wb"))

# Example usage
if __name__ == "__main__":
    run_noisy_parameters_multiple_episode_length()