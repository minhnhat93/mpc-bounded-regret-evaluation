from utilities import plot_results
from ltv_system import LTVSystem
import numpy as np
from mpc import MPCResult
import pickle
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
        print(f"Seed={seed}. Total true cost: {sum(results.step_costs)}")
        plot_results(results)
        pickle.dump(results, open(f"./data/offline/ltv/seed-{seed}", "wb"))


# Example usage
if __name__ == "__main__":
    collect_ltv_trajectories()