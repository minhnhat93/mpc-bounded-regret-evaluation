import numpy as np
from dataclasses import dataclass


@dataclass
class MPCResult:
    """Class to store MPC simulation results"""
    states: np.ndarray  # Shape: (T+1, state_dim)
    inputs: np.ndarray  # Shape: (T, input_dim)
    step_costs: np.ndarray  # Shape: (T,)
    objectives: np.ndarray  # Shape: (T,)
    opt_datas: list


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
        # This cost is wrong because we are using the cost matrices of the prediction model
        # however, we can store the parameters, the inputs and the true states and re-compute the correct cost later
        costs[t] = opt_data.step_costs[0]
        # Use the true dynamic of the reference model to compute the (real) next state
        states[t+1] = system.true_dynamic(states[t], inputs[t], opt_data.parameters[0], t, system.dt)

    costs[episode_length] = opt_data.step_costs[-1]

    return MPCResult(states, inputs, costs, objectives, opt_datas)
