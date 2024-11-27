import numpy as np
from dataclasses import dataclass


@dataclass
class MPCResult:
    """Class to store MPC simulation results"""
    states: np.ndarray  # Shape: (T+1, state_dim)
    inputs: np.ndarray  # Shape: (T, input_dim)
    costs: np.ndarray  # Shape: (T,)


def run_mpc_simulation(system, initial_state: np.ndarray, dt: float, num_steps: int) -> MPCResult:
    """Run MPC simulation and return results"""
    states = np.zeros((num_steps + 1, 2))
    inputs = np.zeros((num_steps, 2))
    costs = np.zeros(num_steps)

    states[0] = initial_state

    for t in range(num_steps):
        tt = t * dt
        # Solve MPC problem
        program_dict = system.get_optimization_prog(states[t], tt, dt, num_steps - t)
        opt = program_dict["opt"]
        objective = opt.solve()

        # Apply optimal input
        opt_inputs = program_dict["u"]
        inputs[t] = opt_inputs[0].value
        costs[t] = objective

        # Simulate one step forward
        opt_parameters = program_dict["parameters"]
        parameters = opt_parameters[0]
        states[t + 1] = system.dynamic(states[t], inputs[t], parameters, dt)

    return MPCResult(states, inputs, costs)


# Example usage
if __name__ == "__main__":
    # System configuration
    # Create system and controller
    from systems.ltv import LTVSystem
    system = LTVSystem()

    # Create simulation
    # Run simulation
    initial_state = np.array([1.0, 0.0])
    results = run_mpc_simulation(system, initial_state, 0.1, 100)

    # Plot results
    from utilities import plot_results
    plot_results(results)