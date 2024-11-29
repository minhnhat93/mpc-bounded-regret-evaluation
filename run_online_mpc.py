import numpy as np
from mpc import MPCResult

def run_mpc_simulation(system, initial_state: np.ndarray, dt: float, num_steps: int) -> MPCResult:
    """Run MPC simulation and return results"""
    states = np.zeros((num_steps + 1, 2))
    inputs = np.zeros((num_steps, 2))
    costs = np.zeros(num_steps+1)
    objectives = np.zeros(num_steps)
    opt_datas = []

    states[0] = initial_state

    for t in range(num_steps):
        tt = t * dt
        # Solve MPC problem
        opt, opt_data = system.get_mpc_optimization(states[t], tt, dt, num_steps - t)
        opt_datas.append(opt_data)
        objectives[t] = opt.value

        opt_inputs = opt_data.inputs
        inputs[t] = opt_inputs[0]
        costs[t] = opt_data.step_costs[0]
        states[t+1] = opt_data.states[1]

    costs[num_steps] = opt_data.step_costs[-1]

    return MPCResult(states, inputs, costs, objectives, opt_datas)


# Example usage
if __name__ == "__main__":
    # System configuration
    # Create system and controller
    from systems.ltv import LTVSystem
    system = LTVSystem(noise_scale=0.1)

    # Create simulation
    # Run simulation
    initial_state = np.array([0.0, 0.0])
    results = run_mpc_simulation(system, initial_state, 0.1, 20)

    # Plot results
    from utilities import plot_results
    plot_results(results)
