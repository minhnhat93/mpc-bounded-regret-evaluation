import numpy as np
from mpc import MPCResult

def run_offline_mpc(system, initial_state: np.ndarray) -> MPCResult:
    """Run MPC simulation and return results"""
    # Solve MPC problem
    opt, opt_data = system.get_mpc_optimization(initial_state, 0)
    objectives = np.cumsum(opt_data.step_costs[::-1])[::-1]
    return MPCResult(opt_data.states, opt_data.inputs, opt_data.step_costs, objectives, opt_data)


# Example usage
if __name__ == "__main__":
    # System configuration
    # Create system and controller
    from systems.ltv import LTVSystem
    system = LTVSystem(dt=0.1, num_steps=20, noise_scale=0.1)

    # Create simulation
    # Run simulation
    initial_state = np.array([0.0, 0.0])
    results = run_offline_mpc(system, initial_state)

    # Plot results
    from utilities import plot_results
    plot_results(results)
