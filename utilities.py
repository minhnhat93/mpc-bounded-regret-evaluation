import matplotlib.pyplot as plt
import numpy as np


def plot_results(results):
    """Plot simulation results"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    state_dim = np.prod(results.states[0].shape)
    input_dim = np.prod(results.inputs[0].shape)
    simulation_steps = len(results.inputs)

    # Plot states
    time_steps = np.arange(simulation_steps + 1)
    for i in range(state_dim):
        axs[0].plot(time_steps, results.states[:, i], label=f'State {i + 1}')
    axs[0].set_title('States')
    axs[0].legend()
    axs[0].grid(True)

    # Plot inputs
    time_steps = np.arange(simulation_steps)
    for i in range(input_dim):
        axs[1].plot(time_steps, results.inputs[:, i], label=f'Input {i + 1}')
    axs[1].set_title('Control Inputs')
    axs[1].legend()
    axs[1].grid(True)

    # Plot costs
    axs[2].plot(time_steps, results.costs)
    axs[2].set_title('Objective Values')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
