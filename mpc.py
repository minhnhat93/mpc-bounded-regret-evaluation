import time
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import matplotlib.pyplot as plt


@dataclass
class SystemConfig:
    """Configuration class for system parameters"""
    state_dim: int
    input_dim: int
    horizon: int
    dt: float


@dataclass
class MPCResult:
    """Class to store MPC simulation results"""
    states: np.ndarray  # Shape: (T+1, state_dim)
    inputs: np.ndarray  # Shape: (T, input_dim)
    costs: np.ndarray  # Shape: (T,)
    solve_times: np.ndarray  # Shape: (T,)


class LTVSystem:
    """Linear Time-Varying System Class"""

    def __init__(self, config: SystemConfig):
        self.config = config

    def get_parameter(self):
        return

    def get_dynamics(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return A(t), B(t) matrices for the LTV system"""
        # Example: Simple time-varying matrices
        A = np.array([[np.cos(t), 1], [-1, np.sin(t)]])
        B = np.array([[0], [1]])
        return A, B

    def get_disturbance(self, t: float) -> np.ndarray:
        """Return disturbance w(t)"""
        # Example: Simple time-varying disturbance
        return np.zeros((self.config.state_dim))


class MPCController:
    def __init__(
            self,
            system: LTVSystem,
            Q: np.ndarray,
            R: np.ndarray,
            Q_terminal: Optional[np.ndarray] = None
    ):
        self.system = system
        self.config = system.config
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal if Q_terminal is not None else Q

    def solve_mpc(
            self,
            current_state: np.ndarray,
            current_time: float
    ) -> Tuple[np.ndarray, float]:
        """
        Solve MPC optimization problem
        Returns optimal input and objective value
        """
        # Initialize optimization variables
        x = cp.Variable((self.config.horizon + 1, self.config.state_dim))
        u = cp.Variable((self.config.horizon, self.config.input_dim))

        # Initialize cost
        cost = 0
        constraints = []

        # Initial condition constraint
        constraints.append(x[0] == current_state)

        # Dynamics constraints over prediction horizon
        for t in range(self.config.horizon):
            A_t, B_t = self.system.get_dynamics(current_time + t * self.config.dt)
            w_t = self.system.get_disturbance(current_time + t * self.config.dt)

            # State cost
            cost += cp.quad_form(x[t], self.Q)
            # Input cost
            cost += cp.quad_form(u[t], self.R)

            # Dynamics constraint
            constraints.append(
                x[t + 1] == A_t @ x[t] + B_t @ u[t] + w_t
            )

        # Terminal cost
        cost += cp.quad_form(x[self.config.horizon], self.Q_terminal)

        # Solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        objective = problem.solve()

        return u[0].value, objective


class MPCSimulation:
    def __init__(
            self,
            system: LTVSystem,
            controller: MPCController,
            simulation_steps: int
    ):
        self.system = system
        self.controller = controller
        self.simulation_steps = simulation_steps

    def run(
            self,
            initial_state: np.ndarray
    ) -> MPCResult:
        """Run MPC simulation and return results"""
        states = np.zeros((self.simulation_steps + 1, self.system.config.state_dim))
        inputs = np.zeros((self.simulation_steps, self.system.config.input_dim))
        costs = np.zeros(self.simulation_steps)
        solve_times = np.zeros(self.simulation_steps)

        states[0] = initial_state

        for t in range(self.simulation_steps):
            # Solve MPC problem
            start_time = time.time()
            optimal_input, objective = self.controller.solve_mpc(
                states[t],
                t * self.system.config.dt
            )
            solve_times[t] = time.time() - start_time

            # Apply optimal input
            inputs[t] = optimal_input
            costs[t] = objective

            # Simulate one step forward
            A_t, B_t = self.system.get_dynamics(t * self.system.config.dt)
            w_t = self.system.get_disturbance(t * self.system.config.dt)

            states[t + 1] = (A_t @ states[t] + B_t @ inputs[t] + w_t).flatten()

        return MPCResult(states, inputs, costs, solve_times)

    def plot_results(self, results: MPCResult):
        """Plot simulation results"""
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Plot states
        time_steps = np.arange(self.simulation_steps + 1)
        for i in range(self.system.config.state_dim):
            axs[0].plot(time_steps, results.states[:, i], label=f'State {i + 1}')
        axs[0].set_title('States')
        axs[0].legend()
        axs[0].grid(True)

        # Plot inputs
        time_steps = np.arange(self.simulation_steps)
        for i in range(self.system.config.input_dim):
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


# Example usage
if __name__ == "__main__":
    # System configuration
    config = SystemConfig(
        state_dim=2,
        input_dim=1,
        horizon=10,
        dt=0.1
    )

    # Create system and controller
    system = LTVSystem(config)
    Q = np.eye(config.state_dim)
    R = np.eye(config.input_dim)
    controller = MPCController(system, Q, R)

    # Create simulation
    sim = MPCSimulation(system, controller, simulation_steps=50)

    # Run simulation
    initial_state = np.array([1.0, 0.0])
    results = sim.run(initial_state)

    # Plot results
    sim.plot_results(results)