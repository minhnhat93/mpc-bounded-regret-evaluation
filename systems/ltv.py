import numpy as np
import cvxpy as cp

class LTVSystem:
    """Linear Time-Varying System Class"""

    def __init__(self, reference_traj_func=None, noise_scale=0.1, rng_seed=101):
        if reference_traj_func:
            self.reference_traj_func = reference_traj_func
        else:
            self.reference_traj_func = lambda t: np.asarray([np.cos(t), np.sin(t)])
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(rng_seed)

    def get_program_parameters(self, t: float):
        """Return A(t), B(t) matrices for the LTV system"""
        # Example: Simple time-varying matrices
        Q = np.eye(2)
        R = np.eye(2)
        A = np.array([
            [np.cos(t), np.sin(t)],
            [-np.sin(t), np.cos(t)]
        ])
        B = np.array([
            [1, 0],
            [0, np.exp(-t)]
        ])
        w = self.rng.normal(loc=0.0, scale=self.noise_scale, size=(self.state_dim(),))
        x_bar = self.reference_traj_func(t)
        Q_terminal = Q
        return Q, R, A, B, w, x_bar, Q_terminal

    def state_dim(self):
        return 2

    def input_dim(self):
        return 2

    def dynamic(self, x, u, parameters, dt):
        Q_t, R_t, A_t, B_t, w_t, x_bar, Q_terminal = parameters
        dxdt = A_t @ x + B_t @ u + w_t
        x_next = x + dxdt * dt
        return x_next

    def get_opt(self, x_start, t_start, dt, num_steps):
        states = cp.Variable((num_steps + 1, self.state_dim()))
        inputs = cp.Variable((num_steps, self.input_dim()))

        # Initialize
        step_costs = []
        constraints = []
        parameters = []

        # Initial condition constraint
        constraints.append(states[0] == x_start)

        # Dynamics constraints over prediction horizon
        for t in range(num_steps):
            tt = t_start + t * dt
            parameters_t = self.get_program_parameters(tt)
            Q_t, R_t, A_t, B_t, w_t, x_bar, Q_terminal = parameters_t
            parameters.append(parameters_t)

            # State cost + Input cost
            cost = cp.quad_form(states[t] - x_bar, Q_t) + cp.quad_form(inputs[t], R_t)
            step_costs.append(cost)

            # Dynamics constraint
            constraints.append(
                states[t + 1] == self.dynamic(states[t], inputs[t], parameters_t, dt)
            )

        # Terminal cost
        terminal_cost = cp.quad_form(states[-1], Q_terminal)
        step_costs.append(terminal_cost)

        total_cost = sum(step_costs)

        # Construct the problem
        opt = cp.Problem(cp.Minimize(total_cost), constraints)

        return {
            "opt": opt,
            "parameters": parameters,
            "x": states,
            "u": inputs,
            "cost": step_costs,
        }
