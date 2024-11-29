import numpy as np
import cvxpy as cp

from dataclasses import dataclass


@dataclass
class LTVParameter:
    Q: np.ndarray
    R: np.ndarray
    x_bar: np.ndarray
    A: np.ndarray
    B: np.ndarray
    w: np.ndarray
    Q_terminal: np.ndarray


@dataclass
class OptData:
    parameters: list[LTVParameter]
    states: list
    inputs: list
    step_costs: list


@dataclass
class LTVPrediction:
    def Q(self, _Q):
        return _Q

    def R(self, _R):
        return _R

    def x_bar(self, _x_bar):
        return _x_bar

    def A(self, _A):
        return _A

    def B(self, _B):
        return _B

    def w(self, _w):
        return _w

    Q_terminal = Q


class NoisyDisturbanceLTVPrediction(LTVPrediction):
    def __init__(self, noise_scale, rng_seed):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(rng_seed)

    def w(self, _w):
        _w = np.copy(_w)
        _w += self.scale * self.rng.normal(np.zeros_like(_w))
        return _w


class NoisyLTVPrediction(LTVPrediction):
    def __init__(self, noise_scale, rng_seed):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(rng_seed)

    def Q(self, _Q):
        _Q = np.copy(_Q)
        _Q[[0, 1],[0, 1]] += self.scale * self.rng.normal((2,))
        return _Q

    def R(self, _R):
        _R = np.copy(_R)
        _R[[0, 1],[0, 1]] += self.scale * self.rng.normal((2,))
        return _R

    def x_bar(self, _x_bar):
        _x_bar = np.copy(_x_bar)
        _x_bar += self.scale * self.rng.normal(np.zeros_like(_x_bar))
        return _x_bar

    def A(self, _A):
        _A = np.copy(_A)
        _A += self.scale * self.rng.normal(np.zeros_like(_A))
        return _A

    def B(self, _B):
        _B = np.copy(_B)
        _B[[0, 1],[0, 1]] += self.scale * self.rng.normal((2,))
        return _B

    def w(self, _w):
        _w = np.copy(_w)
        _w += self.scale * self.rng.normal(np.zeros_like(_w))
        return _w

    Q_terminal = Q


class LTVSystem:
    """Linear Time-Varying System Class"""

    def __init__(self, dt, num_steps, reference_traj_func=None, noise_scale=0.1, rng_seed=101):
        if reference_traj_func:
            self.reference_traj_func = reference_traj_func
        else:
            self.reference_traj_func = lambda t: np.asarray([np.sin(t), np.cos(t)])
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(rng_seed)
        self.num_steps = num_steps
        self.dt = dt
        self.t = np.arange(num_steps) * dt
        self.x_bar = np.array([self.reference_traj_func(t) for t in self.t])
        self.A = np.array([
            [[np.cos(t), np.sin(t)],
             [-np.sin(t), np.cos(t)]]
        for t in self.t])
        self.B = np.array([
            [[1, 0],
             [0, np.exp(-t)]
        ] for t in self.t])
        self.w = self.rng.normal(loc=0.0, scale=self.noise_scale, size=(num_steps, self.state_dim(),))

    def get_program_parameters(self, time_step):
        """Return A(t), B(t) matrices for the LTV system"""
        # Example: Simple time-varying matrices
        Q = np.eye(2)
        R = np.eye(2)
        x_bar = self.x_bar[time_step]
        A = self.A[time_step]
        B = self.B[time_step]
        w = self.w[time_step]
        Q_terminal = np.copy(Q)
        return LTVParameter(Q=Q, R=R, x_bar=x_bar, A=A, B=B, w=w, Q_terminal=Q_terminal)

    def state_dim(self):
        return 2

    def input_dim(self):
        return 2

    def dynamic(self, x, u, parameters, dt):
        dxdt = parameters.A @ x + parameters.B @ u + parameters.w
        x_next = x + dxdt * dt
        return x_next

    def get_mpc_optimization(self, x_start, step):
        traj_length = self.num_steps - step - 1
        states = cp.Variable((traj_length + 1, self.state_dim()))
        inputs = cp.Variable((traj_length, self.input_dim()))

        # Initialize
        step_costs = []
        constraints = []
        parameters = []

        # Initial condition constraint
        constraints.append(states[0] == x_start)

        # Dynamics constraints over prediction horizon
        for t in range(traj_length):
            parameters_t = self.get_program_parameters(t)
            parameters.append(parameters_t)

            # State cost + Input cost
            cost = cp.quad_form(states[t] - parameters_t.x_bar, parameters_t.Q) + cp.quad_form(inputs[t], parameters_t.R)
            step_costs.append(cost)

            # Dynamics constraint
            constraints.append(
                states[t + 1] == self.dynamic(states[t], inputs[t], parameters_t, self.dt)
            )

        # Terminal cost
        terminal_cost = cp.quad_form(states[-1], parameters_t.Q_terminal)
        step_costs.append(terminal_cost)

        total_cost = sum(step_costs)
        opt = cp.Problem(cp.Minimize(total_cost), constraints)
        opt.solve()

        step_costs = np.asarray([e.value for e in step_costs])
        states = states.value
        inputs = inputs.value

        return opt, OptData(parameters=parameters, states=states, inputs=inputs, step_costs=step_costs)

class LTVSystemWithParameterNoise(LTVSystem):
    def __init__(self, reference_parameters: list[LTVParameter], noise_func: LTVPrediction, **kwargs):
        super().__init__(**kwargs)
        self.noise_func = noise_func
        self.reference_parameters = reference_parameters

    def get_program_parameters(self, time_step):
        parameters = self.reference_parameters[time_step]
        parameters.Q = self.noise_func.Q(parameters.Q)
        parameters.R = self.noise_func.R(parameters.R)
        parameters.x_bar = self.noise_func.x_bar(parameters.x_bar)
        parameters.A = self.noise_func.A(parameters.A)
        parameters.B = self.noise_func.B(parameters.B)
        parameters.w = self.noise_func.w(parameters.w)
        parameters.Q_terminal = self.noise_func.Q_terminal(parameters.Q_terminal)
