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
    x_bar_terminal: np.ndarray


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
    x_bar_terminal = x_bar


class NoisyDisturbanceLTVPrediction(LTVPrediction):
    def __init__(self, noise_scale, rng_seed):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(rng_seed)

    def w(self, _w):
        _w = np.copy(_w)
        _w += self.noise_scale * self.rng.uniform(-1, 1, _w.shape)
        return _w


class NoisyLTVPrediction(LTVPrediction):
    def __init__(self, noise_scale, rng_seed):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(rng_seed)

    def Q(self, _Q):
        _Q = np.copy(_Q)
        _Q[[0, 1],[0, 1]] += self.noise_scale * self.rng.uniform(-1, 1, (2,))
        return _Q

    def R(self, _R):
        _R = np.copy(_R)
        _R[[0, 1],[0, 1]] += self.noise_scale * self.rng.uniform(-1, 1, (2,))
        return _R

    def x_bar(self, _x_bar):
        _x_bar = np.copy(_x_bar)
        _x_bar += self.noise_scale * self.rng.uniform(-1, 1, _x_bar.shape)
        return _x_bar

    def A(self, _A):
        _A = np.copy(_A)
        _A += self.noise_scale * self.rng.uniform(-1, 1, _A.shape)
        return _A

    def B(self, _B):
        _B = np.copy(_B)
        _B[[0, 1],[0, 1]] += self.noise_scale * self.rng.uniform(-1, 1, (2,))
        return _B

    def w(self, _w):
        _w = np.copy(_w)
        _w += self.noise_scale * self.rng.uniform(-1, 1, _w.shape)
        return _w

    Q_terminal = Q
    x_bar_terminal = x_bar


class LTVSystem:
    """Linear Time-Varying System Class"""

    def __init__(self, dt, episode_length, reference_traj_func=None, disturbance_strength=0.1, rng_seed=101):
        if reference_traj_func:
            self.reference_traj_func = reference_traj_func
        else:
            self.reference_traj_func = lambda t: np.asarray([np.sin(t), np.cos(t)])
        self.disturbance_strength = disturbance_strength
        self.rng = np.random.default_rng(rng_seed)
        self.episode_length = episode_length
        self.dt = dt
        self.t = np.arange(episode_length) * dt
        self.x_bar = np.array([self.reference_traj_func(t) for t in np.arange(episode_length + 1) * dt])
        self.A = np.array([
            [[np.cos(t), np.sin(t)],
             [-np.sin(t), np.cos(t)]]
        for t in self.t])
        self.B = np.array([
            [[1, 0],
             [0, np.exp(-t)]
        ] for t in self.t])
        self.w = self.rng.normal(loc=0.0, scale=self.disturbance_strength, size=(episode_length, self.state_dim(),))

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
        x_bar_terminal = self.x_bar[time_step + 1]
        return LTVParameter(Q=Q, R=R, x_bar=x_bar, A=A, B=B, w=w, Q_terminal=Q_terminal, x_bar_terminal=x_bar_terminal)

    def state_dim(self):
        return 2

    def input_dim(self):
        return 2

    def dynamic(self, x, u, parameters, dt):
        dxdt = parameters.A @ x + parameters.B @ u + parameters.w
        x_next = x + dxdt * dt
        return x_next

    def get_mpc_optimization(self, x_start, start_step, num_steps):
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
            tt = start_step + t
            parameters_t = self.get_program_parameters(tt)
            parameters.append(parameters_t)

            # State cost + Input cost
            cost = cp.quad_form(states[t] - parameters_t.x_bar, parameters_t.Q) + cp.quad_form(inputs[t], parameters_t.R)
            step_costs.append(cost)

            # Dynamics constraint
            constraints.append(
                states[t + 1] == self.dynamic(states[t], inputs[t], parameters_t, self.dt)
            )

        # Terminal cost
        terminal_cost = cp.quad_form(states[-1] - parameters_t.x_bar_terminal, parameters_t.Q_terminal)
        step_costs.append(terminal_cost)

        total_cost = sum(step_costs)
        opt = cp.Problem(cp.Minimize(total_cost), constraints)
        opt.solve()

        step_costs = np.asarray([e.value for e in step_costs])
        states = states.value
        inputs = inputs.value

        return opt, OptData(parameters=parameters, states=states, inputs=inputs, step_costs=step_costs)

class LTVSystemWithParameterNoise(LTVSystem):
    def __init__(self, reference_parameters: list[LTVParameter], add_noise_funcs: LTVPrediction, **kwargs):
        super().__init__(**kwargs)
        self.add_noise_funcs = add_noise_funcs
        self.reference_parameters = reference_parameters

    def get_program_parameters(self, time_step):
        parameters = self.reference_parameters[time_step]
        parameters.Q = self.add_noise_funcs.Q(parameters.Q)
        parameters.R = self.add_noise_funcs.R(parameters.R)
        parameters.x_bar = self.add_noise_funcs.x_bar(parameters.x_bar)
        parameters.A = self.add_noise_funcs.A(parameters.A)
        parameters.B = self.add_noise_funcs.B(parameters.B)
        parameters.w = self.add_noise_funcs.w(parameters.w)
        parameters.Q_terminal = self.add_noise_funcs.Q_terminal(parameters.Q_terminal)
        parameters.x_bar_terminal = self.add_noise_funcs.x_bar_terminal(parameters.x_bar_terminal)

        return parameters