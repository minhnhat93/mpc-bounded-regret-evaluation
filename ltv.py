import gym
from gym import spaces
import numpy as np


class LTVSystemEnvWithLipschitz(gym.Env):
    """
    Custom OpenAI Gym environment for a 2D Linear Time-Varying (LTV) system
    with controllable Lipschitzness.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=1000, scale_A=1.0, scale_B=1.0):
        super(LTVSystemEnvWithLipschitz, self).__init__()

        # Define action and observation spaces
        # Actions: control inputs u1 and u2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations: states x1 and x2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Time step and maximum number of steps
        self.dt = 0.01  # Time step for integration
        self.max_steps = max_steps

        # Scaling factors to control Lipschitzness
        self.scale_A = scale_A
        self.scale_B = scale_B

        # Initialize state, time, and step counter
        self.state = None
        self.t = 0.0
        self.steps = 0

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        # Random initial state in range [-1, 1]
        self.state = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        self.t = 0.0
        self.steps = 0
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """
        Apply an action to the system and advance the state.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        x = self.state
        u = action

        # Define time-varying matrices A(t) and B(t) with scaling
        A = self.scale_A * np.array([
            [np.cos(self.t), np.sin(self.t)],
            [-np.sin(self.t), np.cos(self.t)]
        ])
        B = self.scale_B * np.array([
            [1, 0],
            [0, np.exp(-self.t)]
        ])

        # Compute state derivative: dx/dt = A(t)x + B(t)u
        dxdt = A @ x + B @ u

        # Euler integration for state update
        self.state = x + self.dt * dxdt

        # Update time and step counter
        self.t += self.dt
        self.steps += 1

        # Define reward as negative state magnitude (to minimize deviation from zero)
        reward = -np.linalg.norm(self.state)

        # Check if episode is done
        done = self.steps >= self.max_steps

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        """
        Render the environment (not implemented here).
        """
        print(f"Time: {self.t:.2f}, State: {self.state}, Scale A: {self.scale_A}, Scale B: {self.scale_B}")

    def close(self):
        """
        Close the environment (if necessary).
        """
        pass

    def set_scales(self, scale_A, scale_B):
        """
        Update the scaling factors for A(t) and B(t) to control Lipschitzness.
        """
        self.scale_A = scale_A
        self.scale_B = scale_B
