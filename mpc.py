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
