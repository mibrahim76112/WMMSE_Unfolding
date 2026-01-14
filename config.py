
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class ExperimentConfig:
    # System dimensions
    nr_of_users: int = 4
    nr_of_BS_antennas: int = 4
    scheduled_users: Tuple[int, ...] = (0, 1, 2, 3)
    dnn_hidden = (512, 512, 512)   # you can change this
    pgd_steps: int = 4



    # WMMSE stopping
    epsilon: float = 1e-4
    power_tolerance: float = 1e-4

    # Power / noise
    total_power: float = 10.0
    noise_power: float = 1.0

    # Channel / pathloss
    path_loss_option: bool = False
    path_loss_range: Tuple[float, float] = (-5.0, 5.0)  # dB

    # Training / testing
    nr_of_batches_training: int = 10000
    nr_of_batches_test: int = 1000
    nr_of_samples_per_batch: int = 100

    # Iterations
    nr_of_iterations_wmmse: int = 1          # baseline truncated iterations (as in notebook)
    nr_of_iterations_nn: int = 1             # unfolded iterations (layers)

    # Optimizer
    learning_rate: float = 1e-3

    # Seeds
    train_seed: int = 0
    test_seed: int = 1234

    def user_weights_batch(self) -> np.ndarray:
        # Shape: (B, K, 1)
        return np.reshape(
            np.ones(self.nr_of_users * self.nr_of_samples_per_batch, dtype=np.float64),
            (self.nr_of_samples_per_batch, self.nr_of_users, 1),
        )

    def user_weights_regular(self) -> np.ndarray:
        return np.ones(self.nr_of_users, dtype=np.float64)
