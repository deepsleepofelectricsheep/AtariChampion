from typing import Any, Tuple
import numpy as np


class ReinforcementLearning:
    def get_action(self, state: np.ndarray) -> int:
        raise NotImplementedError()
    
    def incorporate_feedback(self, state: np.ndarray, action: int, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        raise NotImplementedError()