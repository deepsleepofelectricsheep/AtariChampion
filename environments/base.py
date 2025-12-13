import gymnasium as gym
import ale_py
import numpy as np
from typing import Tuple, Any


gym.register_envs(ale_py)


class AtariMDP:
    def __init__(self, env: str = "ALE/Boxing-v5", mode: str = "training", discount: float = 0.99, max_steps: int = 1000) -> None:
        self.env = gym.make(env) if mode == "training" else gym.make("ALE/Boxing-v5", render_mode="human")
        self._discount = discount
        self._max_steps = max_steps
        self._actions = list(range(self.env.action_space.n))

    def start_state(self) -> Tuple[np.ndarray, dict]:
        return self.env.reset()
    
    def actions(self) -> list[Any]:
        return self._actions
    
    def max_steps(self) -> int:
        return self._max_steps
    
    def discount(self) -> float:
        return self._discount
    
    def transition(self, action: Any) -> Tuple[np.ndarray, float, bool]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated
    
    def close(self) -> None:
        self.env.close()