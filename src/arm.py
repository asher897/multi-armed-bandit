import numpy as np
import math

class Arm():
    def __init__(self, index: int):
        self.reward_func = None
        self.mean: float = np.random.normal(0, math.sqrt(3))
        self.variance: int = 1
        self.expected_value: float = 0.0
        self.rewards: [] = []
        self.index = index

    def get_reward(self) -> float:
        reward: float = np.random.normal(self.mean, math.sqrt(self.variance))
        self.rewards.append(reward)

        self.get_average_rewards()

        return reward

    def get_average_rewards(self) -> None:
        amount_of_rewards: int = len(self.rewards)
        reward_total: float = 0.0

        for reward in self.rewards:
            reward_total += reward

        self.expected_value = reward_total/amount_of_rewards

    def get_expected_value(self) -> float:
        return self.expected_value
