import numpy as np
import math

class Arm():
    def __init__(self, index: int, mean: float, op: bool = False):
        self.reward_func = None
        self.mean: float = mean
        self.variance: int = 1
        self.expected_value: float = 5.0 if op else 0.0
        self.rewards: [] = [self.expected_value] if op else []
        self.index = index

    def get_reward(self) -> float:
        reward: float = np.random.normal(self.mean, self.variance)
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

    def reset_arm(self):
        self.expected_value = 0.0
        self.rewards = []

    def reset_arm_op(self):
        self.expected_value = 5.0
        self.rewards = [self.expected_value]
