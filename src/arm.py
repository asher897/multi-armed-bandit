import numpy as np
import math

from typing import List

class Arm(object):
    def __init__(self, index: int, mean: float, op: bool = False):
        self.reward_func = None
        self.mean: float = mean
        self.variance: int = 1
        self.expected_value: float = 5.0 if op else 0.0
        self.rewards: List[float] = [self.expected_value] if op else []
        self.index = index
        self.times_chosen: float = 0.00001

    def get_reward(self) -> float:
        reward: float = np.random.normal(self.mean, self.variance)
        self.rewards.append(reward)

        self.times_chosen += 1

        self.get_average_rewards()

        return reward

    def ucb_algorithm(self, time: int, c: int) -> float:
        exploit: float = self.get_expected_value()
        log = math.log(time)
        times_chosen = self.get_times_chosen()

        return exploit + c*math.sqrt(log/times_chosen)

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

        return self

    def reset_arm_op(self):
        self.expected_value = 5.0
        self.rewards = [self.expected_value]

    def get_times_chosen(self) -> int:
        return self.times_chosen
