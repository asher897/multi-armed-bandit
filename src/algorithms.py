import random
import numpy as np
from arm import Arm

from typing import List

class BanditAlgorithm(object):
    def run(self, arms: List[Arm], epsilon: float = 0.0, time: int = 0):
        pass


class OptimisticInit(BanditAlgorithm):

    def run(self, arms: List[Arm], epsilon: float = 0.0, time: int = 0):
        return self.exploit(arms=arms)
    @staticmethod
    def exploit(arms: List[Arm]):
        max_reward: float = -100.0
        best_arm = None

        for arm in arms:
            val: float = arm.get_expected_value()
            if val > max_reward:
                best_arm = arm.index
                max_reward = val

        return arms[best_arm].get_reward()


class EpsilonGreedy(BanditAlgorithm):

    def run(self, arms: List[Arm], epsilon: float = 0.0, time: int = 0):
        return self.epsilon_greedy(epsilon=epsilon, arms=arms)

    def epsilon_greedy(self, epsilon: float, arms: List[Arm]):
        func = np.random.choice(a=[self.exploit, self.explore], p=[1-epsilon, epsilon])
        return func(arms)

    @staticmethod
    def exploit(arms: List[Arm]):
        max_reward: float = -100.0
        best_arm = None

        for arm in arms:
            val: float = arm.get_expected_value()
            if val > max_reward:
                best_arm = arm.index
                max_reward = val

        return arms[best_arm].get_reward()

    @staticmethod
    def explore(arms: List[Arm]):
        return arms[random.randint(0, 9)].get_reward()


class UCB(BanditAlgorithm):
    def run(self, arms: List[Arm], epsilon: float = 0.0, time: int = 0):
        return self.ucb_algorithm(time=time, arms=arms)

    @staticmethod
    def ucb_algorithm(arms: [], time: int):
        max_reward: float = -10000
        max_arm: int = 0
        for arm in arms:
            reward = arm.ucb_algorithm(time=time, c=2)
            if reward > max_reward:
                max_reward = reward
                max_arm = arm.index

        return arms[max_arm].get_reward()
    