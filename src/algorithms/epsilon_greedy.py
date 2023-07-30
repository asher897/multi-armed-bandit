import random
import numpy as np

class EpsilonGreedy():
    def epsilon_greedy(self, epsilon: float, arms: []):
        func = np.random.choice(a=[self.exploit, self.explore], p=[1-epsilon, epsilon])
        return func(arms)

    @staticmethod
    def exploit(arms: []):
        max_reward: float = -100.0
        best_arm = None

        for arm in arms:
            val: float = arm.get_expected_value()
            if val > max_reward:
                best_arm = arm.index
                max_reward = val

        return arms[best_arm].get_reward()

    @staticmethod
    def explore(arms: []):
        return arms[random.randint(0, 9)].get_reward()
