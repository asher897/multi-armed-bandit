import math

class UCB():
    def __init__(self):
        self.reward_func = None
        self.c = 2

    def ucb_algorithm(self, arms: [], time: int):
        max_reward: float = -10000
        max_arm: int = 0
        reward: float = 0.0
        for arm in arms:
            reward = arm.ucb_algorithm(time=time, c=2)
            if reward > max_reward:
                max_reward = reward
                max_arm = arm.index

        return arms[max_arm].get_reward()

