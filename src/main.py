# # Multi-Armed Bandit Lab for Reinforcement Learning
# Willem Enslin Roux 2630054
# David Jellin 2742458
# Aaron Sher 2568961

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import math
import random

from fractions import Fraction


class Arm(object):
    def __init__(self, index: int, mean: float, op: bool = False):
        self.reward_func = None
        self.mean: float = mean
        self.variance: int = 1
        self.expected_value: float = 5.0 if op else 0.0
        self.rewards: [] = [self.expected_value] if op else []
        self.index = index
        self.times_chosen: float = 0.00001

    def get_reward(self) -> float:
        reward: float = np.random.normal(self.mean, self.variance)
        self.rewards.append(reward)

        self.times_chosen += 1

        self.get_average_rewards()

        return reward

    def ucb_algorithm(self, time: int, c: float) -> float:
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

    def reset_arm_ucb(self):
        self.expected_value = 0.0
        self.rewards = []
        self.times_chosen = 0.00001

    def reset_arm_op(self, op_val: int):
        self.expected_value = op_val
        self.rewards = [self.expected_value]

    def get_times_chosen(self) -> int:
        return self.times_chosen


class EpsilonGreedy(object):
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


class OptimisticInit(object):

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


class UCB(object):
    def __init__(self, c: float):
        self.reward_func = None
        self.c = c

    def ucb_algorithm(self, arms: [], time: int):
        max_reward: float = -10000
        max_arm: int = 0
        reward: float = 0.0
        for arm in arms:
            reward = arm.ucb_algorithm(time=time, c=self.c)
            if reward > max_reward:
                max_reward = reward
                max_arm = arm.index

        return arms[max_arm].get_reward()


def algorithm(val: float, arms_ep, arms_op, arms_ucb):
    return_vals_ep: [] = []
    return_vals_op: [] = []
    return_vals_ucb: [] = []
    total_returns_ep: [] = []
    total_returns_op: [] = []
    total_returns_ucb: [] = []

    ep = EpsilonGreedy()
    op = OptimisticInit()
    ucb = UCB(c=val if val > 0 else 2.0)
    for z in range(100):
        for i in range(len(arms_ep)):
            arms_op[i].reset_arm_op(op_val=val if val > 0 else 5.0)
            arms_ucb[i].reset_arm_ucb()
            arms_ep[i].reset_arm()
        for j in range(1000):
            if val <= 1:
                return_vals_ep.append(ep.epsilon_greedy(val if val > 0 else 0.1, arms_ep))
            return_vals_op.append(op.exploit(arms_op))
            return_vals_ucb.append(ucb.ucb_algorithm(arms_ucb, time=j + 1))

        if val == 0.0:
            total_returns_ep.append(return_vals_ep)
            total_returns_op.append(return_vals_op)
            total_returns_ucb.append(return_vals_ucb)
        else:
            total_returns_ep.append(np.mean(return_vals_ep))
            total_returns_op.append(np.mean(return_vals_op))
            total_returns_ucb.append(np.mean(return_vals_ucb))
        return_vals_op = []
        return_vals_ucb = []
        return_vals_ep = []

    return total_returns_op, total_returns_ucb, total_returns_ep


def format_as_fraction(x, pos):
    frac = Fraction(x).limit_denominator()
    if frac.denominator == 1:
        return f'{frac.numerator}'
    return f'{frac.numerator}/{frac.denominator}'

def main():
    arms_ep: [] = []
    arms_op: [] = []
    arms_ucb: [] = []
    means_ep: [] = []
    means_op: [] = []
    means_ucb: [] = []

    for i in range(10):
        mean = np.random.normal(0, math.sqrt(3))
        arms_ep.append(Arm(i, mean))
        arms_op.append(Arm(i, mean, op=True))
        arms_ucb.append(Arm(i, mean))

    total_op, total_ucb, total_ep = algorithm(val=0.0, arms_ep=arms_ep, arms_op=arms_op, arms_ucb=arms_ucb)

    avg_ep = np.mean(total_ep, axis=0)
    avg_op = np.mean(total_op, axis=0)
    avg_ucb = np.mean(total_ucb, axis=0)

    plt.plot(avg_ep, label='Epsilon Greedy')
    plt.plot(avg_op, label="Optimistic Initialization")
    plt.plot(avg_ucb, label="UCB")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()

    vals: [] = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
    vals_str = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4', '8', '16']

    for val in vals:
        op, ucb, ep = algorithm(val, arms_ep=arms_ep, arms_op=arms_op, arms_ucb=arms_ucb)
        means_op.append(np.mean(op))
        means_ucb.append(np.mean(ucb))
        means_ep.append(np.mean(ep))

    plt.plot(means_ep, label='Epsilon Greedy')
    plt.plot(means_op, label="Optimistic Initialization")
    plt.plot(means_ucb, label="UCB")
    plt.xlabel("epsilon/c/Q0")
    plt.ylabel("Average Reward")
    plt.xticks([i for i in range(12)], vals_str)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()