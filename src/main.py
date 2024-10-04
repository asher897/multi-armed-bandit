from arm import Arm
from algorithms import BanditAlgorithm, EpsilonGreedy, OptimisticInit, UCB
import matplotlib.pyplot as plt

import numpy as np
import math
from typing import List, Dict


def multi_arm_bandit(algorithm: BanditAlgorithm, arms_amount=10, time_steps=1000, runs=100, epsilon=0.1):
    '''
    Runs a multi arm bandit agent for a given algorithm.
    Each arm is initiated with a random sample drawn from a normal distribution.
    Each arm is run for a given amount of time steps.
    There are a given amount of iterations to perform each run to take an average over
    :param algorithm: The algorithm to use for the multi-arm bandit.
                        One of OptimisticInit, EpsilonGreedy, UCB
    :param arms_amount: How many arms to initialize
    :param time_steps: The amount of time steps to run for the multi-arm bandit agent.
    :param runs: How many runs to perform the multi-arm bandit to take an average over
    :param epsilon: Epsilon parameter for the epsilon-greedy algorithm

    :return: The average values for all the multi-arm bandits run over the amount of runs
    '''
    arms: List[Arm] = [Arm(i, np.random.normal(0, math.sqrt(1)), op=type(algorithm) == OptimisticInit) for i in range(arms_amount)]
    total_returns: List[List[int]] = []

    for i in range(runs):
        return_vals: List[int]  = []
        arms = [arm.reset_arm() for arm in arms]
        for j in range(time_steps):
            result = algorithm.run(arms=arms, epsilon=epsilon, time=j+1)
            return_vals.append(result)
        total_returns.append(return_vals)

    avg_vals = np.mean(total_returns, axis=0)
    return avg_vals


def main():
    ep = EpsilonGreedy()
    op = OptimisticInit()
    ucb = UCB()

    avg_vals_ep = multi_arm_bandit(ep)
    avg_vals_op = multi_arm_bandit(op)
    avg_vals_ucb = multi_arm_bandit(ucb)

    plt.plot(avg_vals_ep, label='Epsilon Greedy')
    plt.plot(avg_vals_op, label="Optimistic Initialization")
    plt.plot(avg_vals_ucb, label="UCB")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()