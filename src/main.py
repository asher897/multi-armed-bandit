from arm import Arm
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.optimistic_initialization import OptimisticInit
from algorithms.ucb import UCB
import matplotlib.pyplot as plt

import numpy as np
import math

def main():
    arms_ep: [] = []
    arms_op: [] = []
    arms_ucb: [] = []
    return_vals_ep: [] = []
    return_vals_op: [] = []
    return_vals_ucb: [] = []
    total_returns_ep: [] = []
    total_returns_op: [] = []
    total_returns_ucb: [] = []
    ep = EpsilonGreedy()
    op = OptimisticInit()
    ucb = UCB()
    for i in range(10):
        mean = np.random.normal(0, math.sqrt(3))
        arms_ep.append(Arm(i, mean))
        arms_op.append(Arm(i, mean, op=True))
        arms_ucb.append(Arm(i, mean))

    for i in range(100):
        for i in range(len(arms_ep)):
            arms_ep[i].reset_arm()
            arms_op[i].reset_arm()
            arms_ucb[i].reset_arm()
        for j in range(1000):
            return_vals_ep.append(ep.epsilon_greedy(0.1, arms_ep))
            return_vals_op.append(op.exploit(arms_op))
            return_vals_ucb.append(ucb.ucb_algorithm(arms_ucb, time=j+1))
        total_returns_ep.append(return_vals_ep)
        total_returns_op.append(return_vals_op)
        total_returns_ucb.append(return_vals_ucb)
        return_vals_ep = []
        return_vals_op = []
        return_vals_ucb = []

    avg_vals_ep = np.mean(total_returns_ep, axis=0)
    avg_vals_op = np.mean(total_returns_op, axis=0)
    avg_vals_ucb = np.mean(total_returns_ucb, axis=0)

    plt.plot(avg_vals_ep, label='Epsilon Greedy')
    plt.plot(avg_vals_op, label="Optimistic Initialization")
    plt.plot(avg_vals_ucb, label="UCB")
    plt.legend()
    plt.show()

    # for arm in arms:
    #     print(arm.get_expected_value())


if __name__ == "__main__":
    main()