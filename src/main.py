from arm import Arm
from algorithms.epsilon_greedy import EpsilonGreedy
import matplotlib.pyplot as plt

import numpy as np

def main():
    arms: [] = []
    return_vals: [] = []
    avg_vals: [] = []
    for i in range(10):
        arms.append(Arm(i))

    ep = EpsilonGreedy()
    for j in range(100):
        for i in range(10):
            return_vals.append(ep.epsilon_greedy(0.1, arms))
        avg_vals.append(np.mean(return_vals))
        return_vals = []

    plt.plot(avg_vals)
    plt.show()

    # for arm in arms:
    #     print(arm.get_expected_value())


if __name__ == "__main__":
    main()