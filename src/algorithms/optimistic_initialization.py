

class OptimisticInit():

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