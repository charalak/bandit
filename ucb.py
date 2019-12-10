import math
from util_functions import ind_max


class UCB1():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        n_arms = len(self.counts)
        # UCB insure that it has played every single arm available to it at least once.
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            # - bonus: a measure of how much less we know about that arm than we know about the other arms
            # - explicitly curious algorithm that tries to seek out the unknown
            # - these rescaling terms allow the algorithm to define a confidence interval that has a reasonable chance
            # of containing the true value of the arm inside of it.
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ind_max(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        # alpha = 0.8
        # new_value = (1 - alpha) * value + (alpha) * reward
        self.values[chosen_arm] = new_value
        return