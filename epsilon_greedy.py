import random
import math
from util_functions import ind_max


class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        '''
        :param epsilon: The frequency with which we explore one of the available arms
        :param counts:  vector of integers of length N that tells us how many times we’ve
    played each of the N arms available to us.
        :param values: A vector of floating point numbers that defines the average amount of reward we’ve gotten
    when playing each of the N arms available to us.
        '''

        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return self

    def select_arm(self):
        if random.random() > self.epsilon:
            return ind_max(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        '''
        We update the estimated value of the chosen arm to be a weighted average
        of the previously estimated value  and the reward we just received.
        This weighting is important, because it means that single observations mean
        less and less to the algorithm when we already have a lot of experience
        with any specific option.

        :param chosen_arm:
        :param reward:
        '''

        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        # counts of best arm updated
        n = self.counts[chosen_arm]
        # value of best arm
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        # alpha = 0.8
        # new_value = (1 - alpha) * value + (alpha) * reward
        self.values[chosen_arm] = new_value
        return


class AnnealingEpsilonGreedy():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        t = sum(self.counts) + 1
        epsilon = 1 / math.log(t + 0.0000001)

        if random.random() > epsilon:
            return ind_max(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        # alpha = 0.8
        # new_value = (1 - alpha) * value + (alpha) * reward
        self.values[chosen_arm] = new_value
        return

