import pandas as pd
from epsilon_greedy import EpsilonGreedy
from softmax import Softmax
from softmax import AnnealingSoftmax
from ucb import UCB1
from arms import BernoulliArm


def test_algo_monte_carlo(algo,
                          mean_probs,
                          n_sim=5000,
                          horizon=250,
                          filename='',
                          store_it=True):

    n_arms = len(mean_probs)
    # The number of times each algorithm is allowed to pull on arms during each simulationx
    sim_nums = []
    times = []
    chosen_arms = []
    cumulative_rewards = []
    rewards = []

    for sim in range(n_sim):
        sim += 1
        algo.initialize(n_arms)
        for t in range(horizon):
            t += 1
            sim_nums.append(sim)
            times.append(t)
            chosen_arm = algo.select_arm()
            chosen_arms.append(chosen_arm)
            mu = mean_probs[chosen_arm]
            reward = BernoulliArm(mu).draw()
            rewards.append(reward)
            if t == 1:
                cumulative_rewards.append(reward)
            else:
                cumulative_rewards.append(cumulative_rewards[- 1] + reward)
            algo.update(chosen_arm, reward)

    col_names = ['sim_nums', 'horizon', 'chosen_arms',
                 'cumulative_rewards', 'rewards']
    results = [sim_nums, times, chosen_arms, cumulative_rewards, rewards]
    results = pd.DataFrame(results)
    results = results.T
    results.columns = col_names
    if store_it:
        results.to_csv('results/results_{}.csv'.format(filename))


def epsilon_greedy_algo():
    # epsilon = 0.0: profit-maximization: only good options but you will never explore
    #
    # epsilon = 1.0: A/B test: wastes resources acquiring data about bad options

    epsilon = 0.1
    n_sim = 5000
    horizon = 250
    filename = 'EG'
    mean_probs = [0.1, 0.1, 0.1, 0.1, 0.9]
    algo = EpsilonGreedy(epsilon, [], [])

    test_algo_monte_carlo(algo,
                          mean_probs,
                          n_sim=n_sim,
                          horizon=horizon,
                          filename=filename,
                          store_it=True)


def softmax_algo():
    '''
    The Softmax algorithm tries to cope with arms differing in estimated value by explicitly
    incorporating information about the reward rates of the available arms into its method
    for choosing which arm to select when it explores.

    :temperatue: the Softmax algorithm at low temperatures behaves orderly,
     while it behaves essentially randomly at high temperatures
    '''
    n_sim = 5000
    horizon = 250
    temperature = 0.5
    algo = Softmax(temperature, [], [])
    mean_probs = [0.3, 0.35, 0.4, 0.5, 0.55]
    filename = 'softmax_temp0.5'

    test_algo_monte_carlo(algo,
                          mean_probs,
                          n_sim=n_sim,
                          horizon=horizon,
                          filename=filename,
                          store_it=True)


def annealing_softmax_algo():
    '''
    The Softmax algorithm tries to cope with arms differing in estimated value by explicitly
    incorporating information about the reward rates of the available arms into its method
    for choosing which arm to select when it explores.

    The annealing term is to enforce the algorithm to explore less over time. This is done
    by reducing the temperature at each time-step.

    :temperatue: the in the Annealing Softmax algorithm  is set to very high value automatically.
    '''

    n_sim = 5000
    horizon = 250
    algo = AnnealingSoftmax([], [])
    mean_probs = [0.1, 0.1, 0.5, 0.1, 0.9]
    filename = 'annealing_softmax_temp0.5'

    test_algo_monte_carlo(algo,
                          mean_probs,
                          n_sim=n_sim,
                          horizon=horizon,
                          filename=filename,
                          store_it=True)


def UCB1_algo():
    '''
    UCB1 algorithm pays attention to not only what it knows, but also how much it knows.
    It can make decisions to explore that are driven by our confidence in the estimated
    value of the arms we’ve selected.
    Characteristics: a) It doesn’t use randomness at all. b) Itdoesn’t have any free parameters
    that you need to configure before you can deploy it. c) It uses an explicit measure of
    confidence.
    '''

    n_sim = 5000
    horizon = 250
    algo = UCB1([], [])
    mean_probs = [0.1, 0.1, 0.5, 0.1, 0.9]
    filename = 'ucb1'

    test_algo_monte_carlo(algo,
                          mean_probs,
                          n_sim=n_sim,
                          horizon=horizon,
                          filename=filename,
                          store_it=True)


if __name__ == "__main__":
    # epsilon_greedy_algo()
    # softmax_algo()
    # annealing_softmax_algo()
    UCB1_algo()
