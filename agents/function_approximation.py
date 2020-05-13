import random
import numpy as np
import matplotlib.pyplot as plt
from env import Environment


class SarsaApproxAgent:
    def __init__(self):
        self.actions = [0, 1]
        self.feature_vector = np.zeros([3, 6, 2])
        self.weights = np.random.randn(3 * 6 * 2)
        self.cuboid = [[[1, 4], [4, 7], [7, 10]], [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]], [[0, 1]]]
        self.step_size = 0.01
        self.epsilon = 0.05

    def train(self, my_lambda, true_star_value):
        environment = Environment()
        terminated = False
        eligibility = np.zeros([36])

        # initial state where both player and dealer draw black card
        card = round(random.uniform(1, 10))
        dealer_card = round(random.uniform(1, 10))

        states = []
        actions = []

        current_state = [dealer_card, card]
        current_action = self.epsilon_action(current_state)

        while not terminated:
            current_action_index = 1 if current_action == 'hit' else 0

            states.append(current_state)
            actions.append(current_action_index)

            old_card = card
            state_prime, reward = environment.step(current_state, current_action)
            card = state_prime[1]

            # reward is 2 when state is not terminated because 0 overlaps with termination state draw reward
            # handle terminal state future action values to be 0
            current_reward = 0
            action_prime = None
            if reward != 2:
                current_reward = reward
                delta = current_reward + (0 - self.do_approximation([dealer_card, old_card], current_action_index))
            else:
                action_prime = self.epsilon_action([dealer_card, card])
                action_prime_index = 1 if action_prime == 'hit' else 0
                delta = current_reward + (self.do_approximation([dealer_card, card], action_prime_index) -
                                          self.do_approximation([dealer_card, old_card], current_action_index))

            eligibility = (my_lambda * eligibility + self.feature_vector.flatten())

            self.weights += self.step_size * delta * eligibility

            current_state = [dealer_card, card]
            current_action = action_prime

            if reward == -1 or reward == 1 or reward == 0:
                terminated = True

        mse = self.calculate_mse(true_star_value)
        return mse

    def epsilon_action(self, approximate_value):
        greedy_action = np.argmax(approximate_value)
        greedy_action = 'hit' if greedy_action == 1 else 'stick'

        explore_action = np.random.choice(self.actions)
        explore_action = 'hit' if explore_action == 1 else 'stick'

        prob = np.random.random()
        if prob <= self.epsilon:
            return explore_action
        else:
            return greedy_action

    def compute_feature_vector(self, state, action):
        # reset the feature vector
        self.feature_vector = np.zeros([3, 6, 2])

        dealer_card, card = state[0], state[1]

        dealer_feature_space = [x[0] <= dealer_card <= x[1] for x in self.cuboid[0]]
        player_feature_space = [x[0] <= card <= x[1] for x in self.cuboid[1]]

        for i in range(3):
            if dealer_feature_space[i]:
                for j in range(6):
                    if player_feature_space[j]:
                        self.feature_vector[i][j][action] = 1

    def do_approximation(self, state, action):
        self.compute_feature_vector(state, action)
        result = np.dot(self.feature_vector.flatten(), self.weights)

        return result

    def reset(self):
        self.weights = np.random.randn(3 * 6 * 2)

    def calculate_mse(self, true_star_value):
        mse = 0

        # loop through all Q state-action pairs
        for i in range(1, 11):
            for j in range(1, 22):
                for action in range(2):
                    mse += np.square(self.do_approximation([i, j], action) - true_star_value[i][j][action])

        return mse / (10 * 21 * 2)

    def plot_mse(self, episodes, mses, report_mses):
        colors = ['k', 'gray', 'brown', 'r', 'tomato', 'y', 'g', 'c', 'b', 'm', 'cyan']

        # plot report mse vs lambda
        lambda_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        plt.plot(lambda_range, report_mses)
        plt.title('Mean Squared Error vs Lambda')
        plt.show()

        plt.figure()

        idx = 0
        for i in range(len(mses)):
            plt.plot(episodes, mses[i], colors[i], label='lambda {0}'.format(lambda_range[i]))
            idx += 0.1

        plt.title('Mean Squared Error vs Episodes')
        plt.legend()
        plt.show()
