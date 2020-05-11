import random
import numpy as np
import matplotlib.pyplot as plt
from env import Environment


class SarsaAgent:
    def __init__(self):
        self.actions = [0, 1]
        self.value_steps = np.zeros([11, 22])
        self.action_value_function = np.zeros([11, 22, len(self.actions)])
        self.action_value_steps = np.zeros([11, 22, len(self.actions)])
        self.n_zero = 100

    def train(self, my_lambda, true_star_value):
        environment = Environment()
        terminated = False
        eligibility = np.zeros([11, 22, len(self.actions)])

        # initial state where both player and dealer draw black card
        card = round(random.uniform(1, 10))
        dealer_card = round(random.uniform(1, 10))

        states = []
        actions = []

        current_state = [dealer_card, card]
        current_action = self.epsilon_action(current_state)

        while not terminated:
            current_action_index = 1 if current_action == 'hit' else 0

            self.value_steps[dealer_card][card] += 1
            self.action_value_steps[dealer_card][card][current_action_index] += 1

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
                delta = current_reward + (0 - self.action_value_function[dealer_card][old_card][current_action_index])
            else:
                action_prime = self.epsilon_action([dealer_card, card])
                action_prime_index = 1 if action_prime == 'hit' else 0
                delta = current_reward + (self.action_value_function[dealer_card][card][action_prime_index] -
                                          self.action_value_function[dealer_card][old_card][current_action_index])

            eligibility[dealer_card][old_card][current_action_index] += 1

            # loop through all Q state-action pairs in the episode
            for i in range(len(states)):
                current_dealer_card, current_card = states[i][0], states[i][1]
                current_action_value_step = self.action_value_steps[current_dealer_card][current_card][actions[i]]
                current_eligibility = eligibility[current_dealer_card][current_card][actions[i]]

                self.action_value_function[current_dealer_card][current_card][actions[i]] += (
                        (1 / current_action_value_step) * delta * current_eligibility)

                eligibility[current_dealer_card][current_card][actions[i]] = my_lambda * current_eligibility

            current_state = [dealer_card, card]
            current_action = action_prime

            if reward == -1 or reward == 1 or reward == 0:
                terminated = True

        mse = self.calculate_mse(true_star_value)
        return mse

    def epsilon_action(self, state):
        epsilon = self.n_zero / (self.n_zero + self.value_steps[state[0]][state[1]])

        greedy_action = np.argmax(self.action_value_function[state[0]][state[1]])
        greedy_action = 'hit' if greedy_action == 1 else 'stick'

        explore_action = np.random.choice(self.actions)
        explore_action = 'hit' if explore_action == 1 else 'stick'

        prob = np.random.random()
        if prob <= epsilon:
            return explore_action
        else:
            return greedy_action

    def reset(self):
        self.value_steps = np.zeros([11, 22])
        self.action_value_function = np.zeros([11, 22, len(self.actions)])
        self.action_value_steps = np.zeros([11, 22, len(self.actions)])

    def calculate_mse(self, true_star_value):
        mse = 0

        # loop through all Q state-action pairs
        for i in range(1, 11):
            for j in range(1, 22):
                for action in range(2):
                    mse += np.square(self.action_value_function[i][j][action] - true_star_value[i][j][action])

        return mse / (10 * 21 * 2)

    def plot_mse(self, episodes, mses):
        plt.plot(episodes, mses[0], 'r', label='lambda 0')
        plt.plot(episodes, mses[1], 'b', label='lambda 1')
        plt.legend()
        plt.show()
