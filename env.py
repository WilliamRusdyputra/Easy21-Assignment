import random


class Environment:
    def __init__(self):
        self.dealer_first_card = 0
        self.dealer_sum = 0
        self.player_sum = 0

    def step(self, state, action):
        self.dealer_first_card = self.dealer_sum = state[0]
        self.player_sum = state[1]

        if action == 'hit':
            card = self.draw_card()
            self.player_sum += card

            if self.player_sum > 21 or self.player_sum < 1:
                return [self.dealer_first_card, self.player_sum], -1
            else:
                return [self.dealer_first_card, self.player_sum], 2
        else:
            while 17 > self.dealer_sum > 0:
                card = self.draw_card()
                self.dealer_sum += card

            if self.dealer_sum > self.player_sum:
                return [self.dealer_first_card, self.player_sum], -1
            elif self.dealer_sum < self.player_sum:
                return [self.dealer_first_card, self.player_sum], 1
            else:
                return [self.dealer_first_card, self.player_sum], 0

    def draw_card(self):
        card = round(random.uniform(1, 10))
        prob = random.random()
        if prob <= (1 / 3):
            card = -card
        return card
