import random


class RandomAgent(object):
    def act(self, state):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return directions[random.randint(0, 3)]

    def transition(self, state, actions, new_state, reward):
        pass