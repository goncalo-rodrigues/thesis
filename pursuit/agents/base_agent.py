class Agent(object):
    def __init__(self, id):
        self.id = id

    def act(self, state):
        raise NotImplementedError()

    def transition(self, state, actions, new_state, reward):
        pass
