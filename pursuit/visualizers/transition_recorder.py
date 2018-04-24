class TransitionRecorder(object):
    def __init__(self):
        self.transitions = []

    def start(self, initial_state):
        pass

    def update(self, state, actions, next_state, reward):
        self.transitions.append((state, actions, next_state, reward))

    def end(self):
        pass