from pursuit.agents.ad_hoc.adhoc import AdhocAgent


class AdhocAfterNAgent(AdhocAgent):

    def __init__(self, initial_behavior, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_agent = initial_behavior
        self.n = n
        self.episodes = 0

    def act(self, state):
        if self.episodes < self.n:
            return self.initial_agent.act(state)
        else:
            return super().act(state)

    def transition(self, state, actions, new_state, reward, *args, **kwargs):
        if self.episodes < self.n:
            self.initial_agent.transition(state, actions, new_state, reward)

        if new_state.terminal:
            self.episodes += 1
        super().transition(state, actions, new_state, reward, *args, **kwargs)



