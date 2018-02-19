import collections
import time


class World(object):
    def __init__(self, initial_state, agents, transition_f, reward_f, visualizers=()):
        assert(isinstance(agents, collections.Iterable))
        assert(callable(transition_f))
        assert(callable(reward_f))
        assert(initial_state is not None)

        self.current_state = self.initial_state = initial_state
        self.agents = agents
        self.total_reward = 0
        self.transition_f = transition_f
        self.reward_f = reward_f
        self.visualizers = visualizers

    def next(self):
        if self.current_state.terminal:
            for visualizer in self.visualizers:
                visualizer.end()
            return False

        actions = []
        for agent in self.agents:
            actions.append(agent.act(self.current_state))

        next_state = self.transition_f(self.current_state, actions)
        reward = self.reward_f(self.current_state, actions, next_state)

        for agent in self.agents:
            agent.transition(self.current_state, actions, next_state, reward)

        for visualizer in self.visualizers:
            visualizer.update(self.current_state, actions, next_state, reward)

        self.current_state = next_state
        self.total_reward += reward

        return True

    def reset(self):
        self.total_reward = 0
        self.current_state = self.initial_state

        for visualizer in self.visualizers:
            visualizer.start(self.current_state)

    def run(self, interval=0):
        self.reset()
        i = 0
        while self.next():
            i += 1
            time.sleep(interval)

        return i, self.total_reward


