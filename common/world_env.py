import gym
from gym import spaces

from common.world import World
from pursuit.agents.ad_hoc.adhoc import ACTIONS
from pursuit.agents.base_agent import Agent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function


class WorldEnv(gym.Env):
    def __init__(self, agents, world_size = (5,5), max_steps=1000):
        self.world_size = world_size
        self.agent = DummyAgent(3)
        initial_state = self._get_new_state()
        transition_f = get_transition_function(4, world_size)
        reward_f = get_reward_function(4, world_size)
        self.world = World(initial_state, agents + [self.agent], transition_f, reward_f)
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(world_size), shape=(8,))
        self.max_steps = max_steps
        self.i = 0

    def _get_new_state(self):
        return PursuitState.random_state(4, self.world_size)

    def step(self, action):
        self.agent.action = action
        self.world.next()
        observation = self.world.current_state.features_relative_prey()
        reward = self.agent.reward
        done = self.world.current_state.terminal or self.i >= self.max_steps

        self.i += 1

        return observation, reward, done, {}

    def reset(self):
        self.i = 0
        self.world.reset()
        self.world.current_state = self._get_new_state()
        return self.world.current_state.features_relative_prey()

    def render(self, mode='human'):
        pass


class DummyAgent(Agent):

    def __init__(self, id):
        super().__init__(id)
        self.action = None
        self.reward = None

    def act(self, state):
        return ACTIONS[self.action]

    def transition(self, state, actions, new_state, reward):
        self.reward = reward