import random

from common.world import World
from mcts.mcts.backups import monte_carlo
from mcts.mcts.default_policies import RandomKStepRollOut
from mcts.mcts.graph import StateNode
from mcts.mcts.mcts import MCTS
from mcts.mcts.tree_policies import UCB1
from pursuit.agents.base_agent import Agent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function
from pursuit.reward import get_reward_function

world_size = (10, 10)

rollouts = {10: [], 100: [], 1000: []}
rewards = {10: [], 100: [], 1000: []}

class RandomKStepRollOut2(RandomKStepRollOut):

    def __call__(self, state_node):
        result = super().__call__(state_node)
        world = World(state_node.state, [GreedyAgent(i) for i in range(4)],
                      get_transition_function(4, world_size),
                      get_reward_function(4, world_size))
        ts, reward = world.run(0, 1000)
        rollouts[self.k].append(result)
        rewards[self.k].append(reward)
        return result


class GameState(PursuitState):
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, agent_positions, prey_positions, world_size, agents, reward_f, transition_f):
        super().__init__(agent_positions, prey_positions, world_size)
        self.reward_f = reward_f
        self.transition_f = transition_f
        self.agents = agents

    def perform(self, action):
        actions = [agent.act(self) for agent in self.agents[:-1]] + [action]
        new_state = self.transition_f(self, actions)
        return GameState(new_state.agent_positions, new_state.prey_positions, self.world_size,
                         self.agents, self.reward_f, self.transition_f)

    def is_terminal(self):
        return self.terminal

    def reward(self, parent, action):
        #actions = [agent.act(self) for agent in self.agents[:-1]] + [action]
        actions = []
        reward = self.reward_f(parent, actions, self)
        return reward


class MCTSAgent(Agent):
    def __init__(self, id, n, k, c):
        super().__init__(id)
        self.root = None
        self.prev_action = None
        self.mcts_k = k
        self.mcts_n = n
        self.mcts_c = c

    def act(self, state):
        game_state = GameState(state.agent_positions, state.prey_positions, world_size,
                               agents, get_reward_function(len(agents), world_size),
                               get_transition_function(len(agents), world_size))
        if self.root is not None and state in self.root.children[self.prev_action].children:
            self.root = self.root.children[self.prev_action].children[state]
            self.root.parent = None
            n = self.mcts_n - self.root.n
        else:
            self.root = StateNode(None, game_state)
            n = self.mcts_n
        # print(self.mcts_n)

        tree = MCTS(tree_policy=UCB1(c=self.mcts_c), default_policy=RandomKStepRollOut2(self.mcts_k), backup=monte_carlo)
        self.prev_action = tree(self.root, n=n)
        # print([[y.n for y in x.children.values()] for x in self.root.children.values()])
        return self.prev_action


for k in (10, 100, 1000):
    for n in (1000, ):
        for c in (100,):
            agents = [GreedyAgent(i) for i in range(4)]
            random.seed(100)
            agents[-1] = MCTSAgent(3, n, k, c*k)
            results = []
            for i in range(1):
                world = World(PursuitState.random_state(len(agents), world_size), agents,
                              get_transition_function(len(agents), world_size),
                              get_reward_function(len(agents), world_size))
                timesteps, reward = world.run(0, 1000)
                results.append(timesteps)

            print("k: " + str(k))
            print("n: " + str(n))
            print("c: " + str(c))
            print("avg: " + str(sum(results)/len(results)))

print(rollouts)
print(rewards)

