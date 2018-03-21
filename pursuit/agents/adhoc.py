import random
from collections import defaultdict, deque

import numpy as np
from keras.layers import Dense, Input
from keras import Model
import matplotlib.pyplot as plt

from mcts.mcts.backups import monte_carlo
from mcts.mcts.default_policies import immediate_reward, random_terminal_roll_out, RandomKStepRollOut
from mcts.mcts.graph import StateNode
from mcts.mcts.mcts import MCTS
from mcts.mcts.tree_policies import UCB1
from pursuit.agents.greedy import GreedyAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState

ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MEMO = {}
class AdhocAgent(GreedyAgent):

    def __init__(self, id):
        super().__init__(id)
        self.id = id
        self.behavior_model = None
        self.environment_model = None
        self.test = self.get_test_dataset()
        self.x = None
        self.y = None
        self.state_and_actions = None
        self.env_metric = []
        self.behavior_metric = []

    def init_models(self, num_features, num_agents):
        input = Input(shape=(num_features,))
        dense = Dense(64, activation='selu')(input)
        outputs = []
        for i in range(num_agents):
            outputs.append(Dense(4, activation='softmax')(dense))

        model = Model(inputs=(input, ), outputs=outputs)
        model.compile(optimizer='adam', loss=['categorical_crossentropy']*num_agents, metrics=['accuracy'])
        self.behavior_model = model

        input = Input(shape=(num_features + 4*(num_agents + 1), ))
        dense = Dense(64, activation='selu')(input)
        output = Dense(num_features, activation='linear')(dense)
        model = Model(input, output)
        model.compile(optimizer='adam', loss='mae')
        self.environment_model = model

    def act(self, state):
        if self.behavior_model is not None:
            tree = MCTS(tree_policy=UCB1(c=1.41), default_policy=RandomKStepRollOut(1000), backup=monte_carlo)
            game_state = GameState(state.agent_positions, state.world_size, self.behavior_model, self.environment_model,
                                   self.id, state.prey_positions)
            best_action = tree(StateNode(None, game_state), n=1000)
            return best_action
        else:
            return random.choice(ACTIONS)

    def transition(self, state, actions, new_state, reward):
        oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        newstatefeatures = new_state.features_relative_prey().reshape(1, -1)

        # some initialization
        if self.x is None:
            self.x = np.zeros((0, oldstatefeatures.shape[1]))
        if self.y is None:
            self.y = [np.zeros((0, 4)) for _ in range(len(actions)-1)]
        if self.state_and_actions is None:
            self.state_and_actions = np.zeros((0, oldstatefeatures.shape[1] + len(actions)*4))
        if self.behavior_model is None:
            self.init_models(oldstatefeatures.shape[1], len(actions)-1)

        # update x
        self.x = np.concatenate((self.x, oldstatefeatures))

        # update y
        my_action = np.zeros((1, 4))
        j = 0
        for i in range(len(actions)):
            if i == self.id:
                my_action[0, ACTIONS.index(actions[i])] = 1
                continue
            self.y[j] = np.concatenate((self.y[j], np.zeros((1, 4))))
            self.y[j][-1, ACTIONS.index(actions[i])] = 1
            j += 1

        # update x+y (used for the environment model)
        state_and_actions = np.concatenate((self.x[-1:], *[a[-1:] for a in self.y], my_action), axis=1)
        self.state_and_actions = np.concatenate((self.state_and_actions, state_and_actions))

        predicted = [ACTIONS[np.argmax(a)] for a in self.behavior_model.predict(self.x[-1:])]
        hits = [predicted[i] == actions[i] for i in range(3)]
        self.behavior_metric.append(sum(hits)/3)

        predicted_state = self.environment_model.predict(self.state_and_actions[-1:])
        self.env_metric.append(np.sum(np.abs(predicted_state - newstatefeatures)))

        self.behavior_model.fit(self.x, self.y, verbose=0)
        self.environment_model.fit(self.state_and_actions, np.concatenate((self.x[1:], newstatefeatures)), verbose=0)
        # if len(self.hits) > 10:
        #     plt.clf()
        #     plt.plot([np.mean(self.hits[k:k + 10]) for k in range(len(self.hits) - 10)])
        #     plt.draw()
        #     plt.pause(0.02)
        #print(self.behavior_model.evaluate(self.test[0], self.test[1], verbose=0))
        MEMO.clear()

    def get_test_dataset(self, filename='greedy_dataset.npy'):
        raw = np.load(filename).item()
        assert(isinstance(raw, dict))
        x = [] # array with dims NxF, N = |dataset|, F = # features
        y = None # list of NxA arrays, N = |dataset|, A = |actions|
        for state, actions in raw.items():
            if y is None:
                y = [[] for _ in range(len(actions))]

            x.append(state.features())
            for i in range(len(actions)):
                y[i].append((np.zeros(4)))
                y[i][-1][ACTIONS.index(actions[i])] = 1

        for i in range(len(y)):
            y[i] = np.array(y[i])

        x = np.array(x)
        return x, y


class GameState(PursuitState):
    actions = ACTIONS

    def __init__(self, agent_positions, world_size, behavior_model, env_model, adhoc_id, prey=None):
        # if prey is None, that means agent_positions are relative to a prey
        if prey is None:
            mid = (world_size[0] // 2, world_size[1] // 2)
            for i, (x, y) in enumerate(agent_positions):
                agent_positions[i] = (mid[0]+x, mid[1]+y)
            agent_positions = tuple(tuple(pos) for pos in agent_positions)
            super().__init__(agent_positions, (mid, ), world_size)
        else:
            super().__init__(tuple(agent_positions), prey, world_size)
        self.behavior_model = behavior_model
        self.env_model = env_model
        self.adhoc_id = adhoc_id
        self.reward_fn = get_reward_function(len(agent_positions), world_size)

    def perform(self, action):
        if (self, action) not in MEMO:
            state_features = self.features_relative_prey().reshape(1, -1)
            # predict what other agents are going to do
            other_actions = [np.argmax(a) for a in self.behavior_model.predict(state_features)]
            my_action = ACTIONS.index(action)
            # build the joint action
            actions_idx = other_actions[:self.adhoc_id] + [my_action] + other_actions[self.adhoc_id:]
            # 1 hot encode it
            actions = np.zeros((len(other_actions)+1, 4))
            actions[range(len(actions_idx)), actions_idx] = 1
            # build X for the env model which is just a single row with all the features
            x = np.concatenate((state_features, actions.reshape(1, -1)), axis=1)
            # predict new state
            new_state = self.env_model.predict(x)
            MEMO[(self, action)] = GameState(np.round(new_state).astype(np.int).reshape(-1, 2), self.world_size, self.behavior_model,
                      self.env_model, self.adhoc_id)
        return MEMO[(self, action)]

    def is_terminal(self):
        return self.terminal

    def reward(self, parent, action):
        return self.reward_fn(parent, None, self)

