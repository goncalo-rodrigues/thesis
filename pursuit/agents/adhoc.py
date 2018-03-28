import random
from collections import defaultdict, deque

import numpy as np
from keras.layers import Dense, Input
from keras import Model
import matplotlib.pyplot as plt

from mcts.mcts.backups import monte_carlo
from mcts.mcts.default_policies import RandomKStepRollOut
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
        # self.behavior_model = None
        # self.environment_model = None
        self.test = self.get_test_dataset()
        # self.x = None
        # self.y = None
        # self.state_and_actions = None
        # self.env_metric = []
        # self.behavior_metric = []

        self.first = True
        self.e_model = EnvironmentModel()
        self.b_model = BehaviorModel()

    # def init_models(self, num_features, num_agents):
    #     input = Input(shape=(num_features,))
    #     dense = Dense(64, activation='selu')(input)
    #     outputs = []
    #     for i in range(num_agents):
    #         outputs.append(Dense(4, activation='softmax')(dense))
    #
    #     model = Model(inputs=(input, ), outputs=outputs)
    #     model.compile(optimizer='adam', loss=['categorical_crossentropy']*num_agents, metrics=['accuracy'])
    #     self.behavior_model = model
    #
    #     input = Input(shape=(num_features + 4*(num_agents + 1), ))
    #     dense = Dense(64, activation='selu')(input)
    #     output = Dense(num_features, activation='linear')(dense)
    #     model = Model(input, output)
    #     model.compile(optimizer='adam', loss='mae')
    #     self.environment_model = model

    def act(self, state):
        if not self.first:
            tree = MCTS(tree_policy=UCB1(c=1.41), default_policy=RandomKStepRollOut(10), backup=monte_carlo)
            game_state = GameState(state, self.b_model, self.e_model, self.id)
            best_action = tree(StateNode(None, game_state), n=100)
            return best_action
        else:
            self.first = False
            return random.choice(ACTIONS)

    def transition(self, state, actions, new_state, reward):
        # oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        # newstatefeatures = new_state.features_relative_prey().reshape(1, -1)
        #
        # # some initialization
        # if self.x is None:
        #     self.x = np.zeros((0, oldstatefeatures.shape[1]))
        # if self.y is None:
        #     self.y = [np.zeros((0, 4)) for _ in range(len(actions)-1)]
        # if self.state_and_actions is None:
        #     self.state_and_actions = np.zeros((0, oldstatefeatures.shape[1] + len(actions)*4))
        # if self.behavior_model is None:
        #     self.init_models(oldstatefeatures.shape[1], len(actions)-1)
        #
        # # update x
        # self.x = np.concatenate((self.x, oldstatefeatures))
        #
        # # update y
        # my_action = np.zeros((1, 4))
        # j = 0
        # for i in range(len(actions)):
        #     if i == self.id:
        #         my_action[0, ACTIONS.index(actions[i])] = 1
        #         continue
        #     self.y[j] = np.concatenate((self.y[j], np.zeros((1, 4))))
        #     self.y[j][-1, ACTIONS.index(actions[i])] = 1
        #     j += 1
        #
        # # update x+y (used for the environment model)
        # state_and_actions = np.concatenate((self.x[-1:], *[a[-1:] for a in self.y], my_action), axis=1)
        # self.state_and_actions = np.concatenate((self.state_and_actions, state_and_actions))
        #
        # predicted = [ACTIONS[np.argmax(a)] for a in self.behavior_model.predict(self.x[-1:])]
        # hits = [predicted[i] == actions[i] for i in range(3)]
        # self.behavior_metric.append(sum(hits)/3)
        #
        # predicted_state = self.environment_model.predict(self.state_and_actions[-1:])
        # self.env_metric.append(np.sum(np.abs(predicted_state - newstatefeatures)))
        #
        # self.behavior_model.fit(self.x, self.y, verbose=0)
        # self.environment_model.fit(self.state_and_actions, np.concatenate((self.x[1:], newstatefeatures)), verbose=0)
        # # if len(self.hits) > 10:
        # #     plt.clf()
        # #     plt.plot([np.mean(self.hits[k:k + 10]) for k in range(len(self.hits) - 10)])
        # #     plt.draw()
        # #     plt.pause(0.02)
        # #print(self.behavior_model.evaluate(self.test[0], self.test[1], verbose=0))

        actions_idx = [ACTIONS.index(a) for a in actions]
        self.b_model.train(state, actions_idx[:self.id] + actions_idx[self.id+1:])
        self.e_model.train(state, actions_idx, new_state)
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

    # def __init__(self, agent_positions, world_size, behavior_model, env_model, adhoc_id, prey=None):
    #     # if prey is None, that means agent_positions are relative to a prey
    #     if prey is None:
    #         mid = (world_size[0] // 2, world_size[1] // 2)
    #         for i, (x, y) in enumerate(agent_positions):
    #             agent_positions[i] = (mid[0]+x, mid[1]+y)
    #         agent_positions = tuple(tuple(pos) for pos in agent_positions)
    #         super().__init__(agent_positions, (mid, ), world_size)
    #     else:
    #         super().__init__(tuple(agent_positions), prey, world_size)
    #     self.behavior_model = behavior_model
    #     self.env_model = env_model
    #     self.adhoc_id = adhoc_id
    #     self.reward_fn = get_reward_function(len(agent_positions), world_size)


    def __init__(self, state, behavior_model, environment_model, adhoc_id):
        super().__init__(state.agent_positions, state.prey_positions, state.world_size)
        self.behavior_model = behavior_model
        self.env_model = environment_model
        self.adhoc_id = adhoc_id
        self.reward_fn = get_reward_function(len(state.agent_positions), state.world_size)
        pass

    def perform(self, action):
        if (self, action) not in MEMO:
            # predict teammate's actions
            predicted_actions = self.behavior_model.predict(self)
            my_action = ACTIONS.index(action)
            # build the joint action
            actions_idx = np.concatenate((predicted_actions[:self.adhoc_id], [my_action], predicted_actions[self.adhoc_id:]))
            # predict new state
            new_state = self.env_model.predict(self, actions_idx)
            MEMO[(self, action)] = GameState(new_state, self.behavior_model, self.env_model, self.adhoc_id)
        return MEMO[(self, action)]

    def is_terminal(self):
        return self.terminal

    def reward(self, parent, action):
        return self.reward_fn(parent, None, self)


class EnvironmentModel(object):
    def __init__(self):
        self.world_size = None
        self.x = None
        self.y = None
        self.model = None
        self.metric = []

    def init(self, num_state_features, output_size, num_agents):
        self.x = np.zeros((0, num_state_features + 4 * num_agents))
        self.y = np.zeros((0, output_size))

        input = Input(shape=(num_state_features + 4 * num_agents,))
        dense = Dense(64, activation='selu')(input)
        output = Dense(output_size, activation='linear')(dense)
        model = Model(input, output)
        model.compile(optimizer='adam', loss='mae')
        self.model = model

    def train(self, state, actions, new_state):
        oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        diff_features = (new_state - state).reshape(1, -1)
        num_agents = len(actions)

        if self.x is None:
            self.init(oldstatefeatures.shape[1], diff_features.shape[1], num_agents)
            self.world_size = state.world_size

        # 1-hot encode
        actions_array = np.zeros((num_agents, 4))
        actions_array[range(num_agents), actions] = 1
        actions_array = actions_array.reshape(1, -1)

        # append to dataset
        self.x = np.append(self.x, np.concatenate((oldstatefeatures, actions_array), axis=1), axis=0)
        self.y = np.append(self.y, diff_features, axis=0)

        # compute accuracy
        predicted = self.predict(state, actions).features()
        hits = [predicted[i] == new_state.features()[i] for i in range(len(predicted))]
        self.metric.append(sum(hits)/diff_features.shape[1])

        # train
        self.model.fit(self.x, self.y, verbose=0)

    def predict(self, state, actions):
        oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        num_agents = len(actions)

        # 1-hot encode
        actions_array = np.zeros((num_agents, 4))
        actions_array[range(num_agents), actions] = 1
        actions_array = actions_array.reshape(1, -1)

        predicted_diff = self.model.predict(np.concatenate((oldstatefeatures, actions_array), axis=1))
        predicted_diff = np.round(predicted_diff).astype(np.int)[0]

        return state+predicted_diff


class BehaviorModel(object):

    def __init__(self):
        self.x = None
        self.y = None
        self.model = None
        self.metric = []

    def init(self, num_state_features, num_agents):
        self.x = np.zeros((0, num_state_features))
        self.y = np.zeros((num_agents, 0, 4))

        input = Input(shape=(num_state_features,))
        dense = Dense(64, activation='selu')(input)
        outputs = []
        for i in range(num_agents):
            outputs.append(Dense(4, activation='softmax')(dense))

        model = Model(inputs=(input, ), outputs=outputs)
        model.compile(optimizer='adam', loss=['categorical_crossentropy']*num_agents, metrics=['accuracy'])
        self.model = model

    def train(self, state, actions):
        oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        num_agents = len(actions)

        if self.x is None:
            self.init(oldstatefeatures.shape[1], num_agents)

        # 1-hot encode
        actions_array = np.zeros((num_agents, 1,  4))
        actions_array[range(num_agents), 0, actions] = 1

        # append to dataset
        self.x = np.append(self.x, oldstatefeatures, axis=0)
        self.y = np.append(self.y, actions_array, axis=1)

        # compute accuracy
        predicted = self.predict(state)
        hits = [predicted[i] == actions[i] for i in range(num_agents)]
        self.metric.append(sum(hits)/num_agents)

        # train
        self.model.fit(self.x, list(self.y), verbose=0)



    def predict(self, state):
        oldstatefeatures = state.features_relative_prey().reshape(1, -1)

        predicted_y = np.array(self.model.predict(oldstatefeatures))[:, 0, :]

        return np.argmax(predicted_y, axis=1)


