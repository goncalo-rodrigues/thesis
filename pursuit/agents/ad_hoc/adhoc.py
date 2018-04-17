import random

import numpy as np
from keras.layers import Dense, Input
from keras import Model

from mcts.mcts.backups import monte_carlo
from mcts.mcts.default_policies import RandomKStepRollOut
from mcts.mcts.graph import StateNode
from mcts.mcts.mcts import MCTS
from mcts.mcts.tree_policies import UCB1
from pursuit.agents.base_agent import Agent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState

ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MEMO = {}


class AdhocAgent(Agent):

    def __init__(self, id, mcts_c=1.41, mcts_n=100, mcts_k=10, behavior_model_size=(64, ), environment_model_size=(64, )):
        super().__init__(id)
        self.id = id
        self.first = True
        self.e_model = EnvironmentModel(behavior_model_size)
        self.b_model = BehaviorModel(environment_model_size)
        self.mcts_c = mcts_c
        self.mcts_n = mcts_n
        self.mcts_k = mcts_k

    def act(self, state):
        if not self.first:
            tree = MCTS(tree_policy=UCB1(c=self.mcts_c), default_policy=RandomKStepRollOut(self.mcts_k), backup=monte_carlo)
            game_state = GameState(state, self.b_model, self.e_model, self.id)
            best_action = tree(StateNode(None, game_state), n=self.mcts_n)
            return best_action
        else:
            self.first = False
            return random.choice(ACTIONS)

    def transition(self, state, actions, new_state, reward):

        actions_idx = [ACTIONS.index(a) for a in actions]
        self.b_model.train(state, [(i, action) for i, action in enumerate(actions_idx) if i != self.id])
        self.e_model.train(state, actions_idx, new_state)
        MEMO.clear()


class GameState(PursuitState):
    actions = ACTIONS

    def __init__(self, state, behavior_model, environment_model, adhoc_id):
        super().__init__(state.agent_positions, state.prey_positions, state.world_size)
        self.behavior_model = behavior_model
        self.env_model = environment_model
        self.adhoc_id = adhoc_id
        self.reward_fn = get_reward_function(len(state.agent_positions), state.world_size)

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
    def __init__(self, model_size):
        self.world_size = None
        self.x = None
        self.y = None
        self.model = None
        self.model_size = model_size
        self.metric = []
        self.metric_prey = []
        self.cache = {}

    def init(self, num_state_features, output_size, num_agents):
        self.x = np.zeros((0, num_state_features + 4 * num_agents))
        self.y = np.zeros((0, output_size))

        input = Input(shape=(num_state_features + 4 * num_agents,))
        previous_layer = input
        for size in self.model_size:
            previous_layer = Dense(size, activation='selu')(previous_layer)
        output = Dense(output_size, activation='linear')(previous_layer)
        model = Model(input, output)
        model.compile(optimizer='adam', loss='mae')
        self.model = model

    def train(self, state, actions, new_state):
        self.cache.clear()
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
        self.metric.append(sum(hits[:-2])/(diff_features.shape[1]-2))
        self.metric_prey.append(sum(hits[-2:]) / 2)

        # train
        self.model.fit(self.x, self.y, verbose=0)

    def predict(self, state, actions):
        oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        dictkey = (str(oldstatefeatures), str(actions))
        if dictkey not in self.cache:
            num_agents = len(actions)

            # 1-hot encode
            actions_array = np.zeros((num_agents, 4))
            actions_array[range(num_agents), actions] = 1
            actions_array = actions_array.reshape(1, -1)

            predicted_diff = self.model.predict(np.concatenate((oldstatefeatures, actions_array), axis=1))
            predicted_diff = np.round(predicted_diff).astype(np.int)[0]
            self.cache[dictkey] = state+predicted_diff

        return self.cache[dictkey]


class BehaviorModel(object):

    def __init__(self, model_size):
        self.x = None
        self.y = None
        self.model = None
        self.model_size = model_size
        self.metric = []
        self.ids = None
        self.cache = {}

    def init(self, num_state_features, actions):
        self.x = np.zeros((0, num_state_features))
        self.y = np.zeros((0, 4))

        input = Input(shape=(num_state_features,))
        previous_layer = input
        for size in self.model_size:
            previous_layer = Dense(size, activation='selu')(previous_layer)
        output = Dense(4, activation='softmax')(previous_layer)

        model = Model(input, output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        self.ids = [x[0] for x in actions]

    def train(self, state, actions):
        self.cache.clear()
        for agent_id, action in actions:
            state_features = state.features_relative_agent(agent_id).reshape(1, -1)
            if self.x is None:
                self.init(state_features.shape[1], actions)
            # 1-hot encode
            actions_array = np.zeros((1,  4))
            actions_array[0, action] = 1

            # append to dataset
            self.x = np.append(self.x, state_features, axis=0)
            self.y = np.append(self.y, actions_array, axis=0)

        # compute accuracy
        predicted_y = self.predict(state)
        hits = [predicted_y[i] == actions[i][1] for i in range(len(actions))]
        self.metric.append(sum(hits)/len(actions))

        # train
        self.model.fit(self.x, self.y, verbose=0)

    def predict(self, state):
        if state not in self.cache:
            state_features = np.zeros((len(self.ids), len(self.x[1])))
            for i, agent_id in enumerate(self.ids):
                state_features[i] = state.features_relative_agent(agent_id).reshape(1, -1)

            predicted_y = np.array(self.model.predict(state_features))
            self.cache[state] = np.argmax(predicted_y, axis=1)

        return self.cache[state]

