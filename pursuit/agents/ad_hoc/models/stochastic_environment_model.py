import pickle as pickle

import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model

from pursuit.agents.ad_hoc.models.base_model import BaseModel


class StochasticEnvironmentModel(BaseModel):
    diffs = np.array(((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)))
    def __init__(self, model_size):
        super().__init__(model_size)
        self.world_size = None
        self.x = None
        self.y = None
        self.model = None
        self.metric = []
        self.metric_prey = []
        self.cache = {}
        self.is_init = False

    def init(self, num_state_features, output_size, num_agents, num_preys=1):
        self.x = np.zeros((0, num_state_features + 4 * num_agents))
        self.y = np.zeros((num_agents+num_preys, 0, output_size))

        input = Input(shape=(num_state_features + 4 * num_agents,))
        previous_layer = input
        for size in self.model_size:
            previous_layer = Dense(size, activation='selu')(previous_layer)

        outputs = []
        for _ in range(num_agents+num_preys):
            outputs.append(Dense(output_size, activation='softmax')(previous_layer))

        model = Model(input, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model = model
        self.is_init = True

    def train(self, state, actions, new_state, fit=True, compute_metrics=True):

        def one_hot_encode(vector):
            return ((self.diffs[None,:] == vector[:,None]).all(axis=2)).astype('int')

        oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        diff_features = (new_state - state).reshape(-1, 2)
        num_agents = len(actions)
        if self.x is None:
            self.init(oldstatefeatures.shape[1], len(self.diffs), num_agents)
            self.world_size = state.world_size

        # 1-hot encode
        actions_array = np.zeros((num_agents, 4))
        actions_array[range(num_agents), actions] = 1
        actions_array = actions_array.reshape(1, -1)

        # append to dataset
        self.x = np.append(self.x, np.concatenate((oldstatefeatures, actions_array), axis=1), axis=0)
        self.y = np.append(self.y, one_hot_encode(diff_features)[:,None], axis=1)

        # compute accuracy
        if compute_metrics:
            predicted = self.predict(state, actions).features().reshape(-1, 2)
            hits = [(predicted[i] == new_state.features().reshape(-1,2)[i]).all() for i in range(len(predicted))]
            self.metric.append(sum(hits[:-1])/(len(hits)-1))
            self.metric_prey.append(sum(hits[-1:]) / 1)

        # train
        if fit:
            self.cache.clear()
            self.model.fit(self.x, list(self.y), verbose=0)

    def predict(self, state, actions):
        oldstatefeatures = state.features_relative_prey().reshape(1, -1)
        dictkey = (str(oldstatefeatures), str(actions))
        def one_hot_decode(vector):
            return self.diffs[np.where(vector)[1]]
        if dictkey not in self.cache:
            num_agents = len(actions)

            # 1-hot encode
            actions_array = np.zeros((num_agents, 4))
            actions_array[range(num_agents), actions] = 1
            actions_array = actions_array.reshape(1, -1)

            predicted_diff = self.model.predict(np.concatenate((oldstatefeatures, actions_array), axis=1))

            self.cache[dictkey] = [a[0] for a in predicted_diff]

        predicted_diff = self.cache[dictkey]
        sampled_diff = np.zeros((len(predicted_diff), 2))

        for i, distribution in enumerate(predicted_diff):
            sampled_diff[i] = self.diffs[np.random.choice(range(len(distribution)), p=distribution)]

        return state+sampled_diff.reshape(-1)