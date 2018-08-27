import pickle as pickle

import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model

from pursuit.agents.ad_hoc.models.base_model import BaseModel


class EnvironmentModel(BaseModel):
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
        self.is_init = True

    def train(self, state, actions, new_state, fit=True, compute_metrics=True):

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
        if compute_metrics:
            predicted = self.predict(state, actions).features()
            hits = [predicted[i] == new_state.features()[i] for i in range(len(predicted))]
            self.metric.append(sum(hits[:-2])/(diff_features.shape[1]-2))
            self.metric_prey.append(sum(hits[-2:]) / 2)

        # train
        if fit:
            self.cache.clear()
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

    # def save(self, filename):
    #     if self.model is not None:
    #         self.model.save(filename + '.model')
    #     d = dict(self.__dict__)
    #     d.pop('model')
    #     f = open(filename, 'wb')
    #     pickle.dump(d, f)
    #     f.close()

    # @staticmethod
    # def load(filename):
    #     model = load_model(filename + '.model')
    #     f = open(filename, 'rb')
    #     attrs = pickle.load(f)
    #     f.close()
    #     obj = EnvironmentModel(attrs['model_size'])
    #     for key, value in attrs.items():
    #         setattr(obj, key, value)
    #     obj.model = model
    #     return obj

