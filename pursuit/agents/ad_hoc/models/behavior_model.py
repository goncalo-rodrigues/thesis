import pickle as pickle

import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model

from pursuit.agents.ad_hoc.models.base_model import BaseModel


class BehaviorModel(BaseModel):

    def __init__(self, model_size):
        super().__init__(model_size)
        self.x = None
        self.y = None
        self.model = None
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

    def train(self, state, actions, fit=True, compute_metrics=True):
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
        if compute_metrics:
            predicted_y = self.predict(state)
            hits = [predicted_y[i] == actions[i][1] for i in range(len(actions))]
            self.metric.append(sum(hits)/len(actions))

        # train
        if fit:
            self.cache.clear()
            self.model.fit(self.x, self.y, epochs=100, verbose=1)

    def predict(self, state):
        if len(self.cache) >= 10*1000:
            print('cleared cache')
            self.cache.clear()

        if state not in self.cache:
            state_features = np.zeros((len(self.ids), len(self.x[0])))
            for i, agent_id in enumerate(self.ids):
                state_features[i] = state.features_relative_agent(agent_id).reshape(1, -1)

            predicted_y = np.array(self.model.predict(state_features))
            self.cache[state] = predicted_y

        predicted_y = self.cache[state]
        # return np.argmax(predicted_y, axis=1)
        return np.array([np.random.choice(range(4), p=p) for p in predicted_y])

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
    #     obj = BehaviorModel(attrs['model_size'])
    #     for key, value in attrs.items():
    #         setattr(obj, key, value)
    #     obj.model = model
    #     return obj