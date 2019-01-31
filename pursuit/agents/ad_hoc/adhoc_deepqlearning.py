import random
from collections import defaultdict
import scipy.stats as st

import numpy as np
from rl.agents import DQNAgent

from common.world_env import WorldEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.agents.handcoded.teammate_aware import TeammateAwareAgent

for world_size in ((20,20),):
    agents = [GreedyAgent(i) for i in range(3)]
    env = WorldEnv(agents, world_size, max_steps=200000)
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dense(64))
    # model.add(Activation('selu'))
    # model.add(Dense(64))
    # model.add(Activation('selu'))
    model.add(Dense(4))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=4, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-4), metrics=['mae', 'accuracy'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    nb_steps = {(5,5): 10000, (10,10): 15000, (20,20): 1000000}
    dqn.fit(env, nb_steps=nb_steps[world_size], visualize=False, verbose=2)

    hist = dqn.test(env, nb_episodes=200, visualize=False)
    results = hist.history['nb_steps']
    print(f'\n####{world_size}####')
    print(np.mean(results))
    print(st.t.interval(0.9, len(results)-1, loc=np.mean(results), scale=st.sem(results)))
    print('##########')