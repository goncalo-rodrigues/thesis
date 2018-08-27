import random

import numpy as np

from mcts.mcts.backups import monte_carlo, Bellman
from mcts.mcts.default_policies import RandomKStepRollOut
from mcts.mcts.graph import StateNode
from mcts.mcts.mcts import MCTS
from mcts.mcts.tree_policies import UCB1
from pursuit.agents.ad_hoc.models.behavior_model import BehaviorModel
from pursuit.agents.ad_hoc.models.environment_model import EnvironmentModel
from pursuit.agents.ad_hoc.models.stochastic_environment_model import StochasticEnvironmentModel
from pursuit.agents.base_agent import Agent
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
import _pickle as pickle

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
        self.tree = None

    def act(self, state):
        if not self.first:
            game_state = GameState(state, self.b_model, self.e_model, self.id)
            root = StateNode(None, game_state)
            best_n = 0
            # if self.tree is not None and state in \
            #         [x for child in self.tree.children.values() for x in child.children.keys()]:
            #
            #     for child in self.tree.children.values():
            #         for key_state, statenode in child.children.items():
            #             if statenode.n > best_n:
            #                 best_n = statenode.n
            #                 root = statenode
            #     root.parent = None

            tree = MCTS(tree_policy=UCB1(c=self.mcts_c), default_policy=RandomKStepRollOut(self.mcts_k), backup=monte_carlo)
            best_action = tree(root, n=self.mcts_n-best_n)
            self.tree = root
            return best_action
        else:
            self.first = False
            return random.choice(ACTIONS)

    def transition(self, state, actions, new_state, reward, fit=True, compute_metrics=True):
        if fit is None:
            fit = new_state.terminal

        actions_idx = [ACTIONS.index(a) for a in actions]
        self.b_model.train(state, [(i, action) for i, action in enumerate(actions_idx) if i != self.id],
                           fit=fit, compute_metrics=compute_metrics)
        self.e_model.train(state, actions_idx, new_state, fit=fit, compute_metrics=compute_metrics)


    def save(self, filename):
        self.e_model.save(filename+'.emodel')
        self.b_model.save(filename+'.bmodel')
        d = dict(self.__dict__)
        if 'e_model' in d:
            d.pop('e_model')
        if 'b_model' in d:
            d.pop('b_model')
        if 'tree' in d:
            d.pop('tree')
        f = open(filename, 'wb')
        pickle.dump(d, f)
        f.close()


    @staticmethod
    def load(filename):
        f = open(filename, 'rb')
        attrs = pickle.load(f)
        f.close()
        obj = AdhocAgent(attrs['id'])
        for key, value in attrs.items():
            setattr(obj, key, value)
        obj.e_model = EnvironmentModel.load(filename + '.emodel')
        obj.b_model = BehaviorModel.load(filename + '.bmodel')
        return obj


class GameState(PursuitState):
    actions = ACTIONS

    def __init__(self, state, behavior_model, environment_model, adhoc_id):
        super().__init__(state.agent_positions, state.prey_positions, state.world_size)
        self.behavior_model = behavior_model
        self.env_model = environment_model
        self.adhoc_id = adhoc_id
        self.reward_fn = get_reward_function(len(state.agent_positions), state.world_size)

    def perform(self, action):
        # predict teammate's actions
        predicted_actions = self.behavior_model.predict(self)
        my_action = ACTIONS.index(action)
        # build the joint action
        actions_idx = np.concatenate((predicted_actions[:self.adhoc_id], [my_action], predicted_actions[self.adhoc_id:]))
        # predict new state
        new_state = self.env_model.predict(self, actions_idx)
        return GameState(new_state, self.behavior_model, self.env_model, self.adhoc_id)

    def is_terminal(self):
        return self.terminal

    def reward(self, parent, action):
        return self.reward_fn(parent, None, self)


