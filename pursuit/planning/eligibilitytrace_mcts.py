import random

from mcts.mcts.tree_policies import UCB1
from mcts.mcts.utils import rand_max


class ETMCTS(object):
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, default_policy, confidence_weight, discount_factor=0.95, eligibility_trace=0.95):
        self.default_policy = default_policy
        self.gamma = discount_factor
        self.lambd = eligibility_trace
        self.c = confidence_weight
        self.traces = {}

    def __call__(self, root, n=1500):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """

        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            node = self._get_next_node(root)
            node.reward = self.default_policy(node)

        return rand_max(root.children.values(), key=lambda x: x.q).action

    def _expand(self, state_node):
        action = random.choice(state_node.untried_actions)
        action_node = state_node.children[action]
        action_node.n = 1
        return action_node.sample_state()


    def _best_child(self, state_node):
        best_action_node = rand_max(state_node.children.values(),
                                          key=UCB1(self.c))
        best_action_node.n += 1
        return best_action_node.sample_state()


    def _get_next_node(self, state_node):
        prev_Q = None
        while not state_node.state.is_terminal():
            state_node.n += 1
            if state_node.untried_actions:
                return self._expand(state_node)
            else:
                next_state_node = self._best_child(state_node)
                best_action_node = next_state_node.parent
                self.traces[best_action_node] = 1.0
                current_Q = best_action_node.q
                if prev_Q is not None:
                    reward = state_node.state.reward(None, best_action_node.action)
                    delta = reward + self.gamma * current_Q - prev_Q
                    for action_node in self.traces:
                        action_node.q += self.traces[action_node] * delta / action_node.n
                        self.traces[action_node] *= self.lambd
                prev_Q = current_Q
                state_node = next_state_node
        return state_node