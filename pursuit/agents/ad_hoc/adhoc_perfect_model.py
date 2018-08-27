from mcts.mcts.backups import monte_carlo, Bellman
from mcts.mcts.default_policies import RandomKStepRollOut
from mcts.mcts.graph import StateNode
from mcts.mcts.mcts import MCTS
from mcts.mcts.tree_policies import UCB1
from pursuit.agents.base_agent import Agent
from pursuit.agents.handcoded.greedy import GreedyAgent
from pursuit.planning.eligibilitytrace_mcts import ETMCTS
from pursuit.reward import get_reward_function
from pursuit.state import PursuitState
from pursuit.transition import get_transition_function

class AdhocPerfectModel(Agent):
    def __init__(self, id, mcts_c=1.41, mcts_n=100, mcts_k=10, agent_type=GreedyAgent):
        super().__init__(id)
        self.agent_type = agent_type
        self.mcts_n = mcts_n
        self.mcts_c = mcts_c
        self.mcts_k = mcts_k

    def act(self, state):
        game_state = GameState(state, self.id, self.agent_type)
        tree = ETMCTS(confidence_weight=self.mcts_c, default_policy=RandomKStepRollOut(self.mcts_k))
        best_action = tree(StateNode(None, game_state), n=self.mcts_n)
        return best_action


class GameState(PursuitState):
    actions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    def __init__(self, state, adhoc_id, agent_type):
        super().__init__(state.agent_positions, state.prey_positions, state.world_size)
        self.adhoc_id = adhoc_id
        self.reward_fn = get_reward_function(len(state.agent_positions), state.world_size)
        self.transi_fn = get_transition_function(len(state.agent_positions), state.world_size)
        self.agents = [agent_type(i) for i in range(3)]
        self.agent_type = agent_type

    def perform(self, action):
        actions = [self.agents[i].act(self) for i in range(3)]
        actions = actions[:self.adhoc_id] + [action] + actions[self.adhoc_id:]
        new_state = self.transi_fn(self, actions)
        return GameState(new_state, self.adhoc_id, self.agent_type)

    def is_terminal(self):
        return self.terminal

    def reward(self, parent, action):
        return self.reward_fn(parent, None, self)

