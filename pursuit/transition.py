import random

from pursuit.helper import move
from pursuit.state import PursuitState


def get_transition_function(num_agents, world_size, random_instance=None, prey_moves=None):
    if random_instance is None:
        random_instance = random._inst

    def transition(state, actions):
        assert(len(actions) == num_agents)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

        def choose_prey_move():
            if not prey_moves:
                return random_instance.choice(directions)
            else:
                result = prey_moves[0]
                if len(prey_moves) > 1:
                    prey_moves.pop(0)
                return result

        occupied_positions = set(state.prey_positions) | set(state.agent_positions)

        num_preys = len(state.prey_positions)

        apos_array = [None] * num_agents
        ppos_array = [None] * num_preys
        agents_indexs = [(i, True) for i in range(num_agents)] + \
                        [(i, False) for i in range(num_preys)]
        random_instance.shuffle(agents_indexs)

        for i, is_agent in agents_indexs:
            if is_agent:
                position = state.agent_positions[i]
                action = actions[i]
            else:
                position = state.prey_positions[i]
                action = choose_prey_move()
            new_position = move(position, action, world_size)

            # if collision is detected, just go to the original position
            if new_position in occupied_positions:
                new_position = position

            occupied_positions.remove(position)
            occupied_positions.add(new_position)

            if is_agent:
                apos_array[i] = new_position
            else:
                ppos_array[i] = new_position

        return PursuitState(prey_positions=tuple(ppos_array), agent_positions=tuple(apos_array), world_size=tuple(world_size))

    return transition
