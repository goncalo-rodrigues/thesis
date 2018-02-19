import random

from pursuit.helper import neighbors


class PursuitState(object):
    def __init__(self, agent_positions, prey_positions, world_size):
        self.agent_positions = agent_positions
        self.prey_positions = prey_positions
        self.terminal = True
        for prey in prey_positions:
            for pos in neighbors(prey, world_size):
                if pos not in agent_positions:
                    self.terminal = False
                    return

    @staticmethod
    def random_state(num_agents, world_size):
        assert(num_agents >= 4)
        num_preys = num_agents // 4

        assert(world_size[0]*world_size[1] > num_agents + num_preys)
        filled_positions = set()

        ppos_array = [(0,0)] * num_preys
        apos_array = [(0,0)] * num_preys
        for i in range(num_preys):
            while True:
                pos = (random.randint(0, world_size[0]), random.randint(0, world_size[1]))
                if pos not in filled_positions:
                    break

            ppos_array[i] = pos
            filled_positions.add(pos)

        for i in range(num_agents):
            while True:
                pos = (random.randint(0, world_size[0]), random.randint(0, world_size[1]))
                if pos not in filled_positions:
                    break

            apos_array[i] = pos
            filled_positions.add(pos)

        return PursuitState(prey_positions=ppos_array, agent_positions=apos_array, world_size=world_size)
