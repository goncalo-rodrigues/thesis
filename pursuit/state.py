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
