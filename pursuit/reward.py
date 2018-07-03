def get_reward_function(num_agents, world_size):
    def reward(state, actions, next_state):
        return 20 if next_state.terminal else -1
    return reward
