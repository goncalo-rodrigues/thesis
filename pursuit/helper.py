def neighbors(pos, world_size):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    result = []
    for d in directions:
        result.append(((pos[0]+d[0]) % world_size[0], (pos[1]+d[1]) % world_size[1]))
    return result



