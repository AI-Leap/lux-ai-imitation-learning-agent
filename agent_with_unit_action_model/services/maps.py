import numpy as np


def calculate_unit_map(game_state, player):
    unit_map = np.zeros((32, 32))
    x_shift = (32 - game_state.map.width) // 2
    y_shift = (32 - game_state.map.height) // 2

    for unit in player.units:
        unit_map[unit.pos.x + x_shift][unit.pos.y + y_shift] += 1

    return unit_map


def check_resource_adjacent(game_state, pos):
    adjacent_cells = game_state.map.get_adjacent_cells_by_pos(pos)

    for cell in adjacent_cells:
        if cell.has_resource():
            return True

    return False


def check_resource_adjacent_build_spot(game_state, pos):
    if game_state.map.get_cell_by_pos(pos).has_resource():
        return False

    adjacent_cells = game_state.map.get_adjacent_cells_by_pos(pos)

    for cell in adjacent_cells:
        if cell.has_resource():
            return True

    return False


def check_opponent_adjacent(pos, unit_map, map_height):
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    shift = (32 - map_height) // 2

    for direction in directions:
        x = pos.x + direction[0] + shift
        y = pos.y + direction[1] + shift

        if 0 <= x < 32 and 0 <= y < 32:
            if unit_map[x][y] > 0:
                return direction

    return None


def get_two_sides_of_opposition(game_state, unit_pos, opponent_pos):
    position_diff = (abs(unit_pos.x - opponent_pos.x), abs(unit_pos.y - opponent_pos.y))

    if position_diff[0] == 0 and position_diff[1] == 1: # it is N or S
        # return east and west of opponent
        return [
            game_state.map.get_cell(opponent_pos.x - 1, opponent_pos.y),
            game_state.map.get_cell(opponent_pos.x + 1, opponent_pos.y),
            game_state.map.get_cell(unit_pos.x, unit_pos.y)
        ]
    if position_diff[0] == 1 and position_diff[1] == 0: # it is E or W
        # return north and south of opponent
        return [
            game_state.map.get_cell(opponent_pos.x, opponent_pos.y - 1),
            game_state.map.get_cell(opponent_pos.x, opponent_pos.y + 1),
            game_state.map.get_cell(unit_pos.x, unit_pos.y)
        ]

    return []
