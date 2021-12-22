import numpy as np


def calculate_unit_map(game_state, player, model_input_width):
    unit_map = np.zeros((model_input_width, model_input_width))
    x_shift = (model_input_width - game_state.map.width) // 2
    y_shift = (model_input_width - game_state.map.height) // 2

    for unit in player.units:
        unit_map[unit.pos.x + x_shift][unit.pos.y + y_shift] += 1

    return unit_map


def check_resource_adjacent(game_state, pos):
    adjacent_cells = game_state.map.get_adjacent_cells_by_pos(pos)

    for cell in adjacent_cells:
        if cell.has_resource():
            return True

    return False


def check_resource_adjacent_with_corner(game_state, pos):
    adjacent_cells = game_state.map.get_adjacent_cells_with_corner_by_pos(pos)

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


def check_opponent_adjacent(pos, unit_map, map_height, model_input_width):
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    shift = (model_input_width - map_height) // 2

    for direction in directions:
        x = pos.x + direction[0] + shift
        y = pos.y + direction[1] + shift

        if 0 <= x < model_input_width and 0 <= y < model_input_width:
            if unit_map[x][y] > 0:
                return direction

    return None


def check_opponent_corner(pos, unit_map, map_height, model_input_width):
    directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    shift = (model_input_width - map_height) // 2

    for direction in directions:
        x = pos.x + direction[0] + shift
        y = pos.y + direction[1] + shift

        if 0 <= x < model_input_width and 0 <= y < model_input_width:
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


def check_collisions(movement_records, units):
    if len(movement_records) == 0:
        return []

    collided_units = []
    for (r_unit, r_pos) in movement_records:

        try:
            unit = next(
                (u for u in units if u.id == r_unit.id),
            )

            if unit is not None:
                if unit.pos != r_pos:
                    collided_units.append((unit, unit.pos, r_pos))
        except StopIteration:
            continue

    return collided_units


def filter_friendly_collision(player, collided_units):
    opponent_collisions = []

    for (unit, unit_pos, r_pos) in collided_units:
        try:
            friend_at_r_pos = next(
                (u for u in player.units if u.pos == r_pos),
            )
            if friend_at_r_pos is None:
                opponent_collisions.append((unit_pos.x, unit_pos.y))
            
        except StopIteration:
            continue

    return opponent_collisions
