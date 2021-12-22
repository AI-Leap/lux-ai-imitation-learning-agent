import numpy as np


unit_actions = [
    ('move', 'n'),
    ('move', 's'),
    ('move', 'w'),
    ('move', 'e'),
    ('build_city',)
]


def in_city(game_state, pos):
    try:
        city = game_state.map.get_cell(pos.x, pos.y).citytile
        return city is not None and city.team == game_state.id
    except:
        return False


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)


def get_action(game_state, unit, policy, dest):
    for label in np.argsort(policy)[::-1]:
        action_prediction = unit_actions[label]

        if action_prediction[0] == 'build_city':
            cell = game_state.map.get_cell_by_pos(unit.pos)
            if cell.resource is not None:
                continue


            if unit.get_cargo_space_left() > 0:
                adjacent_cells = game_state.map.get_adjacent_cells_by_pos(
                    unit.pos
                )
                for cell in adjacent_cells:
                    if cell.resource is not None:
                        return unit.move('c'), unit.pos

            if (game_state.turn % 40) > 28:
                return unit.move('c'), unit.pos

        pos = unit.pos.translate(action_prediction[-1], 1) or unit.pos

        if (pos.x < 0 or pos.x >= game_state.map.width) or \
                (pos.y < 0 or pos.y >= game_state.map.height):
            continue

        if (pos.x, pos.y) not in dest or in_city(game_state, pos):
            action, pos = call_func(unit, *action_prediction), pos

            return action, pos

    return unit.move('c'), unit.pos


def get_actions(game_state, unit_policies, occupied_pos):
    actions = []

    dest = occupied_pos

    for (unit, policy) in unit_policies:
        action, pos = get_action(game_state, unit, policy, dest)
        dest.append((pos.x, pos.y))
        actions.append(action)

    return actions
