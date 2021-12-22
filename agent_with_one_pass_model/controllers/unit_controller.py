import numpy as np


unit_actions = [
    ('move', 'n'),
    ('move', 'w'),
    ('move', 's'),
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

def get_action(game_state, unit, policy, dest, player, opponent, movement_records):
    for label in np.argsort(policy)[::-1]:
        action_prediction = unit_actions[label]

        if action_prediction[0] == 'build_city':
            # DON'T BY GREEDY
            player_cities = sum([len(x.citytiles) for x in player.cities.values()])
            opponent_cities = sum([len(x.citytiles) for x in opponent.cities.values()])

            if game_state.turn > 320 and player_cities > (opponent_cities + 10):
                return unit.move('c'), unit.pos, movement_records


            cell = game_state.map.get_cell_by_pos(unit.pos)
            if cell.resource is not None:
                continue


            if unit.get_cargo_space_left() > 0:
                adjacent_cells = game_state.map.get_adjacent_cells_by_pos(
                    unit.pos
                )
                for cell in adjacent_cells:
                    if cell.resource is not None:
                        return unit.move('c'), unit.pos, movement_records

                continue

            if (game_state.turn % 40) > 28:
                return unit.move('c'), unit.pos, movement_records

        pos = unit.pos.translate(action_prediction[-1], 1) or unit.pos

        if (pos.x < 0 or pos.x >= game_state.map.width) or \
                (pos.y < 0 or pos.y >= game_state.map.height):
            continue

        if (pos.x, pos.y) not in dest or in_city(game_state, pos):
            action, pos = call_func(unit, *action_prediction), pos
            movement_records.append((unit, pos))
            return action, pos, movement_records

    return unit.move('c'), unit.pos, movement_records


def get_actions(game_state, player, opponent, policies, occupied_pos, overwritten_units, model_input_size, movement_records):
    shift = (model_input_size - game_state.map.width) // 2

    actions = []

    dest = occupied_pos

    for unit in player.units:
        if unit in overwritten_units:
            continue
        if unit.can_act():
            policy = policies[:, unit.pos.x + shift, unit.pos.y + shift]
            action, pos, movement_records = get_action(game_state, unit, policy, dest, player, opponent, movement_records)
            dest.append((pos.x, pos.y))
            actions.append(action)

    return actions, movement_records
