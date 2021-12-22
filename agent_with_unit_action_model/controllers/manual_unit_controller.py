from lux.game_map import Position
import services.maps as MapService


def get_actions(game_state, manual_policies):
    actions = []
    overwritten_units = []
    occupied_positions = []

    for (unit, policy, direction) in manual_policies:
        if policy == "WAIT_TO_BUILD":
            actions.append(unit.move('c'))
            overwritten_units.append(unit)
            occupied_positions.append((unit.pos.x, unit.pos.y))

        if policy == "PROXIMITY_ALERT":
            opponent_position = Position(unit.pos.x + direction[0], unit.pos.y + direction[1])

            adjacent_cells = MapService.get_two_sides_of_opposition(
                game_state, unit.pos, opponent_position
            )

            is_needed_to_protect = True
            # we will protect only if there is no resource on the adjacent cells of opponent
            # except our unit position
            for cell in adjacent_cells:
                if cell is None:
                    continue

                if cell.has_resource():
                    is_needed_to_protect = False

            if is_needed_to_protect:
                # print(f'I am protecting at ({unit.pos.x}, {unit.pos.y}) at TURN {game_state.turn}')
                if unit.get_cargo_space_left() == 0:
                    my_cell = game_state.map.get_cell(unit.pos.x, unit.pos.y)
                    if not my_cell.has_resource() and ((game_state.turn % 40) < 28):
                        actions.append(unit.build_city())
                        overwritten_units.append(unit)
                        occupied_positions.append((unit.pos.x, unit.pos.y))
                        continue

                actions.append(unit.move('c'))
                overwritten_units.append(unit)
                occupied_positions.append((unit.pos.x, unit.pos.y))

    return actions, overwritten_units, occupied_positions