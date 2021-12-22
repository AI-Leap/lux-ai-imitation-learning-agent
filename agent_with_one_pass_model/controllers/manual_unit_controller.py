from lux.game_map import Position
import services.maps as MapService


def get_actions(game_state, manual_policies, occupied_positions):
    movement_records = []
    actions = []
    overwritten_units = []

    for (unit, policy, direction) in manual_policies:
        if policy == "WAIT_TO_BUILD":
            actions.append(unit.move('c'))
            overwritten_units.append(unit)
            occupied_positions.append((unit.pos.x, unit.pos.y))

        if policy == "PROXIMITY_ALERT":
            opponent_position = Position(unit.pos.x + direction[0], unit.pos.y + direction[1])

            # if opponent is in resource, we are outside, so we need to trap him
            is_needed_to_trap = False 
            opponent_cell = game_state.map.get_cell(opponent_position.x, opponent_position.y)
            unit_cell = game_state.map.get_cell(unit.pos.x, unit.pos.y)

            if opponent_cell.has_resource() and not unit_cell.has_resource():
                # print(f'Trapping opponent at ({unit.pos.x} {unit.pos.y}) TURN {game_state.turn}')
                is_needed_to_trap = True

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

            if is_needed_to_protect or is_needed_to_trap:
                # print(f'I am protecting at ({unit.pos.x}, {unit.pos.y}) at TURN {game_state.turn}')
                if unit.get_cargo_space_left() == 0 and ((game_state.turn % 40) < 28):
                    my_cell = game_state.map.get_cell(unit.pos.x, unit.pos.y)
                    if not my_cell.has_resource():
                        actions.append(unit.build_city())
                        overwritten_units.append(unit)
                        occupied_positions.append((unit.pos.x, unit.pos.y))
                        continue

                actions.append(unit.move('c'))
                overwritten_units.append(unit)
                occupied_positions.append((unit.pos.x, unit.pos.y))

        if policy == 'CORNER_ALERT':
            inter_pos_1 = (0, direction[1])
            inter_pos_2 = (direction[0], 0)

            inter_cells = [
                game_state.map.get_cell(unit.pos.x + inter_pos_1[0], unit.pos.y + inter_pos_1[1]),
                game_state.map.get_cell(unit.pos.x + inter_pos_2[0], unit.pos.y + inter_pos_2[1]),
            ]

            if inter_cells[0] is not None and inter_cells[0].has_resource():
                continue
            if inter_cells[1] is not None and inter_cells[1].has_resource():
                continue

            adverse_targets = []

            if direction == (-1, -1):
                adverse_targets = [(0, 2), (2, 0)]
            elif direction == (-1, 1):
                adverse_targets = [(0, -2), (2, 0)]
            elif direction == (1, -1):
                adverse_targets = [(0, 2), (-2, 0)]
            elif direction == (1, 1):
                adverse_targets = [(0, -2), (-2, 0)]

            opponent_position = Position(unit.pos.x + direction[0], unit.pos.y + direction[1])

            adverse_cells = [
                game_state.map.get_cell(opponent_position.x + adverse_targets[0][0], opponent_position.y + adverse_targets[0][1]),
                game_state.map.get_cell(opponent_position.x + adverse_targets[1][0], opponent_position.y + adverse_targets[1][1]),
            ]

            for cell in adverse_cells:
                if cell is None:
                    continue
                if cell.has_resource() or game_state.map.get_cell(opponent_position.x, opponent_position.y).has_resource():
                    interspace_x = (opponent_position.x + cell.pos.x) / 2
                    interspace_y = (opponent_position.y + cell.pos.y) / 2

                    if interspace_x == unit.pos.x:
                        if interspace_y > unit.pos.y:
                            next_pos = unit.pos.translate('s', 1)
                            if (next_pos.x, next_pos.y) in occupied_positions:
                                continue

                            next_cell = game_state.map.get_cell(next_pos.x, next_pos.y)
                            if next_cell.citytile is not None:
                                continue

                            actions.append(unit.move('s'))
                            movement_records.append((unit, next_pos))
                            overwritten_units.append(unit)
                            occupied_positions.append((next_pos.x, next_pos.y))

                            # print(f'I am moving to SOUTH from ({unit.pos.x} {unit.pos.y}) at TURN {game_state.turn}')
                            break
                        elif interspace_y < unit.pos.y:
                            next_pos = unit.pos.translate('n', 1)
                            if (next_pos.x, next_pos.y) in occupied_positions:
                                continue

                            next_cell = game_state.map.get_cell(next_pos.x, next_pos.y)
                            if next_cell.citytile is not None:
                                continue

                            actions.append(unit.move('n'))
                            movement_records.append((unit, next_pos))
                            overwritten_units.append(unit)
                            occupied_positions.append((next_pos.x, next_pos.y))

                            # print(f'I am moving to NORTH from ({unit.pos.x} {unit.pos.y}) at TURN {game_state.turn}')
                            break
                    
                    if interspace_y == unit.pos.y:
                        if interspace_x > unit.pos.x:
                            next_pos = unit.pos.translate('e', 1)
                            if (next_pos.x, next_pos.y) in occupied_positions:
                                continue

                            next_cell = game_state.map.get_cell(next_pos.x, next_pos.y)
                            if next_cell.citytile is not None:
                                continue

                            actions.append(unit.move('e'))
                            movement_records.append((unit, next_pos))
                            overwritten_units.append(unit)
                            occupied_positions.append((next_pos.x, next_pos.y))

                            # print(f'I am moving to EAST from ({unit.pos.x} {unit.pos.y}) at TURN {game_state.turn}')
                            break
                        elif interspace_x < unit.pos.x:
                            next_pos = unit.pos.translate('w', 1)
                            if (next_pos.x, next_pos.y) in occupied_positions:
                                continue

                            next_cell = game_state.map.get_cell(next_pos.x, next_pos.y)
                            if next_cell.citytile is not None:
                                continue

                            actions.append(unit.move('w'))
                            movement_records.append((unit, next_pos))
                            overwritten_units.append(unit)
                            occupied_positions.append((next_pos.x, next_pos.y))

                            # print(f'I am moving to WEST from ({unit.pos.x} {unit.pos.y}) at TURN {game_state.turn}')
                            break

            continue

    return actions, overwritten_units, occupied_positions, movement_records
