import os
from lux.game_map import Resource
import numpy as np
import torch
from lux.game import Game
import controllers.city_controller as CityController
import controllers.unit_controller as UnitController
import controllers.manual_unit_controller as ManualUnitController
import services.maps as MapService
import services.resources as ResourceService
import time
import random


path = '/kaggle_simulations/agent' if os.path.exists(
    '/kaggle_simulations') else '.'

model = torch.jit.load(f'{path}/rl_model.pth')
model.eval()


# Input for Neural Network
def make_base_input(obs):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((14, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift

            # Units
            team = int(strs[2])
            if team == obs['player']:
                b[2, x, y] = 1
            else:
                b[3, x, y] = 1
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift

            if team == obs['player']:
                b[4, x, y] = 1
                b[5, x, y] = cities[city_id]
            else:
                b[6, x, y] = 1
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 7, 'coal': 8, 'uranium': 9}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])

            if rp < 50:
                rs = 1
            elif rp < 200:
                rs = 2
            else:
                rs = 3

            if team == obs['player']:
                b[10, :] = rs / 3.0
            else:
                b[11, :] = rs / 3.0
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[12, :] = obs['step'] % 40 / 40
    # Turns
    nth_day = obs['step'] / 40
    b[13, :] = nth_day / 8.0

    return b

game_state = None

def get_game_state(observation):
    global game_state

    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation["player"]
    else:
        game_state._update(observation["updates"])
    return game_state

def in_city(game_state, pos):
    try:
        city = game_state.map.get_cell(pos.x, pos.y).citytile
        return city is not None and city.team == game_state.id
    except:
        return False

def agent(observation, configuration):
    global game_state

    turn_start_time = time.time()


    observation['player'] = observation.player
    game_state = get_game_state(observation)
    observation['width'] = game_state.map.width
    observation['height'] = game_state.map.height
    player = game_state.players[observation.player]
    opponent = game_state.players[0 if observation.player == 1 else 1]

    unit_map = MapService.calculate_unit_map(game_state, player)
    opponent_unit_map = MapService.calculate_unit_map(game_state, opponent)

    base_state = make_base_input(observation)
    x_shift = (32 - game_state.map.width) // 2
    y_shift = (32 - game_state.map.height) // 2
    manual_policies = []

    unit_policies = []
    for unit in player.units:
        current_used_time = time.time() - turn_start_time
        if current_used_time > 2.5:
            if game_state.time_pool < (current_used_time - 2):
                break

        if unit.can_act():
            opponent_direction = MapService.check_opponent_adjacent(
                unit.pos,
                opponent_unit_map,
                game_state.map.width,
            )
 
            if opponent_direction is not None and \
                    not in_city(game_state, unit.pos) and \
                    MapService.check_resource_adjacent(
                        game_state,
                        unit.pos
                    ):
                manual_policies.append((unit, 'PROXIMITY_ALERT', opponent_direction))
                continue

            if unit.get_cargo_space_left() < 40 and \
                unit.get_cargo_space_left() > 0 and \
                MapService.check_resource_adjacent_build_spot(
                game_state,
                unit.pos
            ):
                manual_policies.append((unit, 'WAIT_TO_BUILD', None))
                continue

            state_from_base_state = np.copy(base_state)
            state_from_base_state[0, unit.pos.x + x_shift, unit.pos.y + y_shift] = 1
            state_from_base_state[
                1,
                unit.pos.x + x_shift,
                unit.pos.y + y_shift
            ] = (100 - unit.get_cargo_space_left()) / 100
            state_from_base_state[
                2,
                unit.pos.x + x_shift,
                unit.pos.y + y_shift
            ] = 0 if unit_map[unit.pos.x + x_shift][unit.pos.y + y_shift] == 1 else 1

            with torch.no_grad():
                p = model(torch.from_numpy(state_from_base_state).unsqueeze(0))

            policy = p.squeeze(0).numpy()

            unit_policies.append((unit, policy))

    occupied_pos = []
    for city in opponent.cities.values():
        for city_tile in city.citytiles:
            occupied_pos.append((city_tile.pos.x, city_tile.pos.y))

    actions = []
    manual_actions, overwritten_units, manual_occupied_pos = ManualUnitController.get_actions(game_state, manual_policies)
    actions.extend(manual_actions)
    occupied_pos.extend(manual_occupied_pos)

    unit_policies = [unit_policy for unit_policy in unit_policies if unit_policy[0] not in overwritten_units]

    actions.extend(UnitController.get_actions(game_state, unit_policies, occupied_pos))


    resource_cells = ResourceService.get_resources(game_state, player)
    city_actions = CityController.get_city_actions(game_state, player, resource_cells)
    actions.extend(city_actions)

    turn_time = time.time() - turn_start_time
    # print(f'it takes {turn_time} seconds to {len(player.units)} units')
    # print(f'total_manual_policies {len(manual_policies)}')

    if turn_time > 3:
        game_state.time_pool -= (turn_time - 3)

    return actions
