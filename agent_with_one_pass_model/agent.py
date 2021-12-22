import os
import numpy as np
import torch
from lux.game import Game
import controllers.unit_controller as UnitController
import controllers.city_controller as CityController
import controllers.manual_unit_controller as ManualUnitController
import services.resources as ResourceService
import services.maps as MapService
import time
import random


path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'

model = torch.jit.load(f'{path}/lily_one.pth')
model.eval()


def make_input(obs, model_input_size):
    width, height = obs['width'], obs['height']
    x_shift = (model_input_size - width) // 2
    y_shift = (model_input_size - height) // 2
    cities = {}
    
    b = np.zeros((13, model_input_size, model_input_size), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            
            # Units
            team = int(strs[2])

            if team == obs['player']:
                b[0, x, y] = 1
                b[1, x, y] = (wood + coal + uranium) / 100
            else:
                b[2, x, y] = 1
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift

            if team == obs['player']:
              b[3, x, y] = 1
              b[4, x, y] = cities[city_id]
            else:
              b[5, x, y] = 1
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 6, 'coal': 7, 'uranium': 8}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[9 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[11, :] = (obs['step'] % 40) / 40
    # Turns
    b[12, :] = obs['step'] / 360
    
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

    model_input_size = 32

    player = game_state.players[observation.player]
    opponent = game_state.players[0 if observation.player == 1 else 1]

    collided_units = MapService.check_collisions(game_state.movement_records, player.units)

    opponent_collisions = MapService.filter_friendly_collision(player, collided_units)


    # Input for Neural Network
    state = make_input(observation, model_input_size)
    with torch.no_grad():
        p = model(torch.from_numpy(state).unsqueeze(0))

        policies = p.squeeze(0).numpy()

    occupied_pos = opponent_collisions
    for city in opponent.cities.values():
        for city_tile in city.citytiles:
            occupied_pos.append((city_tile.pos.x, city_tile.pos.y))

    actions = []

    manual_policies = []
    opponent_unit_map = MapService.calculate_unit_map(game_state, opponent, model_input_size)

    for unit in player.units:
        if unit.can_act():
            opponent_direction = MapService.check_opponent_adjacent(
                unit.pos,
                opponent_unit_map,
                game_state.map.width,
                model_input_size
            )
 
            if opponent_direction is not None and \
                    not in_city(game_state, unit.pos) and \
                    MapService.check_resource_adjacent(
                        game_state,
                        unit.pos
                    ):
                manual_policies.append((unit, 'PROXIMITY_ALERT', opponent_direction))
                continue

            opponent_corner_direction = MapService.check_opponent_corner(
                unit.pos,
                opponent_unit_map,
                game_state.map.width,
                model_input_size
            )

            if opponent_corner_direction is not None and \
                    MapService.check_resource_adjacent_with_corner(
                        game_state,
                        unit.pos,
                    ):
                manual_policies.append((unit, 'CORNER_ALERT', opponent_corner_direction))
                continue

            if unit.get_cargo_space_left() < 40 and \
                unit.get_cargo_space_left() > 0 and \
                MapService.check_resource_adjacent_build_spot(
                game_state,
                unit.pos
            ):
                manual_policies.append((unit, 'WAIT_TO_BUILD', None))
                continue

    manual_actions, overwritten_units, manual_occupied_pos, manual_movement_records = ManualUnitController.get_actions(game_state, manual_policies, occupied_pos)

    actions.extend(manual_actions)
    occupied_pos.extend(manual_occupied_pos)

    movement_records = []
    model_actions, movement_records = UnitController.get_actions(game_state, player, opponent, policies, occupied_pos, overwritten_units, model_input_size, movement_records)
    actions.extend(model_actions)

    movement_records.extend(manual_movement_records)
    game_state.update_movement_records(movement_records)

    resource_cells = ResourceService.get_resources(game_state, player)
    city_actions = CityController.get_city_actions(game_state, player, resource_cells)
    actions.extend(city_actions)

    turn_time = time.time() - turn_start_time
    # print('I cant believe what is happening!!', turn_time)
    if turn_time > 3:
        game_state.time_pool -= (turn_time - 3)


    return actions
