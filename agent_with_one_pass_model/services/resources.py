def get_resources(game_state, player):
    '''
    Get all resource cells in the game map.
    '''
    minable_resource_types = ['wood']
    if player.researched_coal():
        minable_resource_types.append('coal')
    if player.researched_uranium():
        minable_resource_types.append('uranium')

    resource_cells = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource() and cell.resource.type in minable_resource_types:
                resource_cells.append(cell)

    return resource_cells


def get_closest_resource_cell_direction(pos, resource_cells):
    closest_distance = 10000
    closest_resource_cell = None

    for resource_tile in resource_cells:
        dist = resource_tile.pos.distance_to(pos)
        if dist < closest_distance:
            closest_distance = dist
            closest_resource_cell = resource_tile

    return closest_resource_cell.pos.direction_to(pos)

