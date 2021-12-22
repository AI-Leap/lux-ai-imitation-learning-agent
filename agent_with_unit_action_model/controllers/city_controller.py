from functools import cmp_to_key


def get_total_surrounding_fuel(game_state, pos):
    total_projected_fuel = 0
    i = 0
    for j in range(pos.y - 2, pos.y + 3):
        for k in range(pos.x - 2, pos.x + 3):
            cell = game_state.map.get_cell(k, j)
            if cell is not None and cell.has_resource():
                fuel_score = 1
                if cell.resource.type == 'coal':
                    fuel_score = 10
                if cell.resource.type == 'uranium':
                    fuel_score = 40

                fuel = cell.resource.amount * fuel_score
                total_projected_fuel += fuel

            i += 1

    return total_projected_fuel


def meta_resource_score(pos, resource_cells):
    score = 0
    for rc in resource_cells:
        distance = pos.distance_to(rc.pos)
        amount = rc.resource.amount

        score += amount / (distance * 2)
    return score


def get_city_actions(
    game_state,
    player,
    resource_cells
):
    '''
    This is actually simple. We greedily build worker if possible.
    The only trick is if two citytiles can build only one worker,
    we decide which gets to build by calculating the score.
    '''
    actions = []
    units_capacity = sum([len(x.citytiles) for x in player.cities.values()])
    units_count = len(player.units)

    actionable_citytiles = []
    for city in player.cities.values():
        for citytile in city.citytiles:
            if citytile.can_act():
                actionable_citytiles.append(citytile)

    citytiles_to_be_sorted = []
    for citytile in actionable_citytiles:
        citytile_score = meta_resource_score(citytile.pos, resource_cells)
        # print(f'citytile_score: {citytile_score}')

        citytiles_to_be_sorted.append({
            'citytile': citytile,
            'score': citytile_score
        })

    def compare(citytile1, citytile2):
        return citytile2['score'] - citytile1['score']

    sorted_citytiles = sorted(
        citytiles_to_be_sorted,
        key=cmp_to_key(compare)
    )

    for citytile in sorted_citytiles:
        if (player.research_points < 50 and player.research_points >= 45) or \
                (player.research_points < 200 and player.research_points > 190):
            actions.append(
                citytile['citytile'].research()
            )
        elif units_count < units_capacity:
            actions.append(
                citytile['citytile'].build_worker()
            )
            units_count += 1
        else:
            if not player.researched_uranium():
                actions.append(
                    citytile['citytile'].research()
                )

    return actions