"""Named building variants for adaptation experiments (all other simulator parts are shared).

Each profile changes the physics a controller has to cope with; the reward, rules and
scenarios stay the same, so differences between policies come from the house alone.
"""
from smart_house.environment.indoor_state import PhysicsParams

REALISTIC_CO2 = dict(co2_per_person_ppm_h=250.0, co2_infiltration_per_h=0.5)

HOUSES = {
    # The simulator every earlier experiment used.
    'default': PhysicsParams(source='house:default'),
    # A person adds hundreds of ppm per hour in a real room; natural air exchange removes some.
    'realistic_co2': PhysicsParams(**REALISTIC_CO2, source='house:realistic_co2'),
    # Weaker envelope: about 1.5x the default loss; in frosts even strong heating barely keeps up.
    'poor_insulation': PhysicsParams(heat_loss_per_h=0.14, source='house:poor_insulation'),
    # Floor heating / radiators reacting over ~1.5 h, and people with daily routines.
    'lagged_schedule': PhysicsParams(hvac_lag_h=1.5, occupancy='schedule', source='house:lagged_schedule'),
    # All of the above together, plus faster humidity exchange.
    'combined': PhysicsParams(heat_loss_per_h=0.14, humidity_exchange_per_h=0.3, hvac_lag_h=1.5,
                              occupancy='schedule', **REALISTIC_CO2, source='house:combined'),
}


def house(name):
    """PhysicsParams of a named house; ValueError for an unknown name."""
    if name not in HOUSES:
        raise ValueError(f"Unknown house {name!r}; choose from: {', '.join(HOUSES)}")
    return HOUSES[name]
