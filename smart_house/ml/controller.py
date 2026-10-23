"""Deterministic teacher and versioned residual action semantics."""
from smart_house.environment.indoor_state import DeviceState
from smart_house.ml.rule_layer import RuleLayer

RESIDUAL_ACTIONS = [(None, None)] + [
    (field, value) for field, values in (
        ('hvac_mode', range(5)), ('ventilation', range(4)),
        ('air_purifier', range(4)), ('humidifier_mode', range(3)),
        ('cleaner_bot', range(2)),
    ) for value in values
]


def _threshold_devices(indoor):
    d = DeviceState()
    if indoor.temperature_c < 20:
        d.hvac_mode = 1
    elif indoor.temperature_c > 24:
        d.hvac_mode = 3
    d.ventilation = 1 if indoor.co2_ppm > 1000 else 0
    d.air_purifier = 1 if indoor.pm25_ugm3 > 12 else 0
    d.humidifier_mode = 1 if indoor.humidity_pct < 40 else (2 if indoor.humidity_pct > 60 else 0)
    d.cleaner_bot = int(indoor.dirt_level > 30 and indoor.occupant_activity != 'sleeping')
    return d


def deterministic_devices(indoor, weather):
    return RuleLayer().apply(_threshold_devices(indoor), indoor, weather)


def teacher_explanations(indoor, weather):
    """Explain inherited controller decisions as well as its final safety constraints."""
    from dataclasses import asdict
    raw = _threshold_devices(indoor)
    reasons = {
        'hvac_mode': f'Термостат: {indoor.temperature_c:g} °C; обігрів нижче 20, охолодження вище 24.',
        'ventilation': f'CO₂ {indoor.co2_ppm:g} ppm; вентиляція 1 при значенні понад 1000.',
        'air_purifier': f'PM2.5 {indoor.pm25_ugm3:g}; очищувач 1 при значенні понад 12.',
        'humidifier_mode': f'Вологість {indoor.humidity_pct:g}%; зволоження нижче 40%, осушення вище 60%.',
        'cleaner_bot': f'Забруднення {indoor.dirt_level:g}%, активність {indoor.occupant_activity}; прибирання понад 30%, крім сну.',
    }
    records = [{'source': 'teacher', 'stage': 'deterministic', 'field': field,
                'before': getattr(DeviceState(), field), 'after': getattr(raw, field), 'reason': reason}
               for field, reason in reasons.items()]
    _, changes = RuleLayer().apply_with_explanations(raw, indoor, weather)
    records.extend({**change, 'source': 'teacher', 'stage': 'deterministic_' + change['stage']} for change in changes)
    return records


def residual_devices(index, indoor, weather):
    if not 0 <= index < len(RESIDUAL_ACTIONS):
        raise ValueError('Invalid residual action index')
    devices = deterministic_devices(indoor, weather)
    field, value = RESIDUAL_ACTIONS[index]
    if field is not None:
        setattr(devices, field, value)
    return devices
