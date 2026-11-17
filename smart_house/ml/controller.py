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


def _tuned_devices(indoor):
    """Stronger staged thresholds: the base of residual_v2 and a fair, hand-tuned baseline."""
    d = _threshold_devices(indoor)
    t = indoor.temperature_c
    d.hvac_mode = 2 if t < 19.5 else 1 if t < 20.5 else 4 if t > 25 else 3 if t > 24 else 0
    c = indoor.co2_ppm
    d.ventilation = 3 if c > 1300 else 2 if c > 1000 else 1 if c > 800 else 0
    p = indoor.pm25_ugm3
    d.air_purifier = 3 if p > 35 else 2 if p > 20 else 1 if p > 12 else 0
    d.cleaner_bot = int(indoor.dirt_level > 15 and indoor.occupant_activity != 'sleeping')
    return d


# Residual action modes: same 19 corrections, different deterministic base controller.
BASE_CONTROLLERS = {'residual_v1': _threshold_devices, 'residual_v2': _tuned_devices}
RESIDUAL_MODES = frozenset(BASE_CONTROLLERS)


def deterministic_devices(indoor, weather):
    return RuleLayer().apply(_threshold_devices(indoor), indoor, weather)


def tuned_deterministic_devices(indoor, weather):
    return RuleLayer().apply(_tuned_devices(indoor), indoor, weather)


def _reasons(indoor, mode):
    if mode == 'residual_v2':
        return {
            'hvac_mode': f'Термостат: {indoor.temperature_c:g} °C; сильний обігрів нижче 19.5, слабкий нижче 20.5, '
                         'охолодження вище 24, сильне вище 25.',
            'ventilation': f'CO₂ {indoor.co2_ppm:g} ppm; вентиляція 1/2/3 понад 800/1000/1300.',
            'air_purifier': f'PM2.5 {indoor.pm25_ugm3:g}; очищувач 1/2/3 понад 12/20/35.',
            'humidifier_mode': f'Вологість {indoor.humidity_pct:g}%; зволоження нижче 40%, осушення вище 60%.',
            'cleaner_bot': f'Забруднення {indoor.dirt_level:g}%, активність {indoor.occupant_activity}; прибирання понад 15%, крім сну.',
        }
    return {
        'hvac_mode': f'Термостат: {indoor.temperature_c:g} °C; обігрів нижче 20, охолодження вище 24.',
        'ventilation': f'CO₂ {indoor.co2_ppm:g} ppm; вентиляція 1 при значенні понад 1000.',
        'air_purifier': f'PM2.5 {indoor.pm25_ugm3:g}; очищувач 1 при значенні понад 12.',
        'humidifier_mode': f'Вологість {indoor.humidity_pct:g}%; зволоження нижче 40%, осушення вище 60%.',
        'cleaner_bot': f'Забруднення {indoor.dirt_level:g}%, активність {indoor.occupant_activity}; прибирання понад 30%, крім сну.',
    }


def teacher_explanations(indoor, weather, mode='residual_v1'):
    """Explain inherited controller decisions as well as its final safety constraints."""
    if mode not in RESIDUAL_MODES:
        raise ValueError('Unknown residual action mode')
    raw = BASE_CONTROLLERS[mode](indoor)
    reasons = _reasons(indoor, mode)
    records = [{'source': 'teacher', 'stage': 'deterministic', 'field': field,
                'before': getattr(DeviceState(), field), 'after': getattr(raw, field), 'reason': reason}
               for field, reason in reasons.items()]
    _, changes = RuleLayer().apply_with_explanations(raw, indoor, weather)
    records.extend({**change, 'source': 'teacher', 'stage': 'deterministic_' + change['stage']} for change in changes)
    return records


def residual_devices(index, indoor, weather, mode='residual_v1'):
    if not 0 <= index < len(RESIDUAL_ACTIONS):
        raise ValueError('Invalid residual action index')
    if mode not in RESIDUAL_MODES:
        raise ValueError('Unknown residual action mode')
    devices = RuleLayer().apply(BASE_CONTROLLERS[mode](indoor), indoor, weather)
    field, value = RESIDUAL_ACTIONS[index]
    if field is not None and (mode == 'residual_v1' or correction_allowed(field, value, devices, indoor)):
        setattr(devices, field, value)
    return devices


def correction_allowed(field, value, base, indoor):
    """residual_v2 corrections may not weaken the base where a reading is already outside comfort.

    Without this, a learned policy can save heat by idling ventilation at night and let CO2 hover
    at the 1500 ppm safety threshold, relying on the final rule layer. A disallowed correction is
    ignored: the base decision is kept.
    """
    if field == 'ventilation' and indoor.co2_ppm > 1000:
        return value >= base.ventilation
    if field == 'air_purifier' and indoor.pm25_ugm3 > 35:
        return value >= base.air_purifier
    if field == 'hvac_mode' and indoor.temperature_c < 19:
        return value in (1, 2) and (value == 2 or base.hvac_mode != 2)
    if field == 'hvac_mode' and indoor.temperature_c > 25:
        return value in (3, 4) and (value == 4 or base.hvac_mode != 4)
    if field == 'humidifier_mode' and indoor.humidity_pct < 35:
        return value == 1
    if field == 'humidifier_mode' and indoor.humidity_pct > 65:
        return value == 2
    return True
