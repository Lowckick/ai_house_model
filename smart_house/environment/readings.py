"""Validated sensor readings from UTF-8 JSON or comma-separated text."""
import csv
import datetime as dt
import io
import json
import math
from pathlib import Path

from smart_house.environment.indoor_state import IndoorState, ROOMS

# Indoor measurements: units are part of each field name.
LIMITS = {
    'temperature_c': (-60, 80), 'humidity_pct': (0, 100),
    'co2_ppm': (0, 100000), 'voc_ppb': (0, 100000),
    'pm25_ugm3': (0, 10000), 'pm10_ugm3': (0, 10000),
    'air_quality_index': (0, 500), 'dirt_level': (0, 100),
    'dust_level': (0, 100), 'total_occupants': (0, 100),
    'power_consumption_kw': (0, 1000), 'energy_today_kwh': (0, 24000),
}
REQUIRED = {'timestamp', 'temperature_c', 'humidity_pct'}


MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_RECORDS = 5000


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}')
        result[key] = value
    return result


def inspect_readings(text, suffix):
    """Validate every row and return (states, JSON-safe report). Never skip bad rows."""
    report = {'valid': False, 'count': 0, 'errors': [], 'warnings': [], 'preview': []}
    errors, warnings = report['errors'], report['warnings']
    try:
        if len(text.encode('utf-8')) > MAX_FILE_BYTES:
            raise ValueError('File exceeds 5 MiB')
        text = text.lstrip('\ufeff')
        if suffix.lower() == '.json':
            rows = json.loads(text, object_pairs_hook=_unique_object)
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list):
                raise ValueError('JSON must be an object or an array of objects')
        elif suffix.lower() in {'.csv', '.txt'}:
            reader = csv.DictReader(io.StringIO(text), strict=True)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('CSV requires a header with unique column names')
            rows = list(reader)
        else:
            raise ValueError('Supported file extensions: .json, .csv, .txt')
        if not rows:
            raise ValueError('The input contains no readings')
        if len(rows) > MAX_RECORDS:
            raise ValueError('Maximum 5000 records per file')
    except (ValueError, csv.Error) as exc:
        errors.append({'record': None, 'message': str(exc)})
        return [], report
    report['count'] = len(rows)
    states, seen, previous = [], set(), None
    for index, row in enumerate(rows, 1):
        # Validate fields independently, so one bad column does not hide another.
        row_errors = []
        if isinstance(row, dict):
            unknown = set(row) - (set(LIMITS) | {'timestamp', 'occupant_activity'})
            if unknown:
                row_errors.append('unknown fields: ' + ', '.join(str(k) for k in unknown))
            for field in REQUIRED:
                if field not in row or row[field] in ('', None):
                    row_errors.append('missing field: ' + field)
            for field in (set(LIMITS) | {'timestamp', 'occupant_activity'}) & set(row):
                if field in REQUIRED and row[field] in ('', None):
                    continue
                probe = {'timestamp': '2026-01-01T12:00:00', 'temperature_c': 21, 'humidity_pct': 50}
                probe[field] = row[field]
                try:
                    _parse(probe)
                except (ValueError, TypeError, OverflowError) as exc:
                    row_errors.append(str(exc))
        else:
            row_errors.append('expected an object')
        if row_errors:
            errors.extend({'record': index, 'message': message} for message in row_errors)
            continue
        state = _parse(row)
        states.append(state)
        if len(report['preview']) < 10:
            report['preview'].append({'timestamp': state.timestamp.isoformat(),
                                      'temperature_c': state.temperature_c,
                                      'humidity_pct': state.humidity_pct, 'co2_ppm': state.co2_ppm})
        if state.temperature_c < 5 or state.temperature_c > 32 or state.co2_ppm > 1500 or state.pm25_ugm3 > 55:
            warnings.append({'record': index, 'message': 'Measurements exceed safety thresholds; verify the sensors'})
        defaults = sorted((set(LIMITS) | {'occupant_activity'}) - set(row))
        defaults += sorted(k for k, value in row.items() if value == '' and k not in REQUIRED)
        if defaults:
            warnings.append({'record': index, 'message': 'Default values used: ' + ', '.join(defaults)})
        timestamp = state.timestamp
        if timestamp.tzinfo is None:
            warnings.append({'record': index, 'message': 'Timestamp has no timezone; treated as UTC for ordering'})
            timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
        normalized = timestamp.astimezone(dt.timezone.utc)
        if normalized in seen:
            warnings.append({'record': index, 'message': 'Duplicate timestamp; record will be retained'})
        if previous is not None and normalized < previous:
            warnings.append({'record': index, 'message': 'Timestamp is out of order; input order will be retained'})
        seen.add(normalized)
        previous = normalized
    report['valid'] = not errors
    return (states if not errors else []), report


def load_readings(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('File exceeds 5 MiB')
    states, report = inspect_readings(path.read_text(encoding='utf-8-sig'), path.suffix)
    if not report['valid']:
        raise ValueError('; '.join(f"Record {e['record']}: {e['message']}" for e in report['errors']))
    return states


def _parse(row):
    if not isinstance(row, dict):
        raise ValueError('expected an object')
    unknown = set(row) - (set(LIMITS) | {'timestamp', 'occupant_activity'})
    if unknown:
        raise ValueError(f'unknown fields: {", ".join(str(k) for k in unknown)}')
    missing = [key for key in REQUIRED if key not in row or row[key] in ('', None)]
    if missing:
        raise ValueError(f'missing fields: {", ".join(sorted(missing))}')
    stamp = str(row['timestamp'])
    if 'T' not in stamp and ' ' not in stamp:
        raise ValueError('timestamp must contain ISO 8601 date and time')
    state = IndoorState(timestamp=dt.datetime.fromisoformat(stamp))
    for name, (low, high) in LIMITS.items():
        if name not in row or row[name] == '':
            continue
        if isinstance(row[name], bool) or row[name] is None:
            raise ValueError(f'{name} must be a number')
        try:
            value = float(row[name])
        except (ValueError, TypeError):
            raise ValueError(f'{name} must be a number') from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
        if name == 'total_occupants':
            if not value.is_integer():
                raise ValueError('total_occupants must be an integer')
            value = int(value)
        setattr(state, name, value)
    activity = row.get('occupant_activity', 'idle') or 'idle'
    if activity not in {'idle', 'cooking', 'exercising', 'sleeping'}:
        raise ValueError('invalid occupant_activity')
    state.occupant_activity = activity
    state.room_temps = {room: state.temperature_c for room in ROOMS}
    # No room-level presence information: use a documented whole-house assumption.
    state.occupancy = {room: state.total_occupants > 0 for room in ROOMS}
    return state
