"""Live sensor readings in shadow mode: the model recommends and logs, it never switches devices.

Sensors push one reading at a time (HTTP from an ESP32 or Home Assistant, or MQTT). Each
reading is validated like an uploaded file, gets a recommendation (continuing from the same
sensor's previous recommendation when it is recent), threshold alerts, and is stored. The
shadow report compares what the model recommended with what the room actually did.
"""
import datetime as dt
import json
import re
import statistics
from dataclasses import fields

from smart_house.analytics.history import encode
from smart_house.environment.indoor_state import DeviceState, device_power_kw
from smart_house.environment.readings import inspect_readings
from smart_house.environment.tariff import price_uah_per_kwh
from smart_house.environment.weather_archive import fixed_weather

SENSOR_ID = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')
CONTINUITY_MINUTES = 30      # previous recommendation is the device state if this recent
MAX_INTERVAL_MINUTES = 30    # longer gaps are not integrated into time shares or energy
DEVICE_FIELDS = ('hvac_mode', 'ventilation', 'air_purifier', 'humidifier_mode', 'cleaner_bot')

# (metric, level, predicate, message); critical levels match the simulator's unsafe thresholds.
ALERT_RULES = (
    ('co2_ppm', 'critical', lambda v: v > 1500, 'CO₂ {v:g} ppm > 1500: негайно провітріть'),
    ('co2_ppm', 'warning', lambda v: 1000 < v <= 1500, 'CO₂ {v:g} ppm > 1000: варто провітрити'),
    ('pm25_ugm3', 'critical', lambda v: v > 55, 'PM2.5 {v:g} мкг/м³ > 55: увімкніть очищувач'),
    ('pm25_ugm3', 'warning', lambda v: 35 < v <= 55, 'PM2.5 {v:g} мкг/м³ > 35'),
    ('temperature_c', 'critical', lambda v: v < 5 or v > 32, 'Температура {v:g} °C поза 5–32 °C'),
    ('temperature_c', 'warning', lambda v: 5 <= v < 16 or 28 < v <= 32, 'Температура {v:g} °C поза 16–28 °C'),
    ('humidity_pct', 'warning', lambda v: v < 30 or v > 70, 'Вологість {v:g}% поза 30–70%'),
)
COMFORT = {'temperature_c': (19.0, 25.0), 'humidity_pct': (35.0, 65.0), 'co2_ppm': (None, 1000.0),
           'pm25_ugm3': (None, 35.0)}


def alerts_for(indoor):
    found = []
    for metric, level, predicate, message in ALERT_RULES:
        value = getattr(indoor, metric)
        if predicate(value):
            found.append({'metric': metric, 'level': level, 'value': value, 'message': message.format(v=value)})
    return found


def _in_band(value, band):
    low, high = band
    return (low is None or value >= low) and (high is None or value <= high)


def _corrective(indoor, devices):
    """Which out-of-comfort readings the recommendation addresses with the right device."""
    checks = {
        'temperature_low': (indoor['temperature_c'] < 19, devices['hvac_mode'] in (1, 2)),
        'temperature_high': (indoor['temperature_c'] > 25, devices['hvac_mode'] in (3, 4)),
        'co2_high': (indoor['co2_ppm'] > 1000, devices['ventilation'] >= 1),
        'pm25_high': (indoor['pm25_ugm3'] > 35, devices['air_purifier'] >= 1),
        'humidity_low': (indoor['humidity_pct'] < 35, devices['humidifier_mode'] == 1),
        'humidity_high': (indoor['humidity_pct'] > 65, devices['humidifier_mode'] == 2),
    }
    return {name: addressed for name, (needed, addressed) in checks.items() if needed}


def _devices(payload):
    names = {f.name for f in fields(DeviceState)}
    return DeviceState(**{k: v for k, v in payload.items() if k in names})


def _parse_time(value):
    stamp = dt.datetime.fromisoformat(value)
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


class LiveService:
    """Uses the ImportService's model, weather source and history database."""

    def __init__(self, imports, notifier=None, clock=None):
        self.imports, self.notifier = imports, notifier
        self.clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        with imports.history.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS live_readings (
                    id INTEGER PRIMARY KEY, sensor_id TEXT NOT NULL, received_at TEXT NOT NULL,
                    timestamp TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS live_by_sensor_time ON live_readings(sensor_id, timestamp);
            ''')

    def _last(self, sensor_id):
        with self.imports.history.connect() as db:
            row = db.execute('SELECT payload FROM live_readings WHERE sensor_id=? ORDER BY timestamp DESC, id DESC LIMIT 1',
                             (sensor_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def ingest(self, sensor_id, reading):
        from smart_house.analytics.ingestion import ValidationError, recommend
        if not isinstance(sensor_id, str) or not SENSOR_ID.match(sensor_id):
            raise ValueError('sensor_id: 1-64 letters, digits, "_", "." or "-"')
        if not isinstance(reading, dict):
            raise ValueError('A live reading must be a JSON object')
        received = self.clock()
        reading = dict(reading)
        reading.setdefault('timestamp', received.isoformat())
        states, report = inspect_readings(json.dumps(reading), '.json')
        if not report['valid']:
            raise ValidationError(report)
        indoor = states[0]
        previous = self._last(sensor_id)
        devices = None
        if previous:
            gap = (_parse_time(reading['timestamp']) - _parse_time(previous['timestamp'])).total_seconds() / 60
            if 0 < gap <= CONTINUITY_MINUTES:
                devices = _devices(previous['recommended_devices'])
        with self.imports._lock:
            agent = self.imports._load()
            weather_source = self.imports.weather
            if weather_source:
                weather_source.prefetch([indoor.timestamp])
                weather, source = weather_source.get(indoor.timestamp)
            else:
                weather, source = fixed_weather(indoor.timestamp), 'fixed'
            record = recommend(agent, indoor, weather, source, devices)
            digest = self.imports._digest
        record['model_sha256'] = digest
        record['alerts'] = alerts_for(indoor)
        started = {(a['metric'], a['level']) for a in record['alerts']} - {
            (a['metric'], a['level']) for a in (previous or {}).get('alerts', [])}
        record['new_alerts'] = [a for a in record['alerts'] if (a['metric'], a['level']) in started]
        with self.imports.history.connect() as db:
            ident = db.execute('INSERT INTO live_readings(sensor_id,received_at,timestamp,payload) VALUES(?,?,?,?)',
                               (sensor_id, received.isoformat(), record['timestamp'], encode(record))).lastrowid
        if self.notifier and record['new_alerts']:
            try:
                self.notifier(sensor_id, record)
            except Exception as exc:  # a failed notification must not lose the reading
                record['notification_error'] = f'{type(exc).__name__}: {exc}'
        return {'id': ident, 'sensor_id': sensor_id, **record}

    def sensors(self):
        with self.imports.history.connect() as db:
            return [dict(row) for row in db.execute(
                'SELECT sensor_id, COUNT(*) AS count, MIN(timestamp) AS first, MAX(timestamp) AS last '
                'FROM live_readings GROUP BY sensor_id ORDER BY sensor_id')]

    def readings(self, sensor_id, since=None, limit=500):
        if not 1 <= limit <= 5000:
            raise ValueError('limit must be from 1 to 5000')
        query, args = 'SELECT id, payload FROM live_readings WHERE sensor_id=?', [sensor_id]
        if since:
            query += ' AND timestamp >= ?'
            args.append(since.isoformat())
        with self.imports.history.connect() as db:
            rows = db.execute(query + ' ORDER BY timestamp DESC, id DESC LIMIT ?', (*args, limit)).fetchall()
        return [{'id': row[0], **json.loads(row[1])} for row in reversed(rows)]

    def report(self, sensor_id, hours=24):
        """Shadow-mode summary of observed conditions and of the recommendations made."""
        if not 1 <= hours <= 24 * 90:
            raise ValueError('hours must be from 1 to 2160')
        end = self.clock()
        rows = self.readings(sensor_id, since=end - dt.timedelta(hours=hours), limit=5000)
        result = {'sensor_id': sensor_id, 'hours': hours, 'until': end.isoformat(), 'readings': len(rows),
                  'mode': 'shadow: recommendations were logged, devices were not switched'}
        if not rows:
            return result
        times = [_parse_time(r['timestamp']) for r in rows]
        steps = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:])]
        weights = [min(step, MAX_INTERVAL_MINUTES) for step in steps] + [0.0]
        covered = sum(weights)
        result.update({'first': times[0].isoformat(), 'last': times[-1].isoformat(),
                       'median_interval_min': statistics.median(steps) if steps else None,
                       'gaps_over_30_min': sum(step > MAX_INTERVAL_MINUTES for step in steps),
                       'covered_hours': round(covered / 60, 3)})

        def share(predicate):
            return round(sum(w for r, w in zip(rows, weights) if predicate(r)) / covered, 4) if covered else None

        result['observed'] = {
            'time_in_comfort': {metric: share(lambda r, m=metric, b=band: _in_band(r['indoor'][m], b))
                                for metric, band in COMFORT.items()},
            'time_all_in_comfort': share(lambda r: all(_in_band(r['indoor'][m], b) for m, b in COMFORT.items())),
            'unsafe_readings': sum(any(a['level'] == 'critical' for a in r.get('alerts', [])) for r in rows),
            'mean': {m: round(statistics.fmean(r['indoor'][m] for r in rows), 3) for m in COMFORT}}
        energy = cost = baseline_energy = 0.0
        for r, w in zip(rows, weights):
            hours_ = w / 60
            hour = _parse_time(r['timestamp']).hour
            for target, key in ((None, 'recommended_devices'), ('base', 'deterministic_devices')):
                kwh = device_power_kw(_devices(r[key])) * hours_
                if target is None:
                    energy += kwh
                    cost += kwh * price_uah_per_kwh(hour)
                else:
                    baseline_energy += kwh
        switches = sum(any(a['recommended_devices'][f] != b['recommended_devices'][f] for f in DEVICE_FIELDS)
                       for a, b in zip(rows, rows[1:]))
        corrective = [_corrective(r['indoor'], r['recommended_devices']) for r in rows]
        needed = sum(len(c) for c in corrective)
        result['recommendations'] = {
            'differs_from_simple_controller': sum(any(r['recommended_devices'][f] != r['deterministic_devices'][f]
                                                      for f in DEVICE_FIELDS) for r in rows),
            'corrections_used': sum(r['action_index'] != 0 for r in rows),
            'switches': switches,
            'out_of_comfort_conditions': needed,
            'addressed_by_recommendation': sum(sum(c.values()) for c in corrective),
            'estimated_energy_kwh_if_followed': round(energy, 3),
            'estimated_cost_uah_if_followed': round(cost, 2),
            'simple_controller_energy_kwh': round(baseline_energy, 3),
            'hvac_modes': {str(m): sum(r['recommended_devices']['hvac_mode'] == m for r in rows) for m in range(5)},
            'note': 'Energy assumes the recommended devices ran between readings (power model of the '
                    'simulator, 0.3 kW baseline included); the house was not controlled.'}
        result['alerts'] = [{'timestamp': r['timestamp'], **a} for r in rows for a in r.get('new_alerts', [])][-50:]
        return result


def telegram_notifier(token, chat_id, session=None, timeout=10):
    """Send new alerts to a Telegram chat; token and chat id come from the environment, not the CLI."""
    if session is None:
        import requests
        session = requests

    def notify(sensor_id, record):
        lines = [f'🏠 {sensor_id} · {record["timestamp"]}'] + [
            ('🔴 ' if a['level'] == 'critical' else '🟡 ') + a['message'] for a in record['new_alerts']]
        response = session.post(f'https://api.telegram.org/bot{token}/sendMessage', timeout=timeout,
                                json={'chat_id': chat_id, 'text': '\n'.join(lines)})
        response.raise_for_status()
    return notify


def handle_mqtt_message(service, topic, payload, topic_pattern):
    """Map an MQTT message to ``ingest``: the sensor id is the topic level matched by '+'."""
    pattern, parts = topic_pattern.split('/'), topic.split('/')
    if len(pattern) != len(parts) or any(p not in ('+', t) for p, t in zip(pattern, parts)):
        raise ValueError(f'Topic {topic} does not match {topic_pattern}')
    wildcards = [t for p, t in zip(pattern, parts) if p == '+']
    sensor_id = wildcards[0] if wildcards else parts[-1]
    reading = json.loads(payload.decode('utf-8') if isinstance(payload, bytes) else payload)
    return service.ingest(sensor_id, reading)
