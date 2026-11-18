import datetime as dt
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from smart_house.analytics.history import History
from smart_house.analytics.ingestion import ImportService, ValidationError
from smart_house.analytics.live import LiveService, alerts_for, handle_mqtt_message, telegram_notifier
from smart_house.environment.indoor_state import IndoorState
from smart_house.web.server import make_server

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'data/models/rollout-v2.pt'
T0 = dt.datetime(2026, 1, 10, 6, 0, tzinfo=dt.timezone.utc)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.imports = ImportService(MODEL, History(Path(self.tmp.name) / 'live.db'))
        self.sent = []
        self.clock = Clock(T0)
        self.live = LiveService(self.imports, lambda sensor, record: self.sent.append((sensor, record)), self.clock)

    def push(self, minutes, **values):
        reading = {'timestamp': (T0 + dt.timedelta(minutes=minutes)).isoformat(),
                   'temperature_c': 21, 'humidity_pct': 45, **values}
        self.clock.now = T0 + dt.timedelta(minutes=minutes)
        return self.live.ingest('bedroom', reading)

    def test_alert_levels(self):
        levels = {(a['metric'], a['level']) for a in alerts_for(IndoorState(co2_ppm=1600, temperature_c=17, humidity_pct=75))}
        self.assertEqual(levels, {('co2_ppm', 'critical'), ('humidity_pct', 'warning')})
        self.assertEqual(alerts_for(IndoorState()), [])

    def test_ingest_continuity_alert_onset_and_report(self):
        first = self.push(0, co2_ppm=900)
        self.assertEqual((first['sensor_id'], first['weather_source']), ('bedroom', 'fixed'))
        self.assertEqual(len(first['top_corrections']), 3)
        second = self.push(10, co2_ppm=1200)
        third = self.push(20, co2_ppm=1300)
        # The alert is reported once when it starts, not on every reading while it lasts.
        self.assertEqual([a['level'] for a in second['new_alerts']], ['warning'])
        self.assertEqual(third['new_alerts'], [])
        self.assertEqual([s for s, _ in self.sent], ['bedroom'])
        self.push(30, co2_ppm=800, temperature_c=17)
        self.push(120, co2_ppm=700)   # 90-minute gap: not integrated into time shares
        report = self.live.report('bedroom', hours=24)
        self.assertEqual((report['readings'], report['gaps_over_30_min'], report['covered_hours']), (5, 1, 1.0))
        observed = report['observed']
        # Each reading holds until the next one, at most 30 min: 10+10+10+30 = 60 covered minutes.
        self.assertEqual(observed['time_in_comfort']['co2_ppm'], 0.6667)     # 1200/1300 ppm for 20 of 60 min
        self.assertEqual(observed['time_in_comfort']['temperature_c'], 0.5)  # 17 °C for the last 30 min
        recommendations = report['recommendations']
        self.assertEqual(recommendations['out_of_comfort_conditions'], 3)
        self.assertGreaterEqual(recommendations['addressed_by_recommendation'], 2)
        self.assertGreater(recommendations['estimated_energy_kwh_if_followed'], 0.3 * 40 / 60)
        self.assertEqual(len(report['alerts']), 1)
        self.assertEqual([s['sensor_id'] for s in self.live.sensors()], ['bedroom'])
        self.assertEqual(len(self.live.readings('bedroom', limit=2)), 2)
        self.assertEqual(self.live.report('kitchen')['readings'], 0)

    def test_rejects_bad_input_and_survives_notifier_failure(self):
        with self.assertRaises(ValueError):
            self.live.ingest('bad id!', {'temperature_c': 20, 'humidity_pct': 40})
        with self.assertRaises(ValidationError):
            self.live.ingest('bedroom', {'temperature_c': 200, 'humidity_pct': 40})
        failing = LiveService(self.imports, lambda *_: (_ for _ in ()).throw(RuntimeError('offline')), self.clock)
        record = failing.ingest('hall', {'temperature_c': 20, 'humidity_pct': 40, 'co2_ppm': 1700})
        self.assertIn('RuntimeError', record['notification_error'])
        self.assertEqual(record['timestamp'], T0.isoformat())   # missing timestamp = time received
        self.assertEqual(len(failing.readings('hall')), 1)

    def test_mqtt_topic_mapping_and_telegram_message(self):
        record = handle_mqtt_message(self.live, 'home/office/climate',
                                     json.dumps({'temperature_c': 22, 'humidity_pct': 50}).encode(), 'home/+/climate')
        self.assertEqual(record['sensor_id'], 'office')
        with self.assertRaises(ValueError):
            handle_mqtt_message(self.live, 'other/office', b'{}', 'home/+/climate')

        class Session:
            def post(self, url, json, timeout):
                self.request = (url, json)
                return type('R', (), {'raise_for_status': lambda self: None})()
        session = Session()
        telegram_notifier('TOKEN', 42, session)('office', {'timestamp': 't', 'new_alerts': [
            {'level': 'critical', 'message': 'CO₂ 1600 ppm'}]})
        self.assertIn('/botTOKEN/sendMessage', session.request[0])
        self.assertEqual(session.request[1]['chat_id'], 42)
        self.assertIn('CO₂ 1600 ppm', session.request[1]['text'])

    def test_http_sensor_push_requires_token(self):
        server = make_server('127.0.0.1', 0, MODEL, Path(self.tmp.name) / 'web.db', live_token='s3cret')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'

        def call(path, body=None, token=None):
            headers = {'Content-Type': 'application/json'}
            if token:
                headers['Authorization'] = 'Bearer ' + token
            data = None if body is None else json.dumps(body).encode()
            try:
                with urllib.request.urlopen(urllib.request.Request(base + path, data, headers), timeout=30) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as exc:
                return exc.code, json.loads(exc.read())

        reading = {'temperature_c': 18.5, 'humidity_pct': 40, 'co2_ppm': 1100}
        self.assertEqual(call('/api/live/kitchen', reading)[0], 401)
        self.assertEqual(call('/api/live/kitchen', reading, 'wrong')[0], 401)
        status, record = call('/api/live/kitchen', reading, 's3cret')
        self.assertEqual((status, record['sensor_id']), (201, 'kitchen'))
        self.assertEqual(call('/api/live/kitchen', {'temperature_c': 'hot'}, 's3cret')[0], 422)
        self.assertEqual(call('/api/live')[1]['sensors'][0]['count'], 1)
        self.assertEqual(len(call('/api/live/kitchen?hours=1')[1]['readings']), 1)
        self.assertEqual(call('/api/live/kitchen/report?hours=1')[1]['readings'], 1)
        self.assertEqual(call('/api/live/kitchen/report?hours=0')[0], 400)


if __name__ == '__main__':
    unittest.main()
