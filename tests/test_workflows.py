import datetime as dt
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import torch

from smart_house.analytics.comparison import compare_controllers, load_agent, deterministic_devices
from smart_house.analytics.history import History
from smart_house.analytics.ingestion import ImportService, ValidationError
from smart_house.environment.indoor_state import IndoorState, DeviceState, IndoorPhysicsSimulator
from smart_house.environment.readings import inspect_readings
from smart_house.ml.agent import action_index_to_devices
from smart_house.ml.rule_layer import RuleLayer
from smart_house.analytics.comparison import assumed_weather
from smart_house.web.server import make_server

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'data/model.pt'
SAMPLE = (ROOT / 'examples/readings.json').read_text()
torch.set_num_threads(2)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'history.db'

    def test_all_validation_errors_and_warnings(self):
        row = {'timestamp': '2026-10-23T08:00:00', 'temperature_c': 22, 'humidity_pct': 50}
        states, report = inspect_readings(json.dumps([row | {'co2_ppm': -1, 'humidity_pct': 101},
                                                     row | {'temperature_c': False}]), '.json')
        self.assertEqual(states, [])
        self.assertEqual(len(report['errors']), 3)
        self.assertEqual({e['record'] for e in report['errors']}, {1, 2})
        _, report = inspect_readings(json.dumps([row, row, row | {'timestamp': '2026-10-22T08:00:00'}]), '.json')
        self.assertTrue(report['valid'])
        warnings = str(report['warnings'])
        for text in ['Duplicate', 'out of order', 'Default values', 'no timezone']:
            self.assertIn(text, warnings)

    def test_limits_and_duplicate_columns(self):
        for text, suffix in [('{"temperature_c":21,"temperature_c":22}', '.json'),
                             ('timestamp,humidity_pct,humidity_pct\na,1,2', '.csv'),
                             ('[]', '.xml')]:
            with self.subTest(text=text):
                self.assertFalse(inspect_readings(text, suffix)[1]['valid'])
        self.assertFalse(inspect_readings(' ' * (5 * 1024 * 1024 + 1), '.json')[1]['valid'])
        self.assertFalse(inspect_readings(json.dumps([{}] * 5001), '.json')[1]['valid'])

    def test_import_persist_explanations_and_atomic_rejection(self):
        history = History(self.db)
        service = ImportService(MODEL, history)
        with self.assertRaises(ValidationError):
            service.import_text('bad.json', '[{"humidity_pct":200}]')
        self.assertEqual(history.list_imports(), [])
        result = service.import_text('<script>.json', SAMPLE)
        self.assertEqual(result['count'], 2)
        self.assertEqual(len(result['model_sha256']), 64)
        self.assertGreaterEqual(result['readings'][0]['recommended_devices']['ventilation'], 2)
        self.assertEqual(result['readings'][0]['indoor']['co2_ppm'], 1800)
        self.assertIn('explanations', result['readings'][0])
        restored = History(self.db).get_import(result['id'])
        self.assertEqual(result, restored)
        self.assertEqual(len(restored['readings']), 2)
        self.assertEqual(restored['readings'][0]['explanations'][0]['source'], 'model')
        # Failure while serializing rows rolls back the parent import as well.
        with self.assertRaises(ValueError):
            history.save_import('bad', 'x', 'y', {}, [{'timestamp': 'x', 'bad': float('nan')}])
        self.assertEqual(len(history.list_imports()), 1)

    def test_safety_wins_over_sleep_and_explanations_match(self):
        state = IndoorState(temperature_c=35, co2_ppm=2000, occupant_activity='sleeping')
        raw = DeviceState()
        final, explanations = RuleLayer().apply_with_explanations(raw, state, assumed_weather(state))
        self.assertEqual(final.ventilation, 2)
        self.assertEqual(final.hvac_mode, 4)
        self.assertEqual(raw.hvac_mode, 0)
        self.assertTrue(any(e['field'] == 'ventilation' and '2000' in e['reason'] for e in explanations))
        self.assertEqual(final, RuleLayer().apply(raw, state, assumed_weather(state)))

    def test_comparison_repeatable_and_identical_policies_equal(self):
        agent = load_agent(MODEL)
        first = compare_controllers(agent, episodes=2, steps=20, seed=123)
        self.assertEqual(first, compare_controllers(agent, episodes=2, steps=20, seed=123))
        with patch.object(agent, 'select_action', return_value=0), patch(
            'smart_house.analytics.comparison.deterministic_devices',
            side_effect=lambda i, w: RuleLayer().apply(action_index_to_devices(0, DeviceState()), i, w)):
            same = compare_controllers(agent, episodes=2, steps=144)
        self.assertEqual(same['summary']['model'], same['summary']['deterministic'])
        for invalid in [0, 21, True]:
            with self.assertRaises(ValueError):
                compare_controllers(agent, episodes=invalid)

    def test_simulation_clock_and_baseline_thresholds(self):
        start = dt.datetime(2026, 1, 1)
        physics = IndoorPhysicsSimulator(seed=42, start_time=start)
        state = physics.reset()
        physics.step(assumed_weather(state), DeviceState())
        self.assertEqual(state.timestamp, start + dt.timedelta(minutes=10))
        for temp, expected in [(18, 1), (22, 0), (26, 3)]:
            state = IndoorState(temperature_c=temp, timestamp=start.replace(hour=12))
            self.assertEqual(deterministic_devices(state, assumed_weather(state)).hvac_mode, expected)

    def test_http_upload_validation_history_and_evaluation(self):
        server = make_server('127.0.0.1', 0, MODEL, self.db)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            worker.join()
        self.addCleanup(cleanup)
        base = f'http://127.0.0.1:{server.server_port}'
        def request(path, body=None, origin=None):
            headers = {'Content-Type': 'application/json'}
            if origin:
                headers['Origin'] = origin
            req = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                return exc.code, json.load(exc)
        with urllib.request.urlopen(base) as response:
            self.assertIn('Перевірити файл', response.read().decode())
        status, validation = request('/api/validate', {'filename': 'a.json', 'text': SAMPLE})
        self.assertEqual(status, 200)
        self.assertTrue(validation['valid'])
        status, _ = request('/api/import', {'filename': 'bad.json', 'text': '[{}]'})
        self.assertEqual(status, 422)
        self.assertEqual(request('/api/history')[1]['imports'], [])
        status, imported = request('/api/import', {'filename': 'a.json', 'text': SAMPLE})
        self.assertEqual(status, 201)
        self.assertEqual(request(f"/api/imports/{imported['id']}")[1], imported)
        status, evaluated = request('/api/evaluate', {'episodes': 1, 'steps': 12, 'seed': 4})
        self.assertEqual(status, 201)
        self.assertEqual(request(f"/api/evaluations/{evaluated['id']}")[1]['summary'], evaluated['summary'])
        history = request('/api/history')[1]
        self.assertEqual(len(history['imports']), 1)
        self.assertEqual(len(history['evaluations']), 1)
        self.assertEqual(request('/api/imports/9999')[0], 404)
        self.assertEqual(request('/api/evaluate', {'episodes': 100})[0], 400)
        self.assertEqual(request('/api/import', {'filename': 'x.json', 'text': SAMPLE}, 'https://other.example')[0], 403)


if __name__ == '__main__':
    unittest.main()
