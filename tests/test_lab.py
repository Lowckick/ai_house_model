"""End-to-end lab tests with real local checkpoints and isolated databases."""
import json
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import torch

from smart_house.analytics.history import History
from smart_house.analytics.ingestion import ValidationError
from smart_house.analytics.lab import ModelLab, QueueFull
from smart_house.web.server import make_server

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'data/model.pt'
READING = {'timestamp': '2026-10-26T12:00:00+02:00', 'temperature_c': 22, 'humidity_pct': 50}
torch.set_num_threads(2)


class LabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.history = History(Path(self.tmp.name) / 'history.db')
        self.lab = ModelLab(MODEL, self.history)
        self.addCleanup(self.lab.close)

    def finished(self, ident):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            job = self.lab.job(ident)
            if job['status'] not in ('queued', 'running'):
                self.assertEqual(job['status'], 'completed', job)
                return job
            time.sleep(.02)
        self.fail('Background evaluation timed out')

    def test_predictions_cache_ratings_and_validation(self):
        body = {'reading': READING}
        first = self.lab.predict(body)
        second = self.lab.predict(body)
        self.assertFalse(first['cache_hit'])
        self.assertTrue(second['cache_hit'])
        self.assertEqual(first['readings'], second['readings'])
        self.assertEqual(self.history.list_imports(), [])
        self.lab.rate(second['run_id'], 4)
        self.assertEqual(self.lab.stats()['totals']['avg_rating'], 4)
        with self.assertRaises(ValueError):
            self.lab.predict({'reading': READING | {'temperature_c': float('nan')}})
        with self.assertRaises(ValueError):
            self.lab.predict({'model_id': '../../etc/passwd', 'reading': READING})
        # NaN fails before the sensor validator because durable JSON forbids NaN.

    def test_invalid_sensor_values_are_reported(self):
        with self.assertRaises(ValidationError):
            self.lab.predict({'reading': READING | {'humidity_pct': 101}})
        self.assertEqual(self.lab.stats()['totals']['errors'], 1)
        for rating in [True, 0, 6, '5']:
            with self.assertRaises(ValueError):
                self.lab.rate(1, rating)
        with self.assertRaises(KeyError):
            self.lab.rate(9999, 5)

    def test_checkpoint_replacement_invalidates_cached_result(self):
        path = Path(self.tmp.name) / 'selected.pt'
        shutil.copyfile(MODEL, path)
        lab = ModelLab(path, self.history)
        self.addCleanup(lab.close)
        first = lab.predict({'reading': READING})
        self.assertTrue(lab.predict({'reading': READING})['cache_hit'])
        shutil.copyfile(ROOT / 'data/models/improved-v2.pt', path)
        replaced = lab.predict({'reading': READING})
        self.assertFalse(replaced['cache_hit'])
        self.assertNotEqual(first['model_sha256'], replaced['model_sha256'])

    def test_background_evaluation_does_not_lock_playground(self):
        from smart_house.analytics.comparison import compare_controllers
        started, release, predicted = threading.Event(), threading.Event(), threading.Event()
        def slow_compare(*args, **kwargs):
            started.set()
            release.wait(5)
            return compare_controllers(*args, **kwargs)
        with patch('smart_house.analytics.lab.compare_controllers', side_effect=slow_compare):
            job = self.lab.submit({'episodes': 1, 'steps': 2})
            self.assertTrue(started.wait(5))
            def predict():
                try:
                    self.lab.predict({'reading': READING})
                    predicted.set()
                finally:
                    release.set()
            worker = threading.Thread(target=predict)
            worker.start()
            try:
                self.assertTrue(predicted.wait(3), 'Playground blocked by evaluation')
            finally:
                release.set()
                worker.join()
            self.finished(job['id'])

    def test_paired_models_safety_regressions_and_persistence(self):
        ids = [m['id'] for m in self.lab.catalog()][:2]
        self.assertEqual(len(ids), 2)
        options = {'model_ids': ids, 'episodes': 1, 'steps': 12, 'seed': 42}
        job = self.finished(self.lab.submit(options)['id'])
        results = job['result']['models']
        self.assertEqual(results[0]['trials'][0]['scenario'], results[1]['trials'][0]['scenario'])
        for result in results:
            self.assertEqual(len(result['checks']), 7)
            self.assertTrue(all(c['passed'] for c in result['checks']))
        repeated = self.finished(self.lab.submit(options | {'model_ids': ids[:1], 'reference_job': job['id']})['id'])
        self.assertTrue(all(c['passed'] for c in repeated['result']['regression'][0]['checks']))
        with self.assertRaises(ValueError):
            self.lab.submit(options | {'reference_job': job['id'], 'seed': 43})
        restored = ModelLab(MODEL, self.history)
        self.addCleanup(restored.close)
        self.assertEqual(restored.job(job['id'])['result'], job['result'])
        self.assertEqual(restored.stats()['totals']['requests'], 2)

    def test_reject_invalid_jobs_and_limit_queue(self):
        for body in [{'model_ids': []}, {'model_ids': ['model-0']*2}, {'model_ids': ['bad']},
                     {'episodes': True}, {'steps': 145}, {'seed': -1}]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.lab.submit(body)
        with self.history.connect() as db:
            for i in range(8):
                db.execute('INSERT INTO lab_jobs VALUES(?,?,?,?,?,?,?)',
                           (str(i), 'now', 'now', 'queued', '{}', None, None))
        with self.assertRaises(QueueFull):
            self.lab.submit({})
        restored = ModelLab(MODEL, self.history)
        self.addCleanup(restored.close)
        self.assertEqual(restored.job('0')['status'], 'interrupted')

    def test_http_lab_routes_and_security(self):
        server = make_server('127.0.0.1', 0, MODEL, Path(self.tmp.name) / 'web.db')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join()
        self.addCleanup(cleanup)
        def request(path, body=None, origin=None):
            headers = {'Content-Type': 'application/json'}
            if origin:
                headers['Origin'] = origin
            req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,
                                         data=None if body is None else json.dumps(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                return exc.code, json.load(exc)
        self.assertGreaterEqual(len(request('/api/lab/models')[1]['models']), 2)
        status, prediction = request('/api/lab/predict', {'reading': READING})
        self.assertEqual(status, 200)
        self.assertEqual(request('/api/lab/rating', {'run_id': prediction['run_id'], 'rating': 5})[0], 200)
        status, job = request('/api/lab/jobs', {'episodes': 1, 'steps': 2})
        self.assertEqual(status, 202)
        self.assertEqual(request('/api/lab/jobs/'+job['id'])[0], 200)
        self.assertEqual(request('/api/lab/jobs/missing')[0], 404)
        self.assertEqual(request('/api/lab/predict', {'reading': READING}, 'https://other.example')[0], 403)
        self.assertEqual(request('/api/lab/predict', {'reading': READING | {'co2_ppm': -1}})[0], 422)
        self.assertEqual(request('/api/lab/stats')[1]['totals']['avg_rating'], 5)
        self.assertTrue(request('/api/lab/jobs')[1]['jobs'])


if __name__ == '__main__':
    unittest.main()
