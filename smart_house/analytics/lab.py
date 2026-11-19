"""Local model playground, bounded background evaluations and durable telemetry."""
import datetime as dt
import json
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from uuid import uuid4

from smart_house.analytics.comparison import compare_controllers
from smart_house.analytics.history import encode
from smart_house.analytics.ingestion import ImportService

LOGGER = logging.getLogger(__name__)


def now():
    """Current UTC time as ISO 8601 text."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


class QueueFull(ValueError):
    """The background evaluation queue is full; retry later."""
    pass


class ModelLab:
    """Model laboratory: manual tests with an LRU cache, background comparisons, ratings and statistics."""
    def __init__(self, model, history):
        self.history = history
        self.lock = RLock()
        self.services, self.cache = OrderedDict(), OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='model-lab')
        selected = Path(model).resolve()
        data = Path(__file__).resolve().parents[2] / 'data'
        candidates = [selected, data / 'model.pt', *sorted((data / 'models').glob('*.pt')),
                      *sorted((data / 'user/training').glob('**/best.pt'))]
        self.models = {}
        for path in candidates:
            path = path.resolve()
            if path.is_file() and path not in self.models.values() and len(self.models) < 20:
                self.models[f'model-{len(self.models)}'] = path
        with history.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS lab_runs (
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL,
                    model_id TEXT NOT NULL, latency_ms REAL NOT NULL, cache_hit INTEGER NOT NULL,
                    error TEXT, rating INTEGER, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lab_jobs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    status TEXT NOT NULL, request TEXT NOT NULL, result TEXT, error TEXT
                );
            ''')
            db.execute("UPDATE lab_jobs SET status='interrupted', error='Server restarted; submit again', updated_at=? WHERE status IN ('queued','running')", (now(),))

    def close(self):
        """Wait for running background jobs and stop the worker."""
        self.executor.shutdown(wait=True)

    def catalog(self):
        """Models available to the lab (selected model, data/models, local experiments)."""
        return [{'id': key, 'name': str(path.relative_to(path.parents[1])) if path.parent.name == 'models'
                 else path.name, 'default': key == 'model-0'} for key, path in self.models.items()]

    def _service(self, model_id):
        if not isinstance(model_id, str) or model_id not in self.models:
            raise ValueError('Select a model from /api/lab/models')
        path = self.models[model_id]
        signature = (path.stat().st_mtime_ns, path.stat().st_ctime_ns, path.stat().st_size)
        key = (model_id, signature)
        with self.lock:
            if key not in self.services:
                service = ImportService(path, self.history)
                service._load()
                self.services[key] = service
                while len(self.services) > 3:
                    self.services.popitem(last=False)
            self.services.move_to_end(key)
            return self.services[key]

    def _record(self, kind, model_id, started, payload, hit=False, error=None):
        with self.history.connect() as db:
            ident = db.execute('INSERT INTO lab_runs(created_at,kind,model_id,latency_ms,cache_hit,error,payload) VALUES(?,?,?,?,?,?,?)',
                               (now(), kind, model_id, (time.perf_counter()-started)*1000,
                                int(hit), error, encode(payload))).lastrowid
        return ident

    def predict(self, body):
        """Recommendation of one model for one reading, cached by model, file signature and reading."""
        started = time.perf_counter()
        model_id = body.get('model_id', 'model-0')
        try:
            reading = body.get('reading')
            if not isinstance(reading, dict):
                raise ValueError('reading must be an object of sensor values')
            # No arbitrary files or checkpoints can be supplied by the browser.
            service = self._service(model_id)
            text = encode(reading)
            key = (service._digest, text)
            with self.lock:
                hit = key in self.cache
                if hit:
                    result = self.cache[key]
                    self.cache.move_to_end(key)
                else:
                    result = service.analyze_text('playground.json', text)
                    self.cache[key] = result
                    while len(self.cache) > 128:
                        self.cache.popitem(last=False)
            run_id = self._record('predict', model_id, started, result, hit)
            return {**result, 'run_id': run_id, 'cache_hit': hit,
                    'latency_ms': (time.perf_counter()-started)*1000,
                    'cost': None, 'cost_note': 'Local CPU inference; no paid API. Energy is measured in simulation only.'}
        except Exception as exc:
            self._record('predict', str(model_id), started, {}, error=str(exc))
            raise

    def submit(self, body):
        """Queue a background comparison of up to three models; returns the job id."""
        ids = body.get('model_ids', ['model-0'])
        if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or any(
                not isinstance(i, str) or i not in self.models for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Select 1–3 distinct known models')
        options = {k: body.get(k, default) for k, default in [('episodes', 5), ('steps', 144), ('seed', 42)]}
        for name, low, high in [('episodes', 1, 20), ('steps', 1, 144), ('seed', 0, 2**32-21)]:
            if type(options[name]) is not int or not low <= options[name] <= high:
                raise ValueError(f'{name} must be an integer from {low} to {high}')
        reference = body.get('reference_job')
        if reference is not None:
            if not isinstance(reference, str):
                raise ValueError('reference_job must be a job ID')
            previous = self.job(reference)
            if previous['status'] != 'completed':
                raise ValueError('Reference job must be completed')
            if previous['result']['options'] != options:
                raise ValueError('Regression comparison requires identical episodes, steps and seed')
        request = {'model_ids': ids, **options, 'reference_job': reference}
        with self.lock, self.history.connect() as db:
            pending = db.execute("SELECT count(*) FROM lab_jobs WHERE status IN ('queued','running')").fetchone()[0]
            if pending >= 8:
                raise QueueFull('Evaluation queue is full; retry after a job finishes')
            ident = uuid4().hex
            db.execute('INSERT INTO lab_jobs VALUES(?,?,?,?,?,?,?)',
                       (ident, now(), now(), 'queued', encode(request), None, None))
        self.executor.submit(self._work, ident, request)
        return self.job(ident)

    def _work(self, ident, request):
        started = time.perf_counter()
        try:
            self._update(ident, 'running')
            options = {k: request[k] for k in ('episodes', 'steps', 'seed')}
            results = []
            for model_id in request['model_ids']:
                # The worker owns a separate agent: a long simulation must not lock
                # the playground's cached agent or mutate its evaluation state.
                service = ImportService(self.models[model_id], self.history)
                agent = service._load()
                result = compare_controllers(agent, **options)
                result.update({'model_id': model_id, 'model_name': self.models[model_id].name,
                               'model_sha256': service._digest})
                checks = self._safety_checks(service)
                results.append(result | {'checks': checks})
            regression = []
            if request['reference_job']:
                previous = self.job(request['reference_job'])['result']['models']
                # Each candidate is compared to the first (explicit reference) model of that job.
                baseline = previous[0]['summary']['model']
                for result in results:
                    metrics = result['summary']['model']
                    regression.append({'model_id': result['model_id'], 'reference_sha256': previous[0]['model_sha256'],
                                       'checks': [
                                           {'name': 'unsafe_steps ≤ reference', 'passed': metrics['unsafe_steps'] <= baseline['unsafe_steps']},
                                           {'name': 'comfort ≥ reference − 1 percentage point', 'passed': metrics['avg_comfort'] >= baseline['avg_comfort'] - 1},
                                           {'name': 'energy ≤ reference + 5%', 'passed': metrics['energy_kwh'] <= baseline['energy_kwh'] * 1.05 + 1e-9}]})
            payload = {'options': options, 'models': results, 'regression': regression,
                       'duration_ms': (time.perf_counter()-started)*1000}
            run_id = self._record('evaluation', ','.join(request['model_ids']), started, payload)
            self._update(ident, 'completed', payload | {'run_id': run_id})
        except Exception as exc:
            LOGGER.exception('Background model evaluation failed')
            self._record('evaluation', ','.join(request['model_ids']), started, {}, error=str(exc))
            self._update(ident, 'failed', error='Evaluation failed; check server log and selected checkpoint')

    @staticmethod
    def _safety_checks(service):
        cases = [('cold', {'temperature_c': 0}, 'hvac_mode', lambda v: v == 2),
                 ('hot while sleeping', {'temperature_c': 35, 'occupant_activity': 'sleeping'}, 'hvac_mode', lambda v: v == 4),
                 ('CO2 while sleeping', {'co2_ppm': 2000, 'occupant_activity': 'sleeping'}, 'ventilation', lambda v: v >= 2),
                 ('smoke', {'pm25_ugm3': 80}, 'air_purifier', lambda v: v >= 2),
                 ('dry', {'humidity_pct': 20}, 'humidifier_mode', lambda v: v == 1),
                 ('humid', {'humidity_pct': 80}, 'humidifier_mode', lambda v: v == 2),
                 ('quiet sleep', {'dirt_level': 90, 'occupant_activity': 'sleeping'}, 'cleaner_bot', lambda v: v == 0)]
        checks = []
        for name, override, field, predicate in cases:
            reading = {'timestamp': '2026-01-01T12:00:00+02:00', 'temperature_c': 22, 'humidity_pct': 50, **override}
            result = service.analyze_text('safety.json', encode(reading))
            actual = result['readings'][0]['recommended_devices'][field]
            checks.append({'name': name, 'field': field, 'actual': actual, 'passed': predicate(actual)})
        return checks

    def _update(self, ident, status, result=None, error=None):
        with self.history.connect() as db:
            db.execute('UPDATE lab_jobs SET status=?,updated_at=?,result=?,error=? WHERE id=?',
                       (status, now(), encode(result) if result else None, error, ident))

    def job(self, ident):
        """Status and result of one background job."""
        with self.history.connect() as db:
            row = db.execute('SELECT * FROM lab_jobs WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise KeyError('Job not found')
        result = dict(row)
        for key in ('request', 'result'):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def jobs(self):
        """Recent background jobs."""
        with self.history.connect() as db:
            return [dict(r) for r in db.execute('SELECT id,created_at,status,error FROM lab_jobs ORDER BY created_at DESC LIMIT 50')]

    def rate(self, run_id, rating):
        """Store a 1–5 user rating for a manual test."""
        if type(run_id) is not int or type(rating) is not int or not 1 <= rating <= 5:
            raise ValueError('run_id must be an integer; rating must be from 1 to 5')
        with self.history.connect() as db:
            if not db.execute('UPDATE lab_runs SET rating=? WHERE id=?', (rating, run_id)).rowcount:
                raise KeyError('Run not found')
        return {'saved': True}

    def stats(self):
        """Request counts, errors, cache hits, latencies and ratings of the lab."""
        with self.history.connect() as db:
            totals = dict(db.execute('SELECT count(*) requests, coalesce(sum(error IS NOT NULL),0) errors, '
                                     'coalesce(sum(cache_hit),0) cache_hits, avg(latency_ms) avg_latency_ms, avg(rating) avg_rating FROM lab_runs').fetchone())
            recent = [dict(r) for r in db.execute('SELECT id,created_at,kind,model_id,latency_ms,cache_hit,error,rating FROM lab_runs ORDER BY id DESC LIMIT 100')]
            grouped = [dict(r) for r in db.execute('SELECT model_id,kind,count(*) requests,avg(latency_ms) avg_latency_ms FROM lab_runs GROUP BY model_id,kind')]
        return {'totals': totals, 'by_model': grouped, 'recent': recent}
