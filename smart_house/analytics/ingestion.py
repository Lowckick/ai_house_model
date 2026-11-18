"""Shared import workflow for browser and CLI, without touching training data."""
import hashlib
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from smart_house.analytics.comparison import assumed_weather, load_agent, model_digest, deterministic_devices, compare_controllers
from smart_house.environment.readings import inspect_readings
from smart_house.ml.agent import encode_state
from smart_house.ml.controller import teacher_explanations
from smart_house.environment.indoor_state import DeviceState
from smart_house.ml.rule_layer import RuleLayer


class ValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('File validation failed')


class ImportService:
    def __init__(self, model_path, history, weather=None):
        """``weather``: an ``ArchiveWeather`` for real hourly weather; None keeps the fixed assumption."""
        self.model_path, self.history, self.weather = Path(model_path), history, weather
        self._agent = None
        self._lock = Lock()
        self._digest = None

    def _load(self):
        if self._agent is None:
            self._agent = load_agent(self.model_path)
            self._digest = model_digest(self.model_path)
        return self._agent

    def analyze_text(self, filename, text):
        states, report = inspect_readings(text, Path(filename).suffix)
        if not report['valid']:
            raise ValidationError(report)
        with self._lock:
            agent = self._load()
            results = []
            failed_requests = self.weather.prefetch([s.timestamp for s in states]) if self.weather else 0
            for indoor in states:
                weather, weather_source = self.weather.get(indoor.timestamp) if self.weather else (assumed_weather(indoor), 'fixed')
                action = agent.select_action(encode_state(indoor, weather, DeviceState()))
                raw = agent.decode_action(action, DeviceState(), indoor, weather)
                recommended, changes = RuleLayer().apply_with_explanations(raw, indoor, weather)
                results.append({'timestamp': indoor.timestamp.isoformat(), 'indoor': asdict(indoor),
                                'weather': asdict(weather), 'weather_source': weather_source, 'action_index': action, 'action_mode': agent.action_mode,
                                'model_devices': asdict(raw), 'recommended_devices': asdict(recommended),
                                'deterministic_devices': asdict(deterministic_devices(indoor, weather)),
                                'explanations': [
                                    {'source': 'model', 'reason': 'Обрано дію з найбільшою Q-оцінкою моделі (epsilon=0). ' +
                                     ('Політика коригує детермінований контролер; дія 0 залишає його рішення. ' if agent.residual else '') +
                                     'Причини нейронної оцінки не інтерпретуються; зміни правил наведено окремо.'},
                                    *(teacher_explanations(indoor, weather, agent.action_mode) if agent.residual else []),
                                    *changes]})
            digest = self._digest
        sources = [r['weather_source'] for r in results]
        weather_summary = {'mode': 'open-meteo' if self.weather else 'fixed',
                           'open_meteo_readings': sources.count('open-meteo'), 'fixed_readings': sources.count('fixed')}
        if self.weather:
            weather_summary.update({'latitude': self.weather.lat, 'longitude': self.weather.lon,
                                    'failed_requests': failed_requests, 'errors': self.weather.errors[-5:]})
        return {'count': len(results), 'validation': report, 'readings': results,
                'model_sha256': digest, 'weather': weather_summary}

    def import_text(self, filename, text):
        result = self.analyze_text(filename, text)
        ident = self.history.save_import(Path(filename).name, hashlib.sha256(text.encode('utf-8')).hexdigest(),
                                         result['model_sha256'], result['validation'], result['readings'],
                                         result['weather'])
        return self.history.get_import(ident)

    def evaluate(self, episodes=5, steps=144, seed=42):
        with self._lock:
            payload = compare_controllers(self._load(), episodes, steps, seed)
            payload.update({'model': str(self.model_path), 'model_sha256': self._digest})
        ident = self.history.save_evaluation(payload)
        return {'id': ident, **payload}
