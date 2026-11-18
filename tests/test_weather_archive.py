import datetime as dt
import tempfile
import unittest
from pathlib import Path

from smart_house.analytics.history import History
from smart_house.analytics.ingestion import ImportService
from smart_house.environment.weather_archive import ARCHIVE_URL, FORECAST_URL, ArchiveWeather

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'data/models/rollout-v2.pt'


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f'HTTP {self.status}')

    def json(self):
        return self.payload


class FakeSession:
    """Serves -5 °C for every requested hour; records calls; can fail."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def get(self, url, params, timeout):
        self.calls.append((url, params['start_date'], params['end_date']))
        if self.fail:
            raise ConnectionError('blocked')
        start, end = dt.date.fromisoformat(params['start_date']), dt.date.fromisoformat(params['end_date'])
        times = [f'{start + dt.timedelta(days=d)}T{h:02d}:00' for d in range((end - start).days + 1) for h in range(24)]
        n = len(times)
        return FakeResponse({'hourly': {'time': times, 'temperature_2m': [-5.0] * n,
                                        'relative_humidity_2m': [85.0] * n, 'wind_speed_10m': [12.0] * n,
                                        'cloud_cover': [90.0] * n, 'precipitation': [0.5] * n,
                                        'is_day': [int(7 <= int(t[11:13]) < 16) for t in times]}})


class ArchiveWeatherTests(unittest.TestCase):
    TODAY = dt.date(2026, 11, 18)

    def test_lookup_batches_requests_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'weather.json'
            session = FakeSession()
            weather = ArchiveWeather(50.45, 30.52, cache_path=cache, session=session, today=self.TODAY)
            kyiv = dt.timezone(dt.timedelta(hours=2))
            stamps = [dt.datetime(2026, 1, 10, 8, 30, tzinfo=kyiv), dt.datetime(2026, 1, 11, 23, 0),
                      dt.datetime(2026, 11, 17, 12, 0, tzinfo=dt.timezone.utc)]
            self.assertEqual(weather.prefetch(stamps), 0)
            # Old days from the archive in one request, recent days from the forecast endpoint.
            self.assertEqual(session.calls, [(ARCHIVE_URL, '2026-01-10', '2026-01-11'),
                                             (FORECAST_URL, '2026-11-17', '2026-11-17')])
            snapshot, source = weather.get(stamps[0])  # 08:30 +02:00 is 06 UTC: before daylight
            self.assertEqual((source, snapshot.temperature_c, snapshot.condition, snapshot.is_daytime),
                             ('open-meteo', -5.0, 'snow', False))
            self.assertEqual(snapshot.timestamp, stamps[0])
            self.assertTrue(weather.get(stamps[2])[0].is_daytime)
            # A second instance reads the cache and makes no request.
            offline = ArchiveWeather(50.45, 30.52, cache_path=cache, session=FakeSession(fail=True), today=self.TODAY)
            self.assertEqual(offline.prefetch(stamps), 0)
            self.assertEqual(offline.get(stamps[1])[1], 'open-meteo')

    def test_failures_fall_back_to_fixed_weather(self):
        weather = ArchiveWeather(session=FakeSession(fail=True), today=self.TODAY)
        stamp = dt.datetime(2026, 1, 10, 12, 0)
        self.assertEqual(weather.prefetch([stamp]), 1)
        snapshot, source = weather.get(stamp)
        self.assertEqual((source, snapshot.temperature_c, snapshot.humidity_pct), ('fixed', 15.0, 60.0))
        self.assertIn('ConnectionError', weather.errors[0])
        with self.assertRaises(ValueError):
            ArchiveWeather(lat=91, session=FakeSession())

    def test_import_uses_real_weather_and_reports_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = History(Path(tmp) / 'h.db')
            text = ('timestamp,temperature_c,humidity_pct\n'
                    '2026-01-10T08:00:00+02:00,19,40\n2026-01-10T09:00:00+02:00,19.5,41\n')
            weather = ArchiveWeather(session=FakeSession(), today=self.TODAY)
            payload = ImportService(MODEL, history, weather).import_text('w.csv', text)
            self.assertEqual(payload['weather']['open_meteo_readings'], 2)
            self.assertEqual({r['weather_source'] for r in payload['readings']}, {'open-meteo'})
            self.assertEqual(payload['readings'][0]['weather']['temperature_c'], -5.0)
            stored = history.get_import(payload['id'])
            self.assertEqual(stored['readings'][0]['weather_source'], 'open-meteo')
            fixed = ImportService(MODEL, history).analyze_text('w.csv', text)
            self.assertEqual((fixed['weather']['mode'], fixed['readings'][0]['weather']['temperature_c']), ('fixed', 15.0))


if __name__ == '__main__':
    unittest.main()
