import datetime as dt
import tempfile
import unittest
from pathlib import Path

from smart_house.analytics.demo import seed_demo, simulated_day
from smart_house.analytics.history import History
from smart_house.analytics.ingestion import ImportService
from smart_house.analytics.live import LiveService

ROOT = Path(__file__).resolve().parents[1]
NOW = dt.datetime(2026, 1, 15, 18, 0, tzinfo=dt.timezone.utc)


class DemoTests(unittest.TestCase):
    def test_simulated_day_is_valid_and_ends_now(self):
        readings = simulated_day(2, 11, NOW, hours=2)
        self.assertEqual(len(readings), 12)
        self.assertEqual(readings[-1]['timestamp'], NOW.isoformat())
        self.assertEqual(readings, simulated_day(2, 11, NOW, hours=2))

    def test_seed_demo_fills_every_tab_and_replaces_old_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'demo.db'
            model = ROOT / 'data/models/rollout-v2.pt'
            seed_demo(db, model, now=NOW)
            summary = seed_demo(db, model, now=NOW)   # a second run starts from scratch
            self.assertEqual(summary['sensors'], {'bedroom-demo': 144, 'kitchen-demo': 144})
            history = History(db)
            self.assertEqual(len(history.list_imports()), 1)
            self.assertEqual(len(history.list_evaluations()), 1)
            live = LiveService(ImportService(model, history), clock=lambda: NOW)
            self.assertEqual([s['count'] for s in live.sensors()], [144, 144])
            report = live.report('bedroom-demo', hours=24)
            self.assertEqual(report['readings'], 144)
            self.assertTrue(report['alerts'])


if __name__ == '__main__':
    unittest.main()
