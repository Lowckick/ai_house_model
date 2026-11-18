import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from smart_house.analytics.calibration import DEFAULT_HEAT_LOSS, calibrate, load_params
from smart_house.analytics.comparison import compare_controllers, load_agent
from smart_house.environment.indoor_state import (DeviceState, IndoorPhysicsSimulator, PhysicsParams,
                                                  set_physics)
from smart_house.environment.weather import SyntheticWeatherGenerator

ROOT = Path(__file__).resolve().parents[1]
TRUE = PhysicsParams(heat_loss_per_h=0.2, co2_per_person_ppm_h=250.0, co2_infiltration_per_h=0.5,
                     humidity_exchange_per_h=0.3)


def recorded_week(params, days=7, seed=3):
    """Readings of a simulated house with all devices off, as a sensor would record them."""
    start = dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc)
    sim = IndoorPhysicsSimulator(n_occupants=3, seed=seed, start_time=start, params=params)
    state = sim.reset()
    weather = SyntheticWeatherGenerator(seed=seed)
    rows = []
    for step in range(days * 144):
        time = start + dt.timedelta(minutes=10 * step)
        w = weather.next(hour_of_day=time.hour, day_of_year=time.timetuple().tm_yday)
        rows.append({'time': time, 'temperature_c': state.temperature_c, 'humidity_pct': state.humidity_pct,
                     'co2_ppm': state.co2_ppm, 'total_occupants': float(state.total_occupants),
                     'outdoor_temperature_c': w.temperature_c, 'outdoor_humidity_pct': w.humidity_pct})
        state = sim.step(w, DeviceState())
    return rows


class CalibrationTests(unittest.TestCase):
    def tearDown(self):
        set_physics(None)

    def test_recovers_known_parameters(self):
        report = calibrate(recorded_week(TRUE))
        params = report['params']
        self.assertEqual(params['source'], 'calibrated')
        self.assertAlmostEqual(params['heat_loss_per_h'], 0.2, delta=0.04)
        self.assertAlmostEqual(params['co2_infiltration_per_h'], 0.5, delta=0.1)
        self.assertAlmostEqual(params['co2_per_person_ppm_h'], 250.0, delta=50.0)
        # Occupant moisture correlates with the driving difference: expect estimates 20-30 % low.
        self.assertAlmostEqual(params['humidity_exchange_per_h'], 0.3, delta=0.1)
        self.assertTrue(all(fit['accepted'] for fit in (report['fits']['temperature'], report['fits']['humidity'])))
        json.dumps(report, allow_nan=False)

    def test_insufficient_or_uninformative_data_keeps_defaults(self):
        short = calibrate(recorded_week(TRUE)[:10])
        self.assertEqual(short['params']['source'], 'default (insufficient data)')
        self.assertIn('at least', short['warnings'][0])
        flat = recorded_week(TRUE)
        for row in flat:  # constant indoor/outdoor difference: heat loss is not identifiable
            row['outdoor_temperature_c'] = row['temperature_c'] - 5
            row['total_occupants'] = 2.0
        report = calibrate(flat)
        self.assertIsNone(report['params']['heat_loss_per_h'])
        self.assertTrue(any(w.startswith('heat_loss_per_h') for w in report['warnings']))
        self.assertTrue(any('co2_per_person' in w for w in report['warnings']))
        assumed = recorded_week(TRUE)
        for row in assumed:  # outdoor values from the fixed assumption: CO2 only
            row['outdoor_source'] = 'fixed'
        report = calibrate(assumed)
        self.assertIsNone(report['params']['heat_loss_per_h'])
        self.assertEqual(report['params']['humidity_exchange_per_h'], 0.05)
        self.assertGreater(report['params']['co2_infiltration_per_h'], 0.4)

    def test_params_file_drives_evaluation_and_defaults_stay_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cal.json'
            path.write_text(json.dumps({'params': {**TRUE.__dict__, 'source': 'calibrated'}}))
            loaded = load_params(path)
            self.assertEqual(loaded.heat_loss_per_h, 0.2)
            (Path(tmp) / 'bad.json').write_text(json.dumps({'heat_loss_per_h': 5}))
            with self.assertRaises(ValueError):
                load_params(Path(tmp) / 'bad.json')
            agent = load_agent(ROOT / 'data/models/rollout-v2.pt')
            default = compare_controllers(agent, 1, 48, seed=1, split='validation')
            set_physics(loaded)
            calibrated = compare_controllers(agent, 1, 48, seed=1, split='validation')
            self.assertEqual(calibrated['physics']['heat_loss_per_h'], 0.2)
            self.assertNotEqual(default['summary'], calibrated['summary'])
            set_physics(None)
            self.assertEqual(compare_controllers(agent, 1, 48, seed=1, split='validation')['summary'], default['summary'])
            self.assertIsNone(default['physics']['heat_loss_per_h'])
            self.assertAlmostEqual(DEFAULT_HEAT_LOSS, 0.0933, places=4)


if __name__ == '__main__':
    unittest.main()
