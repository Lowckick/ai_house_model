import datetime as dt
import json
import random
import tempfile
import unittest

from smart_house.environment.houses import HOUSES, house
from smart_house.environment.indoor_state import (
    DeviceState,
    IndoorPhysicsSimulator,
    IndoorState,
    PhysicsParams,
    ScheduleOccupancy,
    set_physics,
)
from smart_house.environment.scenarios import make_scenario
from smart_house.ml.adaptation import RuleParams, evaluate_policies, rule_grid, rule_policy, run_adaptation
from smart_house.ml.controller import _tuned_devices


class AdaptationTests(unittest.TestCase):
    def tearDown(self):
        set_physics(None)

    def test_default_rule_params_are_the_tuned_controller(self):
        rng = random.Random(1)
        for _ in range(300):
            state = IndoorState(temperature_c=rng.uniform(15, 28), co2_ppm=rng.uniform(400, 2000),
                                pm25_ugm3=rng.uniform(0, 70), humidity_pct=rng.uniform(20, 80),
                                dirt_level=rng.uniform(0, 60), occupant_activity=rng.choice(['idle', 'sleeping']))
            self.assertEqual(RuleParams().devices(state), _tuned_devices(state))
        grid = list(rule_grid())
        self.assertEqual(len(grid), 540)
        self.assertIn(RuleParams(), grid)

    def test_house_profiles(self):
        self.assertEqual(set(HOUSES), {'default', 'realistic_co2', 'poor_insulation', 'lagged_schedule', 'combined'})
        for params in HOUSES.values():
            params.validate()
        with self.assertRaises(ValueError):
            house('castle')
        with self.assertRaises(ValueError):
            PhysicsParams(occupancy='random-walk').validate()

    def test_schedule_occupancy_is_persistent_and_hides_the_plan(self):
        sim = ScheduleOccupancy(3)
        sim._rng.seed(5)
        days = []
        for _day in range(2):
            totals = [sim.update(h, m)[1] for h in range(24) for m in (0, 30)]
            days.append(totals)
            # Presence changes only at leave/return times: few switches per day, home at night.
            self.assertLessEqual(sum(a != b for a, b in zip(totals, totals[1:])), 6)
            self.assertEqual(totals[2 * 3], 3)   # 03:00
        sim.update(12, 0)
        before = list(sim.plan)
        away = [p for p in before if p is not None and p[0] <= 12 < p[1]]
        sim.reseed_future(random.Random(9), 12.0)
        self.assertEqual(sum(p is None for p in sim.plan), sum(p is None for p in before))
        if away:
            self.assertNotEqual(sim.plan, before)

    def test_lagged_heating_responds_gradually(self):
        start = dt.datetime(2026, 1, 1, 12, tzinfo=dt.timezone.utc)
        weather = make_scenario('validation', 0, 1).weather_trace(1)[0]
        weather.temperature_c = 20.0
        gains = {}
        for lag in (None, 1.5):
            sim = IndoorPhysicsSimulator(n_occupants=0, seed=1, start_time=start, params=PhysicsParams(hvac_lag_h=lag))
            sim.reset().temperature_c = 20.0
            first = sim.step(weather, DeviceState(hvac_mode=2)).temperature_c
            gains[lag] = first - 20.0
        self.assertLess(gains[1.5], gains[None] / 4)

    def test_small_adaptation_run_reports_all_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_adaptation(tmp, ['lagged_schedule', 'default'], seeds=[1], rounds=1, episodes=1, workers=1,
                                    validation_episodes=1, test_episodes=2, test_seeds=[31000], steps=8, verbose=False)
            self.assertEqual(list(report['houses']), ['default', 'lagged_schedule'])
            test = report['houses']['lagged_schedule']['tests']['31000']
            self.assertEqual(set(test['summary']), {'rules_default', 'rules_retuned', 'nn_default', 'nn_retrained'})
            self.assertEqual(report['houses']['lagged_schedule']['physics']['occupancy'], 'schedule')
            self.assertIn('nn_retrained - rules_retuned', test['paired'])
            default = report['houses']['default']['tests']['31000']['summary']
            self.assertEqual(default['nn_default'], default['nn_retrained'])
            # Resuming skips finished houses.
            again = run_adaptation(tmp, ['default'], seeds=[1], rounds=1, episodes=1, workers=1, validation_episodes=1,
                                   test_episodes=2, test_seeds=[31000], steps=8, verbose=False)
            self.assertEqual(again['houses']['default'], json.loads(json.dumps(report['houses']['default'])))
            json.dumps(again, allow_nan=False)
        self.assertIsNone(IndoorPhysicsSimulator.PARAMS.hvac_lag_h)

    def test_paired_policies_share_scenarios(self):
        result = evaluate_policies({'a': rule_policy(RuleParams()), 'b': rule_policy(RuleParams())}, 2, 3, 'validation',
                                   steps=12)
        self.assertEqual(result['summary']['a'], result['summary']['b'])


if __name__ == '__main__':
    unittest.main()
