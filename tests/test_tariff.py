import datetime as dt
import unittest

from smart_house.analytics.comparison import assumed_weather, compare_controllers
from smart_house.environment.indoor_state import DeviceState, IndoorState
from smart_house.environment.tariff import MEAN_PRICE_UAH, interval_hour, is_night, price_uah_per_kwh, price_weight
from smart_house.ml.agent import RewardFunction, SmartHomeDQNAgent
from smart_house.ml.rule_layer import RuleLayer


class TariffTests(unittest.TestCase):
    def test_two_zone_prices_and_normalised_weights(self):
        self.assertEqual([h for h in range(24) if is_night(h)], [0, 1, 2, 3, 4, 5, 6, 23])
        self.assertEqual((price_uah_per_kwh(12), price_uah_per_kwh(3)), (4.32, 2.16))
        self.assertAlmostEqual(sum(price_weight(h) for h in range(24)) / 24, 1.0)
        self.assertAlmostEqual(price_weight(12) / price_weight(3), 2.0)
        self.assertAlmostEqual(MEAN_PRICE_UAH, 3.6)
        # The 06:50-07:00 interval ends at 07:00 but is billed at the night price.
        self.assertEqual(interval_hour(dt.datetime(2026, 1, 1, 7, 0), 10), 6)

    def test_tou_reward_differs_only_in_energy_weight(self):
        with self.assertRaises(ValueError):
            RewardFunction('unknown')
        flat, tou = RewardFunction(), RewardFunction('comfort_safety_tou_v3')
        for hour, weight in ((13, 1.2), (2, 0.6)):
            state = IndoorState(timestamp=dt.datetime(2026, 1, 1, hour, 10), power_consumption_kw=2.0)
            difference = tou.compute(state, DeviceState()) - flat.compute(state, DeviceState())
            self.assertAlmostEqual(difference, -flat.ENERGY_WEIGHT * 2.0 * (weight - 1.0))

    def test_reward_version_round_trips_and_overrides_scoring(self):
        import tempfile
        from pathlib import Path
        from smart_house.analytics.comparison import load_agent
        agent = SmartHomeDQNAgent(action_mode='residual_v2', device='cpu', reward_version='comfort_safety_tou_v3')
        with tempfile.TemporaryDirectory() as tmp:
            agent.save(str(Path(tmp) / 'tou.pt'))
            restored = load_agent(Path(tmp) / 'tou.pt')
        self.assertEqual(restored.reward_fn.VERSION, 'comfort_safety_tou_v3')
        own = compare_controllers(restored, 1, 24, seed=3, split='validation')
        shared = compare_controllers(restored, 1, 24, seed=3, split='validation', reward_fn=RewardFunction())
        self.assertEqual((own['reward_version'], shared['reward_version']), ('comfort_safety_tou_v3', 'comfort_safety_v2'))
        self.assertEqual(own['summary']['model']['cost_uah'], shared['summary']['model']['cost_uah'])
        self.assertNotEqual(own['summary']['model']['total_reward'], shared['summary']['model']['total_reward'])
        trial = own['trials'][0]['model']
        self.assertGreater(trial['cost_uah'], 2.16 * trial['energy_kwh'] - 1e-9)
        self.assertLess(trial['cost_uah'], 4.32 * trial['energy_kwh'] + 1e-9)
        self.assertEqual(own['rules_version'], 'rules_v2')

    def test_sleep_mode_no_longer_cools_comfortable_bedrooms(self):
        rules = RuleLayer()
        for temperature, expected in ((21.0, 0), (23.5, 3)):
            state = IndoorState(temperature_c=temperature, occupant_activity='sleeping', total_occupants=1)
            self.assertEqual(rules.apply(DeviceState(), state, assumed_weather(state)).hvac_mode, expected)
        cold = IndoorState(temperature_c=19.0, occupant_activity='sleeping', total_occupants=1)
        self.assertEqual(rules.apply(DeviceState(hvac_mode=1), cold, assumed_weather(cold)).hvac_mode, 1)


if __name__ == '__main__':
    unittest.main()
