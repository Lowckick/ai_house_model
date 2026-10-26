import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from smart_house.environment.scenarios import make_scenario
from smart_house.environment.indoor_state import DeviceState
from smart_house.ml.agent import RewardFunction
from smart_house.ml.planning import rollout_scores, train_planned


class PlanningTests(unittest.TestCase):
    def test_rollouts_are_reproducible_and_do_not_mutate_source(self):
        scenario = make_scenario('train', 0, 42)
        physics = scenario.environment()
        physics.state.temperature_c = 10
        before = asdict(physics.state)
        random_state = physics._rng.getstate()
        occupants = physics._occupancy_sim._rng.getstate()
        trace = scenario.weather_trace(2)
        scores = rollout_scores(physics, DeviceState(), trace, RewardFunction(), 2)
        self.assertEqual(scores, rollout_scores(physics, DeviceState(), trace, RewardFunction(), 2))
        self.assertGreater(max(scores), scores[0])
        self.assertEqual(asdict(physics.state), before)
        self.assertEqual(physics._rng.getstate(), random_state)
        self.assertEqual(physics._occupancy_sim._rng.getstate(), occupants)

    def test_invalid_training_settings_create_no_run(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'run'
            for settings in [{'epochs': 0}, {'horizon': True}, {'validation_seed': 1, 'test_seed': 1}]:
                with self.assertRaises(ValueError):
                    train_planned(target, **settings)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
