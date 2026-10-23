import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from smart_house.analytics.comparison import compare_controllers, load_agent
from smart_house.environment.scenarios import make_scenario
from smart_house.environment.indoor_state import DeviceState, IndoorState
from smart_house.ml.agent import SmartHomeDQNAgent, RewardFunction, encode_state
from smart_house.ml.controller import RESIDUAL_ACTIONS, deterministic_devices
from smart_house.ml.experiments import run_experiment
from smart_house.ml.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class TrainingTests(unittest.TestCase):
    def test_disjoint_scenarios_clock_and_trace(self):
        for split, years in [('train', {2023, 2024}), ('validation', {2025}), ('test', {2026})]:
            scenarios = [make_scenario(split, i) for i in range(12)]
            self.assertEqual(len({s.start.month for s in scenarios}), 12)
            self.assertTrue(all(s.start.year in years for s in scenarios))
            for scenario in scenarios:
                self.assertEqual(scenario, make_scenario(split, scenario.index))
                self.assertEqual(scenario.weather_trace(3), scenario.weather_trace(3))
        self.assertEqual(len({make_scenario(split, 0).seed for split in ('train','validation','test')}), 3)
        scenario = make_scenario('train', 0)
        physics = scenario.environment()
        weather = scenario.weather_trace(145)
        self.assertTrue(all(t == physics.state.temperature_c for t in physics.state.room_temps.values()))
        for w in weather[:144]:
            self.assertEqual(w.timestamp, physics.state.timestamp)
            physics.step(w, DeviceState())
        self.assertEqual(physics.state.timestamp - scenario.start, dt.timedelta(days=1))
        self.assertEqual(weather[-1].timestamp, physics.state.timestamp)
        self.assertGreater(len({w.temperature_c for w in weather}), 1)

    def test_residual_semantics_and_checkpoint_compatibility(self):
        agent = SmartHomeDQNAgent(action_mode='residual_v1')
        self.assertEqual(agent.n_actions, 19)
        scenario = make_scenario('test', 1)
        state = scenario.environment().state
        weather = scenario.weather_trace(1)[0]
        self.assertEqual(agent.decode_action(0, DeviceState(), state, weather), deterministic_devices(state, weather))
        for index, (field, value) in enumerate(RESIDUAL_ACTIONS[1:], 1):
            self.assertEqual(getattr(agent.decode_action(index, DeviceState(), state, weather), field), value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new.pt'
            agent.save(path, {'seed': 42})
            restored = load_agent(path)
            self.assertEqual(restored.action_mode, 'residual_v1')
            observation = encode_state(state, weather, DeviceState())
            agent.epsilon = 0
            self.assertEqual(agent.select_action(observation), restored.select_action(observation))
            self.assertEqual(restored.checkpoint_metadata, {'seed': 42})
        legacy = load_agent(ROOT / 'data/model.pt')
        self.assertEqual(legacy.action_mode, 'legacy')
        self.assertEqual(legacy.n_actions, 1920)

    def test_reward_prioritizes_comfort_and_safety(self):
        reward = RewardFunction()
        good = IndoorState(temperature_c=22, humidity_pct=50, co2_ppm=500, comfort_score=95)
        poor = IndoorState(temperature_c=10, humidity_pct=20, co2_ppm=2000, comfort_score=25)
        self.assertGreater(reward.compute(good, DeviceState(hvac_mode=2)), reward.compute(poor, DeviceState()))
        safe = IndoorState(temperature_c=22, co2_ppm=1500, comfort_score=90)
        unsafe = IndoorState(temperature_c=22, co2_ppm=1501, comfort_score=90)
        self.assertLess(reward.compute(unsafe, DeviceState()), reward.compute(safe, DeviceState()) - 2.9)
        self.assertLess(reward.compute(good, DeviceState(hvac_mode=2), DeviceState()),
                        reward.compute(good, DeviceState(hvac_mode=2)))

    def test_validation_preserves_training_state(self):
        agent = SmartHomeDQNAgent(action_mode='residual_v1')
        agent.epsilon = 0.13
        random_state = agent._rng.getstate()
        before = {key: value.clone() for key, value in agent.policy_net.state_dict().items()}
        compare_controllers(agent, 1, 4, split='validation')
        self.assertEqual(agent.epsilon, 0.13)
        self.assertTrue(agent.policy_net.training)
        self.assertEqual(random_state, agent._rng.getstate())
        for key, value in agent.policy_net.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))

    def test_teacher_rollouts_learn_and_no_test_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(output_dir=directory, seed=7, warmup_episodes=2, warmup_epochs=2,
                              steps_per_episode=40, validation_episodes=1)
            trainer.warmup()
            self.assertEqual(trainer.history['warmup']['samples'], 80)
            self.assertTrue(all(t.demonstration for t in trainer.agent.replay._buf))
            loss = trainer.agent.learn()
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(trainer.agent.steps_done, 1)
            # Selection must use only validation, even if subsequent training deteriorates.
            def result(comfort):
                return {'summary': {'model': {'unsafe_steps': 0, 'avg_comfort': comfort,
                                              'energy_kwh': 10, 'total_reward': 1}}, 'split': 'validation'}
            with patch('smart_house.analytics.comparison.compare_controllers', side_effect=[result(90), result(50)]) as compare:
                trainer.validate(0)
                trainer.validate(1)
                self.assertEqual(trainer.history['best_episode'], 0)
                self.assertTrue(all(c.kwargs['split'] == 'validation' for c in compare.call_args_list))
            self.assertTrue((Path(directory) / 'best.pt').exists())

    def test_training_reproducible_and_next_weather_aligned(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            snapshots = []
            for index in range(2):
                trainer = Trainer(seed=123, output_dir=Path(directory) / str(index),
                                  warmup_episodes=1, warmup_epochs=1, steps_per_episode=64)
                trainer.warmup()
                summary = trainer.run_episode(1)
                snapshots.append((summary, {k: v.clone() for k, v in trainer.agent.policy_net.state_dict().items()}))
                scenario = trainer._scenario(1)
                expected_weather = scenario.weather_trace(65)[1]
                first_rl = list(trainer.agent.replay._buf)[64]
                self.assertAlmostEqual(float(first_rl.next_state[0]), expected_weather.temperature_c / 40, places=5)
                with sqlite3.connect(trainer._db_path) as db:
                    stamps = [dt.datetime.fromisoformat(row[0]) for row in db.execute('SELECT timestamp FROM episodes ORDER BY id')]
                self.assertEqual(len(stamps), 64)
                self.assertTrue(all(b - a == dt.timedelta(minutes=10) for a, b in zip(stamps, stamps[1:])))
            self.assertEqual(snapshots[0][0]['total_reward'], snapshots[1][0]['total_reward'])
            for key, value in snapshots[0][1].items():
                self.assertTrue(torch.equal(value, snapshots[1][1][key]), key)

    def test_experiment_run_persistence_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'experiment'
            report = run_experiment(output, seeds=[7, 8], episodes=1, steps=8,
                                    warmup_episodes=1, warmup_epochs=1,
                                    validation_interval=1, validation_episodes=1, test_episodes=1)
            self.assertEqual(len(report['runs']), 2)
            self.assertIn(report['selected_seed'], [7, 8])
            self.assertIsNotNone(report['test_across_training_seeds']['avg_comfort']['stddev'])
            self.assertTrue((output / 'best.pt').is_file())
            self.assertTrue((output / 'experiment.json').is_file())
            for run in report['runs']:
                self.assertEqual(run['validation']['split'], 'validation')
                self.assertEqual(run['test']['split'], 'test')
                history = json.loads((output / f"seed-{run['seed']}" / 'training.json').read_text())
                self.assertEqual(history['train'][0]['steps'], 8)
                self.assertTrue((output / f"seed-{run['seed']}" / 'last.pt').is_file())
            with self.assertRaises(ValueError):
                run_experiment(output, seeds=[7], episodes=1)


if __name__ == '__main__':
    unittest.main()
