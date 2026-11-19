import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from smart_house.analytics.comparison import assumed_weather, compare_controllers, load_agent
from smart_house.analytics.ingestion import ImportService
from smart_house.environment.indoor_state import DeviceState, IndoorState
from smart_house.environment.scenarios import make_scenario
from smart_house.ml.agent import SmartHomeDQNAgent
from smart_house.ml.controller import (
    RESIDUAL_ACTIONS,
    residual_devices,
    teacher_explanations,
    tuned_deterministic_devices,
)
from smart_house.ml.experiments import run_rollout_experiment
from smart_house.ml.rollout import collect_episode, correction_values, distill, soft_targets, train_rollout

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class RolloutTests(unittest.TestCase):
    def test_tuned_base_and_residual_v2_semantics(self):
        cold = IndoorState(temperature_c=18, co2_ppm=1400, pm25_ugm3=40, dirt_level=20)
        weather = assumed_weather(cold)
        tuned = tuned_deterministic_devices(cold, weather)
        self.assertEqual((tuned.hvac_mode, tuned.ventilation, tuned.air_purifier, tuned.cleaner_bot), (2, 3, 3, 1))
        self.assertEqual(residual_devices(0, cold, weather, 'residual_v2'), tuned)
        self.assertEqual(residual_devices(0, cold, weather, 'residual_v1').hvac_mode, 1)
        with self.assertRaises(ValueError):
            residual_devices(0, cold, weather, 'residual_v9')
        # Corrections cannot weaken the base where readings are already outside comfort.
        vent0, hvac0 = RESIDUAL_ACTIONS.index(('ventilation', 0)), RESIDUAL_ACTIONS.index(('hvac_mode', 0))
        self.assertEqual(residual_devices(vent0, cold, weather, 'residual_v2'), tuned)
        self.assertEqual(residual_devices(hvac0, cold, weather, 'residual_v2'), tuned)
        self.assertEqual(residual_devices(vent0, cold, weather, 'residual_v1').ventilation, 0)
        mild = IndoorState(temperature_c=21, co2_ppm=900)
        self.assertEqual(residual_devices(vent0, mild, assumed_weather(mild), 'residual_v2').ventilation, 0)
        explanations = teacher_explanations(cold, weather, 'residual_v2')
        self.assertIn('19.5', next(e['reason'] for e in explanations if e['field'] == 'hvac_mode'))

    def test_margin_keeps_base_and_checkpoint_round_trip(self):
        agent = SmartHomeDQNAgent(action_mode='residual_v2', device='cpu')
        agent.epsilon = 0.0
        with torch.no_grad():
            layer = agent.policy_net.advantage_stream[-1]
            layer.weight.zero_()
            layer.bias.zero_()
            layer.bias[5] = 1.0  # correction 5 is preferred by a logit margin of exactly 1
        state = np.zeros(len(agent.policy_net.shared[0].weight[0]), dtype=np.float32)
        self.assertEqual(agent.select_action(state), 5)
        agent.correction_margin = 2.0
        self.assertEqual(agent.select_action(state), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'v2.pt'
            agent.save(str(path))
            restored = load_agent(path)
            self.assertEqual((restored.action_mode, restored.correction_margin), ('residual_v2', 2.0))
            self.assertEqual(restored.select_action(state), 0)

    def test_correction_values_are_pure_repeatable_and_paired(self):
        scenario = make_scenario('train', 0, 42)
        trace = scenario.weather_trace(20)
        physics = scenario.environment()
        before = copy.deepcopy(asdict(physics.state))
        values = correction_values(physics, DeviceState(), trace, 0, 'residual_v2', horizon=6, noise_seed=7)
        self.assertEqual(asdict(physics.state), before)
        self.assertEqual(values.shape, (len(RESIDUAL_ACTIONS),))
        np.testing.assert_array_equal(values, correction_values(physics, DeviceState(), trace, 0, 'residual_v2',
                                                                horizon=6, noise_seed=7))
        # A correction that equals the base decision is the same trajectory, hence the same value.
        base = residual_devices(0, physics.state, trace[0], 'residual_v2')
        same = RESIDUAL_ACTIONS.index(('hvac_mode', base.hvac_mode))
        self.assertEqual(values[0], values[same])
        self.assertGreater(np.ptp(values), 0)
        held = correction_values(physics, DeviceState(), trace, 0, 'residual_v2', horizon=6, noise_seed=7, hold=3)
        self.assertEqual(held[0], values[0])  # holding "no correction" is the base trajectory
        self.assertFalse(np.array_equal(held, values))
        with self.assertRaises(ValueError):
            correction_values(physics, DeviceState(), trace, 0, 'residual_v2', horizon=6, hold=7)

    def test_soft_targets_and_distillation_fit_labels(self):
        values = np.array([[0.0, 1.0, 0.5], [2.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(soft_targets(values, 0).numpy(), [[0, 1, 0], [1, 0, 0]])
        soft = soft_targets(values, 1.0).numpy()
        np.testing.assert_allclose(soft.sum(1), 1, rtol=1e-6)
        self.assertEqual(int(soft[0].argmax()), 1)
        states, labels = collect_episode((0, 42, 'residual_v2', 12, 4, 1, None, 0.0, 'comfort_safety_v2'))
        self.assertEqual(states.shape[0], labels.shape[0])
        agent = SmartHomeDQNAgent(action_mode='residual_v2', device='cpu')
        fit = distill(agent, states, labels, epochs=60, temperature=0)
        self.assertGreater(fit['agreement'], 0.9)
        self.assertGreaterEqual(fit['mean_regret'], 0)

    def test_rollout_training_and_experiment_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = train_rollout(Path(tmp) / 'single', rounds=2, episodes=1, steps=8, horizon=3,
                                    epochs=2, workers=1, validation_episodes=1, verbose=False)
            self.assertEqual([h['behaviour'] for h in history['history']], ['rollout expert', 'network (DAgger)'])
            self.assertEqual(history['history'][1]['samples'], 16)
            self.assertIn('tuned', history['best_validation']['summary'])
            with self.assertRaises(ValueError):
                train_rollout(Path(tmp) / 'single', rounds=1, episodes=1, workers=1)
            report = run_rollout_experiment(Path(tmp) / 'experiment', seeds=[1, 2], rounds=1, episodes=1, steps=6,
                                            horizon=2, epochs=1, workers=1, validation_episodes=1, test_episodes=1,
                                            test_seeds=[31000], reference_models=[ROOT / 'data/models/improved-v2.pt'])
            selected = json.loads((Path(tmp) / 'experiment/selection.json').read_text())
            self.assertNotIn('references', selected)
            self.assertEqual(load_agent(report['selected_checkpoint']).action_mode, 'residual_v2')
            self.assertEqual(set(report['test_across_training_seeds']), {'31000'})
            reference = report['references'][0]['test']['31000']
            self.assertEqual(reference['summary']['model'], reference['summary']['deterministic'])
            # Residual v2 with no correction is exactly the tuned baseline.
            agent = load_agent(report['selected_checkpoint'])
            agent.correction_margin = float('inf')
            result = compare_controllers(agent, 2, 24, seed=5, split='validation', include_tuned=True)
            self.assertEqual(result['summary']['model'], result['summary']['tuned'])
            self.assertTrue(all(v == 0 for v in result['delta_model_minus_tuned'].values()))
            # Imports explain the tuned base controller for residual_v2 checkpoints.
            service = ImportService(report["selected_checkpoint"], None)
            text = 'timestamp,temperature_c,humidity_pct\n2026-01-01T08:00:00+02:00,18,50\n'
            reading = service.analyze_text('r.csv', text)['readings'][0]
            self.assertEqual(reading['action_mode'], 'residual_v2')
            self.assertTrue(any('19.5' in e['reason'] for e in reading['explanations'] if e.get('source') == 'teacher'))


if __name__ == '__main__':
    unittest.main()
