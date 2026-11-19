"""Published results must stay reproducible: default physics, rules and models give the same trials."""
import json
import unittest
from pathlib import Path

from smart_house.analytics.comparison import compare_controllers, load_agent

ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def test_published_tariff_trials_are_reproduced_exactly(self):
        report = json.loads((ROOT / 'data/models/rollout-v3-tou-evaluation.json').read_text())
        run = next(r for r in report['runs'] if r['seed'] == report['selected_seed'])
        published = run['test']['31000']['trials'][:2]
        agent = load_agent(ROOT / 'data/models/rollout-v3-tou.pt')
        fresh = compare_controllers(agent, 2, 144, seed=31000, split='test', include_tuned=True)['trials']
        for new, old in zip(fresh, published):
            self.assertEqual(new['scenario'], old['scenario'])
            for policy in ('model', 'tuned', 'deterministic'):
                self.assertEqual(new[policy], old[policy], policy)


if __name__ == '__main__':
    unittest.main()
