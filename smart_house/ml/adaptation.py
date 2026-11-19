"""Adaptation study: do fixed rules, retuned rules and retrained networks carry over to other houses?

For every house profile (smart_house/environment/houses.py):

* ``rules_default``  — the staged-threshold controller tuned for the default house (residual_v2 base);
* ``rules_retuned``  — the same controller with thresholds grid-searched for this house on validation
                       scenarios: a stand-in for an engineer retuning the rules;
* ``nn_default``     — the rollout network trained on the default house, used without retraining;
* ``nn_retrained``   — the same pipeline retrained automatically on this house.

Selection (thresholds, checkpoints, seeds) uses validation scenarios only; every policy is then
evaluated once on the same paired test scenarios.
"""
import itertools
import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev

from smart_house.environment.indoor_state import DeviceState, set_physics
from smart_house.environment.scenarios import make_scenario
from smart_house.environment.tariff import interval_hour, price_uah_per_kwh
from smart_house.ml.agent import RewardFunction, encode_state
from smart_house.ml.controller import _threshold_devices
from smart_house.ml.rule_layer import RuleLayer

METRICS = ('avg_comfort', 'energy_kwh', 'cost_uah', 'total_reward', 'unsafe_steps')


@dataclass(frozen=True)
class RuleParams:
    """Thresholds of the staged controller; the defaults are exactly ``controller._tuned_devices``."""
    heat_high_below: float = 19.5
    heat_low_below: float = 20.5
    cool_low_above: float = 24.0
    cool_high_above: float = 25.0
    vent_levels: tuple = (800.0, 1000.0, 1300.0)
    purifier_levels: tuple = (12.0, 20.0, 35.0)
    clean_above: float = 15.0

    def devices(self, indoor):
        """Device modes chosen by these thresholds before the shared rules."""
        d = _threshold_devices(indoor)
        t = indoor.temperature_c
        d.hvac_mode = (2 if t < self.heat_high_below else 1 if t < self.heat_low_below else
                       4 if t > self.cool_high_above else 3 if t > self.cool_low_above else 0)
        c, (v1, v2, v3) = indoor.co2_ppm, self.vent_levels
        d.ventilation = 3 if c > v3 else 2 if c > v2 else 1 if c > v1 else 0
        p, (p1, p2, p3) = indoor.pm25_ugm3, self.purifier_levels
        d.air_purifier = 3 if p > p3 else 2 if p > p2 else 1 if p > p1 else 0
        d.cleaner_bot = int(indoor.dirt_level > self.clean_above and indoor.occupant_activity != 'sleeping')
        return d


def rule_grid():
    """Candidate thresholds: heating/cooling shifts, ventilation and purifier scales, cleaning level."""
    base = RuleParams()
    for heat, cool, vent, purifier, clean in itertools.product(
            (-1.0, -0.5, 0.0, 0.5, 1.0), (-1.0, 0.0, 1.0), (0.7, 0.85, 1.0, 1.15), (0.7, 1.0, 1.3), (10.0, 15.0, 25.0)):
        yield replace(base, heat_high_below=base.heat_high_below + heat, heat_low_below=base.heat_low_below + heat,
                      cool_low_above=base.cool_low_above + cool, cool_high_above=base.cool_high_above + cool,
                      vent_levels=tuple(round(v * vent, 1) for v in base.vent_levels),
                      purifier_levels=tuple(round(v * purifier, 1) for v in base.purifier_levels), clean_above=clean)


def rule_policy(params):
    """Closed-loop policy of the staged rules with the given thresholds."""
    rules = RuleLayer()
    return lambda indoor, weather, previous: rules.apply(params.devices(indoor), indoor, weather)


def network_policy(agent):
    """Closed-loop policy of a trained network (residual corrections plus rules)."""
    rules = RuleLayer()

    def act(indoor, weather, previous):
        action = agent.select_action(encode_state(indoor, weather, previous))
        return rules.apply(agent.decode_action(action, previous, indoor, weather), indoor, weather)
    return act


def evaluate_policies(policies, episodes, seed, split, reward_fn=None, steps=144):
    """Paired closed-loop evaluation: identical scenarios, weather and seeded noise for every policy."""
    reward_fn = reward_fn or RewardFunction()
    trials = []
    for episode in range(episodes):
        scenario = make_scenario(split, episode, seed)
        trace = scenario.weather_trace(steps)
        outcome = {}
        for name, policy in policies.items():
            physics, previous = scenario.environment(), DeviceState()
            comfort, rewards, energy, cost, unsafe = [], [], 0.0, 0.0, 0
            for weather in trace:
                chosen = policy(physics.state, weather, previous)
                state = physics.step(weather, chosen)
                kwh = state.power_consumption_kw * physics.TIMESTEP_MINUTES / 60
                energy += kwh
                cost += kwh * price_uah_per_kwh(interval_hour(state.timestamp, physics.TIMESTEP_MINUTES))
                comfort.append(state.comfort_score)
                rewards.append(reward_fn.compute(state, chosen, previous))
                unsafe += int(state.temperature_c < 5 or state.temperature_c > 32 or state.co2_ppm > 1500
                              or state.pm25_ugm3 > 55)
                previous = chosen
            outcome[name] = {'avg_comfort': mean(comfort), 'energy_kwh': energy, 'cost_uah': cost,
                             'total_reward': sum(rewards), 'unsafe_steps': unsafe}
        trials.append({'scenario': scenario.describe(), **outcome})
    summary = {name: {m: mean(t[name][m] for t in trials) for m in METRICS} for name in policies}
    return {'split': split, 'seed': seed, 'episodes': episodes, 'summary': summary, 'trials': trials}


def paired(result, a, b):
    """Mean, standard error and wins of a − b over the paired scenarios of ``result``."""
    trials = result['trials']
    out = {}
    for metric in METRICS:
        diffs = [t[a][metric] - t[b][metric] for t in trials]
        better = (lambda d: d < 0) if metric in ('energy_kwh', 'cost_uah', 'unsafe_steps') else (lambda d: d > 0)
        out[metric] = {'mean': mean(diffs),
                       'se': stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else None,
                       'better_in': sum(better(d) for d in diffs), 'worse_in': sum(better(-d) for d in diffs)}
    return out


def tune_rules(episodes=12, seed=10000, steps=144):
    """Grid search on validation: no more unsafe steps than the default thresholds, then highest reward."""
    default = RuleParams()
    candidates = list(rule_grid())
    policies = {i: rule_policy(p) for i, p in enumerate(candidates)}
    result = evaluate_policies(policies, episodes, seed, 'validation', steps=steps)
    summary = result['summary']
    base_index = candidates.index(default)
    base_unsafe = summary[base_index]['unsafe_steps']
    best = max(summary, key=lambda i: (-max(0.0, summary[i]['unsafe_steps'] - base_unsafe), summary[i]['total_reward']))
    return candidates[best], {'candidates': len(candidates), 'validation_episodes': episodes,
                              'default': summary[base_index], 'best': summary[best], 'params': asdict(candidates[best])}


def run_adaptation(output_dir, houses, seeds=(42, 43), *, rounds=3, episodes=48, workers=None,
                   validation_episodes=12, test_episodes=20, test_seeds=(20000, 31000), steps=144, verbose=True):
    """Runs (or resumes) each house in order; 'default' is trained first and reused as nn_default."""
    from smart_house.analytics.comparison import load_agent
    from smart_house.environment.houses import house
    from smart_house.ml.experiments import run_rollout_experiment
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / 'adaptation.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'protocol': 'validation-only selection of thresholds, checkpoints and seeds; one paired test evaluation',
        'seeds': list(seeds), 'rounds': rounds, 'episodes_per_round': episodes, 'test_seeds': list(test_seeds),
        'test_episodes': test_episodes, 'houses': {}}
    houses = list(houses)
    if 'default' in houses:
        houses.remove('default')
        houses.insert(0, 'default')
    try:
        for name in houses:
            if name in report['houses']:
                continue
            started = time.monotonic()
            params = house(name)
            set_physics(params)
            directory = root / name
            if directory.exists() and not (directory / 'experiment.json').exists():
                shutil.rmtree(directory)   # an interrupted training run; start this house again
            if not (directory / 'experiment.json').exists():
                run_rollout_experiment(directory, seeds=list(seeds), rounds=rounds, episodes=episodes, steps=steps,
                                       workers=workers, validation_episodes=validation_episodes,
                                       test_episodes=test_episodes, test_seeds=tuple(test_seeds))
            retuned, tuning = tune_rules(validation_episodes, steps=steps)
            default_model = root / 'default' / 'best.pt'
            if not default_model.exists():
                raise ValueError("Run the 'default' house first: nn_default comes from it")
            policies = {'rules_default': rule_policy(RuleParams()), 'rules_retuned': rule_policy(retuned),
                        'nn_default': network_policy(load_agent(default_model)),
                        'nn_retrained': network_policy(load_agent(directory / 'best.pt'))}
            tests = {}
            for seed in test_seeds:
                result = evaluate_policies(policies, test_episodes, seed, 'test', steps=steps)
                result['paired'] = {
                    'nn_retrained - rules_retuned': paired(result, 'nn_retrained', 'rules_retuned'),
                    'nn_retrained - rules_default': paired(result, 'nn_retrained', 'rules_default'),
                    'nn_default - rules_default': paired(result, 'nn_default', 'rules_default'),
                    'nn_retrained - nn_default': paired(result, 'nn_retrained', 'nn_default'),
                    'rules_retuned - rules_default': paired(result, 'rules_retuned', 'rules_default')}
                tests[str(seed)] = result
            report['houses'][name] = {'physics': asdict(params), 'rule_tuning': tuning, 'tests': tests,
                                      'duration_s': time.monotonic() - started}
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            if verbose:
                for seed, result in tests.items():
                    s = result['summary']
                    print(f"{name} test {seed}: " + ' | '.join(
                        f"{p} comfort {s[p]['avg_comfort']:.2f} kWh {s[p]['energy_kwh']:.2f} "
                        f"reward {s[p]['total_reward']:.1f} unsafe {s[p]['unsafe_steps']:.2f}" for p in s), flush=True)
    finally:
        set_physics(None)
    return report
