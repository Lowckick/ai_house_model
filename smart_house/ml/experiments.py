"""Multiple training seeds; choose by validation, report test outcomes once."""
import json
import shutil
from pathlib import Path
from statistics import mean, stdev

from smart_house.analytics.comparison import compare_controllers, load_agent, model_digest
from smart_house.ml.trainer import Trainer


def selection_key(result):
    metrics = result['summary']['model']
    return (-metrics['unsafe_steps'], metrics['avg_comfort'], -metrics['energy_kwh'], metrics['total_reward'])


def run_experiment(output_dir, seeds=(42, 43, 44), episodes=200, *,
                   steps=144, warmup_episodes=12, warmup_epochs=5,
                   validation_interval=5, validation_episodes=6, test_episodes=12,
                   resume=None, reference_model=None, train_callback=None):
    if not seeds or len(seeds) > 10 or len(set(seeds)) != len(seeds):
        raise ValueError('Provide 1–10 distinct training seeds')
    if any(type(seed) is not int or not 0 <= seed <= 2**32 - 21 for seed in seeds):
        raise ValueError('Training seeds must be integers from 0 to 4294967275')
    if type(episodes) is not int or episodes < 1 or not 1 <= test_episodes <= 20:
        raise ValueError('Episodes must be positive; test episodes from 1 to 20')
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Experiment output directory must be new or empty')
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        trainer = Trainer(seed=seed, output_dir=root / f'seed-{seed}', resume=resume,
                          steps_per_episode=steps, warmup_episodes=warmup_episodes,
                          warmup_epochs=warmup_epochs, validation_interval=validation_interval,
                          validation_episodes=validation_episodes)
        if train_callback:
            train_callback(trainer, episodes)
        else:
            trainer.train(episodes)
        runs.append({'seed': seed, 'best_episode': trainer.history['best_episode'],
                     'checkpoint': str(trainer.output_dir / 'best.pt'),
                     'validation': trainer.history['best_validation']})
    # Fix the winner BEFORE accessing any test result. Test scores never choose a seed/checkpoint.
    winner = max(runs, key=lambda run: selection_key(run['validation']))
    selected_path = root / 'best.pt'
    shutil.copyfile(winner['checkpoint'], selected_path)
    report = {'protocol_version': 2, 'selected_seed': winner['seed'],
              'selected_checkpoint': str(selected_path), 'selected_sha256': model_digest(selected_path),
              'selection': 'validation only: fewest unsafe steps, highest comfort, lowest energy, highest reward',
              'runs': runs, 'note': 'Simulation results; residual policy includes the deterministic teacher and safety rules.'}
    # A completed selection is durable even if an external reference model cannot be evaluated.
    (root / 'selection.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for run in runs:
        run['test'] = compare_controllers(load_agent(run['checkpoint']), test_episodes, 144, seed=20000, split='test')
    metrics = ('avg_comfort', 'energy_kwh', 'total_reward', 'unsafe_steps')
    report['test_across_training_seeds'] = {
        metric: {'mean': mean(run['test']['summary']['model'][metric] for run in runs),
                 'stddev': stdev(run['test']['summary']['model'][metric] for run in runs) if len(runs) > 1 else None}
        for metric in metrics}
    if reference_model:
        report['reference'] = {'model': str(reference_model), 'sha256': model_digest(reference_model),
                               'test': compare_controllers(load_agent(reference_model), test_episodes, 144, seed=20000, split='test')}
    (root / 'experiment.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Selected by validation: {selected_path}; test report: {root / 'experiment.json'}", flush=True)
    return report


def run_rollout_experiment(output_dir, seeds=(42, 43, 44), *, rounds=3, episodes=48, steps=144,
                           horizon=18, samples=1, epochs=30, temperature=0.05, workers=None,
                           validation_episodes=12, test_episodes=20, test_seeds=(20000, 31000),
                           reference_models=(), reward_version='comfort_safety_v2'):
    """Rollout distillation per training seed; winner fixed on validation before any test run."""
    from smart_house.ml.rollout import METHOD, selection_key as rollout_key, train_rollout
    if not seeds or len(seeds) > 10 or len(set(seeds)) != len(seeds):
        raise ValueError('Provide 1–10 distinct training seeds')
    if any(type(seed) is not int or not 0 <= seed <= 2**32 - 21 for seed in seeds):
        raise ValueError('Training seeds must be integers from 0 to 4294967275')
    if not 1 <= test_episodes <= 20 or not test_seeds:
        raise ValueError('Test episodes must be from 1 to 20 and at least one test seed is required')
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Experiment output directory must be new or empty')
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        history = train_rollout(root / f'seed-{seed}', seed=seed, rounds=rounds, episodes=episodes, steps=steps,
                                horizon=horizon, samples=samples, epochs=epochs, temperature=temperature,
                                workers=workers, validation_episodes=validation_episodes,
                                reward_version=reward_version)
        runs.append({'seed': seed, 'best_round': history['best_round'], 'margin': history['best_margin'],
                     'checkpoint': str(root / f'seed-{seed}' / 'best.pt'), 'validation': history['best_validation']})
    winner = max(runs, key=lambda run: rollout_key(run['validation']))
    selected_path = root / 'best.pt'
    shutil.copyfile(winner['checkpoint'], selected_path)
    report = {'protocol_version': 3, 'method': METHOD, 'reward_version': reward_version, 'selected_seed': winner['seed'],
              'selected_checkpoint': str(selected_path), 'selected_sha256': model_digest(selected_path),
              'selection': 'validation only: unsafe steps above the tuned baseline first, then highest reward',
              'test_seeds': list(test_seeds), 'runs': runs,
              'note': 'Simulation results. The network replaces the rollout planner at inference; '
                      'the residual policy includes its deterministic base controller and safety rules.'}
    (root / 'selection.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    metrics = ('avg_comfort', 'energy_kwh', 'cost_uah', 'total_reward', 'unsafe_steps')
    for run in runs:
        agent = load_agent(run['checkpoint'])
        run['test'] = {str(s): compare_controllers(agent, test_episodes, 144, seed=s, split='test', include_tuned=True)
                       for s in test_seeds}
    report['test_across_training_seeds'] = {
        str(s): {metric: {'mean': mean(run['test'][str(s)]['summary']['model'][metric] for run in runs),
                          'stddev': stdev(run['test'][str(s)]['summary']['model'][metric] for run in runs)
                          if len(runs) > 1 else None} for metric in metrics}
        for s in test_seeds}
    from smart_house.ml.agent import RewardFunction
    shared_reward = RewardFunction(reward_version)  # references are scored on the trained model's reward
    report['references'] = [{'model': str(model), 'sha256': model_digest(model),
                             'test': {str(s): compare_controllers(load_agent(model), test_episodes, 144, seed=s,
                                                                  split='test', include_tuned=True,
                                                                  reward_fn=shared_reward)
                                      for s in test_seeds}}
                            for model in reference_models]
    (root / 'experiment.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Selected by validation: {selected_path}; test report: {root / 'experiment.json'}", flush=True)
    return report
