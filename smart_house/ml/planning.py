"""Distil counterfactual simulator rollouts instead of cloning action zero.

The teacher is used only to generate TRAIN labels. At inference the student
uses exactly the existing 54 observations, 19 actions and final safety rules.
This is model-based policy distillation, not evidence of successful online RL.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from smart_house.analytics.comparison import compare_controllers, model_digest
from smart_house.environment.indoor_state import DeviceState
from smart_house.environment.scenarios import make_scenario
from smart_house.ml.agent import SmartHomeDQNAgent, encode_state
from smart_house.ml.controller import residual_devices
from smart_house.ml.rule_layer import RuleLayer


def rollout_scores(physics, previous, weather, reward_fn, horizon=6):
    """Common random numbers across candidates; never mutate source simulator.

    Each candidate repeatedly applies its correction for the short horizon.
    The teacher has simulator/forecast access; the deployed student does not.
    """
    if type(horizon) is not int or horizon < 1 or not weather:
        raise ValueError('Positive horizon and nonempty weather required')
    scores = []
    rules = RuleLayer()
    for action in range(19):
        trial, devices = deepcopy(physics), deepcopy(previous)
        score = 0.0
        for index, snapshot in enumerate(weather[:horizon]):
            chosen = rules.apply(residual_devices(action, trial.state, snapshot), trial.state, snapshot)
            state = trial.step(snapshot, chosen)
            unsafe = state.temperature_c < 5 or state.temperature_c > 32 or state.co2_ppm > 1500 or state.pm25_ugm3 > 55
            score += .99**index * (reward_fn.compute(state, chosen, devices) - 10 * int(unsafe))
            devices = chosen
        scores.append(score)
    return scores


def train_planned(output_dir, *, seed=42, episodes=24, horizon=6, epochs=30,
                  validation_seed=41000, validation_episodes=6, test_seed=61000,
                  test_episodes=20):
    integers = [(seed, 0, 2**32-21), (episodes, 1, 200), (horizon, 1, 24),
                (epochs, 1, 200), (validation_episodes, 1, 20), (test_episodes, 1, 20),
                (validation_seed, 0, 2**32-21), (test_seed, 0, 2**32-21)]
    if any(type(v) is not int or not low <= v <= high for v, low, high in integers):
        raise ValueError('Invalid experiment parameter')
    if validation_seed == test_seed:
        raise ValueError('Use distinct validation and test seeds')
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Output directory must be new or empty')
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = SmartHomeDQNAgent(action_mode='residual_v1', seed=seed, device='cpu')
    agent.epsilon = 0
    states, labels, advantages = [], [], []
    for episode in range(episodes):
        scenario = make_scenario('train', episode, seed)
        physics, devices = scenario.environment(), DeviceState()
        trace = scenario.weather_trace(144 + horizon)
        for step in range(144):
            # Cover both baseline and teacher-induced state distributions.
            if step % 2 == 0:
                scores = rollout_scores(physics, devices, trace[step:step+horizon], agent.reward_fn, horizon)
                best = max(range(19), key=lambda a: (scores[a], -a))
                states.append(encode_state(physics.state, trace[step], devices))
                labels.append(best)
                advantages.append(max(scores)-scores[0])
            action = best if episode % 2 else 0
            chosen = RuleLayer().apply(residual_devices(action, physics.state, trace[step]), physics.state, trace[step])
            physics.step(trace[step], chosen)
            devices = chosen
        print(f'Teacher episode {episode+1}/{episodes}; labels {dict(Counter(labels))}', flush=True)
    if not any(labels):
        raise RuntimeError('Teacher found no corrections; do not publish a baseline clone')
    np.savez_compressed(root/'training-labels.npz', states=np.asarray(states), labels=np.asarray(labels), advantages=np.asarray(advantages))
    history = {'method': 'counterfactual_policy_distillation_v1', 'seed': seed, 'episodes': episodes,
               'horizon': horizon, 'epochs': epochs, 'validation_seed': validation_seed,
               'validation_episodes': validation_episodes, 'test_seed': test_seed,
               'test_episodes': test_episodes, 'labels': dict(Counter(labels)),
               'note': 'Teacher uses simulator state and synthetic future weather on TRAIN only. Student observes only current encoded state. No claim of online RL improvement.',
               'validation': []}
    best_key = None
    for epoch in range(1, epochs+1):
        agent.imitate(states, labels, epochs=1)
        if epoch % 5 == 0 or epoch == epochs:
            result = compare_controllers(agent, validation_episodes, 144, validation_seed, split='validation')
            m = result['summary']['model']
            key = (-m['unsafe_steps'], m['avg_comfort'], -m['energy_kwh'], m['total_reward'])
            if best_key is None or key > best_key:
                best_key = key
                agent.save(root/'best.pt', {'method': history['method'], 'seed': seed,
                                           'epoch': epoch, 'selection_split': 'validation', 'metrics': m})
                history['best_epoch'] = epoch
            history['validation'].append({'epoch': epoch, 'result': result})
            print(f'epoch {epoch}: {m}', flush=True)
    agent.save(root/'last.pt', {'method': history['method'], 'seed': seed, 'epoch': epochs})
    history['selected_sha256'] = model_digest(root/'best.pt')
    # Persist selection before opening the fresh test set.
    (root/'selection.json').write_text(json.dumps(history, indent=2))
    from smart_house.analytics.comparison import load_agent
    selected = load_agent(root/'best.pt')
    counts = Counter()
    original = selected.select_action
    def recorded(state):
        action = original(state)
        counts[action] += 1
        return action
    selected.select_action = recorded
    history['test'] = compare_controllers(selected, test_episodes, 144, test_seed)
    history['test_action_histogram'] = dict(counts)
    (root/'experiment.json').write_text(json.dumps(history, indent=2))
    print('TEST', history['test']['summary'], 'ACTIONS',dict(counts),flush=True)
    return history
