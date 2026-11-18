"""Rollout policy improvement over a deterministic base controller, distilled into the network.

Training only: a copy of the simulator scores each of the 19 residual corrections by a short
closed-loop rollout (the correction now, the base controller afterwards). The network learns
to reproduce the best correction from observations alone. Later rounds label the states that
the network itself visits (DAgger), so it is trained on its own state distribution.
Inference needs only the network: no simulator, no lookahead.

Rollout noise is resampled rather than copied from the scenario, so labels estimate an expected
return instead of exploiting the realised future. All corrections at a state share the same
noise (common random numbers), which keeps their comparison low-variance.
"""
import copy
import hashlib
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from smart_house.environment.indoor_state import DeviceState
from smart_house.environment.scenarios import make_scenario
from smart_house.ml.agent import RewardFunction, SmartHomeDQNAgent, encode_state
from smart_house.ml.controller import RESIDUAL_ACTIONS, RESIDUAL_MODES, residual_devices
from smart_house.ml.rule_layer import RuleLayer

METHOD = 'rollout_distill_v1'
MARGINS = (0.0, 0.5, 1.0, 2.0)


def _noise_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'big')


def _reseed(physics, seed):
    physics._rng.seed(seed)
    physics._occupancy_sim._rng.seed(seed + 1)


def correction_values(physics, devices, trace, t, mode, horizon=18, gamma=0.99, samples=1,
                      noise_seed=0, reward_fn=None, rules=None, hold=1):
    """Mean discounted return of every residual correction at step ``t``; ``physics`` is not changed.

    The correction is applied for ``hold`` steps (a sustained plan such as pre-heating), then the
    base controller continues until the horizon.
    """
    if mode not in RESIDUAL_MODES:
        raise ValueError('Unknown residual action mode')
    if horizon < 1 or samples < 1 or not 1 <= hold <= horizon:
        raise ValueError('horizon and samples must be positive; hold from 1 to horizon')
    reward_fn = reward_fn or RewardFunction()
    rules = rules or RuleLayer()
    values = np.zeros(len(RESIDUAL_ACTIONS), dtype=np.float64)
    for sample in range(samples):
        seed = _noise_seed(noise_seed, sample)
        for action in range(len(RESIDUAL_ACTIONS)):
            sim, previous, total = copy.deepcopy(physics), devices, 0.0
            _reseed(sim, seed)
            for k in range(min(horizon, len(trace) - t)):
                weather = trace[t + k]
                chosen = rules.apply(residual_devices(action if k < hold else 0, sim.state, weather, mode),
                                     sim.state, weather)
                state = sim.step(weather, chosen)
                total += gamma ** k * reward_fn.compute(state, chosen, previous)
                previous = chosen
            values[action] += total / samples
    return values


def _policy(mode, weights, margin):
    agent = SmartHomeDQNAgent(action_mode=mode, device='cpu')  # reward is irrelevant for acting
    agent.policy_net.load_state_dict(weights)
    agent.policy_net.eval()
    agent.epsilon, agent.correction_margin = 0.0, margin
    return agent


def collect_episode(job):
    """Label every state of one training episode. Behaviour: the rollout expert, or the network."""
    torch.set_num_threads(1)
    index, seed, mode, steps, horizon, samples, weights, margin, reward_version, *rest = job
    hold = rest[0] if rest else 1
    agent = _policy(mode, weights, margin) if weights is not None else None
    scenario = make_scenario('train', index, seed)
    trace = scenario.weather_trace(steps)
    physics, devices, rules, reward_fn = scenario.environment(), DeviceState(), RuleLayer(), RewardFunction(reward_version)
    states, values = [], []
    for t, weather in enumerate(trace):
        state = encode_state(physics.state, weather, devices)
        scores = correction_values(physics, devices, trace, t, mode, horizon, samples=samples,
                                   noise_seed=_noise_seed(scenario.seed, t), reward_fn=reward_fn, rules=rules,
                                   hold=min(hold, horizon))
        states.append(state)
        values.append(scores)
        action = int(scores.argmax()) if agent is None else agent.select_action(state)
        chosen = rules.apply(residual_devices(action, physics.state, weather, mode), physics.state, weather)
        physics.step(weather, chosen)
        devices = chosen
    return np.asarray(states, dtype=np.float32), np.asarray(values, dtype=np.float32)


def soft_targets(values, temperature):
    values = torch.as_tensor(values, dtype=torch.float32)
    if temperature <= 0:
        return nn.functional.one_hot(values.argmax(1), values.shape[1]).float()
    return torch.softmax((values - values.max(1, keepdim=True).values) / temperature, dim=1)


def distill(agent, states, values, epochs=30, temperature=0.05, seed=42, batch_size=128):
    """Train the policy head on rollout labels from scratch; returns loss and agreement."""
    torch.manual_seed(seed)
    for layer in agent.policy_net.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    x = torch.as_tensor(states, dtype=torch.float32)
    y = soft_targets(values, temperature)
    optimizer = torch.optim.AdamW(agent.policy_net.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    agent.policy_net.train()
    loss_value = None
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            loss = -(y[batch] * torch.log_softmax(agent.policy_net(x[batch]), dim=1)).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.item())
    agent.policy_net.eval()
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    with torch.no_grad():
        predicted = agent.policy_net(x).argmax(1)
    best = torch.as_tensor(values).argmax(1)
    chosen_value = torch.as_tensor(values).gather(1, predicted.unsqueeze(1)).squeeze(1)
    return {'loss': loss_value, 'agreement': float((predicted == best).float().mean()),
            'mean_regret': float((torch.as_tensor(values).max(1).values - chosen_value).mean())}


def selection_key(result):
    """Highest validation reward; more unsafe steps than the tuned baseline are penalised first."""
    model, tuned = result['summary']['model'], result['summary']['tuned']
    return (-max(0.0, model['unsafe_steps'] - tuned['unsafe_steps']), model['total_reward'])


def train_rollout(output_dir, *, seed=42, mode='residual_v2', rounds=3, episodes=48, steps=144,
                  horizon=18, samples=1, epochs=30, temperature=0.05, workers=None,
                  validation_episodes=12, reward_version='comfort_safety_v2', hold=1, verbose=True):
    """Expert labels, then DAgger rounds; checkpoint and margin are chosen on validation only."""
    from smart_house.analytics.comparison import compare_controllers
    if mode not in RESIDUAL_MODES:
        raise ValueError('mode must be a residual action mode')
    if rounds < 1 or episodes < 1 or not 1 <= steps <= 144 or horizon < 1 or samples < 1 or epochs < 1:
        raise ValueError('rounds, episodes, horizon, samples and epochs must be positive; steps 1-144')
    if not 1 <= hold <= horizon:
        raise ValueError('hold must be from 1 to horizon')
    if not 1 <= validation_episodes <= 20:
        raise ValueError('validation episodes must be from 1 to 20')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Rollout output directory must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    workers = workers or min(4, os.cpu_count() or 1)
    agent = SmartHomeDQNAgent(action_mode=mode, seed=seed, device='cpu', reward_version=reward_version)
    agent.epsilon = 0.0
    history = {'method': METHOD, 'seed': seed, 'action_mode': mode, 'reward_version': agent.reward_fn.VERSION,
               'rounds': rounds, 'episodes_per_round': episodes, 'steps_per_episode': steps, 'horizon': horizon,
               'samples': samples, 'hold': hold, 'epochs': epochs, 'temperature': temperature,
               'validation_episodes': validation_episodes, 'margins': list(MARGINS), 'history': []}
    all_states, all_values, best_key = [], [], None
    pool = get_context().Pool(workers) if workers > 1 else None
    try:
        for round_index in range(rounds):
            started = time.monotonic()
            weights = None if round_index == 0 else {k: v.clone() for k, v in agent.policy_net.state_dict().items()}
            jobs = [(round_index * episodes + i, seed, mode, steps, horizon, samples, weights, agent.correction_margin,
                     reward_version, hold)
                    for i in range(episodes)]
            for states, values in (pool.map(collect_episode, jobs) if pool else map(collect_episode, jobs)):
                all_states.append(states)
                all_values.append(values)
            states, values = np.concatenate(all_states), np.concatenate(all_values)
            fit = distill(agent, states, values, epochs=epochs, temperature=temperature, seed=seed)
            candidates = []
            for margin in MARGINS:
                agent.correction_margin = margin
                result = compare_controllers(agent, validation_episodes, 144, seed=10000,
                                             split='validation', include_tuned=True)
                candidates.append((selection_key(result), margin, result))
            key, margin, result = max(candidates, key=lambda item: item[0])
            agent.correction_margin = margin
            selected = best_key is None or key > best_key
            if selected:
                best_key = key
                history['best_round'], history['best_margin'], history['best_validation'] = round_index, margin, result
                agent.save(str(output / 'best.pt'), {'method': METHOD, 'seed': seed, 'round': round_index,
                                                     'selection_split': 'validation', 'metrics': result['summary']['model']})
            history['history'].append({
                'round': round_index, 'behaviour': 'rollout expert' if round_index == 0 else 'network (DAgger)',
                'samples': int(len(states)), **fit, 'selected': selected, 'margin': margin,
                'validation_by_margin': {str(m): {'key': list(k), 'model': r['summary']['model']}
                                         for k, m, r in candidates},
                'duration_s': time.monotonic() - started})
            if verbose:
                metrics, tuned = result['summary']['model'], result['summary']['tuned']
                print(f"Seed {seed} round {round_index + 1}/{rounds}: samples={len(states)} "
                      f"agreement={fit['agreement']:.3f} margin={margin} val reward={metrics['total_reward']:.1f} "
                      f"(tuned {tuned['total_reward']:.1f}) comfort={metrics['avg_comfort']:.2f} "
                      f"kWh={metrics['energy_kwh']:.2f}", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    agent.save(str(output / 'last.pt'), {'method': METHOD, 'seed': seed, 'round': rounds - 1})
    np.savez_compressed(output / 'dataset.npz', states=np.concatenate(all_states), values=np.concatenate(all_values))
    (output / 'training.json').write_text(json.dumps(history, indent=2, allow_nan=False), encoding='utf-8')
    return history
