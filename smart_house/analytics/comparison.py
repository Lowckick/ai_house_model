"""Paired, repeatable closed-loop comparison of DQN and a simple controller."""
import hashlib
from pathlib import Path
from statistics import mean, stdev

from smart_house.environment.indoor_state import DeviceState
from smart_house.environment.weather import WeatherSnapshot
from smart_house.ml.agent import SmartHomeDQNAgent, encode_state
from smart_house.ml.rule_layer import RuleLayer
from smart_house.ml.controller import deterministic_devices
from smart_house.environment.scenarios import make_scenario


def load_agent(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise ValueError(f'Model not found: {path}')
    agent = SmartHomeDQNAgent(device='cpu')
    agent.load(str(path))
    agent.epsilon = 0.0
    agent.policy_net.eval()
    return agent


def model_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assumed_weather(indoor):
    return WeatherSnapshot(indoor.timestamp, 15.0, 60.0, 0.0, 50.0,
                           0.0, 0.0, 6 <= indoor.timestamp.hour < 20, 'cloudy')


def compare_controllers(agent, episodes=5, steps=144, seed=42, split='test'):
    epsilon, training = agent.epsilon, agent.policy_net.training
    agent.epsilon = 0.0
    agent.policy_net.eval()
    try:
        return _compare(agent, episodes, steps, seed, split)
    finally:
        agent.epsilon = epsilon
        agent.policy_net.train(training)


def _compare(agent, episodes, steps, seed, split):
    if type(episodes) is not int or not 1 <= episodes <= 20:
        raise ValueError('episodes must be an integer from 1 to 20')
    if type(steps) is not int or not 1 <= steps <= 144:
        raise ValueError('steps must be an integer from 1 to 144')
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 21:
        raise ValueError('seed must be an integer from 0 to 4294967275')
    if split not in {'validation', 'test'}:
        raise ValueError('Comparison requires validation or test scenarios')
    trials = []
    for episode in range(episodes):
        scenario = make_scenario(split, episode, seed)
        weather_trace = scenario.weather_trace(steps)
        outcomes = {}
        for name in ('model', 'deterministic'):
            physics = scenario.environment()
            devices = DeviceState()
            comfort, rewards, powers, violations = [], [], [], 0
            for weather in weather_trace:
                state = physics.state
                if name == 'model':
                    action = agent.select_action(encode_state(state, weather, devices))
                    chosen = RuleLayer().apply(agent.decode_action(action, devices, state, weather), state, weather)
                else:
                    chosen = deterministic_devices(state, weather)
                state = physics.step(weather, chosen)
                comfort.append(state.comfort_score)
                powers.append(state.power_consumption_kw)
                rewards.append(agent.reward_fn.compute(state, chosen, devices))
                violations += int(state.temperature_c < 5 or state.temperature_c > 32 or state.co2_ppm > 1500 or state.pm25_ugm3 > 55)
                devices = chosen
            outcomes[name] = {'avg_comfort': mean(comfort),
                              'energy_kwh': sum(powers) * physics.TIMESTEP_MINUTES / 60,
                              'total_reward': sum(rewards), 'unsafe_steps': violations}
        trials.append({'scenario': scenario.describe(), **outcomes})
    metrics = ('avg_comfort', 'energy_kwh', 'total_reward', 'unsafe_steps')
    summary = {name: {metric: mean(t[name][metric] for t in trials) for metric in metrics}
               for name in ('model', 'deterministic')}
    delta = {metric: summary['model'][metric] - summary['deterministic'][metric] for metric in metrics}
    variability = {metric: (stdev(t['model'][metric] - t['deterministic'][metric] for t in trials)
                            if episodes > 1 else None) for metric in metrics}
    return {'protocol_version': 2, 'split': split, 'action_mode': agent.action_mode,
            'reward_version': agent.reward_fn.VERSION, 'episodes': episodes, 'steps_per_episode': steps, 'seed': seed,
            'summary': summary, 'delta_model_minus_deterministic': delta,
            'paired_delta_stddev': variability, 'trials': trials,
            'unsafe_thresholds': {'temperature_c': [5, 32], 'co2_ppm_max': 1500, 'pm25_ugm3_max': 55},
            'baseline': 'HVAC heat below 20 C, cool above 24 C; ventilation above 1000 ppm CO2; '
                        'purifier above 12 ug/m3 PM2.5; humidity target 40-60%; clean above 30% dirt. '
                        'Shared lighting, blinds, security, sleep and final safety rules.',
            'method': 'Paired closed-loop simulation: identical initial states, weather, seeded physics/occupancy noise; '
                      '10-minute steps. Both controllers use the same final safety rules. No training. '
                      'Higher comfort/reward and lower energy/unsafe steps are better. '
                      'Results describe this simulator and these scenarios, not measured real-world savings.'}
