"""
Training loop for the Smart Home DQN agent.
Runs episodes of simulated time and logs progress to SQLite.
"""

import sqlite3
import os
import time
import datetime
import numpy as np

from smart_house.environment.indoor_state import DeviceState
from smart_house.ml.agent import SmartHomeDQNAgent, encode_state
from smart_house.ml.rule_layer import RuleLayer


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "training.db")


def init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode INTEGER,
            step INTEGER,
            reward REAL,
            comfort REAL,
            temperature REAL,
            humidity REAL,
            co2 REAL,
            pm25 REAL,
            dirt REAL,
            power_kw REAL,
            hvac_mode INTEGER,
            cleaner_bot INTEGER,
            air_purifier INTEGER,
            epsilon REAL,
            loss REAL,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episode_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode INTEGER,
            total_reward REAL,
            avg_comfort REAL,
            avg_power REAL,
            steps INTEGER,
            duration_s REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


class Trainer:
    """Seasonal training with teacher warmup and validation-only model selection."""

    STEPS_PER_EPISODE = 144
    MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model.pt")

    def __init__(self, use_api_weather=False, progress_callback=None, *, seed=42,
                 output_dir=None, resume=None, validation_interval=5,
                 validation_episodes=6, warmup_episodes=12, warmup_epochs=5,
                 steps_per_episode=144):
        import random
        import torch
        from pathlib import Path
        seed = int(seed)
        if not 0 <= seed <= 2**32 - 21:
            raise ValueError("seed must be from 0 to 4294967275")
        if not 1 <= steps_per_episode <= 144:
            raise ValueError("steps_per_episode must be from 1 to 144")
        if validation_interval < 1 or not 1 <= validation_episodes <= 20:
            raise ValueError("validation interval must be positive and validation episodes from 1 to 20")
        if warmup_episodes < 0 or warmup_epochs < 0:
            raise ValueError("warmup settings cannot be negative")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed
        self.STEPS_PER_EPISODE = steps_per_episode
        self.validation_interval = validation_interval
        self.validation_episodes = validation_episodes
        self.warmup_episodes, self.warmup_epochs = warmup_episodes, warmup_epochs
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "data/user/training" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / 'training.json').exists() or (self.output_dir / 'best.pt').exists():
            raise ValueError("Training output directory already contains a run; choose a new directory")
        self._db_path = str(self.output_dir / 'training.db')
        init_db(self._db_path)
        self.agent = SmartHomeDQNAgent(action_mode='residual_v1', seed=seed, device='cpu')
        if resume:
            if not Path(resume).is_file():
                raise ValueError(f"Resume checkpoint not found: {resume}")
            self.agent.load(str(resume))
            if self.agent.action_mode != 'residual_v1':
                raise ValueError("Legacy checkpoints cannot be resumed into the residual action space; train fresh")
        self.resume = str(resume) if resume else None
        self.rule_layer = RuleLayer()
        self.progress_callback = progress_callback
        # Training intentionally uses seasonal synthetic traces, even if live API is available.
        self.weather_provider = type('Source', (), {'source': 'Seasonal synthetic training scenarios'})()
        self.physics = None
        self.best_key = None
        self.history = {'seed': seed, 'action_mode': self.agent.action_mode,
                        'reward_version': self.agent.reward_fn.VERSION, 'resume': self.resume,
                        'steps_per_episode': steps_per_episode, 'validation_episodes': validation_episodes,
                        'validation_interval': validation_interval, 'warmup_episodes': warmup_episodes,
                        'warmup_epochs': warmup_epochs, 'train': [], 'validation': []}

    def _scenario(self, index):
        from smart_house.environment.scenarios import make_scenario
        return make_scenario('train', index, self.seed)

    def warmup(self):
        """Demonstrations teach action 0: retain the teacher; RL explores corrections."""
        states, actions = [], []
        for index in range(self.warmup_episodes):
            scenario = self._scenario(index)
            physics = scenario.environment()
            devices = DeviceState()
            weather_trace = scenario.weather_trace(self.STEPS_PER_EPISODE + 1)
            for step in range(self.STEPS_PER_EPISODE):
                indoor, weather = physics.state, weather_trace[step]
                state = encode_state(indoor, weather, devices)
                chosen = self.rule_layer.apply(self.agent.decode_action(0, devices, indoor, weather), indoor, weather)
                indoor = physics.step(weather, chosen)
                reward = self.agent.reward_fn.compute(indoor, chosen, devices)
                next_state = encode_state(indoor, weather_trace[step + 1], chosen)
                self.agent.store(state, 0, reward, next_state, step == self.STEPS_PER_EPISODE - 1, demonstration=True)
                states.append(state)
                actions.append(0)
                devices = chosen
        losses = self.agent.imitate(states, actions, epochs=self.warmup_epochs)
        self.history['warmup'] = {'samples': len(states), 'last_loss': losses[-1] if losses else None,
                                  'teacher_action': '0 = no correction to deterministic controller'}

    def run_episode(self, episode):
        from copy import deepcopy
        scenario = self._scenario(self.warmup_episodes + episode - 1)
        self.physics = scenario.environment()
        indoor = self.physics.state
        weather_trace = scenario.weather_trace(self.STEPS_PER_EPISODE + 1)
        devices = DeviceState()
        total_reward = comfort_sum = power_sum = 0.0
        unsafe = 0
        ep_start = time.monotonic()
        with sqlite3.connect(self._db_path) as conn:
            for step in range(self.STEPS_PER_EPISODE):
                weather = weather_trace[step]
                state = encode_state(indoor, weather, devices)
                action = self.agent.select_action(state)
                previous = deepcopy(devices)
                devices = self.rule_layer.apply(self.agent.decode_action(action, devices, indoor, weather), indoor, weather)
                indoor = self.physics.step(weather, devices)
                reward = self.agent.reward_fn.compute(indoor, devices, previous)
                next_state = encode_state(indoor, weather_trace[step + 1], devices)
                self.agent.store(state, action, reward, next_state, step == self.STEPS_PER_EPISODE - 1)
                loss = self.agent.learn()
                total_reward += reward
                comfort_sum += indoor.comfort_score
                power_sum += indoor.power_consumption_kw
                unsafe += int(indoor.temperature_c < 5 or indoor.temperature_c > 32 or indoor.co2_ppm > 1500 or indoor.pm25_ugm3 > 55)
                conn.execute("""INSERT INTO episodes
                    (episode,step,reward,comfort,temperature,humidity,co2,pm25,dirt,power_kw,
                     hvac_mode,cleaner_bot,air_purifier,epsilon,loss,timestamp)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (episode, step, reward, indoor.comfort_score, indoor.temperature_c,
                     indoor.humidity_pct, indoor.co2_ppm, indoor.pm25_ugm3, indoor.dirt_level,
                     indoor.power_consumption_kw, devices.hvac_mode, devices.cleaner_bot,
                     devices.air_purifier, self.agent.epsilon, loss, indoor.timestamp.isoformat()))
                if self.progress_callback and step % 10 == 0:
                    self.progress_callback(episode, step, indoor, devices, weather, reward)
            duration = time.monotonic() - ep_start
            avg_comfort = comfort_sum / self.STEPS_PER_EPISODE
            avg_power = power_sum / self.STEPS_PER_EPISODE
            conn.execute("""INSERT INTO episode_summary
                (episode,total_reward,avg_comfort,avg_power,steps,duration_s,timestamp) VALUES (?,?,?,?,?,?,?)""",
                (episode, total_reward, avg_comfort, avg_power, self.STEPS_PER_EPISODE,
                 duration, indoor.timestamp.isoformat()))
        summary = {'episode': episode, 'scenario': scenario.describe(), 'total_reward': total_reward,
                   'avg_comfort': avg_comfort, 'avg_power': avg_power, 'unsafe_steps': unsafe,
                   'epsilon': self.agent.epsilon, 'steps': self.STEPS_PER_EPISODE, 'duration_s': duration}
        self.history['train'].append(summary)
        return summary

    def validate(self, episode):
        from smart_house.analytics.comparison import compare_controllers
        result = compare_controllers(self.agent, episodes=self.validation_episodes,
                                     steps=144, seed=10000, split='validation')
        metrics = result['summary']['model']
        # Safety first, then comfort; energy/reward break ties. Never select on test data.
        key = (-metrics['unsafe_steps'], metrics['avg_comfort'], -metrics['energy_kwh'], metrics['total_reward'])
        improved = self.best_key is None or key > self.best_key
        if improved:
            self.best_key = key
            self.history['best_episode'] = episode
            self.history['best_validation'] = result
            self.agent.save(str(self.output_dir / 'best.pt'), {'seed': self.seed, 'episode': episode,
                            'selection_split': 'validation', 'metrics': metrics})
        self.history['validation'].append({'episode': episode, 'selected': improved, 'result': result})
        self._write_history()
        return metrics

    def _write_history(self):
        import json
        path = self.output_dir / 'training.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.history, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)

    def train(self, n_episodes=200, verbose=True):
        if n_episodes < 1:
            raise ValueError("n_episodes must be positive")
        self.history['episodes_requested'] = n_episodes
        self.warmup()
        self.validate(0)  # The teacher-initialized model is a real candidate, not just the last RL iterate.
        for episode in range(1, n_episodes + 1):
            summary = self.run_episode(episode)
            if verbose:
                print(f"Seed {self.seed} ep {episode}/{n_episodes}: comfort={summary['avg_comfort']:.1f}% "
                      f"reward={summary['total_reward']:.2f} epsilon={summary['epsilon']:.3f}", flush=True)
            if episode % self.validation_interval == 0 or episode == n_episodes:
                self.validate(episode)
        self.agent.save(str(self.output_dir / 'last.pt'), {'seed': self.seed, 'episode': n_episodes})
        self._write_history()
        print(f"Best checkpoint: {self.output_dir / 'best.pt'} (episode {self.history['best_episode']})", flush=True)
        return self.history
