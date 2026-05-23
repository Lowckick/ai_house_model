"""
Training loop for the Smart Home DQN agent.
Runs episodes of simulated time and logs progress to SQLite.
"""

import sqlite3
import os
import time
import datetime
import numpy as np
from typing import Optional, Callable

from smart_house.environment.indoor_state import IndoorPhysicsSimulator, DeviceState
from smart_house.environment.weather import WeatherProvider
from smart_house.ml.agent import SmartHomeDQNAgent, encode_state, action_index_to_devices
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
    """Runs training episodes for the DQN agent."""

    STEPS_PER_EPISODE = 144      # 144 × 10 min = 24 hours simulated
    MODEL_SAVE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "model.pt"
    )

    def __init__(self, use_api_weather: bool = True,
                 progress_callback: Optional[Callable] = None):
        self.agent = SmartHomeDQNAgent()
        self.physics = IndoorPhysicsSimulator(n_occupants=2)
        self.weather_provider = WeatherProvider(use_api=use_api_weather)
        self.rule_layer = RuleLayer()
        self.progress_callback = progress_callback
        self._db_path = os.path.abspath(DB_PATH)
        init_db(self._db_path)

        # Try to load existing model
        model_path = os.path.abspath(self.MODEL_SAVE_PATH)
        self.agent.load(model_path)

    def run_episode(self, episode: int) -> dict:
        """Run one full episode (24 simulated hours). Returns summary dict."""
        indoor = self.physics.reset()
        devices = DeviceState()
        weather = self.weather_provider.get()

        total_reward = 0.0
        comfort_sum = 0.0
        power_sum = 0.0
        loss_val = None
        ep_start = time.time()

        conn = sqlite3.connect(self._db_path)

        for step in range(self.STEPS_PER_EPISODE):
            state = encode_state(indoor, weather, devices)
            action_idx = self.agent.select_action(state)
            prev_devices = DeviceState(
                hvac_mode=devices.hvac_mode,
                lights=dict(devices.lights),
                blinds=dict(devices.blinds),
                air_purifier=devices.air_purifier,
                humidifier_mode=devices.humidifier_mode,
                cleaner_bot=devices.cleaner_bot,
                ventilation=devices.ventilation,
                security=devices.security,
            )

            # Apply RL action, then safety rules on top
            raw_devices = action_index_to_devices(action_idx, devices)
            devices = self.rule_layer.apply(raw_devices, indoor, weather)

            # Refresh weather every 30 steps (~5 hours)
            if step % 30 == 0:
                weather = self.weather_provider.get()

            indoor = self.physics.step(weather, devices)
            next_state = encode_state(indoor, weather, devices)

            reward = self.agent.reward_fn.compute(indoor, devices, prev_devices)
            done = (step == self.STEPS_PER_EPISODE - 1)

            self.agent.store(state, action_idx, reward, next_state, done)
            loss_val = self.agent.learn()

            total_reward += reward
            comfort_sum += indoor.comfort_score
            power_sum += indoor.power_consumption_kw

            # Write per-step data
            conn.execute("""
                INSERT INTO episodes
                (episode, step, reward, comfort, temperature, humidity, co2, pm25,
                 dirt, power_kw, hvac_mode, cleaner_bot, air_purifier, epsilon, loss, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                episode, step, round(reward, 4), indoor.comfort_score,
                indoor.temperature_c, indoor.humidity_pct,
                indoor.co2_ppm, indoor.pm25_ugm3,
                indoor.dirt_level, indoor.power_consumption_kw,
                devices.hvac_mode, devices.cleaner_bot,
                devices.air_purifier, round(self.agent.epsilon, 4),
                round(loss_val, 6) if loss_val else None,
                indoor.timestamp.isoformat()
            ))

            if self.progress_callback and step % 10 == 0:
                self.progress_callback(episode, step, indoor, devices, weather, reward)

        ep_duration = time.time() - ep_start
        avg_comfort = comfort_sum / self.STEPS_PER_EPISODE
        avg_power = power_sum / self.STEPS_PER_EPISODE

        conn.execute("""
            INSERT INTO episode_summary
            (episode, total_reward, avg_comfort, avg_power, steps, duration_s, timestamp)
            VALUES (?,?,?,?,?,?,?)
        """, (episode, round(total_reward, 4), round(avg_comfort, 2),
              round(avg_power, 4), self.STEPS_PER_EPISODE,
              round(ep_duration, 2), datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()

        # Save model every 10 episodes
        if episode % 10 == 0:
            self.agent.save(os.path.abspath(self.MODEL_SAVE_PATH))

        return {
            "episode": episode,
            "total_reward": total_reward,
            "avg_comfort": avg_comfort,
            "avg_power": avg_power,
            "epsilon": self.agent.epsilon,
            "steps": self.STEPS_PER_EPISODE,
            "duration_s": ep_duration,
        }

    def train(self, n_episodes: int = 500, verbose: bool = True):
        """Train the agent for n_episodes."""
        print(f"[Trainer] Starting training for {n_episodes} episodes")
        print(f"[Trainer] Weather source: {self.weather_provider.source}")
        print(f"[Trainer] Action space size: {self.agent.n_actions}")
        print(f"[Trainer] State dimensions: {self.agent.policy_net.shared[0].in_features}")

        for ep in range(1, n_episodes + 1):
            summary = self.run_episode(ep)
            if verbose and ep % 5 == 0:
                print(
                    f"  Ep {ep:4d}/{n_episodes} | "
                    f"Reward: {summary['total_reward']:+8.2f} | "
                    f"Comfort: {summary['avg_comfort']:5.1f}% | "
                    f"Power: {summary['avg_power']:.3f}kW | "
                    f"ε: {summary['epsilon']:.4f} | "
                    f"t: {summary['duration_s']:.1f}s"
                )

        self.agent.save(os.path.abspath(self.MODEL_SAVE_PATH))
        print(f"[Trainer] Training complete. Model saved to {self.MODEL_SAVE_PATH}")
