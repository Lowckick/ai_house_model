"""
Deep Q-Network (DQN) agent for smart home control.
State:  indoor observations + outdoor weather + time features
Action: discrete composite action (encoded as index)
Reward: comfort score + energy penalty + cleanliness bonus
"""

import math
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

from smart_house.environment.indoor_state import IndoorState, DeviceState
from smart_house.environment.weather import WeatherSnapshot


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

HVAC_MODES = [0, 1, 2, 3, 4]            # 5 options
LIGHT_PRESETS = [0, 3, 7, 10]           # brightness presets applied to ALL rooms
BLIND_PRESETS = [0.0, 25.0, 50.0, 100.0]
AIR_PURIFIER_MODES = [0, 1, 2, 3]
HUMIDIFIER_MODES = [0, 1, 2]
CLEANER_BOT_CMDS = [0, 1]               # 0=stop/dock, 1=start cleaning
VENTILATION_SPEEDS = [0, 1, 2, 3]


def build_action_space() -> List[dict]:
    """
    Build a reduced discrete action space by enumerating meaningful combinations.
    Full factorial is too large; we use important combinations + delta actions.
    """
    actions = []
    for hvac in HVAC_MODES:
        for lights in LIGHT_PRESETS:
            for purifier in AIR_PURIFIER_MODES:
                for humidifier in HUMIDIFIER_MODES:
                    for cleaner in CLEANER_BOT_CMDS:
                        for vent in VENTILATION_SPEEDS:
                            actions.append({
                                "hvac_mode": hvac,
                                "lights_preset": lights,
                                "air_purifier": purifier,
                                "humidifier_mode": humidifier,
                                "cleaner_bot": cleaner,
                                "ventilation": vent,
                            })
    return actions


ACTION_SPACE = build_action_space()
N_ACTIONS = len(ACTION_SPACE)


def action_index_to_devices(idx: int, current: DeviceState) -> DeviceState:
    """Convert action index to DeviceState, preserving per-room granularity."""
    a = ACTION_SPACE[idx]
    new_state = DeviceState(
        hvac_mode=a["hvac_mode"],
        lights={room: a["lights_preset"] for room in current.lights},
        blinds=dict(current.blinds),                # blinds handled by rule layer
        air_purifier=a["air_purifier"],
        humidifier_mode=a["humidifier_mode"],
        cleaner_bot=a["cleaner_bot"],
        ventilation=a["ventilation"],
        security=current.security,
        curtains_closed=dict(current.curtains_closed),
    )
    return new_state


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

def encode_state(indoor: IndoorState, weather: WeatherSnapshot,
                 devices: DeviceState) -> np.ndarray:
    """Encode full observation into a flat numpy float32 vector."""
    now = indoor.timestamp
    hour_sin = math.sin(2 * math.pi * now.hour / 24.0)
    hour_cos = math.cos(2 * math.pi * now.hour / 24.0)
    day_sin = math.sin(2 * math.pi * now.timetuple().tm_yday / 365.0)
    day_cos = math.cos(2 * math.pi * now.timetuple().tm_yday / 365.0)

    weather_vec = [
        weather.temperature_c / 40.0,
        weather.humidity_pct / 100.0,
        weather.wind_speed_kmh / 80.0,
        weather.cloud_cover_pct / 100.0,
        weather.precipitation_mm / 20.0,
        weather.uv_index / 12.0,
        float(weather.is_daytime),
    ]

    indoor_vec = indoor.to_vector()
    device_vec = devices.to_vector()
    time_vec = [hour_sin, hour_cos, day_sin, day_cos]

    full = np.array(weather_vec + indoor_vec + device_vec + time_vec, dtype=np.float32)
    return full


STATE_DIM: int = len(encode_state(
    IndoorState(), WeatherSnapshot(
        timestamp=__import__("datetime").datetime.now(),
        temperature_c=15.0, humidity_pct=60.0, wind_speed_kmh=10.0,
        cloud_cover_pct=50.0, precipitation_mm=0.0, uv_index=3.0,
        is_daytime=True, condition="cloudy"
    ), DeviceState()
))


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """Dueling DQN architecture with layer normalization."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden // 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        # Dueling: Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    demonstration: bool = False


class PrioritizedReplayBuffer:
    """Simple uniform replay buffer (priority extension available)."""

    def __init__(self, capacity: int = 50_000, rng=None):
        self._rng = rng or random.Random()
        self._buf: deque = deque(maxlen=capacity)

    def push(self, t: Transition):
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return self._rng.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    Multi-objective reward:
      + comfort (temperature, humidity, CO2, light)
      + cleanliness
      - energy usage
      - rapid device switching (wear penalty)
    """

    VERSION = "comfort_safety_v2"
    COMFORT_WEIGHT = 2.0
    ENERGY_WEIGHT = 0.15
    CLEAN_WEIGHT = 0.2
    SWITCH_PENALTY = 0.03

    def __init__(self):
        self._prev_devices: Optional[DeviceState] = None

    def compute(self, indoor: IndoorState, devices: DeviceState,
                 prev_devices: Optional[DeviceState] = None) -> float:
        reward = 0.0

        # Comfort: scaled comfort score to [-1, +1]
        reward += self.COMFORT_WEIGHT * (indoor.comfort_score / 50.0 - 1.0)

        # Explicit physical deviations prevent energy savings from masking discomfort.
        temp_gap = max(0.0, 19.0 - indoor.temperature_c, indoor.temperature_c - 25.0)
        humidity_gap = max(0.0, 35.0 - indoor.humidity_pct, indoor.humidity_pct - 65.0)
        reward -= 0.15 * temp_gap + 0.015 * humidity_gap
        reward -= 0.3 * max(0.0, indoor.co2_ppm - 1000.0) / 1000.0
        reward -= 0.2 * max(0.0, indoor.pm25_ugm3 - 35.0) / 35.0
        if (indoor.temperature_c < 5 or indoor.temperature_c > 32
                or indoor.co2_ppm > 1500 or indoor.pm25_ugm3 > 55):
            reward -= 3.0

        # Energy penalty
        reward -= self.ENERGY_WEIGHT * indoor.power_consumption_kw

        # Cleanliness bonus/penalty
        clean = 100.0 - (indoor.dirt_level + indoor.dust_level) / 2.0
        reward += self.CLEAN_WEIGHT * (clean / 50.0 - 1.0)

        # Switching penalty
        if prev_devices is not None:
            switches = self._count_switches(devices, prev_devices)
            reward -= self.SWITCH_PENALTY * switches

        return float(reward)

    def _count_switches(self, new: DeviceState, old: DeviceState) -> int:
        n = 0
        if new.hvac_mode != old.hvac_mode: n += 1
        if new.air_purifier != old.air_purifier: n += 1
        if new.humidifier_mode != old.humidifier_mode: n += 1
        if new.cleaner_bot != old.cleaner_bot: n += 1
        if new.ventilation != old.ventilation: n += 1
        return n


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class SmartHomeDQNAgent:
    """
    Double DQN agent with experience replay and epsilon-greedy exploration.
    """

    GAMMA = 0.97
    BATCH_SIZE = 64
    LR = 1e-3
    TARGET_UPDATE_FREQ = 200      # steps
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 device: Optional[str] = None, action_mode="legacy", seed=42):
        from smart_house.ml.controller import RESIDUAL_ACTIONS
        if action_mode not in {"legacy", "residual_v1"}:
            raise ValueError("Unknown action mode")
        self.action_mode = action_mode
        self._rng = random.Random(seed)
        if action_mode == "residual_v1":
            n_actions = len(RESIDUAL_ACTIONS)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_actions = n_actions

        self.policy_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=3e-4 if action_mode == "residual_v1" else self.LR)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = PrioritizedReplayBuffer(rng=self._rng)
        self.reward_fn = RewardFunction()

        self.epsilon = 0.2 if action_mode == "residual_v1" else self.EPSILON_START
        self.steps_done = 0
        self.losses: List[float] = []
        self.episode_rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done, demonstration=False):
        self.replay.push(Transition(state, action, reward, next_state, done, demonstration))

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.BATCH_SIZE:
            return None

        batch = self.replay.sample(self.BATCH_SIZE)
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).to(self.device)

        # Double DQN: policy net selects action, target net evaluates
        current_q = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze()
            target_q = rewards + (0.99 if self.action_mode == "residual_v1" else self.GAMMA) * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        demonstrations = torch.tensor([t.demonstration for t in batch], dtype=torch.bool, device=self.device)
        if demonstrations.any():
            # Retain a small imitation anchor while RL learns useful corrections.
            logits = self.policy_net(states[demonstrations])
            loss = loss + 0.1 * nn.functional.cross_entropy(logits, actions[demonstrations].squeeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(0.02 if self.action_mode == "residual_v1" else self.EPSILON_END,
                           self.epsilon * self.EPSILON_DECAY)
        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def decode_action(self, index, current, indoor, weather):
        if self.action_mode == "residual_v1":
            from smart_house.ml.controller import residual_devices
            return residual_devices(index, indoor, weather)
        return action_index_to_devices(index, current)

    def imitate(self, states, actions, epochs=5):
        """Behavior cloning of teacher rollouts, followed by a target-network sync."""
        if not states or epochs < 1:
            return []
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        losses = []
        for _ in range(epochs):
            order = torch.randperm(len(x), device=self.device)
            for start in range(0, len(x), self.BATCH_SIZE):
                indices = order[start:start + self.BATCH_SIZE]
                loss = nn.functional.cross_entropy(self.policy_net(x[indices]), y[indices])
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.item()))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return losses

    def save(self, path: str, metadata=None):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "schema_version": 2, "action_mode": self.action_mode,
            "n_actions": self.n_actions, "reward_version": self.reward_fn.VERSION,
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon, "steps_done": self.steps_done,
            "rng_state": self._rng.getstate(), "metadata": metadata or {},
        }, temporary)
        os.replace(temporary, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        mode = checkpoint.get("action_mode", "legacy")
        from smart_house.ml.controller import RESIDUAL_ACTIONS
        expected = len(RESIDUAL_ACTIONS) if mode == "residual_v1" else N_ACTIONS
        if mode not in {"legacy", "residual_v1"} or checkpoint.get("n_actions", expected) != expected:
            raise ValueError("Unsupported checkpoint action semantics")
        if self.action_mode != mode or self.n_actions != expected:
            self.__init__(device=str(self.device), action_mode=mode)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.EPSILON_END)
        self.steps_done = checkpoint.get("steps_done", 0)
        if "rng_state" in checkpoint:
            self._rng.setstate(checkpoint["rng_state"])
        self.checkpoint_metadata = checkpoint.get("metadata", {})
