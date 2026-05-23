# 🏠 Smart House ML Control System

A large-scale Python reinforcement learning system that adaptively controls a simulated smart home to maximize occupant comfort, minimize energy use, and maintain cleanliness — all based on real-time or simulated indoor and outdoor conditions.

---

## Architecture

```
kursova/
├── main.py                         ← Entry point (CLI)
├── requirements.txt
├── data/                           ← SQLite DB + saved models + reports
│   ├── training.db
│   ├── model.pt
│   ├── reports/
│   └── screenshots/
└── smart_house/
    ├── environment/
    │   ├── weather.py              ← Open-Meteo API + synthetic generator
    │   └── indoor_state.py         ← Physics simulation (temp, humidity, air, dirt, light, power)
    ├── ml/
    │   ├── agent.py                ← Dueling Double DQN (PyTorch)
    │   ├── rule_layer.py           ← Safety + smart rule overrides
    │   └── trainer.py              ← Training loop + SQLite logging
    ├── dashboard/
    │   └── display.py              ← Rich live terminal dashboard
    └── analytics/
        ├── reporter.py             ← Matplotlib training report generator
        └── screenshots.py          ← PNG dashboard snapshot generator
```

---

## Features

| Feature | Description |
|---|---|
| **Adaptive HVAC** | Learns to heat/cool based on indoor temp, outdoor weather, occupancy |
| **Smart Lighting** | Auto-adjusts per-room brightness based on natural light + occupancy |
| **Blind/Curtain Control** | Opens/closes blinds for solar gain, privacy, or glare reduction |
| **Cleaner Robot** | Launches automatically when dirt exceeds threshold; pauses during sleep |
| **Air Purifier** | Ramps up when PM2.5 or VOC levels are elevated |
| **Humidifier** | Maintains 40–60% RH; dehumidifies in rain/high-humidity weather |
| **Ventilation** | Increases when CO2 > 1000 ppm or occupants are exercising |
| **Security System** | Arms/disarms based on occupancy patterns |
| **Energy Tracking** | Monitors kW per device, daily kWh total |
| **Occupancy Simulation** | Realistic daily patterns (sleeping, cooking, working, away) |
| **Weather Integration** | Live data from [Open-Meteo](https://open-meteo.com) (no API key needed) |
| **Synthetic Fallback** | Seasonal + diurnal physics-based weather generator if API is unavailable |
| **Live Dashboard** | Full-screen Rich terminal UI updating 4×/second |
| **Training Analytics** | Matplotlib charts saved as PNG reports |
| **Report Screenshots** | Generates PNG dashboard snapshots for coursework/demo materials |

---

## ML Model

- **Algorithm**: Dueling Double DQN (PyTorch)
- **State**: 50+ features — indoor conditions, weather, devices, time-of-day encodings
- **Action**: Discrete composite actions (HVAC × Lights × Purifier × Humidifier × Cleaner × Ventilation)
- **Reward**: `+comfort score` `−energy use` `+cleanliness` `−device switching penalty`
- **Enhancements**: Layer normalization, experience replay (50k), epsilon-greedy decay, periodic target network updates

---

## Setup

```bash
pip install -r requirements.txt
```

> For GPU acceleration, install the appropriate PyTorch build from https://pytorch.org

---

## Usage

```bash
# Train with live dashboard (200 episodes by default)
python main.py train

# Train faster without UI
python main.py train --no-ui --episodes 500

# Skip real weather API (use synthetic data)
python main.py train --no-api

# Run live simulation using the trained model
python main.py simulate

# Simulate at faster speed (0.2s per step)
python main.py simulate --speed 0.2

# Generate training analytics report (PNG)
python main.py report

# Generate dashboard screenshots for the coursework report
python main.py screenshots --count 4 --no-api

# Quick 5-episode demo
python main.py demo
```

---

## Weather Data

By default the system uses **Open-Meteo** (free, no API key, location: Kyiv, Ukraine).
If the API is unreachable, it automatically falls back to the **synthetic weather generator** which produces realistic seasonal + diurnal patterns with Gaussian noise.

---

## Extending

- Add new devices → extend `DeviceState`, update `build_action_space()` in `agent.py`
- Change location → set `DEFAULT_LAT`/`DEFAULT_LON` in `weather.py`
- Tune comfort targets → edit `COMFORT_TEMP_MIN/MAX` in `indoor_state.py`
- Adjust reward weights → edit `RewardFunction` in `agent.py`
