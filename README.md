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
- **Action**: New `residual_v1` models choose 19 corrections to a deterministic controller; legacy checkpoints retain 1920 composite actions
- **Reward**: `comfort_safety_v2`: comfort and cleanliness, explicit discomfort/safety penalties, energy and switching costs
- **Enhancements**: Layer normalization, experience replay (50k), epsilon-greedy decay, periodic target network updates

---

## Setup

```bash
pip install -r requirements.txt
```

> For GPU acceleration, install the appropriate PyTorch build from https://pytorch.org

---

## Покращене навчання

Сезонні сценарії, умовний час, 19 дій-корекцій над простим контролером,
експертний warmup, окремі validation/test та відбір найкращої моделі:

```bash
python main.py train --no-ui --seeds 42 43 44 --episodes 200 \
  --output-dir data/user/training/experiment-01 --reference-model data/model.pt
python main.py web --model data/user/training/experiment-01/best.pt
```

[Інструкція навчання, параметри та методика відбору](docs/training.md).
[Результати виконаного експерименту](docs/training-results.md).
Стара `data/model.pt` зберігається; нові checkpoint створюються окремо.
Готовий checkpoint і звіт включено до `data/models/`:

```bash
python main.py web --model data/models/improved-v2.pt
```


## Usage

```bash
# Train with live dashboard (200 episodes by default; new experiment directory)
python main.py train

# Train faster without UI
python main.py train --no-ui --episodes 500

# Skip real weather API (use synthetic data)
python main.py train --no-api

# Run simulation with an explicitly selected trained model
python main.py simulate --model data/user/training/experiment-01/best.pt

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

## Вебінтерфейс: завантаження, історія та оцінка моделі

```bash
python main.py web
```

Локальна сторінка на порту 8000 дозволяє перевіряти та завантажувати JSON/CSV/TXT,
переглядати вимірювання, пояснення рекомендацій і збережену історію SQLite,
а також порівнювати модель із простим детермінованим контролером.

[Покрокова інструкція, методика оцінки, CLI та API](docs/user-data.md).

## Завантаження власних показників

Команда `import-data` читає показники приміщення, виконує inference навченої
моделі, застосовує правила безпеки й зберігає рекомендації для кожного запису.
Вона не змінює модель або базу навчання й не керує фізичними пристроями.
Для навчання на історії потрібні додаткові дані про дії та їхні наслідки;
цей імпорт призначений для аналізу вимірювань.

### Як використати

Після встановлення залежностей, із кореня репозиторію:

```bash
python main.py import-data --input examples/readings.json --output /tmp/recommendations-json.json
python main.py import-data --input examples/readings.txt --output /tmp/recommendations-text.json
# Власна навчена модель:
python main.py import-data --input /path/to/readings.csv --output /tmp/my-results.json --model /path/to/model.pt
```

У підготовленому хмарному середовищі замість `python` використовуйте
`/workspace/.venvs/ai-house/bin/python`. Типова модель — `data/model.pt`;
використовуйте лише довірені PyTorch checkpoint-файли.
Вихідний файл має бути новим: наявні файли не перезаписуються.
Помилки формату/значень завершують команду з кодом 2; часткового імпорту
при помилці валідації немає. Звіт містить номери всіх проблемних записів.
Успішні імпорти додатково зберігаються в історії `data/user/measurements.db`;
шлях можна змінити через `--db`.

### Формат файлів

Кодування — UTF-8 (BOM дозволено). JSON містить один об’єкт або масив об’єктів,
як у [прикладі](examples/readings.json). `.csv` та `.txt` — таблиця з комами,
заголовком і одним записом на рядок, як у [прикладі](examples/readings.txt).
Числовий десятковий роздільник — крапка. Довільний текст не підтримується.

Обов’язкові поля:

| Поле | Значення |
| --- | --- |
| `timestamp` | Дата й час ISO 8601, наприклад `2026-10-23T08:00:00+03:00`; бажано з часовим поясом |
| `temperature_c` | Температура всередині, °C, від −60 до 80 |
| `humidity_pct` | Вологість всередині, %, від 0 до 100 |

Необов’язкові поля та значення за замовчуванням:

| Поле | Одиниці / діапазон | Типове значення |
| --- | --- | --- |
| `co2_ppm` | ppm, 0–100000 | 450 |
| `voc_ppb` | ppb, 0–100000 | 100 |
| `pm25_ugm3`, `pm10_ugm3` | мкг/м³, 0–10000 | 5, 10 |
| `air_quality_index` | 0–500 | 50 |
| `dirt_level`, `dust_level` | %, 0–100 | 0 |
| `total_occupants` | ціле число, 0–100 | 0 |
| `occupant_activity` | `idle`, `cooking`, `exercising`, `sleeping` | `idle` |
| `power_consumption_kw` | кВт, 0–1000 | 0 |
| `energy_today_kwh` | кВт·год, 0–24000 | 0 |

Невідомі назви полів, NaN, нескінченність і значення поза діапазонами
відхиляються. Порожні необов’язкові клітинки CSV використовують типові значення.
Кожен рядок аналізується незалежно, у порядку файлу; інтерполяції немає.
Температура всіх кімнат приймається рівною загальній. Якщо мешканці присутні,
усі кімнати вважаються зайнятими; освітленість за замовчуванням — 0 люкс.

Зовнішня погода для відтворюваного автономного аналізу фіксована: 15 °C,
60% вологості, вітер 0 км/год, хмарність 50%, опади та UV — 0;
денний час — з 06:00 до 20:00 за часом запису. API не викликається.
Ці припущення впливають на рекомендації; вони не замінюють реальні вимірювання.

### Результат

JSON містить `count` та масив `readings`: нормалізовані показники `indoor`,
припущення `weather`, індекс дії моделі `action_index` і пристрої
`recommended_devices` після правил безпеки. Наприклад, CO₂ понад 1500 ppm
примусово задає вентиляцію щонайменше рівня 2.
Числові режими пристроїв описані в `DeviceState` у
`smart_house/environment/indoor_state.py` (HVAC: 0 — вимкнено,
1/2 — слабкий/сильний обігрів, 3/4 — слабке/сильне охолодження;
вентиляція й очищувач: 0–3).

Перевірка імпортера: `python -m unittest discover -s tests -v`.
