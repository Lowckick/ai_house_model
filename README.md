# 🏠 Smart House ML Control System

A large-scale Python reinforcement learning system that adaptively controls a simulated smart home to maximize occupant comfort, minimize energy use, and maintain cleanliness — all based on real-time or simulated indoor and outdoor conditions.

---

## Architecture

```
ai_house_model/
├── main.py                      ← CLI: train, evaluate, web, import-data, mqtt, shadow-report, calibrate, …
├── requirements.txt, constraints.txt, Dockerfile, docker-compose.yml
├── data/
│   ├── models/                  ← rollout-v2.pt (default), rollout-v3-tou.pt, improved-v2.pt + evaluation reports
│   ├── model.pt                 ← legacy 1920-action DQN (kept for comparison)
│   └── user/                    ← local, not in Git: histories, live readings, weather cache, experiments
├── smart_house/
│   ├── environment/
│   │   ├── indoor_state.py      ← building physics (calibratable PhysicsParams), devices, power model
│   │   ├── scenarios.py         ← disjoint train/validation/test seasonal scenarios
│   │   ├── weather.py           ← Open-Meteo forecast + synthetic generator
│   │   ├── weather_archive.py   ← hourly historical weather for recorded readings (cached)
│   │   ├── tariff.py            ← two-zone electricity tariff
│   │   └── readings.py          ← validation of sensor readings (JSON/CSV/TXT)
│   ├── ml/
│   │   ├── agent.py             ← network, rewards (comfort_safety_v2, tou_v3), checkpoints
│   │   ├── controller.py        ← deterministic base controllers and the 19 residual corrections
│   │   ├── rollout.py           ← rollout labels + DAgger distillation (recommended training)
│   │   ├── trainer.py           ← original warmup + DQN training
│   │   ├── experiments.py       ← multi-seed experiments, validation-only selection, test reports
│   │   └── rule_layer.py        ← lighting, blinds, security, sleep and final safety rules
│   ├── analytics/
│   │   ├── comparison.py        ← paired closed-loop evaluation (comfort, energy, cost, safety)
│   │   ├── ingestion.py         ← recommendations for readings, explanations, model confidence
│   │   ├── live.py              ← live sensors, alerts, Telegram, shadow-mode report
│   │   ├── calibration.py       ← fit simulator parameters to recorded readings
│   │   ├── history.py, lab.py   ← SQLite history and the model lab
│   │   └── reporter.py, screenshots.py
│   ├── web/                     ← local web panel (no third-party web framework)
│   └── dashboard/               ← Rich terminal dashboard
├── tests/                       ← unittest suite (CI: .github/workflows/tests.yml)
└── docs/                        ← training, results, sensors, calibration, Docker, model lab, thesis
```

---

## Features

| Feature | Description |
|---|---|
| **Hybrid control** | A network chooses one of 19 corrections to a deterministic controller; safety rules always apply last |
| **Rollout training** | Corrections scored on a simulator copy and distilled into the network (DAgger, validation-only selection) |
| **Two-zone tariff** | Optional tariff-aware reward; every evaluation reports electricity cost in UAH |
| **Live sensors** | HTTP (ESP32, Home Assistant) or MQTT readings in shadow mode: recommend and log, never switch devices |
| **Alerts** | CO₂, PM2.5, temperature and humidity alerts in the browser and optionally Telegram |
| **Real weather** | Hourly Open-Meteo history for imported and live readings, cached for offline reuse |
| **Calibration** | Heat loss, CO₂ generation/infiltration and humidity exchange estimated from your readings |
| **Web panel** | Sensors, uploads, history, explanations with model confidence, paired evaluation, model lab |
| **Simulation** | Seasonal weather, occupancy, temperature, humidity, air quality, dirt, light and power |
| **Terminal tools** | Rich live dashboard, Matplotlib training reports, dashboard screenshots |

---

## ML Model

- **Network**: dueling MLP with layer normalization (PyTorch), 54 input features: indoor conditions, weather, devices, time of day
- **Actions**: `residual_v2` — 19 corrections to a staged-threshold controller, with corrections that would weaken it outside comfort ignored; `residual_v1` and legacy 1920-action checkpoints still load
- **Training**: `--method rollout` (recommended) or the original `--method dqn`
- **Rewards**: `comfort_safety_v2`; `comfort_safety_tou_v3` weights energy by the day/night tariff
- **Results**: [docs/training-results.md](docs/training-results.md)

---

## Setup

Python 3.11+ (перевірено на 3.13). Для роботи на CPU спершу поставте легку
CPU-збірку PyTorch (≈200 МБ замість кількох ГБ зі CUDA), потім решту:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`constraints.txt` фіксує версії, з якими востаннє проходили всі тести:
`pip install -r requirements.txt -c constraints.txt`. Для GPU встановіть
відповідну збірку з https://pytorch.org. Тести: `python -m unittest discover -s tests`
(їх також запускає GitHub Actions на кожен push).

---

## Швидкий старт

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
python main.py web --weather open-meteo          # http://127.0.0.1:8000
```

або `docker compose up -d --build` ([docs/docker.md](docs/docker.md)). Датчики
(ESP32, Home Assistant, MQTT) підключаються в тіньовому режимі: модель радить і
записує, не керуючи пристроями ([docs/sensors.md](docs/sensors.md)). З їхніх
записів можна відкалібрувати симулятор під свій будинок
([docs/calibration.md](docs/calibration.md)).

## Найкраща модель: `rollout-v2`

```bash
python main.py web --model data/models/rollout-v2.pt
```

Мережа навчена відтворювати найкращу з 19 корекцій сильнішого ступінчастого
контролера; корекції оцінюються перебором на копії симулятора лише під час
навчання (`--method rollout`, DAgger, відбір на validation). На 40 test-сценаріях
комфорт 93.8 % проти 92.2 % у сильного контролера та 83–84 % у `improved-v2`,
енергія на 1 кВт·год/добу менша за сильний контролер, небезпечних кроків
не більше.

Для двозонного тарифу (уночі вдвічі дешевше) є `data/models/rollout-v3-tou.pt`:
на 3.3–5 % дешевша за `rollout-v2` за майже того ж комфорту
([результати](docs/training-results.md#двозонний-тариф-і-правила-rules_v2-18-листопада-2026)).
Порівняння в CLI, вебпанелі та звітах показують і вартість у гривнях. [Метод](docs/training.md#метод-rollout-перебір-корекцій-і-дистиляція-в-мережу) ·
[результати](docs/training-results.md#експеримент-rollout-17-листопада-2026).

```bash
python main.py train --method rollout --no-ui --seeds 42 43 44 \
  --output-dir data/user/training/rollout-01
```

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

## Лабораторія моделей

Вебпанель тепер містить ручне тестування показників, порівняння до трьох
checkpoint на однакових сценаріях, сім автоматичних перевірок безпеки,
перевірку регресій відносно попереднього запуску та журнал швидкодії й оцінок.
Тривалі оцінювання працюють у фоні; повторні тести використовують LRU-кеш.

[Запуск, інструкція та API лабораторії](docs/model-lab.md).

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
`/workspace/.venvs/ai-house/bin/python`. Типова модель — `data/models/rollout-v2.pt`;
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
