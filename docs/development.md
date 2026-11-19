# Розробка

## Тести й лінтер

```bash
pip install ruff paho-mqtt
ruff check .
python -m unittest discover -s tests -v
```

GitHub Actions виконує обидві перевірки на кожен push (`.github/workflows/tests.yml`).
Налаштування ruff — у `pyproject.toml`. Нові тести додавайте в `tests/` (unittest).

## Структура

| Каталог | Зміст |
| --- | --- |
| `smart_house/environment` | фізика будинку (`PhysicsParams`), варіанти будинків, сценарії, погода, тариф, валідація показників |
| `smart_house/ml` | мережа й винагороди, базові контролери, rollout-навчання, DQN, експерименти, адаптація, правила |
| `smart_house/analytics` | оцінювання, імпорт і пояснення, живі датчики, калібрування, історія, лабораторія, графіки, демо |
| `smart_house/web` | HTTP-сервер без сторонніх фреймворків, `index.html`, `static/app.js`, `static/style.css` |

## Як розширювати

- **Новий пристрій**: поле в `DeviceState` (`indoor_state.py`), його вплив у `IndoorPhysicsSimulator`
  і в `device_power_kw`, рішення в базових контролерах (`controller.py`), корекції в `RESIDUAL_ACTIONS`.
  Зміна кількості корекцій змінює семантику дій — потрібен новий `action_mode` і нове навчання.
- **Новий варіант будинку**: запис у `HOUSES` (`environment/houses.py`); він одразу доступний для
  `train --house`, `evaluate --house` і `adapt --houses`.
- **Інша винагорода**: нова версія в `RewardFunction.VERSIONS`; версія зберігається в checkpoint.
- **Правила**: `rule_layer.py`; при зміні поведінки збільшуйте `RuleLayer.VERSION`, бо від неї
  залежать порівняння з попередніми результатами.
- **Локація за замовчуванням**: `DEFAULT_LAT`/`DEFAULT_LON` у `weather.py` або `--lat/--lon` у CLI.

Типові значення `PhysicsParams` відтворюють симулятор попередніх експериментів біт-у-біт —
зберігайте цю властивість, додаючи нові параметри (тест `tests/test_reproducibility.py`).
