# Запуск у Docker

Образ містить програму, CPU-збірку PyTorch, `paho-mqtt` і готові моделі з
`data/models/`. Дані (історія, показники датчиків, кеш погоди) зберігаються
в томі `house-data` і переживають оновлення образу.

```bash
cp /dev/null .env
echo "SMART_HOUSE_LIVE_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" >> .env
docker compose up -d --build
```

Вебпанель: http://127.0.0.1:8000. Щоб датчики з домашньої мережі могли
надсилати показники, додайте в `.env` рядок `SMART_HOUSE_BIND=0.0.0.0`
(панель не має входу за паролем — лише довірена мережа). Інші змінні `.env`:

| Змінна | Призначення |
| --- | --- |
| `SMART_HOUSE_LIVE_TOKEN` | обов'язковий токен для `POST /api/live/<кімната>` |
| `SMART_HOUSE_LAT`, `SMART_HOUSE_LON` | координати для погоди Open-Meteo (типово Київ) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | сповіщення про тривоги (необов'язково) |
| `MQTT_BROKER`, `MQTT_TOPIC`, `MQTT_PASSWORD` | для підписника MQTT |

MQTT-підписник: `docker compose --profile mqtt up -d`. Якщо брокер вимагає
ім'я користувача, додайте `"--username", "<ім'я>"` до `command` сервісу `mqtt`.

Інші команди того самого образу:

```bash
docker compose run --rm web shadow-report --sensor bedroom --hours 168 --output data/user/bedroom.json
docker compose run --rm web calibrate --sensor bedroom --output data/user/bedroom-physics.json
docker compose run --rm web evaluate --episodes 12 --seed 20000 --output data/user/eval.json
```

Файли з'являються в томі (`docker compose cp web:/app/data/user/bedroom.json .`).

Без Docker — те саме однією командою після встановлення залежностей:
`python main.py web --weather open-meteo`.

Примітка: у середовищі, де готувався Dockerfile, збірку не вдалося виконати
до кінця (обмеження мережі для Docker Hub і download.pytorch.org); перевірено
синтаксис `docker compose config` та обов'язковість токена.
