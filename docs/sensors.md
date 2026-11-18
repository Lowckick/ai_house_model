# Підключення датчиків: тіньовий режим

Датчики надсилають показники по одному: HTTP-запитом (ESP32, Arduino, Home
Assistant) або через MQTT. Для кожного показника програма перевіряє значення,
формує рекомендацію моделі, перевіряє пороги тривог і зберігає все в історії.
**Пристроями вона не керує** — це тіньовий режим: модель лише радить і записує,
що зробила б. Звіт тіньового режиму порівнює ці рекомендації з тим, як насправді
змінювались умови в кімнаті (матеріал для натурного експерименту, підрозділ 5.8).

## Формат показника

Той самий, що для файлів ([user-data.md](user-data.md#формат-файлів)), але один
JSON-об'єкт. Обов'язкові `temperature_c` і `humidity_pct`; `timestamp` можна не
надсилати — тоді береться час отримання (UTC). Невідомі поля відхиляються.

```json
{"temperature_c": 21.4, "humidity_pct": 44, "co2_ppm": 870, "pm25_ugm3": 6}
```

Ідентифікатор датчика (кімнати) — 1–64 символи: латиниця, цифри, `_ . -`.
Якщо попередній показник того самого датчика надійшов не раніше ніж 30 хв тому,
його рекомендація використовується як поточний стан пристроїв (як у симуляторі).

## HTTP

```bash
export SMART_HOUSE_LIVE_TOKEN='довгий-випадковий-рядок'
python main.py web --host 0.0.0.0 --weather open-meteo --lat 50.45 --lon 30.52
```

Сервер, доступний з мережі, без `SMART_HOUSE_LIVE_TOKEN` не запуститься.
Датчик надсилає `POST /api/live/<sensor_id>` із заголовками
`Content-Type: application/json` та `Authorization: Bearer <токен>`:

```bash
curl -X POST http://192.168.1.10:8000/api/live/bedroom \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $SMART_HOUSE_LIVE_TOKEN" \
  -d '{"temperature_c": 21.4, "humidity_pct": 44, "co2_ppm": 870}'
```

Відповідь `201` містить рекомендацію, `top_corrections` (впевненість мережі)
і тривоги; `401` — немає або неправильний токен; `422` — помилки значень.
Вебпанель (імпорт, історія, лабораторія) не має входу за паролем — відкривайте
сервер лише в довіреній домашній мережі.

### ESP32 / Arduino (приклад)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>

void sendReading(float t, float h, float co2) {
  HTTPClient http;
  http.begin("http://192.168.1.10:8000/api/live/bedroom");
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", "Bearer " SMART_HOUSE_TOKEN);  // #define у secrets.h
  String body = String("{\"temperature_c\":") + t + ",\"humidity_pct\":" + h +
                ",\"co2_ppm\":" + co2 + "}";
  int code = http.POST(body);   // 201 = прийнято
  http.end();
}
```

Інтервал 1–10 хв. Частіші показники не шкодять, але й не дають нової інформації.

### Home Assistant

`configuration.yaml`:

```yaml
rest_command:
  smart_house_reading:
    url: "http://192.168.1.10:8000/api/live/{{ room }}"
    method: POST
    headers:
      Authorization: !secret smart_house_bearer   # "Bearer <токен>" у secrets.yaml
    content_type: "application/json"
    payload: >
      {"temperature_c": {{ states('sensor.' ~ room ~ '_temperature') }},
       "humidity_pct": {{ states('sensor.' ~ room ~ '_humidity') }},
       "co2_ppm": {{ states('sensor.' ~ room ~ '_co2') }}}
```

Автоматизація з тригером `time_pattern` (`minutes: "/5"`) викликає
`rest_command.smart_house_reading` з `room: bedroom`. Назви сутностей змініть
на свої; якщо датчика CO₂ немає, приберіть поле — підставиться типове значення.

## MQTT

```bash
pip install paho-mqtt
export MQTT_PASSWORD='...'          # якщо брокер вимагає пароль
python main.py mqtt --broker 192.168.1.5 --topic 'home/+/climate' --username house \
  --weather open-meteo --lat 50.45 --lon 30.52
```

Рівень теми на місці `+` — ідентифікатор датчика (`home/bedroom/climate` →
`bedroom`), вміст повідомлення — JSON-показник. `--tls` вмикає TLS. Помилкові
повідомлення відхиляються з поясненням у журналі й не зупиняють підписку.
Zigbee2MQTT публікує інший формат; перетворіть його правилом брокера чи
автоматизацією Home Assistant на поля вище.

## Тривоги та Telegram

| Показник | Попередження | Критично |
| --- | --- | --- |
| CO₂ | > 1000 ppm | > 1500 ppm |
| PM2.5 | > 35 мкг/м³ | > 55 мкг/м³ |
| Температура | < 16 або > 28 °C | < 5 або > 32 °C |
| Вологість | < 30 або > 70 % | — |

Тривога надсилається один раз, коли вона з'являється, а не на кожен показник.
Для Telegram створіть бота через @BotFather і задайте змінні середовища (не
аргументи командного рядка, щоб токен не потрапив в історію shell):

```bash
export TELEGRAM_BOT_TOKEN='123456:ABC...'
export TELEGRAM_CHAT_ID='123456789'
```

Збій надсилання не губить показник: помилка записується в `notification_error`.

## Звіт тіньового режиму

```bash
python main.py shadow-report --sensor bedroom --hours 168 --output /tmp/bedroom-week.json
```

Або `GET /api/live/bedroom/report?hours=168`. Звіт містить:

- покриття: кількість показників, медіанний інтервал, розриви понад 30 хв;
- **спостережене**: частку часу в зоні комфорту (19–25 °C, 35–65 %, CO₂ ≤ 1000,
  PM2.5 ≤ 35) окремо й разом, середні значення, кількість критичних показників;
- **рекомендації**: скільки разів модель відрізнялась від простого контролера,
  скільки використала корекцій, скільки перемикань пристроїв радила, скільки
  виходів за комфорт «покрила» правильним пристроєм;
- оцінку енергії й вартості (двозонний тариф), якби рекомендації виконувались,
  поряд з енергією простого контролера — за моделлю потужностей симулятора;
- журнал тривог.

Кожен показник утримується до наступного, але не довше 30 хв: довші розриви
не враховуються в частках часу й енергії. Звіт не доводить ефекту керування:
пристрої не перемикались. Щоб оцінити ефект, порівняйте періоди, коли
рекомендації виконували вручну, з періодами, коли ні, і зафіксуйте це в журналі.

`GET /api/live` — список датчиків; `GET /api/live/<id>?hours=24&limit=500` — показники.
