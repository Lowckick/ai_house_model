# ДОДАТОК А. СТРУКТУРА ПРОГРАМИ Й ВАЖЛИВІ ФРАГМЕНТИ КОДУ

Фрагменти відповідають версії коду {{commit}}. Ідентифікацію програмних артефактів наведено в електронному пакеті окремо від списку наукової літератури. Наведені функції є редагованим текстом, а не зображенням. Довгі рядки можуть переноситися засобами редактора без зміни їхнього програмного змісту.

## А.1 Модулі проєкту

```text
main.py                              CLI та маршрути робочих режимів
smart_house/environment/readings.py  сенсорний контракт і валідація
smart_house/environment/scenarios.py сезонні сценарії й seed
smart_house/environment/indoor_state.py фізика та структури стану
smart_house/environment/weather.py   погодний провайдер і генератор
smart_house/ml/agent.py              мережа, replay, reward, checkpoint
smart_house/ml/controller.py         вчитель і дискретні корекції
smart_house/ml/rule_layer.py         остаточні правила та пояснення
smart_house/ml/trainer.py            загальний навчальний цикл
smart_house/ml/planning.py           модельні мітки та дистиляція політики
smart_house/ml/experiments.py        multi-seed і підсумковий відбір
smart_house/analytics/comparison.py  парне оцінювання
smart_house/analytics/ingestion.py   спільний сервіс аналізу
smart_house/analytics/history.py     транзакційна історія
smart_house/analytics/lab.py         playground, кеш, worker, telemetry
smart_house/web/server.py            локальний HTTP API
smart_house/web/index.html           браузерна панель
```

## А.2 Кодування стану

Файл: smart_house/ml/agent.py.

```python
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
```

## А.3 Формування residual-команди

Файл: smart_house/ml/controller.py.

```python
def residual_devices(index, indoor, weather):
    if not 0 <= index < len(RESIDUAL_ACTIONS):
        raise ValueError('Invalid residual action index')
    devices = deterministic_devices(indoor, weather)
    field, value = RESIDUAL_ACTIONS[index]
    if field is not None:
        setattr(devices, field, value)
    return devices
```

## А.4 Оцінка корекцій у копіях симулятора

Файл: smart_house/ml/planning.py. Учитель використовується для підготовки навчальних міток. Фрагмент є точним витягом поточної функції; він не виконується у вебрежимі застосування моделі.

```python
def rollout_scores(physics, previous, weather, reward_fn, horizon=6):
    """Common random numbers across candidates; never mutate source simulator.

    Each candidate repeatedly applies its correction for the short horizon.
    The teacher has simulator/forecast access; the deployed student does not.
    """
    if type(horizon) is not int or horizon < 1 or not weather:
        raise ValueError('Positive horizon and nonempty weather required')
    scores = []
    rules = RuleLayer()
    for action in range(19):
        trial, devices = deepcopy(physics), deepcopy(previous)
        score = 0.0
        for index, snapshot in enumerate(weather[:horizon]):
            chosen = rules.apply(residual_devices(action, trial.state, snapshot), trial.state, snapshot)
            state = trial.step(snapshot, chosen)
            unsafe = state.temperature_c < 5 or state.temperature_c > 32 or state.co2_ppm > 1500 or state.pm25_ugm3 > 55
            score += .99**index * (reward_fn.compute(state, chosen, devices) - 10 * int(unsafe))
            devices = chosen
        scores.append(score)
    return scores
```

У train_planned індекс обирається через max(range(19), key=lambda a: (scores[a], -a)). За рівності переважає менший індекс. Збереження selection.json відбувається до compare_controllers на test. Мережа навчається функцією imitate через cross_entropy; виходи є логітами політики.

## А.5 Спільний сервіс аналізу вимірювань

Файл: smart_house/analytics/ingestion.py. У першому витягу пропущено цикл рекомендацій; місце скорочення позначено коментарем. Поле distilled перемикає пояснення на найбільший логіт політики. Для інших підтримуваних checkpoint збережено повідомлення про Q-оцінку.

```python
def analyze_text(self, filename, text):
    states, report = inspect_readings(text, Path(filename).suffix)
    if not report['valid']:
        raise ValidationError(report)
    with self._lock:
        agent = self._load()
        distilled = getattr(agent, 'checkpoint_metadata', {}).get(
            'method') == 'counterfactual_policy_distillation_v1'
        # Далі: select_action, decode_action, RuleLayer,
        # формування recommendations і пояснень.

```

Метод import_text повторно використовує analyze_text і зберігає результат через History. Вебрежим не викликає планувальник і не перенавчає модель. Для цілісності пакета зберігаються digest checkpoint, валідаційний звіт і фактичні режими до та після правил.

## А.6 Кеш і запис ручного тесту

Файл: smart_house/analytics/lab.py.

```python
    def predict(self, body):
        started = time.perf_counter()
        model_id = body.get('model_id', 'model-0')
        try:
            reading = body.get('reading')
            if not isinstance(reading, dict):
                raise ValueError('reading must be an object of sensor values')
            # No arbitrary files or checkpoints can be supplied by the browser.
            service = self._service(model_id)
            text = encode(reading)
            key = (service._digest, text)
            with self.lock:
                hit = key in self.cache
                if hit:
                    result = self.cache[key]
                    self.cache.move_to_end(key)
                else:
                    result = service.analyze_text('playground.json', text)
                    self.cache[key] = result
                    while len(self.cache) > 128:
                        self.cache.popitem(last=False)
            run_id = self._record('predict', model_id, started, result, hit)
            return {**result, 'run_id': run_id, 'cache_hit': hit,
                    'latency_ms': (time.perf_counter()-started)*1000,
                    'cost': None, 'cost_note': 'Local CPU inference; no paid API. Energy is measured in simulation only.'}
        except Exception as exc:
            self._record('predict', str(model_id), started, {}, error=str(exc))
            raise
```

## А.7 Фонове порівняння та регресії

Файл: smart_house/analytics/lab.py.

```python
    def _work(self, ident, request):
        started = time.perf_counter()
        try:
            self._update(ident, 'running')
            options = {k: request[k] for k in ('episodes', 'steps', 'seed')}
            results = []
            for model_id in request['model_ids']:
                # The worker owns a separate agent: a long simulation must not lock
                # the playground's cached agent or mutate its evaluation state.
                service = ImportService(self.models[model_id], self.history)
                agent = service._load()
                result = compare_controllers(agent, **options)
                result.update({'model_id': model_id, 'model_name': self.models[model_id].name,
                               'model_sha256': service._digest})
                checks = self._safety_checks(service)
                results.append(result | {'checks': checks})
            regression = []
            if request['reference_job']:
                previous = self.job(request['reference_job'])['result']['models']
                # Each candidate is compared to the first (explicit reference) model of that job.
                baseline = previous[0]['summary']['model']
                for result in results:
                    metrics = result['summary']['model']
                    regression.append({'model_id': result['model_id'], 'reference_sha256': previous[0]['model_sha256'],
                                       'checks': [
                                           {'name': 'unsafe_steps ≤ reference', 'passed': metrics['unsafe_steps'] <= baseline['unsafe_steps']},
                                           {'name': 'comfort ≥ reference − 1 percentage point', 'passed': metrics['avg_comfort'] >= baseline['avg_comfort'] - 1},
                                           {'name': 'energy ≤ reference + 5%', 'passed': metrics['energy_kwh'] <= baseline['energy_kwh'] * 1.05 + 1e-9}]})
            payload = {'options': options, 'models': results, 'regression': regression,
                       'duration_ms': (time.perf_counter()-started)*1000}
            run_id = self._record('evaluation', ','.join(request['model_ids']), started, payload)
            self._update(ident, 'completed', payload | {'run_id': run_id})
        except Exception as exc:
            LOGGER.exception('Background model evaluation failed')
            self._record('evaluation', ','.join(request['model_ids']), started, {}, error=str(exc))
            self._update(ident, 'failed', error='Evaluation failed; check server log and selected checkpoint')
```

# ДОДАТОК Б. ГРАФІЧНІ МАТЕРІАЛИ ТА ПОСЛІДОВНІСТЬ ВЗАЄМОДІЙ

## Б.1 UML-діаграма послідовності фонового запуску

На рисунку Б.1 деталізовано обмін повідомленнями. HTTP 202 підтверджує постановку задачі, а не її завершення. Браузер повторює GET; повідомлення опитування на схемі наведено один раз для компактності. Worker оновлює статус незалежно від чергового опитування.

{{FIG:sequence|Рисунок Б.1 — UML-діаграма послідовності порівняння моделей}}

## Б.2 Проєкт каналу фізичних датчиків

Апаратні датчики наявні. До отримання їхнього експорту наведено логічний контракт: фізичний сигнал → показник і timestamp → файл JSON/CSV/TXT → validate-data → ImportService → модель → RuleLayer → історія. Моделі датчиків, топологія підключення, частота та походження записів очікують доповнення. На рисунку Б.2 штрихові зв’язки позначають частину, для якої ще не додано натурний протокол.

{{FIG:sensors|Рисунок Б.2 — Канал сенсорних показників і межа підтвердженої реалізації}}

## Б.3 Редаговані джерела графіки

Компонентна схема, UML класів, pipeline, модель мережі, навчання, стани jobs, схема БД і сенсорний канал мають джерела DOT та SVG. Діаграма послідовності й дослідницькі графіки мають SVG. PNG вставлено у документ для стабільного відображення; змінити композицію можна через відповідне векторне джерело в пакеті.

# ДОДАТОК В. КОНТРАКТ ВИМІРЮВАНЬ ТА СЕМАНТИКА ДІЙ

## В.1 Мінімальний сенсорний запис

```json
{
  "timestamp": "2026-10-26T12:00:00+02:00",
  "temperature_c": 22.0,
  "humidity_pct": 50.0,
  "co2_ppm": 800,
  "total_occupants": 2
}
```

Це приклад формату, а не запис фактичного фізичного вимірювання. CSV використовує ті самі назви полів у заголовку й десяткову крапку. Порожня необов’язкова клітинка задає типове значення. Відсутні сенсорні параметри потрібно описувати у звіті, а не приховувати як виміряні.

## В.2 Повна карта 19 корекцій

Таблиця В.1 — Семантика action_index для residual_v1

| Індекс | Поле | Значення | Зміст |
| --- | --- | --- | --- |
| 0 | без заміни | — | залишити детерміноване рішення |
| 1 | hvac_mode | 0 | замінити один режим; далі RuleLayer |
| 2 | hvac_mode | 1 | замінити один режим; далі RuleLayer |
| 3 | hvac_mode | 2 | замінити один режим; далі RuleLayer |
| 4 | hvac_mode | 3 | замінити один режим; далі RuleLayer |
| 5 | hvac_mode | 4 | замінити один режим; далі RuleLayer |
| 6 | ventilation | 0 | замінити один режим; далі RuleLayer |
| 7 | ventilation | 1 | замінити один режим; далі RuleLayer |
| 8 | ventilation | 2 | замінити один режим; далі RuleLayer |
| 9 | ventilation | 3 | замінити один режим; далі RuleLayer |
| 10 | air_purifier | 0 | замінити один режим; далі RuleLayer |
| 11 | air_purifier | 1 | замінити один режим; далі RuleLayer |
| 12 | air_purifier | 2 | замінити один режим; далі RuleLayer |
| 13 | air_purifier | 3 | замінити один режим; далі RuleLayer |
| 14 | humidifier_mode | 0 | замінити один режим; далі RuleLayer |
| 15 | humidifier_mode | 1 | замінити один режим; далі RuleLayer |
| 16 | humidifier_mode | 2 | замінити один режим; далі RuleLayer |
| 17 | cleaner_bot | 0 | замінити один режим; далі RuleLayer |
| 18 | cleaner_bot | 1 | замінити один режим; далі RuleLayer |

## В.3 Режими пристроїв

HVAC: 0 — вимкнено; 1 — слабкий обігрів; 2 — сильний обігрів; 3 — слабке охолодження; 4 — сильне охолодження. Вентиляція й очищувач: рівні 0–3. Зволожувач: 0 — вимкнено, 1 — зволоження, 2 — осушення. Робот: 0 — зупинено, 1 — прибирання. Це програмні режими симулятора, а не електричні команди конкретного обладнання.

Після декодування правила можуть змінити режим. Тому action_index сам по собі не визначає остаточну фізичну дію. Для аудиту потрібно зберігати model_devices, recommended_devices і explanations. Для residual_v1 частина рішень успадкована від детермінованого вчителя.

# ДОДАТОК Г. ПЕРВИННІ ДАНІ СИМУЛЯЦІЙНОГО ДОСЛІДЖЕННЯ

## Г.1 Початкові умови контрольного тестового набору

Наведено перші 20 test-сценаріїв із базовим seed 61000. Це синтетичні значення, а не записи фізичних датчиків. Дати є модельним часом 2026 року та не задають календар фактичних вимірювань. Повні початкові умови всіх 60 сценаріїв збережено в evaluation.json.

Таблиця Г.1 — Початкові умови test seed 61000

| № | Початок UTC | T, °C | RH, % | CO₂, ppm | PM2.5 | Мешканців |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-01-20 18:00 | 24,01 | 74,40 | 1693,2 | 12,65 | 0 |
| 2 | 2026-06-24 00:00 | 20,95 | 28,76 | 475,9 | 19,89 | 2 |
| 3 | 2026-11-21 06:00 | 20,97 | 70,74 | 1323,2 | 15,49 | 1 |
| 4 | 2026-04-18 18:00 | 25,39 | 75,94 | 1147,2 | 4,22 | 3 |
| 5 | 2026-09-11 12:00 | 28,76 | 56,14 | 1958,1 | 47,66 | 3 |
| 6 | 2026-02-18 00:00 | 22,73 | 48,21 | 1658,6 | 42,14 | 0 |
| 7 | 2026-07-20 12:00 | 18,42 | 28,74 | 652,1 | 27,65 | 4 |
| 8 | 2026-12-18 06:00 | 27,37 | 78,17 | 1877,8 | 22,81 | 2 |
| 9 | 2026-05-19 12:00 | 23,56 | 40,24 | 1291,4 | 21,60 | 2 |
| 10 | 2026-10-09 12:00 | 23,58 | 69,52 | 2112,4 | 23,20 | 0 |
| 11 | 2026-03-16 06:00 | 16,75 | 25,55 | 1445,9 | 61,20 | 4 |
| 12 | 2026-08-20 00:00 | 21,79 | 70,62 | 947,0 | 23,84 | 1 |
| 13 | 2026-01-04 18:00 | 27,40 | 28,06 | 1098,8 | 20,58 | 1 |
| 14 | 2026-06-13 12:00 | 16,07 | 55,66 | 674,9 | 35,64 | 1 |
| 15 | 2026-11-13 00:00 | 16,19 | 33,50 | 2180,5 | 34,90 | 4 |
| 16 | 2026-04-22 06:00 | 24,34 | 38,41 | 1033,8 | 35,69 | 4 |
| 17 | 2026-09-06 12:00 | 17,52 | 56,84 | 895,4 | 9,70 | 4 |
| 18 | 2026-02-18 18:00 | 23,43 | 28,17 | 1940,9 | 18,73 | 4 |
| 19 | 2026-07-25 12:00 | 26,35 | 53,34 | 2165,3 | 62,15 | 0 |
| 20 | 2026-12-21 18:00 | 26,34 | 48,26 | 1071,7 | 15,95 | 3 |

## Г.2 Парні результати контрольного набору

Таблиця Г.2 — Модель / детермінована система, test seed 61000

| № | Комфорт, % | Енергія, кВт·год | Винагорода | Кроки поза межами |
| --- | --- | --- | --- | --- |
| 1 | 71,938 / 63,393 | 29,053 / 27,350 | 16,787 / -65,794 | 6 / 8 |
| 2 | 80,093 / 91,327 | 18,648 / 27,033 | 129,992 / 229,661 | 0 / 0 |
| 3 | 79,203 / 88,525 | 28,272 / 27,090 | 108,847 / 197,715 | 0 / 0 |
| 4 | 86,864 / 87,031 | 14,646 / 19,069 | 204,527 / 202,186 | 0 / 0 |
| 5 | 78,431 / 76,642 | 28,067 / 31,889 | 73,550 / 36,035 | 14 / 22 |
| 6 | 88,640 / 62,510 | 35,430 / 27,080 | 172,028 / -64,384 | 5 / 5 |
| 7 | 77,921 / 88,339 | 22,093 / 28,203 | 114,813 / 205,747 | 0 / 0 |
| 8 | 83,826 / 80,801 | 32,048 / 30,283 | 118,759 / 81,967 | 14 / 18 |
| 9 | 82,916 / 90,408 | 19,731 / 25,312 | 157,148 / 222,675 | 0 / 0 |
| 10 | 94,644 / 93,521 | 11,097 / 10,775 | 198,128 / 186,328 | 21 / 22 |
| 11 | 82,032 / 80,759 | 35,913 / 23,753 | 64,725 / 65,398 | 27 / 27 |
| 12 | 96,641 / 94,924 | 14,683 / 18,804 | 277,591 / 259,872 | 0 / 0 |
| 13 | 85,747 / 82,894 | 28,984 / 26,767 | 165,876 / 141,916 | 0 / 0 |
| 14 | 90,997 / 92,088 | 17,904 / 22,235 | 230,614 / 236,931 | 0 / 0 |
| 15 | 76,316 / 72,450 | 33,978 / 25,422 | -32,299 / -64,189 | 44 / 46 |
| 16 | 87,932 / 87,038 | 17,753 / 20,322 | 210,168 / 197,996 | 0 / 1 |
| 17 | 87,190 / 88,916 | 26,884 / 26,646 | 193,370 / 210,131 | 0 / 0 |
| 18 | 59,114 / 26,839 | 46,840 / 31,157 | -148,777 / -452,705 | 26 / 31 |
| 19 | 89,673 / 89,709 | 12,423 / 14,175 | 163,210 / 148,860 | 18 / 24 |
| 20 | 89,057 / 69,690 | 36,648 / 29,286 | 201,928 / 28,407 | 0 / 0 |

## Г.3 Розподіл навчальних міток та остаточні команди

Таблиця Г.3 — Кількість міток учителя для канонічного навчального seed 42

| Індекс корекції | Кількість |
| --- | --- |
| 0 | 21 |
| 1 | 446 |
| 2 | 0 |
| 3 | 313 |
| 4 | 0 |
| 5 | 9 |
| 6 | 282 |
| 7 | 37 |
| 8 | 4 |
| 9 | 82 |
| 10 | 50 |
| 11 | 2 |
| 12 | 3 |
| 13 | 57 |
| 14 | 122 |
| 15 | 1 |
| 16 | 0 |
| 17 | 1 |
| 18 | 298 |

Загалом 1728 міток, з яких лише 21 має індекс 0. Це розподіл учителя на train. Частка відмінних остаточних команд виміряна окремо на test: 2130/2880 та 2190/2880 для seed 71000 і 81000. Збіг чи відмінність індексів не підміняє оцінювання наслідків.

## Г.4 Метадані й відтворення

Підсумковий checkpoint SHA256: {{digest}}. Код: {{commit}}. Канонічний навчальний seed 42, best_epoch=30, validation seed 41000; test seeds 61000, 71000, 81000. Семантика дій residual_v1, метод counterfactual_policy_distillation_v1, звітна винагорода comfort_safety_v2. Підсумковий аналіз охоплює 60 парних добових сценаріїв по 144 кроки.

Файл evaluation.json є повним звітом моделі; aggregate.json і per-episode.csv містять похідні узагальнення та неокруглені пари. training-seed-42/43/44.json і selection-seed-42/43/44.json зберігають окремі навчальні запуски. measurements.json містить нові 40 некешованих і 40 кешованих сервісних викликів та сім перевірок checkpoint. benchmark.json містить 20 обчислювальних замірів учня й учителя. tests.log фіксує успіх 28 тестів 26.10.2026.

Електронний пакет також містить diagnostic-baseline-clone.json із негативною діагностикою початкової політики. Він відокремлений від підсумкового evaluation.json і не використовується для розрахунку нових середніх. Натурних вимірювань у наведених таблицях немає.
