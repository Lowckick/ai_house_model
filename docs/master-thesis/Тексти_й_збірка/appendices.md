# ДОДАТОК А. СТРУКТУРА ПРОГРАМИ Й ВАЖЛИВІ ФРАГМЕНТИ КОДУ

Фрагменти відповідають репозиторію й коміту, зазначеним у списку джерел. Повний код доступний у GitHub. Наведені функції є редагованим текстом, а не зображенням. Довгі рядки можуть переноситися засобами редактора без зміни їхнього програмного змісту.

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
smart_house/ml/trainer.py            навчальний цикл і validation
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

## А.4 Формування Double DQN цілі

Файл: smart_house/ml/agent.py.

```python
    def select_action(self, state: np.ndarray) -> int:
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

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
```

## А.5 Спільний сервіс аналізу вимірювань

Файл: smart_house/analytics/ingestion.py.

```python
    def analyze_text(self, filename, text):
        states, report = inspect_readings(text, Path(filename).suffix)
        if not report['valid']:
            raise ValidationError(report)
        with self._lock:
            agent = self._load()
            results = []
            for indoor in states:
                weather = assumed_weather(indoor)
                action = agent.select_action(encode_state(indoor, weather, DeviceState()))
                raw = agent.decode_action(action, DeviceState(), indoor, weather)
                recommended, changes = RuleLayer().apply_with_explanations(raw, indoor, weather)
                results.append({'timestamp': indoor.timestamp.isoformat(), 'indoor': asdict(indoor),
                                'weather': asdict(weather), 'action_index': action, 'action_mode': agent.action_mode,
                                'model_devices': asdict(raw), 'recommended_devices': asdict(recommended),
                                'deterministic_devices': asdict(deterministic_devices(indoor, weather)),
                                'explanations': [
                                    {'source': 'model', 'reason': 'Обрано дію з найбільшою Q-оцінкою моделі (epsilon=0). ' +
                                     ('Політика коригує детермінований контролер; дія 0 залишає його рішення. ' if agent.action_mode == 'residual_v1' else '') +
                                     'Причини нейронної оцінки не інтерпретуються; зміни правил наведено окремо.'},
                                    *(teacher_explanations(indoor, weather) if agent.action_mode == "residual_v1" else []),
                                    *changes]})
            digest = self._digest
        return {'count': len(results), 'validation': report, 'readings': results,
                'model_sha256': digest}

    def import_text(self, filename, text):
        result = self.analyze_text(filename, text)
        ident = self.history.save_import(Path(filename).name, hashlib.sha256(text.encode('utf-8')).hexdigest(),
                                         result['model_sha256'], result['validation'], result['readings'])
        return self.history.get_import(ident)
```

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

## Г.1 Початкові умови сценаріїв

Таблиця Г.1 — Параметри 12 test-сценаріїв; базовий seed 20000

| № | Початок UTC | T, °C | RH, % | CO₂, ppm | PM2.5 | Мешканців* |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-01-03 00:00 | 26.44 | 50.22 | 806.4 | 46.54 | 3 |
| 2 | 2026-06-13 18:00 | 25.12 | 71.52 | 805.7 | 21.48 | 3 |
| 3 | 2026-11-09 12:00 | 24.46 | 36.05 | 1385.2 | 30.17 | 3 |
| 4 | 2026-04-17 18:00 | 16.73 | 45.18 | 1199.8 | 42.52 | 0 |
| 5 | 2026-09-24 00:00 | 18.68 | 35.44 | 1653.7 | 36.54 | 4 |
| 6 | 2026-02-26 00:00 | 25.35 | 44.98 | 1473.8 | 60.87 | 4 |
| 7 | 2026-07-07 12:00 | 18.93 | 56.59 | 1061.5 | 57.99 | 0 |
| 8 | 2026-12-03 06:00 | 18.99 | 34.67 | 531.7 | 62.72 | 3 |
| 9 | 2026-05-15 06:00 | 22.98 | 58.90 | 939.1 | 39.29 | 2 |
| 10 | 2026-10-04 06:00 | 23.47 | 43.69 | 1345.8 | 32.80 | 1 |
| 11 | 2026-03-12 06:00 | 19.75 | 40.02 | 1431.1 | 25.70 | 1 |
| 12 | 2026-08-26 12:00 | 26.94 | 79.08 | 1812.2 | 27.53 | 0 |

*Поле occupants задає параметр сценарію для моделі присутності. Фактична присутність на першому кроці залежить також від початкової години й моделі активності. PM2.5 подано в мкг/м³. Всі значення в цій таблиці синтетичні.

## Г.2 Посценарні результати

Таблиця Г.2 — Результати гібридної політики; детермінований контролер отримав ті самі значення

| № | Комфорт, % | Енергія, кВт·год | Винагорода | Небезпечні кроки |
| --- | --- | --- | --- | --- |
| 1 | 88.5233 | 27.8825 | 203.8453 | 0 |
| 2 | 87.6551 | 26.6050 | 199.2255 | 0 |
| 3 | 83.4594 | 27.2158 | 160.3119 | 0 |
| 4 | 96.0506 | 17.4067 | 266.1969 | 0 |
| 5 | 81.3343 | 21.0375 | 124.4638 | 11 |
| 6 | 45.6720 | 31.4867 | -197.3409 | 3 |
| 7 | 94.7362 | 18.9117 | 248.1939 | 2 |
| 8 | 74.3176 | 27.6458 | 7.5687 | 22 |
| 9 | 89.2997 | 22.5042 | 214.8501 | 0 |
| 10 | 91.9627 | 21.3017 | 234.9574 | 0 |
| 11 | 89.0640 | 26.5458 | 205.4817 | 0 |
| 12 | 92.5068 | 18.2650 | 201.0417 | 11 |

## Г.3 Метадані та перевірка відтворення

Checkpoint SHA256: {{digest}}. Код: {{commit}}. Параметри: episodes=12, steps=144, seed=20000, split=test, epsilon=0. Версія винагороди comfort_safety_v2. Повні scenario seed та результати без округлення містяться в evaluation.json.

Файл measurements.json містить 40 некешованих і 40 кешованих спостережень, один cold-start, параметри середовища та сім правилових перевірок. tests.log містить протокол 26 автоматизованих тестів. Векторні рисунки побудовано з цих даних. Округлення в друкованих таблицях не змінює первинні JSON.

# ДОДАТОК Д. ПРОТОКОЛ ДОДАВАННЯ НАТУРНИХ ДАНИХ

## Д.1 Паспорт вимірювань: очікує заповнення

| Поле | Значення |
| --- | --- |
| Модель і серійний / умовний ID датчика | [ДОДАТИ] |
| Параметри й одиниці | [ДОДАТИ] |
| Місце встановлення | [ДОДАТИ БЕЗ ПРИВАТНОЇ АДРЕСИ] |
| Період і часовий пояс | [ДОДАТИ] |
| Інтервал опитування | [ДОДАТИ] |
| Похибка за паспортом | [ДОДАТИ ДЖЕРЕЛО] |
| Калібрування / контрольний прилад | [ДОДАТИ АБО НЕ ВИКОНУВАЛОСЯ] |
| Пропуски й відмови | [ДОДАТИ] |
| Фактичні дії обладнання | [ДОДАТИ АБО НЕ ФІКСУВАЛИСЯ] |

## Д.2 Послідовність аналізу після отримання даних

1. Зберегти незмінний оригінальний експорт і його SHA256.
2. Зафіксувати джерело, одиниці, часовий пояс, інтервал і правила пропусків.
3. Привести копію даних до контракту без мовчазного видалення некоректних рядків.
4. Виконати validate-data; окремо зберегти помилки й попередження.
5. Побудувати графіки й описову статистику лише фактичних показників.
6. Виконати inference зафіксованого checkpoint із незмінними правилами.
7. Порахувати частоту команд і остаточних корекцій, показати типові й крайні випадки.
8. Якщо є історія дій і наступних станів, окремо погодити аналіз наслідків.
9. Оновити підрозділ 5.8, висновки й анотацію з чітким відокремленням натурних результатів.

## Д.3 Обмеження майбутньої інтерпретації

Сам набір сенсорних показників дозволяє оцінити вхідну якість і рекомендації. Для причинного висновку про економію потрібні зіставні умови, факти виконаних дій і виміряна енергія. Запланований етап не підміняється симуляцією, а відсутній показник не заповнюється вигаданим значенням. До отримання даних підрозділ 5.8 залишається позначеним як незавершений.
