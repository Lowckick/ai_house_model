const $ = (id) => document.getElementById(id);
// Theme colour from the active CSS tokens (light or dark), so charts follow the theme.
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
let selected = null,
  current = null,
  currentEvaluation = null,
  busy = false;
function node(tag, text) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  return n;
}
function message(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : "ok";
}
function controls() {
  for (const id of ["validate", "evaluate", "refresh", "file"]) $(id).disabled = busy;
  $("import").disabled = busy || !selected;
}
async function task(fn) {
  if (busy) return;
  busy = true;
  controls();
  try {
    await fn();
  } catch (e) {
    message(e.message, true);
    if (e.report) showValidation(e.report);
  } finally {
    busy = false;
    controls();
  }
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
  );
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || "Помилка запиту");
    error.report = data.validation;
    throw error;
  }
  return data;
}
function download(name, data) {
  const a = node("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function showValidation(report) {
  const box = $("validation");
  box.replaceChildren(
    node(
      "p",
      `${report.valid ? "Файл коректний" : "Файл містить помилки"} · записів: ${report.count} · помилок: ${report.errors.length} · попереджень: ${report.warnings.length}`,
    ),
  );
  for (const [label, list] of [
    ["Помилка", report.errors],
    ["Попередження", report.warnings],
  ]) {
    const ul = node("ul");
    for (const item of list.slice(0, 100))
      ul.append(node("li", `${label}${item.record ? " · запис " + item.record : ""}: ${item.message}`));
    box.append(ul);
    if (list.length > 100)
      box.append(node("p", "Показано перші 100 повідомлень. Повний звіт можна завантажити."));
  }
  const b = node("button", "Завантажити звіт перевірки");
  b.className = "secondary";
  b.onclick = () => download("validation.json", report);
  box.append(b);
  if (report.preview && report.preview.length) {
    const detail = node("details"),
      wrap = node("div"),
      table = node("table"),
      header = node("tr");
    detail.append(node("summary", "Попередній перегляд: перші 10 коректних записів"));
    wrap.className = "scroll";
    for (const name of ["Час", "°C", "Вологість, %", "CO₂, ppm"]) header.append(node("th", name));
    table.append(header);
    for (const r of report.preview) {
      const row = node("tr");
      for (const v of [r.timestamp, r.temperature_c, r.humidity_pct, r.co2_ppm])
        row.append(node("td", String(v)));
      table.append(row);
    }
    wrap.append(table);
    detail.append(wrap);
    box.append(detail);
  }
}
$("file").onchange = () => {
  selected = null;
  controls();
  $("validation").replaceChildren();
};
$("validate").onclick = () =>
  task(async () => {
    selected = null;
    const file = $("file").files[0];
    if (!file) throw new Error("Оберіть файл.");
    if (file.size > 5 * 1024 * 1024) throw new Error("Файл перевищує 5 МіБ.");
    const text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
    const body = { filename: file.name, text };
    const report = await api("/api/validate", body);
    showValidation(report);
    if (report.valid) selected = body;
    message(
      report.valid
        ? "Перевірку завершено. Можна зберегти дані."
        : "Виправте помилки й оберіть файл повторно.",
      !report.valid,
    );
  });
$("import").onclick = () =>
  task(async () => {
    const result = await api("/api/import", selected);
    showImport(result);
    selected = null;
    await refresh();
    message("Вимірювання та рекомендації збережено в історії.");
  });
$("sample").onclick = () =>
  download("readings.json", [
    {
      timestamp: "2026-10-23T08:00:00+03:00",
      temperature_c: 22,
      humidity_pct: 45,
      co2_ppm: 1800,
      total_occupants: 2,
    },
  ]);
const hvac = ["вимкнено", "слабкий обігрів", "сильний обігрів", "слабке охолодження", "сильне охолодження"];
function deviceText(d) {
  return `HVAC: ${hvac[d.hvac_mode]}; вентиляція: ${d.ventilation}; очищувач: ${d.air_purifier}; вологість: ${["вимкнено", "зволоження", "осушення"][d.humidifier_mode]}; робот: ${d.cleaner_bot ? "працює" : "вимкнено"}`;
}
const fieldNames = {
  hvac_mode: "HVAC",
  ventilation: "вентиляція",
  air_purifier: "очищувач",
  humidifier_mode: "зволожувач",
  cleaner_bot: "робот",
};
function correctionText(c) {
  if (c.field === null) return "без корекції (рішення контролера)";
  const value =
    c.field === "hvac_mode"
      ? hvac[c.value]
      : c.field === "humidifier_mode"
        ? ["вимкнено", "зволоження", "осушення"][c.value]
        : c.field === "cleaner_bot"
          ? ["стоїть", "прибирає"][c.value]
          : "рівень " + c.value;
  return `${fieldNames[c.field] || c.field} → ${value}`;
}
function correctionsNode(list, margin) {
  const chosen = list.find((c) => c.chosen) || list[0],
    top = list[0];
  const box = node("details");
  box.append(
    node(
      "summary",
      `Модель: «${correctionText(chosen)}» (${(chosen.confidence * 100).toFixed(0)}% впевненості)`,
    ),
  );
  const ul = node("ul");
  for (const c of list)
    ul.append(
      node("li", `${c.chosen ? "✓ " : ""}${correctionText(c)} — ${(c.confidence * 100).toFixed(1)}%`),
    );
  box.append(ul);
  if (chosen !== top)
    box.append(
      node(
        "p",
        `Найімовірніша корекція «${correctionText(top)}» не набрала порогу впевненості${margin ? ` (${margin})` : ""}, тому залишено рішення контролера.`,
      ),
    );
  box.append(node("p", "Ймовірності мережі серед 19 корекцій; правила безпеки застосовуються після вибору."));
  return box;
}
function showImport(result) {
  current = result;
  $("results").hidden = false;
  $("count").textContent = result.count + " записів";
  $("rows").replaceChildren();
  for (const r of result.readings.slice(0, 200)) {
    const tr = node("tr");
    tr.append(
      node("td", r.timestamp),
      node("td", `${r.indoor.temperature_c} / ${r.indoor.humidity_pct} / ${r.indoor.co2_ppm}`),
      node("td", deviceText(r.recommended_devices)),
      node("td", deviceText(r.deterministic_devices)),
    );
    const cell = node("td");
    for (const e of r.explanations) {
      const detail = node("details");
      detail.append(
        node(
          "summary",
          e.source === "model"
            ? "Рішення моделі"
            : `${e.source === "teacher" ? "Контролер · " : ""}${e.stage} · ${e.field}`,
        ),
        node("p", e.reason),
      );
      if (e.source !== "model")
        detail.append(node("pre", JSON.stringify(e.before) + " → " + JSON.stringify(e.after)));
      cell.append(detail);
    }
    if (r.top_corrections && r.top_corrections.length)
      cell.append(correctionsNode(r.top_corrections, r.correction_margin));
    const detail = node("details");
    detail.append(
      node("summary", "Повні режими пристроїв"),
      node("pre", JSON.stringify(r.recommended_devices, null, 2)),
    );
    cell.append(detail);
    tr.append(cell);
    $("rows").append(tr);
  }
  $("table-note").textContent = result.count > 200 ? "Показано перші 200 записів. Повні дані — у JSON." : "";
  draw();
}
function draw() {
  if (!current) return;
  const c = $("chart"),
    ctx = c.getContext("2d"),
    metric = $("metric").value,
    all = current.readings.map((r) => r.indoor[metric]);
  const w = c.clientWidth,
    h = 230,
    dpr = window.devicePixelRatio || 1;
  c.width = w * dpr;
  c.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const low = Math.min(...all),
    high = Math.max(...all),
    range = high - low || 1;
  ctx.font = "13px system-ui";
  ctx.fillStyle = css("--text");
  ctx.fillText(`${$("metric").selectedOptions[0].text}: ${low.toFixed(1)} … ${high.toFixed(1)}`, 5, 20);
  ctx.strokeStyle = css("--accent");
  ctx.lineWidth = 2;
  ctx.beginPath();
  all.forEach((v, i) => {
    const x = 20 + (i * (w - 40)) / Math.max(1, all.length - 1),
      y = h - 35 - ((v - low) / range) * (h - 80);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  if (all.length === 1) {
    ctx.beginPath();
    ctx.arc(20, h - 35, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillText("Запис 1", 5, h - 10);
  ctx.fillText("Запис " + all.length, w - 100, h - 10);
}
window.addEventListener("resize", draw);
$("metric").onchange = draw;
$("download").onclick = () => current && download(`import-${current.id}.json`, current);
function showEvaluation(data) {
  currentEvaluation = data;
  $("download-evaluation").hidden = false;
  const box = $("evaluation");
  box.replaceChildren();
  const table = node("table"),
    head = node("tr");
  for (const h of ["Метрика", "Модель + правила", "Детермінована", "Різниця (модель − правила)"])
    head.append(node("th", h));
  table.append(head);
  for (const [key, label] of [
    ["avg_comfort", "Комфорт, % ↑"],
    ["energy_kwh", "Енергія, кВт·год ↓"],
    ["cost_uah", "Вартість, грн ↓"],
    ["total_reward", "Сумарна винагорода ↑"],
    ["unsafe_steps", "Кроки поза безпечними межами ↓"],
  ]) {
    const row = node("tr");
    row.append(node("td", label));
    for (const v of [
      data.summary.model[key],
      data.summary.deterministic[key],
      data.delta_model_minus_deterministic[key],
    ])
      row.append(node("td", v === undefined ? "—" : v.toFixed(3)));
    table.append(row);
  }
  box.append(
    table,
    node("p", `Середнє за ${data.episodes} епізодами × ${data.steps_per_episode} кроків; seed ${data.seed}.`),
    node(
      "p",
      "Вищі комфорт/винагорода та нижчі енергія/небезпечні кроки кращі. Одна метрика не визначає переможця. Повні результати кожного сценарію та розкид різниць — у JSON. Результати описують симулятор, а не доведену економію в реальному будинку.",
    ),
  );
}
$("evaluate").onclick = () =>
  task(async () => {
    message("Виконується порівняння…");
    const data = await api("/api/evaluate", {
      episodes: Number($("episodes").value),
      steps: Number($("steps").value),
      seed: Number($("seed").value),
    });
    showEvaluation(data);
    await refresh();
    message("Порівняння завершено й збережено.");
  });
$("download-evaluation").onclick = () => currentEvaluation && download("evaluation.json", currentEvaluation);
async function refresh() {
  const data = await api("/api/history");
  for (const [id, items, prefix, show] of [
    ["history", data.imports, "/api/imports/", showImport],
    ["evaluations", data.evaluations, "/api/evaluations/", showEvaluation],
  ]) {
    const box = $(id);
    box.replaceChildren();
    if (!items.length) box.append(node("p", "Ще немає записів."));
    for (const item of items) {
      const row = node("p"),
        b = node(
          "button",
          item.filename ? `#${item.id} · ${item.filename} · ${item.count} записів` : `Оцінка #${item.id}`,
        );
      b.className = "secondary";
      b.onclick = () =>
        task(async () => {
          show(await api(prefix + item.id));
          message("Відкрито збережений результат.");
        });
      row.append(b, node("small", " " + new Date(item.created_at).toLocaleString("uk-UA")));
      box.append(row);
    }
  }
}
$("refresh").onclick = () => task(refresh);
task(refresh);
let labPrediction = null,
  labJob = null,
  labTimer = null;
const labNames = new Map();
function labPreset() {
  const presets = {
    normal: {},
    cold: { temperature_c: 0 },
    co2: { co2_ppm: 2000, occupant_activity: "sleeping" },
    smoke: { pm25_ugm3: 80 },
  };
  $("lab-reading").value = JSON.stringify(
    {
      timestamp: "2026-10-26T12:00:00+02:00",
      temperature_c: 22,
      humidity_pct: 50,
      total_occupants: 2,
      ...presets[$("lab-preset").value],
    },
    null,
    2,
  );
}
$("lab-preset").onchange = labPreset;
labPreset();
async function labAction(button, fn) {
  button.disabled = true;
  try {
    await fn();
  } catch (e) {
    message(e.message, true);
    if (e.report) showValidation(e.report);
  } finally {
    button.disabled = false;
  }
}
function labTable(headings, rows) {
  const table = node("table"),
    head = node("tr");
  for (const h of headings) head.append(node("th", h));
  table.append(head);
  for (const values of rows) {
    const row = node("tr");
    for (const value of values) row.append(node("td", String(value ?? "—")));
    table.append(row);
  }
  return table;
}
function labChecks(checks) {
  const list = node("ul");
  for (const check of checks) {
    const item = node(
      "li",
      (check.passed ? "✓ " : "✗ ") + check.name + (check.field ? ` · ${check.field}: ${check.actual}` : ""),
    );
    item.className = check.passed ? "pass" : "fail";
    list.append(item);
  }
  return list;
}
$("lab-predict").onclick = () =>
  labAction($("lab-predict"), async () => {
    const reading = JSON.parse($("lab-reading").value);
    const data = await api("/api/lab/predict", { model_id: $("lab-model").value, reading });
    labPrediction = data;
    const r = data.readings[0];
    $("lab-output").replaceChildren(
      node(
        "p",
        `${data.latency_ms.toFixed(2)} мс · ${data.cache_hit ? "відповідь із кешу" : "обчислено моделлю"} · дія ${r.action_index} · ${r.action_mode}`,
      ),
      node("p", deviceText(r.recommended_devices)),
    );
    if (r.top_corrections && r.top_corrections.length)
      $("lab-output").append(correctionsNode(r.top_corrections, r.correction_margin));
    const detail = node("details");
    detail.append(node("summary", "Повне рішення та пояснення"), node("pre", JSON.stringify(r, null, 2)));
    $("lab-output").append(detail);
    $("lab-export").hidden = false;
    $("lab-rating-box").hidden = false;
    await labStats();
    message("Тест завершено, результат збережено в журналі.");
  });
$("lab-export").onclick = () => labPrediction && download("model-test.json", labPrediction);
$("lab-rate").onclick = () =>
  labAction($("lab-rate"), async () => {
    if (!labPrediction) return;
    await api("/api/lab/rating", { run_id: labPrediction.run_id, rating: Number($("lab-rating").value) });
    await labStats();
    message("Оцінку збережено.");
  });
function showLabJob(job) {
  labJob = job;
  const labels = {
    queued: "У черзі",
    running: "Виконується",
    completed: "Завершено",
    failed: "Помилка",
    interrupted: "Перервано після перезапуску сервера",
  };
  $("lab-job-status").textContent =
    `${labels[job.status] || job.status} · ${job.id.slice(0, 8)}${job.error ? " · " + job.error : ""}`;
  $("lab-job-result").replaceChildren();
  $("lab-export-job").hidden = !job.result;
  if (!job.result) return;
  const result = job.result;
  const rows = result.models.map((m) => [
    labNames.get(m.model_id) || m.model_id,
    m.model_sha256.slice(0, 12),
    ...["avg_comfort", "energy_kwh", "cost_uah", "total_reward", "unsafe_steps"].map((k) =>
      m.summary.model[k] === undefined ? "—" : m.summary.model[k].toFixed(3),
    ),
  ]);
  const baseline = result.models[0].summary.deterministic;
  rows.push([
    "Детермінована система",
    "—",
    ...["avg_comfort", "energy_kwh", "cost_uah", "total_reward", "unsafe_steps"].map((k) =>
      baseline[k] === undefined ? "—" : baseline[k].toFixed(3),
    ),
  ]);
  $("lab-job-result").append(
    labTable(
      ["Модель", "SHA256", "Комфорт % ↑", "кВт·год ↓", "Грн ↓", "Винагорода ↑", "Небезпечні кроки ↓"],
      rows,
    ),
    node(
      "p",
      `Тривалість: ${(result.duration_ms / 1000).toFixed(2)} с. Середні результати за ${result.options.episodes} епізодами × ${result.options.steps} кроків, seed ${result.options.seed}.`,
    ),
  );
  for (const model of result.models)
    $("lab-job-result").append(
      node("h3", `Перевірки: ${labNames.get(model.model_id) || model.model_id}`),
      labChecks(model.checks),
    );
  for (const regression of result.regression)
    $("lab-job-result").append(
      node("h3", `Регресія: ${labNames.get(regression.model_id) || regression.model_id}`),
      labChecks(regression.checks),
    );
  $("lab-job-result").append(
    node(
      "p",
      "Результати описують симулятор. Нижче енергоспоживання має оцінюватися разом із комфортом і безпекою.",
    ),
  );
}
async function watchLabJob(id) {
  clearTimeout(labTimer);
  try {
    const job = await api("/api/lab/jobs/" + encodeURIComponent(id));
    showLabJob(job);
    if (["queued", "running"].includes(job.status)) labTimer = setTimeout(() => watchLabJob(id), 1200);
    else {
      await Promise.all([labJobs(), labStats()]);
    }
  } catch (e) {
    message(e.message, true);
    $("lab-job-status").textContent = "Не вдалося оновити стан. Відкрийте запуск зі списку повторно.";
  }
}
$("lab-submit").onclick = () =>
  labAction($("lab-submit"), async () => {
    const model_ids = Array.from($("lab-models").querySelectorAll("input:checked")).map((n) => n.value);
    const job = await api("/api/lab/jobs", {
      model_ids,
      episodes: Number($("lab-episodes").value),
      steps: Number($("lab-steps").value),
      seed: Number($("lab-seed").value),
      reference_job: $("lab-reference").value || null,
    });
    await labJobs();
    watchLabJob(job.id);
    message("Завдання додано до черги. Можна продовжувати роботу зі сторінкою.");
  });
$("lab-export-job").onclick = () => labJob && download(`comparison-${labJob.id}.json`, labJob);
async function labJobs() {
  const data = await api("/api/lab/jobs");
  $("lab-jobs").replaceChildren();
  const reference = $("lab-reference"),
    selected = reference.value;
  reference.replaceChildren(new Option("Без порівняння з попереднім запуском", ""));
  for (const job of data.jobs) {
    const b = node(
      "button",
      `${job.id.slice(0, 8)} · ${job.status} · ${new Date(job.created_at).toLocaleString("uk-UA")}`,
    );
    b.className = "secondary";
    b.onclick = () => watchLabJob(job.id);
    const p = node("p");
    p.append(b);
    $("lab-jobs").append(p);
    if (job.status === "completed")
      reference.append(
        new Option(job.id.slice(0, 8) + " · " + new Date(job.created_at).toLocaleString("uk-UA"), job.id),
      );
  }
  if (Array.from(reference.options).some((o) => o.value === selected)) reference.value = selected;
  return data.jobs;
}
async function labStats() {
  const data = await api("/api/lab/stats"),
    t = data.totals;
  $("lab-stats").replaceChildren();
  for (const [label, value] of [
    ["Запити", t.requests],
    ["Помилки", t.errors],
    ["Влучання кешу", t.cache_hits],
    ["Середня оцінка", t.avg_rating?.toFixed(2) ?? "—"],
  ]) {
    const card = node("div");
    card.className = "card";
    card.append(node("strong", String(value)), node("span", label));
    $("lab-stats").append(card);
  }
  $("lab-breakdown").replaceChildren(
    labTable(
      ["Модель", "Тип", "Запити", "Середній час, мс"],
      data.by_model.map((r) => [
        labNames.get(r.model_id) || r.model_id,
        r.kind,
        r.requests,
        r.avg_latency_ms.toFixed(2),
      ]),
    ),
  );
  $("lab-log").replaceChildren(
    labTable(
      ["Час", "Тип", "Модель", "мс", "Кеш", "Оцінка", "Помилка"],
      data.recent.map((r) => [
        new Date(r.created_at).toLocaleString("uk-UA"),
        r.kind,
        labNames.get(r.model_id) || r.model_id,
        r.latency_ms.toFixed(2),
        r.cache_hit ? "так" : "ні",
        r.rating,
        r.error,
      ]),
    ),
  );
}
$("lab-refresh").onclick = () => labAction($("lab-refresh"), () => Promise.all([labStats(), labJobs()]));
async function initLab() {
  try {
    const data = await api("/api/lab/models");
    for (const model of data.models) {
      labNames.set(model.id, model.name);
      $("lab-model").append(new Option(model.name, model.id));
      const label = node("label"),
        input = node("input");
      input.type = "checkbox";
      input.value = model.id;
      input.checked = model.default;
      label.append(input, document.createTextNode(model.name));
      $("lab-models").append(label);
    }
    const [jobs] = await Promise.all([labJobs(), labStats()]);
    const active = jobs.find((j) => ["queued", "running"].includes(j.status));
    if (active) watchLabJob(active.id);
  } catch (e) {
    message(e.message, true);
  }
}
initLab();

const COMFORT = {
  temperature_c: [19, 25],
  humidity_pct: [35, 65],
  co2_ppm: [null, 1000],
  pm25_ugm3: [null, 35],
};
let liveData = null,
  liveReport = null,
  liveSeen = new Set(),
  liveTimer = null,
  liveBusy = false;
function liveCard(label, value) {
  const c = node("div");
  c.className = "card";
  c.append(node("strong", value), node("span", label));
  return c;
}
function pct(v) {
  return v === null || v === undefined ? "—" : (v * 100).toFixed(0) + "%";
}
function drawLive() {
  if (!liveData || !liveData.readings.length) return;
  const c = $("live-chart"),
    ctx = c.getContext("2d"),
    metric = $("live-metric").value,
    rows = liveData.readings;
  const w = c.clientWidth,
    h = 230,
    dpr = window.devicePixelRatio || 1;
  c.width = w * dpr;
  c.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const times = rows.map((r) => Date.parse(r.timestamp)),
    values = rows.map((r) => r.indoor[metric]),
    [lowBand, highBand] = COMFORT[metric];
  const t0 = Math.min(...times),
    t1 = Math.max(...times),
    span = t1 - t0 || 1;
  let low = Math.min(...values),
    high = Math.max(...values);
  if (lowBand !== null) low = Math.min(low, lowBand);
  if (highBand !== null) high = Math.max(high, highBand);
  const range = high - low || 1;
  const X = (t) => 44 + ((t - t0) / span) * (w - 64),
    Y = (v) => h - 30 - ((v - low) / range) * (h - 70);
  ctx.fillStyle = css("--band");
  const top = Y(highBand === null ? high : highBand),
    bottom = Y(lowBand === null ? low : lowBand);
  ctx.fillRect(44, top, w - 64, bottom - top);
  ctx.font = "13px system-ui";
  ctx.fillStyle = css("--text");
  ctx.fillText(
    `${$("live-metric").selectedOptions[0].text}: ${Math.min(...values).toFixed(1)} … ${Math.max(...values).toFixed(1)} · зелене — комфорт`,
    5,
    18,
  );
  ctx.fillStyle = css("--muted");
  ctx.fillText(new Date(t0).toLocaleString("uk-UA"), 40, h - 8);
  const end = new Date(t1).toLocaleString("uk-UA");
  ctx.fillText(end, w - 20 - ctx.measureText(end).width, h - 8);
  ctx.strokeStyle = css("--accent");
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = X(times[i]),
      y = Y(values[i]);
    const gap = i && times[i] - times[i - 1] > 30 * 60000;
    i && !gap ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = css("--error");
  rows.forEach((r, i) => {
    if ((r.new_alerts || []).some((a) => a.metric === metric)) {
      ctx.beginPath();
      ctx.arc(X(times[i]), Y(values[i]), 4, 0, 7);
      ctx.fill();
    }
  });
  ctx.fillStyle = css("--muted");
  ctx.font = "11px system-ui";
  for (const v of [low, high, lowBand, highBand]) {
    if (v === null) continue;
    ctx.fillText(Number(v.toFixed(1)).toString(), 2, Y(v) + 4);
  }
}
function showLatest(r) {
  const box = $("live-latest");
  box.replaceChildren();
  const i = r.indoor;
  box.append(
    node(
      "p",
      `${new Date(r.timestamp).toLocaleString("uk-UA")} · ${i.temperature_c} °C · ${i.humidity_pct}% · CO₂ ${i.co2_ppm} ppm · PM2.5 ${i.pm25_ugm3}`,
    ),
    node("p", "Рекомендація: " + deviceText(r.recommended_devices)),
  );
  if (r.top_corrections && r.top_corrections.length)
    box.append(correctionsNode(r.top_corrections, r.correction_margin));
  const alerts = r.alerts || [];
  box.append(
    node("p", alerts.length ? "Тривоги: " + alerts.map((a) => a.message).join("; ") : "Тривог немає."),
  );
  box.lastChild.className = alerts.some((a) => a.level === "critical") ? "fail" : alerts.length ? "" : "pass";
}
function showLiveReport(rep) {
  const box = $("live-report");
  box.replaceChildren();
  if (!rep.readings) {
    $("live-report-note").textContent = "";
    return;
  }
  const o = rep.observed,
    m = rep.recommendations;
  box.append(
    liveCard("показників", String(rep.readings)),
    liveCard("час у комфорті (усі показники)", pct(o.time_all_in_comfort)),
    liveCard("CO₂ ≤ 1000 ppm", pct(o.time_in_comfort.co2_ppm)),
    liveCard("19–25 °C", pct(o.time_in_comfort.temperature_c)),
    liveCard(
      "виходів за комфорт покрито порадою",
      m.out_of_comfort_conditions ? `${m.addressed_by_recommendation}/${m.out_of_comfort_conditions}` : "—",
    ),
    liveCard("кВт·год, якби виконували поради", m.estimated_energy_kwh_if_followed.toFixed(2)),
    liveCard("грн за тарифом", m.estimated_cost_uah_if_followed.toFixed(2)),
  );
  $("live-report-note").textContent =
    `Покрито ${rep.covered_hours} год, розривів понад 30 хв: ${rep.gaps_over_30_min}. Порад, відмінних від простого контролера: ${m.differs_from_simple_controller}. Пристрої не перемикались — оцінка енергії модельна.`;
  const list = $("live-alerts");
  list.replaceChildren();
  if (!rep.alerts.length) list.append(node("p", "Тривог за період не було."));
  else {
    const ul = node("ul");
    for (const a of rep.alerts.slice().reverse())
      ul.append(
        node(
          "li",
          `${new Date(a.timestamp).toLocaleString("uk-UA")} · ${a.level === "critical" ? "критично" : "увага"} · ${a.message}`,
        ),
      );
    list.append(ul);
  }
}
function notifyNew(rows) {
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  for (const r of rows)
    for (const a of r.new_alerts || []) {
      const key = `${$("live-sensor").value}|${r.timestamp}|${a.metric}|${a.level}`;
      if (!liveSeen.has(key)) {
        liveSeen.add(key);
        if (liveData)
          new Notification(`${$("live-sensor").value}: ${a.level === "critical" ? "критично" : "увага"}`, {
            body: a.message,
          });
      }
    }
}
async function loadLive() {
  if (liveBusy) return;
  liveBusy = true;
  try {
    const sensors = (await api("/api/live")).sensors,
      select = $("live-sensor"),
      chosen = select.value;
    select.replaceChildren(
      ...sensors.map((x) => {
        const o = node("option", `${x.sensor_id} (${x.count})`);
        o.value = x.sensor_id;
        return o;
      }),
    );
    if (sensors.some((x) => x.sensor_id === chosen)) select.value = chosen;
    $("live-empty").hidden = sensors.length > 0;
    $("live-body").hidden = !sensors.length;
    if (!sensors.length) return;
    const sensor = encodeURIComponent(select.value),
      hours = $("live-hours").value;
    const first = liveData === null || liveData.sensor_id !== select.value;
    const data = await api(`/api/live/${sensor}?hours=${hours}&limit=5000`);
    if (first)
      for (const r of data.readings)
        for (const a of r.new_alerts || [])
          liveSeen.add(`${select.value}|${r.timestamp}|${a.metric}|${a.level}`);
    liveData = data;
    notifyNew(data.readings);
    liveReport = await api(`/api/live/${sensor}/report?hours=${hours}`);
    if (data.readings.length) showLatest(data.readings[data.readings.length - 1]);
    else $("live-latest").replaceChildren(node("p", "За вибраний період показників немає."));
    showLiveReport(liveReport);
    drawLive();
  } catch (e) {
    message("Датчики: " + e.message, true);
  } finally {
    liveBusy = false;
  }
}
function liveSchedule() {
  clearInterval(liveTimer);
  if ($("live-auto").checked) liveTimer = setInterval(loadLive, 60000);
}
$("live-refresh").onclick = loadLive;
$("live-sensor").onchange = () => {
  liveData = null;
  loadLive();
};
$("live-hours").onchange = loadLive;
$("live-metric").onchange = drawLive;
$("live-auto").onchange = liveSchedule;
$("live-notify").onclick = async () => {
  if (!("Notification" in window)) {
    message("Цей браузер не підтримує сповіщення.", true);
    return;
  }
  const result = await Notification.requestPermission();
  message(
    result === "granted"
      ? "Сповіщення увімкнено: нові тривоги з’являтимуться, поки сторінка відкрита."
      : "Сповіщення не дозволено в браузері.",
    result !== "granted",
  );
};
window.addEventListener("resize", drawLive);
loadLive();
liveSchedule();

// ---------------------------------------------------------------------------
// Tabs: one panel at a time, remembered in the URL hash (#sensors, #files, …).
const TABS = ["sensors", "files", "compare", "lab", "history"];
function showTab(name, focus = false) {
  if (!TABS.includes(name)) name = "sensors";
  for (const tab of TABS) {
    const button = $("tab-button-" + tab),
      selected = tab === name;
    $("tab-" + tab).hidden = !selected;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  }
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  // Canvases have no size while hidden: redraw when their panel appears.
  if (name === "sensors") drawLive();
  if (name === "files") draw();
}
TABS.forEach((tab, index) => {
  const button = $("tab-button-" + tab);
  button.onclick = () => showTab(tab);
  button.onkeydown = (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (step) showTab(TABS[(index + step + TABS.length) % TABS.length], true);
    if (event.key === "Home") showTab(TABS[0], true);
    if (event.key === "End") showTab(TABS[TABS.length - 1], true);
  };
});
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
showTab(location.hash.slice(1));

// ---------------------------------------------------------------------------
// Theme: automatic (system), light or dark; the choice is a per-browser convenience.
const THEMES = ["auto", "light", "dark"],
  THEME_LABELS = { auto: "Тема: авто", light: "Тема: світла", dark: "Тема: темна" };
function applyTheme(theme) {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  $("theme-toggle").textContent = THEME_LABELS[theme];
  drawLive();
  draw();
}
let theme = "auto";
try {
  theme = localStorage.getItem("smart-house-theme") || "auto";
} catch (e) {}
if (!THEMES.includes(theme)) theme = "auto";
applyTheme(theme);
$("theme-toggle").onclick = () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  try {
    localStorage.setItem("smart-house-theme", theme);
  } catch (e) {}
  applyTheme(theme);
};
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  drawLive();
  draw();
});
