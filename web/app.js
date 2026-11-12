"use strict";

const languages = {
  de: "Alemán",
  en: "Inglés",
  es: "Español",
  ro: "Rumano",
};

const samplePhrases = {
  es: [
    "La música conecta personas de culturas muy distintas sin necesidad de explicar cada palabra.",
    "Hoy probamos nuestro traductor multilingüe entrenado con corpus paralelos en cuatro idiomas.",
    "Aunque el camino sea largo, cada paso nos acerca un poco más a comprender otras voces.",
  ],
  en: [
    "Language is not just a set of rules, but a window into how another culture sees the world.",
    "Even after visiting three other languages, a clear idea should find its way back home.",
  ],
  ro: [
    "Fiecare limbă păstrează o perspectivă unică asupra lumii și a istoriei sale.",
    "Traducerea automată funcționează cel mai bine atunci când datele sunt curate și echilibrate.",
  ],
  de: [
    "Jede Sprache eröffnet einen neuen Blickwinkel auf dieselbe gemeinsame Idee.",
    "Ein einziges Modell übersetzt direkt zwischen vier Sprachen in zwölf Richtungen.",
  ],
};

const state = {
  mode: "directa",
  busy: false,
  modelLoaded: false,
  sampleIndex: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const form = $("#translateForm");
const sourceText = $("#sourceText");
const sourceLang = $("#sourceLang");
const targetLang = $("#targetLang");
const languageControls = $("#languageControls");
const routeSummary = $("#routeSummary");
const routePreview = $("#routePreview");
const modeNote = $("#modeNote");
const resultEmpty = $("#resultEmpty");
const resultContent = $("#resultContent");
const translateButton = $("#translateButton");
const modelChip = $("#modelChip");
const modelStatus = $("#modelStatus");
const systemModel = $("#systemModel");
const clearButton = $("#clearButton");
const sampleButton = $("#sampleButton");

function option(code, prefix = "") {
  const item = document.createElement("option");
  item.value = code;
  item.textContent = `${prefix}${languages[code]}`;
  return item;
}

function populateLanguages() {
  Object.keys(languages).forEach((code) => sourceLang.append(option(code)));
  sourceLang.value = "es";
  updateTargets("en");
}

function updateTargets(preferred) {
  const available = Object.keys(languages).filter((code) => code !== sourceLang.value);
  targetLang.replaceChildren(...available.map((code) => option(code)));
  targetLang.value = available.includes(preferred) ? preferred : available[0];
  buildChainSelects();
  renderRoute();
}

function buildChainSelects(order) {
  const others = Object.keys(languages).filter((code) => code !== sourceLang.value);
  const selected =
    order && [...order].sort().join() === [...others].sort().join() ? order : others;
  const elements = [];

  selected.forEach((value, index) => {
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Paso ${index + 1} de la cadena`);
    others.forEach((code) => select.append(option(code, `${index + 1}. `)));
    select.value = value;
    select.addEventListener("change", () => normalizeChain(select));
    elements.push(select);

    if (index < selected.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "route-arrow";
      arrow.textContent = "→";
      arrow.setAttribute("aria-hidden", "true");
      elements.push(arrow);
    }
  });

  $("#chainSelects").replaceChildren(...elements);
}

function normalizeChain(changed) {
  const selects = $$("#chainSelects select");
  const duplicates = selects.filter((item) => item !== changed && item.value === changed.value);
  if (duplicates.length) {
    const used = new Set(selects.filter((item) => item !== duplicates[0]).map((item) => item.value));
    duplicates[0].value = Object.keys(languages).find(
      (code) => code !== sourceLang.value && !used.has(code),
    );
  }
  renderRoute();
}

function chainOrder() {
  return $$("#chainSelects select").map((select) => select.value);
}

function pill(text, cls = "", title = "") {
  const span = document.createElement("span");
  span.className = `route-pill ${cls}`.trim();
  span.textContent = text;
  if (title) span.title = title;
  return span;
}

function arrow() {
  const span = document.createElement("span");
  span.className = "route-arrow";
  span.textContent = "→";
  span.setAttribute("aria-hidden", "true");
  return span;
}

function renderRoute() {
  const origin = sourceLang.value;
  if (state.mode === "todas") {
    const others = Object.keys(languages)
      .filter((code) => code !== origin)
      .map((code) => code.toUpperCase())
      .join("·");
    routePreview.replaceChildren(
      pill(origin, "is-origin", languages[origin]),
      arrow(),
      pill(`6 órdenes (${others})`, "is-text"),
      arrow(),
      pill(origin, "is-return", `Regreso a ${languages[origin]}`),
    );
    return;
  }

  const route = [origin, ...chainOrder(), origin];
  const nodes = [];
  route.forEach((code, idx) => {
    const cls = idx === 0 ? "is-origin" : idx === route.length - 1 ? "is-return" : "";
    nodes.push(pill(code, cls, languages[code]));
    if (idx < route.length - 1) nodes.push(arrow());
  });
  routePreview.replaceChildren(...nodes);
}

function setMode(mode) {
  state.mode = mode;
  languageControls.dataset.mode = mode;

  $$(".mode-button").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });

  const isDirect = mode === "directa";
  const isChain = mode === "cadena";

  $("#directControl").hidden = !isDirect;
  $("#swapLanguages").hidden = !isDirect;
  routeSummary.hidden = isDirect;
  $("#chainControl").hidden = !isChain;
  $("#sourceLangLabel").textContent = isDirect ? "Origen" : "Origen y regreso";

  modeNote.textContent = isDirect
    ? "Traducción directa entre dos idiomas."
    : isChain
      ? "Recorre los otros tres idiomas en orden y vuelve al idioma inicial."
      : "Compara las seis rutas posibles de ida y vuelta.";

  renderRoute();
}

function setModelStatus(info) {
  const status = info?.estado || "sin_cargar";
  const visual =
    status === "listo"
      ? "ready"
      : status === "cargando"
        ? "loading"
        : status === "error"
          ? "error"
          : "idle";
  modelChip.dataset.state = visual;
  state.modelLoaded = status === "listo";

  if (status === "listo") {
    const source = info.modelo_base ? "Modelo base" : "Modelo listo";
    const dev = String(info.dispositivo || "").toUpperCase();
    modelStatus.textContent = dev ? `${source} · ${dev}` : source;
    systemModel.textContent = info.fuente || "M2M100 afinado";
  } else if (status === "cargando") {
    modelStatus.textContent = "Cargando…";
  } else if (status === "error") {
    modelStatus.textContent = "Error en modelo";
  } else {
    modelStatus.textContent = "Comprobar modelo";
  }
}

function setBusy(busy, message = "Traduciendo…") {
  state.busy = busy;
  translateButton.disabled = busy;
  translateButton.querySelector("span").textContent = busy ? message : "Traducir";
  if (busy && !state.modelLoaded) {
    setModelStatus({ estado: "cargando" });
  }
}

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"]/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char],
  );
}

function routeLabel(steps) {
  return steps.map((step) => step.idioma.toUpperCase()).join(" → ");
}

function isExactMatch(a, b) {
  return String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
}

function stepsHtml(steps) {
  const firstText = steps[0]?.texto || "";
  return `<div class="step-list">${steps
    .map((step, index) => {
      const isLast = index === steps.length - 1;
      const exact = isLast && isExactMatch(firstText, step.texto);
      const badge = isLast
        ? `<span class="match-tag ${exact ? "is-exact" : ""}">${exact ? "Exacta" : "Variación"}</span>`
        : "";
      const label =
        index === 0
          ? `Original · ${languages[step.idioma]}`
          : isLast
            ? `Regreso · ${languages[step.idioma]}`
            : `Paso ${index} · ${languages[step.idioma]}`;
      return `
    <div class="step">
      <span class="step-code">${escapeHtml(step.idioma)}</span>
      <small>${escapeHtml(label)} ${badge}</small>
      <p>${escapeHtml(step.texto)}</p>
    </div>`;
    })
    .join("")}</div>`;
}

function formatMeta(data, elapsedMs) {
  const parts = [];
  if (data.modelo?.dispositivo) parts.push(String(data.modelo.dispositivo).toUpperCase());
  if (data.perfil) parts.push(`calidad ${data.perfil}`);
  if (typeof elapsedMs === "number") parts.push(`${(elapsedMs / 1000).toFixed(2)} s`);
  return escapeHtml(parts.join(" · "));
}

function renderResult(data, elapsedMs) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;

  if (data.modo === "directa") {
    resultContent.innerHTML = `
      <div class="result-head">
        <p class="result-kicker">${escapeHtml(languages[data.origen])} → ${escapeHtml(languages[data.destino])}</p>
        <div class="result-actions">
          <button class="panel-action" id="useAsInputBtn" type="button" title="Usar esta traducción como nueva entrada">Invertir y usar</button>
          <button class="copy-button" type="button">Copiar</button>
        </div>
      </div>
      <p class="translation-output">${escapeHtml(data.traduccion)}</p>
      <div class="result-meta">
        <span>M2M100 afinado</span>
        <span>${formatMeta(data, elapsedMs)}</span>
      </div>`;

    $(".copy-button").addEventListener("click", () => copyText(data.traduccion));
    $("#useAsInputBtn").addEventListener("click", () => {
      sourceText.value = data.traduccion;
      updateCharCount();
      sourceLang.value = data.destino;
      updateTargets(data.origen);
      showToast("Traducción colocada como entrada.");
    });
    return;
  }

  if (data.modo === "cadena") {
    const last = data.pasos[data.pasos.length - 1];
    const fullChainText = data.pasos
      .map((s) => `[${s.idioma.toUpperCase()}] ${s.texto}`)
      .join("\n");

    resultContent.innerHTML = `
      <div class="result-head">
        <p class="result-kicker">${routeLabel(data.pasos)}</p>
        <div class="result-actions">
          <button class="panel-action" id="copyAllStepsBtn" type="button">Copiar todo</button>
          <button class="copy-button" id="copyFinalBtn" type="button">Copiar regreso</button>
        </div>
      </div>
      ${stepsHtml(data.pasos)}
      <div class="result-meta">
        <span>4 saltos completados</span>
        <span>${formatMeta(data, elapsedMs)}</span>
      </div>`;

    $("#copyFinalBtn").addEventListener("click", () => copyText(last.texto));
    $("#copyAllStepsBtn").addEventListener("click", () => copyText(fullChainText));
    return;
  }

  const allRoutesText = data.cadenas
    .map(
      (c, i) =>
        `Ruta ${i + 1} (${routeLabel(c.pasos)}):\n` +
        c.pasos.map((s) => `  [${s.idioma.toUpperCase()}] ${s.texto}`).join("\n"),
    )
    .join("\n\n");

  resultContent.innerHTML = `
    <div class="result-head">
      <p class="result-kicker">6 rutas desde ${escapeHtml(languages[data.origen])}</p>
      <div class="result-actions">
        <button class="panel-action" id="toggleAllBtn" type="button">Desplegar todas</button>
        <button class="copy-button" id="copyRoutesBtn" type="button">Copiar todo</button>
      </div>
    </div>
    <div class="chains-list">
      ${data.cadenas
        .map((chain, index) => {
          const ret = chain.pasos[chain.pasos.length - 1].texto;
          return `<details class="chain-group" ${index === 0 ? "open" : ""}>
            <summary>
              <span class="summary-route">${routeLabel(chain.pasos)}</span>
              <span class="summary-preview">${escapeHtml(ret)}</span>
            </summary>
            ${stepsHtml(chain.pasos)}
          </details>`;
        })
        .join("")}
    </div>
    <div class="result-meta">
      <span>6 permutaciones</span>
      <span>${formatMeta(data, elapsedMs)}</span>
    </div>`;

  let expanded = false;
  $("#toggleAllBtn").addEventListener("click", (ev) => {
    expanded = !expanded;
    $$(".chain-group").forEach((el) => {
      el.open = expanded;
    });
    ev.currentTarget.textContent = expanded ? "Contraer todas" : "Desplegar todas";
  });
  $("#copyRoutesBtn").addEventListener("click", () => copyText(allRoutesText));
}

function renderError(message) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;
  resultContent.innerHTML = `<div class="error-card"><h2>No se pudo traducir</h2><p>${escapeHtml(message)}</p></div>`;
}

async function api(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("El servidor devolvió una respuesta inválida.");
  }
  if (!response.ok) {
    const err = new Error(data.error || `Error HTTP ${response.status}`);
    err.modelo = data.modelo;
    throw err;
  }
  return data;
}

async function submitTranslation(event) {
  if (event) event.preventDefault();
  if (state.busy) return;
  const payload = {
    modo: state.mode,
    texto: sourceText.value,
    origen: sourceLang.value,
    perfil: $("#generationProfile").value,
  };
  if (state.mode === "directa") payload.destino = targetLang.value;
  if (state.mode === "cadena") payload.orden = chainOrder();

  setBusy(true, state.mode === "todas" ? "Calculando rutas…" : "Traduciendo…");
  const start = performance.now();
  try {
    const data = await api("/api/traducir", payload);
    const elapsedMs = performance.now() - start;
    renderResult(data, elapsedMs);
    setModelStatus(data.modelo);
  } catch (error) {
    renderError(error.message);
    if (error.modelo) {
      setModelStatus(error.modelo);
    } else {
      await refreshStatus();
    }
  } finally {
    setBusy(false);
  }
}

async function loadModel() {
  if (state.busy) return;
  setBusy(true, "Comprobando…");
  setModelStatus({ estado: "cargando" });
  try {
    const data = await api("/api/cargar");
    setModelStatus(data.modelo);
    showToast("Modelo listo.");
  } catch (error) {
    setModelStatus({ estado: "error" });
    renderError(error.message);
  } finally {
    setBusy(false);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (response.ok) setModelStatus((await response.json()).modelo);
  } catch {
    setModelStatus({ estado: "error" });
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast("Texto copiado.");
  } catch {
    showToast("No se pudo copiar.");
  }
}

let toastTimer;
function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.hidden = true;
  }, 2400);
}

function updateCharCount() {
  const len = sourceText.value.length;
  $("#charCount").textContent = `${len.toLocaleString("es-MX")} / 5,000`;
  clearButton.hidden = len === 0;
}

populateLanguages();
setMode("directa");
refreshStatus();

$$(".mode-button").forEach((button) =>
  button.addEventListener("click", () => setMode(button.dataset.mode)),
);
sourceLang.addEventListener("change", () => updateTargets(targetLang.value));
targetLang.addEventListener("change", renderRoute);
sourceText.addEventListener("input", updateCharCount);

sourceText.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});

sampleButton.addEventListener("click", () => {
  const list = samplePhrases[sourceLang.value] || samplePhrases.es;
  sourceText.value = list[state.sampleIndex % list.length];
  state.sampleIndex += 1;
  updateCharCount();
  sourceText.focus();
});

clearButton.addEventListener("click", () => {
  sourceText.value = "";
  updateCharCount();
  sourceText.focus();
});

$("#swapLanguages").addEventListener("click", () => {
  const previousSource = sourceLang.value;
  sourceLang.value = targetLang.value;
  updateTargets(previousSource);
});

$("#loadModel").addEventListener("click", loadModel);
form.addEventListener("submit", submitTranslation);
