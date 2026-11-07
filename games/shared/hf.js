// Hands-free input for the arcade.
// The EEG controller (controller/mappings/arcade.json) turns gestures into OS input: head turns / the
// eye tracker move the cursor, jaw clench = left mouse button (a short clench clicks, a clench hold
// keeps the button down until the clench ends), double blink = right click, eyebrow raise = Space.
// This file turns that input back into gestures for the games, and adds look-and-hold selection
// (dwell) for eyes-only play.
//
// Gestures passed to HF.on(fn) as fn(name, info):
//   press          left button down (info.target, info.x, info.y)  - jaw clench starts
//   release        left button up (info.held, info.target under the cursor)
//   jaw_clench     short clench (after release)
//   clench_hold    button held past HOLD_MS
//   double_blink   right click
//   eyebrow_raise  Space
//   escape         Esc (operator)
(() => {
  "use strict";

  const EMBEDDED = new URLSearchParams(location.search).has("embed");
  if (EMBEDDED) document.documentElement.classList.add("embedded");

  const HOLD_MS = 300;       // the controller presses the button down only after its own hold_s
  const GRACE_MS = 250;      // gaze slips off a target shorter than this pause the ring, not reset it
  const DEFAULTS = { input: "eeg", dwellMs: 1000, menuDwellMs: 1500, menuDwell: true, sound: true, showLog: true };

  const settings = Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem("hf.settings") || "{}"));
  const listeners = [];
  const mouse = { x: innerWidth / 2, y: innerHeight / 2, known: false, down: false };

  const HF = {
    settings,
    mouse,
    activate: null,      // fn(el): a [data-dwell="game"] element was pressed (EEG) or looked at (eyes only)
    canDwell: null,      // fn(el) -> bool: may this game element be selected by looking at it right now?
    saveSettings() { localStorage.setItem("hf.settings", JSON.stringify(settings)); },
    eyesOnly() { return settings.input === "gaze"; },
    on(fn) { listeners.push(fn); },
  };

  function emit(name, info = {}) {
    for (const fn of listeners) fn(name, info);
  }

  // ------------------------------------------------------------------ sound
  let actx = null;
  function ensureAudio() {
    if (!actx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (Ctx) actx = new Ctx();
    }
    if (actx && actx.state === "suspended") actx.resume();
  }
  function ready() { return settings.sound && actx && actx.state === "running"; }

  HF.tone = (freq, ms, type = "sine", vol = 0.18) => {
    if (!ready()) return;
    const t = actx.currentTime, end = t + ms / 1000;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = type;
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + 0.012);
    g.gain.setValueAtTime(vol, Math.max(t + 0.02, end - 0.05));
    g.gain.exponentialRampToValueAtTime(0.0001, end);
    o.connect(g).connect(actx.destination);
    o.start(t);
    o.stop(end + 0.02);
  };

  // Tone that lasts until the returned stop() is called (e.g. while a clench is held).
  HF.toneHold = (freq, type = "sine", vol = 0.18) => {
    if (!ready()) return () => {};
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = type;
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + 0.012);
    o.connect(g).connect(actx.destination);
    o.start(t);
    let stopped = false;
    return () => {
      if (stopped) return;
      stopped = true;
      const n = actx.currentTime;
      g.gain.cancelScheduledValues(n);
      g.gain.setValueAtTime(Math.max(g.gain.value, 0.0001), n);
      g.gain.exponentialRampToValueAtTime(0.0001, n + 0.06);
      o.stop(n + 0.08);
    };
  };

  HF.sfx = {
    good: () => { HF.tone(660, 110, "triangle"); setTimeout(() => HF.tone(990, 150, "triangle"), 100); },
    bad: () => HF.tone(110, 500, "sawtooth", 0.14),
    click: () => HF.tone(520, 60, "triangle", 0.1),
    win: () => [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => HF.tone(f, 160, "triangle"), i * 120)),
  };

  // ------------------------------------------------------------------ scores
  const best = JSON.parse(localStorage.getItem("hf.best") || "{}");
  HF.best = (key) => best[key] || 0;
  HF.recordBest = (key, score) => {
    if (score <= (best[key] || 0)) return false;
    best[key] = score;
    localStorage.setItem("hf.best", JSON.stringify(best));
    return true;
  };
  HF.resetBest = () => {
    for (const k of Object.keys(best)) delete best[k];
    localStorage.removeItem("hf.best");
  };

  // ------------------------------------------------------------------ detected-input log
  let logEl = null;
  HF.log = (text, ignored = false) => {
    if (!settings.showLog) return;
    if (!logEl) { logEl = document.createElement("div"); logEl.id = "hf-log"; document.body.appendChild(logEl); }
    const el = document.createElement("div");
    el.textContent = text;
    if (ignored) el.className = "ignored";
    logEl.prepend(el);
    while (logEl.children.length > 5) logEl.lastChild.remove();
    setTimeout(() => el.classList.add("fade"), 2000);
    setTimeout(() => el.remove(), 2800);
  };

  HF.goArcade = () => { location.href = "../index.html"; };

  // ------------------------------------------------------------------ top bar
  HF.topbar = ({ title, hint = "", onMenu = null, arcade = true }) => {
    const bar = document.createElement("header");
    bar.className = "hf-topbar";
    bar.innerHTML = `
      ${arcade && !EMBEDDED ? '<button class="hf-btn" data-dwell="ui" data-hf="arcade">&larr; Arcade</button>' : ""}
      <h1>${title}</h1>
      <span class="hint"></span>
      <span class="status"></span>
      <button class="hf-btn" data-hf="input" title="Operator: switch input mode"></button>
      ${onMenu ? '<button class="hf-btn" data-dwell="ui" data-hf="menu">Menu</button>' : ""}`;
    document.body.prepend(bar);
    const q = (s) => bar.querySelector(s);
    const renderInput = () => {
      const b = q('[data-hf="input"]');
      b.textContent = HF.eyesOnly() ? "Input: Eyes only" : "Input: EEG + Gaze";
      b.classList.toggle("on", HF.eyesOnly());
    };
    renderInput();
    q('[data-hf="input"]').addEventListener("click", () => {
      settings.input = HF.eyesOnly() ? "eeg" : "gaze";
      HF.saveSettings();
      renderInput();
      emit("inputchange");
    });
    if (arcade && !EMBEDDED) q('[data-hf="arcade"]').addEventListener("click", HF.goArcade);
    if (onMenu) q('[data-hf="menu"]').addEventListener("click", () => onMenu());
    const api = {
      hint: (t) => { q(".hint").textContent = t; },
      status: (t) => { q(".status").textContent = t; },
    };
    api.hint(hint);
    return api;
  };

  // ------------------------------------------------------------------ quick menu / result screen
  let overlay = null, menuOnClose = null;
  HF.menuOpen = () => !!(overlay && overlay.classList.contains("open"));

  // items: [{label, fn}]. Picking a tile closes the menu and runs fn; closing any other way
  // (eyebrow raise, Esc) runs onClose (usually "resume").
  HF.openMenu = ({ title = "Menu", sub = "", items = [], onClose = null }) => {
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.className = "hf-overlay";
      document.body.appendChild(overlay);
    }
    overlay.innerHTML = `<h2>${title}<small>${sub}</small></h2><div class="hf-tiles"></div>`;
    const tiles = overlay.querySelector(".hf-tiles");
    for (const it of items) {
      const b = document.createElement("button");
      b.className = "hf-tile";
      b.dataset.dwell = "ui";
      b.textContent = it.label;
      b.addEventListener("click", () => { closeMenu(false); it.fn(); });
      tiles.appendChild(b);
    }
    menuOnClose = onClose;
    overlay.classList.add("open");
  };

  function closeMenu(runOnClose = true) {
    if (!HF.menuOpen()) return;
    overlay.classList.remove("open");
    const cb = menuOnClose;
    menuOnClose = null;
    if (runOnClose && cb) cb();
  }
  HF.closeMenu = closeMenu;

  // ------------------------------------------------------------------ raw input -> gestures
  const clench = { down: false, held: false, timer: null };

  function gameTarget(t) {
    return t && t.closest ? t.closest('[data-dwell="game"]') : null;
  }
  function inChrome(t) {
    return t && t.closest && (t.closest(".hf-topbar") || t.closest(".hf-overlay"));
  }

  document.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX; mouse.y = e.clientY; mouse.known = true;
  });

  document.addEventListener("mousedown", (e) => {
    ensureAudio();
    if (e.button === 2) {
      e.preventDefault();
      if (HF.eyesOnly()) return HF.log("Double blink (right click) - ignored, eyes only", true);
      HF.log("Double blink (right click)");
      if (!HF.menuOpen()) emit("double_blink");
      return;
    }
    if (e.button !== 0) return;
    mouse.down = true;
    if (inChrome(e.target)) return;            // topbar / menu buttons handle their own click
    if (HF.eyesOnly()) return HF.log("Jaw clench (left click) - ignored, eyes only", true);
    clench.down = true;
    clench.held = false;
    clearTimeout(clench.timer);
    clench.timer = setTimeout(() => {
      clench.held = true;
      HF.log("Clench hold (left button held)");
      emit("clench_hold");
    }, HOLD_MS);
    emit("press", { target: e.target, x: e.clientX, y: e.clientY });
    const g = gameTarget(e.target);
    if (g && HF.activate) HF.activate(g, "press");
  });

  document.addEventListener("mouseup", (e) => {
    if (e.button !== 0) return;
    mouse.down = false;
    if (!clench.down) return;
    clench.down = false;
    clearTimeout(clench.timer);
    const target = document.elementFromPoint(e.clientX, e.clientY);
    emit("release", { held: clench.held, target, x: e.clientX, y: e.clientY });
    if (!clench.held) {
      HF.log("Jaw clench (left click)");
      emit("jaw_clench");
    }
  });

  document.addEventListener("contextmenu", (e) => e.preventDefault());

  // A focused button would also "click" on Space, which is the eyebrow-raise key.
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest("button");
    if (b) b.blur();
  });

  document.addEventListener("keydown", (e) => {
    ensureAudio();
    if (e.target.closest && e.target.closest("input, select, textarea")) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (e.repeat) return;
      if (HF.eyesOnly()) return HF.log("Eyebrow raise (Space) - ignored, eyes only", true);
      HF.log("Eyebrow raise (Space)");
      if (HF.menuOpen()) closeMenu();
      else emit("eyebrow_raise");
    } else if (e.code === "Escape") {
      e.preventDefault();
      if (HF.menuOpen()) closeMenu();
      else emit("escape");
    }
  });
  document.addEventListener("keyup", (e) => { if (e.code === "Space") e.preventDefault(); });

  // ------------------------------------------------------------------ look-and-hold (dwell)
  const dwell = { el: null, start: 0, lostAt: 0, last: 0 };

  function dwellMs(el) {
    const kind = el.dataset.dwell;
    if (kind === "ui") return settings.menuDwell || HF.eyesOnly() ? settings.menuDwellMs : 0;
    if (kind === "game") {
      if (!HF.eyesOnly() || HF.menuOpen()) return 0;
      if (HF.canDwell && !HF.canDwell(el)) return 0;
      return settings.dwellMs;
    }
    return 0;
  }

  function dwellTarget() {
    if (!mouse.known) return null;
    const hit = document.elementFromPoint(mouse.x, mouse.y);
    const el = hit && hit.closest && hit.closest("[data-dwell]");
    return el && dwellMs(el) ? el : null;
  }

  function setProgress(el, p) {
    if (!el) return;
    el.style.setProperty("--p", p);
    el.classList.toggle("dwelling", p > 0);
  }

  function tick(now) {
    const dt = now - (dwell.last || now);
    dwell.last = now;
    const el = dwellTarget();
    if (el !== dwell.el) {
      if (dwell.el && document.contains(dwell.el) && dwellMs(dwell.el)) {
        if (!dwell.lostAt) dwell.lostAt = now;
        if (now - dwell.lostAt < GRACE_MS) { dwell.start += dt; requestAnimationFrame(tick); return; }
      }
      setProgress(dwell.el, 0);
      dwell.el = el;
      dwell.start = now;
      dwell.lostAt = 0;
    } else {
      dwell.lostAt = 0;
    }
    if (el) {
      const p = Math.max(0, Math.min(1, (now - dwell.start) / dwellMs(el)));
      setProgress(el, p);
      if (p >= 1) {
        setProgress(el, 0);
        dwell.start = now + 300;      // short pause, then the same target can be selected again
        if (el.dataset.dwell === "game") { if (HF.activate) HF.activate(el, "dwell"); }
        else el.click();
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  window.HF = HF;
})();
