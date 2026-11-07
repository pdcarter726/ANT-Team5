(() => {
  "use strict";

  if (new URLSearchParams(location.search).has("embed")) document.documentElement.classList.add("embedded");

  const B = window.BINDINGS;
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const label = (g) => (B.gestures[g] ? B.gestures[g].label : g);

  // ---------------------------------------------------------------- settings / scores
  const DEFAULTS = {
    padDwellMs: 1000, menuDwellMs: 1500, speed: "normal", headTurnPct: 25,
    saysHeadTurns: true, menuDwell: true, sound: true, showLog: true,
  };
  const settings = Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem("simon.settings") || "{}"));
  const saveSettings = () => localStorage.setItem("simon.settings", JSON.stringify(settings));
  let best = JSON.parse(localStorage.getItem("simon.best") || "{}");

  function recordBest(mode, score) {
    if (score <= (best[mode] || 0)) return false;
    best[mode] = score;
    localStorage.setItem("simon.best", JSON.stringify(best));
    return true;
  }

  // ---------------------------------------------------------------- sound
  let actx = null;
  function ensureAudio() {
    if (!actx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (Ctx) actx = new Ctx();
    }
    if (actx && actx.state === "suspended") actx.resume();
  }
  function tone(freq, ms, type = "sine", vol = 0.2) {
    if (!settings.sound || !actx || actx.state !== "running") return;
    const t = actx.currentTime, end = t + ms / 1000;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = type;
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + 0.015);
    g.gain.setValueAtTime(vol, Math.max(t + 0.02, end - 0.06));
    g.gain.exponentialRampToValueAtTime(0.0001, end);
    o.connect(g).connect(actx.destination);
    o.start(t);
    o.stop(end + 0.02);
  }
  function toneHold(freq, vol = 0.2) {
    if (!settings.sound || !actx || actx.state !== "running") return () => {};
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol, t + 0.015);
    o.connect(g).connect(actx.destination);
    o.start(t);
    return () => {
      const n = actx.currentTime;
      g.gain.cancelScheduledValues(n);
      g.gain.setValueAtTime(Math.max(g.gain.value, 0.0001), n);
      g.gain.exponentialRampToValueAtTime(0.0001, n + 0.06);
      o.stop(n + 0.08);
    };
  }
  const sfx = {
    good: () => { tone(660, 120, "triangle"); setTimeout(() => tone(990, 160, "triangle"), 110); },
    bad: () => tone(110, 600, "sawtooth", 0.15),
    cue: () => tone(523, 120, "triangle", 0.12),
  };

  // ---------------------------------------------------------------- app shell
  const app = { mode: null, overlay: null };
  const MODE_NAMES = { eeg: "EEG + GAZE", gaze: "EYES ONLY", says: "SIMON SAYS" };

  function show(id) {
    $$(".screen").forEach((s) => s.classList.toggle("active", s.id === id));
  }

  function renderBest() {
    $("#best-scores").textContent =
      `Best: Classic ${best.eeg || 0}  |  Look & Hold ${best.gaze || 0}  |  Simon Says ${best.says || 0}`;
  }

  function startMode(mode) {
    app.mode = mode;
    if (mode === "says") {
      show("screen-says");
      says.enter();
    } else {
      show("screen-board");
      $("#screen-board").dataset.mode = mode;
      $("#board-mode").textContent = MODE_NAMES[mode];
      $("#board-hint").textContent = mode === "eeg"
        ? "Aim with head/eyes · jaw clench = press · double blink = replay · eyebrow raise = quick actions"
        : "Hold your gaze on a color to press it · hold on the center for quick actions";
      simon.reset();
    }
  }

  function goMenu() {
    simon.stop();
    says.stop();
    closeOverlay();
    app.mode = null;
    renderBest();
    show("screen-menu");
  }

  // ---------------------------------------------------------------- detected-input log
  const logEl = $("#input-log");
  function logInput(text, ignored = false) {
    if (!settings.showLog) return;
    const el = document.createElement("div");
    el.textContent = text;
    if (ignored) el.className = "ignored";
    logEl.prepend(el);
    while (logEl.children.length > 5) logEl.lastChild.remove();
    setTimeout(() => el.classList.add("fade"), 2200);
    setTimeout(() => el.remove(), 3000);
  }

  // ---------------------------------------------------------------- gesture layer
  // Turns raw browser input (what the EEG app emits) back into gestures.
  const mouse = { x: innerWidth / 2, y: innerHeight / 2, known: false };
  const clench = { down: false, held: false, timer: null };

  function gesture(name) {
    const ignored = app.mode === "gaze";
    const g = B.gestures[name];
    logInput(`${label(name)}  (${g ? g.input : "?"})${ignored ? "  - ignored, eyes only" : ""}`, ignored);
    if (app.mode === "says") says.flashChip(name);
    if (ignored) return;

    if (app.overlay === "actions") {
      if (name === "eyebrow_raise") runAction("resume");
      return;
    }
    if (app.overlay) return;
    if (app.mode === "eeg") {
      if (name === "eyebrow_raise") openActions();
      else if (name === "double_blink") simon.replay();
    } else if (app.mode === "says") {
      says.onGesture(name);
    }
  }

  function mouse1Down(target) {
    if (app.mode !== "eeg" || app.overlay || !target) return;
    const pad = target.closest(".pad");
    if (pad) simon.press(+pad.dataset.i, true);
    else if (target.closest("#hub")) simon.hubAction();
  }

  document.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX; mouse.y = e.clientY; mouse.known = true;
  });

  document.addEventListener("mousedown", (e) => {
    ensureAudio();
    if (e.button === 0) {
      clench.down = true;
      clench.held = false;
      clearTimeout(clench.timer);
      clench.timer = setTimeout(() => { clench.held = true; gesture("clench_hold"); }, B.clenchHoldMs);
      mouse1Down(e.target);
    } else if (e.button === 2) {
      e.preventDefault();
      gesture("double_blink");
    }
  });

  document.addEventListener("mouseup", (e) => {
    if (e.button !== 0) return;
    simon.releaseHold();
    if (!clench.down) return;
    clench.down = false;
    clearTimeout(clench.timer);
    if (!clench.held) gesture("jaw_clench");
  });

  document.addEventListener("contextmenu", (e) => e.preventDefault());

  // A focused button would also "click" on Space, which is the eyebrow-raise key.
  document.addEventListener("click", (e) => {
    if (e.target.closest("button")) e.target.closest("button").blur();
  });

  document.addEventListener("keydown", (e) => {
    ensureAudio();
    if (app.overlay === "settings") {
      if (e.code === "Escape") closeOverlay();
      return;
    }
    if (e.repeat) { if (e.code === "Space") e.preventDefault(); return; }
    if (B.eyebrowKeys.includes(e.code)) {
      e.preventDefault();
      gesture("eyebrow_raise");
      return;
    }
    if (e.code === "Escape") {
      e.preventDefault();
      if (app.overlay === "actions") runAction("resume");
      else if (app.mode) openActions();
      return;
    }
    const sim = B.debugKeys[e.code];
    if (!sim) return;
    e.preventDefault();
    if (sim.startsWith("pad")) {
      if (app.mode === "eeg" && !app.overlay) {
        logInput(`Key ${e.key}: press pad ${+sim.slice(3) + 1}`);
        simon.press(+sim.slice(3));
      }
    } else {
      gesture(sim);
    }
  });
  document.addEventListener("keyup", (e) => { if (B.eyebrowKeys.includes(e.code)) e.preventDefault(); });

  // ---------------------------------------------------------------- look-and-hold (dwell)
  // Gaze from a webcam is 150-350 px off, so targets are huge and short slips off a target
  // (up to GRACE_MS) pause the progress instead of resetting it.
  const GRACE_MS = 250;
  const dwell = { el: null, start: 0, lostAt: 0, last: 0 };

  function dwellTarget() {
    if (!mouse.known) return null;
    const hit = document.elementFromPoint(mouse.x, mouse.y);
    const el = hit && hit.closest("[data-dwell]");
    return el && dwellMs(el) ? el : null;
  }

  function dwellMs(el) {
    const kind = el.dataset.dwell;
    if (kind === "menu") return settings.menuDwell ? settings.menuDwellMs : 0;
    if (app.mode !== "gaze" || app.overlay) return 0;
    if (kind === "pad") return simon.phase === "input" ? settings.padDwellMs : 0;
    if (kind === "hub") {
      if (simon.phase === "idle" || simon.phase === "over") return settings.menuDwellMs;
      if (simon.phase === "input") return settings.menuDwellMs * 1.5;
    }
    return 0;
  }

  function setProgress(el, p) {
    if (!el) return;
    el.style.setProperty("--p", p);
    el.classList.toggle("dwelling", p > 0);
  }

  function fireDwell(el) {
    const kind = el.dataset.dwell;
    if (kind === "pad") simon.press(+el.dataset.i);
    else if (kind === "hub") simon.hubAction();
    else el.click();
  }

  function dwellTick(now) {
    const dt = now - (dwell.last || now);
    dwell.last = now;
    const el = dwellTarget();

    if (el !== dwell.el) {
      if (dwell.el && document.contains(dwell.el) && dwellMs(dwell.el)) {
        if (!dwell.lostAt) dwell.lostAt = now;
        if (now - dwell.lostAt < GRACE_MS) {      // brief slip: hold the progress
          dwell.start += dt;
          return;
        }
      }
      setProgress(dwell.el, 0);
      dwell.el = el;
      dwell.start = now;
      dwell.lostAt = 0;
    } else {
      dwell.lostAt = 0;
    }
    if (!el) return;

    const p = Math.max(0, Math.min(1, (now - dwell.start) / dwellMs(el)));
    setProgress(el, p);
    if (p >= 1) {
      setProgress(el, 0);
      dwell.start = now + 300;   // short pause, then the same target can be selected again
      fireDwell(el);
    }
  }

  // ---------------------------------------------------------------- classic Simon board
  const PADS = $$(".pad");
  const FREQ = [391.995, 329.628, 261.626, 195.998];   // green G4, red E4, yellow C4, blue G3
  const SPEED = { slow: 1.4, normal: 1, fast: 0.7 };
  const padTimers = [];

  function flash(i, ms) {
    const pad = PADS[i];
    clearTimeout(padTimers[i]);
    pad.classList.add("lit");
    tone(FREQ[i], ms);
    padTimers[i] = setTimeout(() => pad.classList.remove("lit"), ms);
  }

  function hub(title, score, msg) {
    $("#hub-title").textContent = title;
    $("#hub-score").textContent = score;
    $("#hub-msg").textContent = msg;
    $("#hub").classList.toggle("clickable", simon.phase === "idle" || simon.phase === "over");
    $("#board-best").textContent = `Best ${best[app.mode] || 0}`;
  }

  const simon = {
    seq: [], idx: 0, phase: "idle", token: 0, pausedPhase: null,

    reset() {
      this.token++;
      this.seq = [];
      this.idx = 0;
      this.phase = "idle";
      hub("SIMON", "", app.mode === "eeg"
        ? "Look here and clench your jaw to start"
        : "Look here and hold to start");
    },

    stop() { this.token++; this.phase = "idle"; PADS.forEach((p) => p.classList.remove("lit")); },

    hubAction() {
      if (this.phase === "idle" || this.phase === "over") this.start();
      else if (this.phase === "input" && app.mode === "gaze") openActions();
    },

    start() {
      this.seq = [];
      this.next();
    },

    next() {
      this.seq.push(Math.floor(Math.random() * 4));
      this.idx = 0;
      this.play();
    },

    async play() {
      this.releaseHold();
      const my = ++this.token;
      this.phase = "showing";
      const n = this.seq.length;
      hub("WATCH", n - 1, `Round ${n}`);
      const k = SPEED[settings.speed] || 1;
      const on = Math.max(260, 620 - 25 * n) * k;
      const gap = Math.max(90, 220 - 8 * n) * k;
      await sleep(900);
      for (const i of this.seq) {
        if (my !== this.token) return;
        flash(i, on);
        await sleep(on + gap);
      }
      if (my !== this.token) return;
      this.phase = "input";
      this.idx = 0;
      hub("YOUR TURN", n - 1, app.mode === "eeg"
        ? `0 / ${n} · double blink to replay`
        : `0 / ${n} · hold on a color`);
    },

    // hold = true: the left button is down (jaw clench). The color stays lit and sounding until the
    // clench ends (releaseHold), like holding a button on a real Simon; the press counts immediately.
    press(i, hold = false) {
      if (this.phase !== "input") return;
      this.releaseHold();
      if (hold) {
        clearTimeout(padTimers[i]);
        PADS[i].classList.add("lit");
        this.held = { i, stop: toneHold(FREQ[i]), t0: performance.now() };
      } else {
        flash(i, 280);
      }
      if (i !== this.seq[this.idx]) return this.fail(i);
      this.idx++;
      const n = this.seq.length;
      if (this.idx < n) {
        $("#hub-msg").textContent = `${this.idx} / ${n}`;
        return;
      }
      const my = ++this.token;
      this.phase = "between";
      hub("NICE!", n, "Next round...");
      const go = () => {
        if (my !== this.token) return;
        if (this.held) return setTimeout(go, 100);     // wait until the clench is released
        this.next();
      };
      setTimeout(go, 1000);
    },

    async fail(wrong) {
      const my = ++this.token;
      this.phase = "over";
      const score = this.seq.length - 1;
      const correct = this.seq[this.idx];
      PADS[wrong].classList.add("wrong");
      setTimeout(() => PADS[wrong].classList.remove("wrong"), 400);
      sfx.bad();
      const newBest = recordBest(app.mode, score);
      hub("GAME OVER", score, (newBest ? "New best! " : "") + (app.mode === "eeg"
        ? "Look here and clench to play again"
        : "Look here and hold to play again"));
      await sleep(700);
      for (let r = 0; r < 3 && my === this.token; r++) {   // show which color was right
        PADS[correct].classList.add("lit");
        await sleep(220);
        PADS[correct].classList.remove("lit");
        await sleep(160);
      }
    },

    held: null,
    releaseHold() {
      const h = this.held;
      if (!h) return;
      this.held = null;
      // a quick click (the controller's short clench) still gets a full flash and tone
      const left = Math.max(0, 260 - (performance.now() - h.t0));
      setTimeout(h.stop, left);
      padTimers[h.i] = setTimeout(() => PADS[h.i].classList.remove("lit"), left);
    },

    replay() {
      if (this.phase !== "input") return;
      this.idx = 0;
      this.play();
    },

    pause() {
      this.pausedPhase = this.phase;
      if (["showing", "input", "between"].includes(this.phase)) {
        this.token++;
        this.phase = "paused";
      }
    },

    resume() {
      const was = this.pausedPhase;
      this.pausedPhase = null;
      if (was === "between") this.next();
      else if (was === "showing" || was === "input") { this.idx = 0; this.play(); }
    },
  };

  // ---------------------------------------------------------------- Simon Says (gestures)
  const says = {
    phase: "ready", lives: 3, score: 0, round: 0, token: 0,
    cue: null, simonSays: true, deadline: 0, windowMs: 0, origin: null, last: null, pausedPhase: null,

    pool() {
      return Object.keys(B.gestures).filter((g) => settings.saysHeadTurns || !g.startsWith("head_"));
    },

    renderLegend() {
      $("#says-legend").innerHTML = this.pool()
        .map((g) => `<span class="chip" data-g="${g}"><b>${label(g)}</b> · ${B.gestures[g].input}</span>`)
        .join("");
    },

    flashChip(g) {
      const chip = document.querySelector(`.chip[data-g="${g}"]`);
      if (!chip) return;
      chip.classList.add("hit");
      setTimeout(() => chip.classList.remove("hit"), 450);
    },

    renderStats() {
      $("#says-lives").textContent = "♥".repeat(Math.max(0, this.lives)) || "-";
      $("#says-score").textContent = this.score;
      $("#says-best").textContent = best.says || 0;
    },

    stage(prefix, action, input, feedback = "", cls = "") {
      $("#says-prefix").textContent = prefix;
      $("#says-action").textContent = action;
      $("#says-input").textContent = input;
      const fb = $("#says-feedback");
      fb.textContent = feedback;
      fb.className = cls;
    },

    tiles(showThem, startLabel = "Start") {
      $("#says-tiles").classList.toggle("show", showThem);
      $('[data-says="start"]').firstChild.textContent = startLabel;
      $(".timer").classList.remove("on");
    },

    enter() {
      this.token++;
      this.phase = "ready";
      this.lives = 3;
      this.score = 0;
      this.renderLegend();
      this.renderStats();
      this.stage("Simon says", "GET READY",
        "Do the gesture only when Simon says. Look at Start and clench (or hold your gaze) to begin.");
      this.tiles(true);
    },

    start() {
      this.lives = 3;
      this.score = 0;
      this.round = 0;
      this.last = null;
      this.renderStats();
      this.tiles(false);
      this.countdown();
    },

    async countdown() {
      const my = ++this.token;
      this.phase = "countdown";
      $(".timer").classList.remove("on");
      for (const n of ["3", "2", "1"]) {
        this.stage("Relax your face", n, "");
        await sleep(700);
        if (my !== this.token) return;
      }
      this.nextCue();
    },

    nextCue() {
      const options = this.pool().filter((g) => g !== this.last);
      const g = options[Math.floor(Math.random() * options.length)];
      this.last = g;
      this.round++;
      this.cue = g;
      this.simonSays = this.round === 1 || Math.random() < 0.7;
      this.windowMs = Math.max(1800, 4200 - this.round * 100);
      this.deadline = performance.now() + this.windowMs;
      this.origin = { x: mouse.x, y: mouse.y };
      this.phase = "cue";
      this.stage(this.simonSays ? "Simon says" : "", B.gestures[g].cue, `(${B.gestures[g].input})`);
      $(".timer").classList.add("on");
      sfx.cue();
    },

    onGesture(g) {
      if (this.phase !== "cue") return;
      if (!this.simonSays) this.result(false, `Simon didn't say! (you did: ${label(g)})`);
      else if (g === this.cue) this.result(true, `Nice! ${label(g)}`);
      else this.result(false, `That was ${label(g)}, Simon said ${label(this.cue)}`);
    },

    tick(now) {
      if (this.phase !== "cue") return;
      const left = Math.max(0, this.deadline - now);
      $("#says-timer-fill").style.width = `${(left / this.windowMs) * 100}%`;

      if (settings.saysHeadTurns && mouse.known) {
        const dx = mouse.x - this.origin.x, dy = mouse.y - this.origin.y;
        const tx = innerWidth * settings.headTurnPct / 100, ty = innerHeight * settings.headTurnPct / 100;
        let g = null;
        if (Math.abs(dx) / tx >= Math.abs(dy) / ty) {
          if (dx <= -tx) g = "head_left"; else if (dx >= tx) g = "head_right";
        } else {
          if (dy <= -ty) g = "head_up"; else if (dy >= ty) g = "head_down";
        }
        if (g) { gesture(g); return; }
      }

      if (left <= 0) {
        if (this.simonSays) this.result(false, "Too slow!");
        else this.result(true, "Good, Simon didn't say!");
      }
    },

    async result(ok, msg) {
      const my = ++this.token;
      this.phase = "feedback";
      $(".timer").classList.remove("on");
      if (ok) { this.score++; sfx.good(); } else { this.lives--; sfx.bad(); }
      this.renderStats();
      $("#says-feedback").textContent = msg;
      $("#says-feedback").className = ok ? "ok" : "bad";
      await sleep(1400);
      if (my !== this.token) return;
      if (this.lives <= 0) this.over();
      else this.nextCue();
    },

    over() {
      this.phase = "over";
      const newBest = recordBest("says", this.score);
      this.renderStats();
      this.stage("Game over", `SCORE ${this.score}`, newBest ? "New best!" : `Best ${best.says || 0}`);
      this.tiles(true, "Play again");
    },

    stop() { this.token++; this.phase = "ready"; },

    pause() {
      this.pausedPhase = this.phase;
      if (["countdown", "cue", "feedback"].includes(this.phase)) {
        this.token++;
        this.phase = "paused";
      }
    },

    resume() {
      const was = this.pausedPhase;
      this.pausedPhase = null;
      if (["countdown", "cue", "feedback"].includes(was)) {
        if (this.lives <= 0) this.over();
        else this.countdown();
      }
    },
  };

  $$("[data-says]").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.says === "start") says.start();
    else goMenu();
  }));

  // ---------------------------------------------------------------- overlays
  function openActions() {
    if (!app.mode || app.overlay) return;
    app.overlay = "actions";
    if (app.mode === "says") says.pause(); else simon.pause();
    const canReplay = app.mode !== "says" && ["showing", "input"].includes(simon.pausedPhase);
    // Keep the 2x2 grid even without Replay: the screen center (where the eyes-only player is
    // looking when the overlay opens) must stay on the gap, not on a tile.
    $('[data-act="replay"]').hidden = !canReplay;
    $("#overlay-actions").classList.add("open");
  }

  function closeOverlay() {
    $$(".overlay").forEach((o) => o.classList.remove("open"));
    app.overlay = null;
  }

  function runAction(act) {
    closeOverlay();
    const game = app.mode === "says" ? says : simon;
    if (act === "resume") game.resume();
    else if (act === "replay") { simon.pausedPhase = "showing"; simon.resume(); }
    else if (act === "restart") { game.pausedPhase = null; app.mode === "says" ? says.start() : simon.start(); }
    else if (act === "menu") goMenu();
  }

  $$("[data-act]").forEach((b) => b.addEventListener("click", () => runAction(b.dataset.act)));
  $("#board-menu").addEventListener("click", openActions);

  // ---------------------------------------------------------------- settings
  const form = $(".settings-panel");
  function fmt(name, v) {
    return name.endsWith("Ms") ? `${(v / 1000).toFixed(1)} s` : name === "headTurnPct" ? `${v}%` : v;
  }
  function loadForm() {
    for (const el of form.querySelectorAll("[name]")) {
      if (el.type === "checkbox") el.checked = !!settings[el.name];
      else el.value = settings[el.name];
      const out = form.querySelector(`output[data-for="${el.name}"]`);
      if (out) out.textContent = fmt(el.name, settings[el.name]);
    }
  }
  form.addEventListener("input", (e) => {
    const el = e.target;
    if (!el.name) return;
    settings[el.name] = el.type === "checkbox" ? el.checked : el.type === "range" ? +el.value : el.value;
    saveSettings();
    loadForm();
    if (el.name === "showLog" && !el.checked) logEl.innerHTML = "";
  });
  $("#btn-settings").addEventListener("click", () => {
    loadForm();
    app.overlay = "settings";
    $("#overlay-settings").classList.add("open");
  });
  $("#settings-close").addEventListener("click", closeOverlay);
  $("#settings-reset").addEventListener("click", () => {
    best = {};
    localStorage.removeItem("simon.best");
    renderBest();
  });

  // ---------------------------------------------------------------- menu + main loop
  $$(".card[data-mode]").forEach((c) => c.addEventListener("click", () => startMode(c.dataset.mode)));
  $("#btn-arcade").addEventListener("click", () => { location.href = "../index.html"; });

  function frame(now) {
    dwellTick(now);
    if (app.mode === "says" && !app.overlay) says.tick(now);
    requestAnimationFrame(frame);
  }

  renderBest();
  const hash = location.hash.slice(1);
  if (MODE_NAMES[hash]) startMode(hash);
  requestAnimationFrame(frame);
})();
