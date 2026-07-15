"use strict";

let gameId = null;
let sel = { decl: null, thumb: null };
let lastView = null;
const mode = "adult";
let tipOn = localStorage.getItem("ys_tip") !== "off";
let meterOn = localStorage.getItem("ys_meter") !== "off";
let soundOn = localStorage.getItem("ys_sound") !== "off";
let difficulty = localStorage.getItem("ys_diff") || "normal";
let prevHands = { human: 2, ai: 2 };

// ---- 音（WebAudioで生成・ファイル不要） ----
let AC = null;
function ac() { if (!AC) AC = new (window.AudioContext || window.webkitAudioContext)(); return AC; }
function beep(freq, dur, type = "sine", gain = 0.15, delay = 0) {
  if (!soundOn) return;
  try {
    const c = ac(), o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.value = freq; o.connect(g); g.connect(c.destination);
    const t = c.currentTime + delay;
    g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.start(t); o.stop(t + dur);
  } catch (e) {}
}
const SND = {
  click: () => beep(520, 0.06, "triangle", 0.12),
  drop:  () => { beep(180, 0.12, "square", 0.18); beep(120, 0.18, "sine", 0.14, 0.04); },
  win:   () => { [523,659,784,1047].forEach((f,i)=>beep(f,0.18,"triangle",0.2,i*0.12)); },
  lose:  () => { [392,330,262].forEach((f,i)=>beep(f,0.28,"sine",0.2,i*0.14)); },
};

const $ = (id) => document.getElementById(id);
const show = (id) => { for (const s of ["start","game","result"]) $(s).classList.toggle("hidden", s !== id); };
const T = (o) => (o ? (o[mode] ?? o.adult ?? "") : "");
const SIDE_JP = { human: "あなた", ai: "AI" };
const STOCK_SET = ["ストック", "チョイス", "オール", "ドロップ"];

async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: {"Content-Type":"application/json"},
                                body: JSON.stringify(body || {}) });
  return r.json();
}

// ---- toggles ----
function refreshToggles() {
  const t = $("tg-tip");
  t.classList.toggle("on", tipOn);
  $("tg-tip-label").textContent = tipOn ? "説明ON" : "説明OFF";
  const meterToggle = $("tg-meter");
  meterToggle.classList.toggle("on", meterOn);
  meterToggle.querySelector(".tg-emoji").textContent = meterOn ? "📊" : "📉";
  $("tg-meter-label").textContent = meterOn ? "形勢判断ON" : "形勢判断OFF";
  $("meter").classList.toggle("hidden", !meterOn);
  const s = $("tg-sound");
  s.classList.toggle("on", soundOn);
  s.querySelector(".tg-emoji").textContent = soundOn ? "🔊" : "🔇";
  $("tg-sound-label").textContent = soundOn ? "サウンドON" : "サウンドOFF";
}
$("tg-tip").onclick = () => { tipOn = !tipOn; localStorage.setItem("ys_tip", tipOn ? "on" : "off");
  refreshToggles(); hideTip(); };
$("tg-meter").onclick = () => { meterOn = !meterOn; localStorage.setItem("ys_meter", meterOn ? "on" : "off");
  refreshToggles(); };
$("tg-sound").onclick = () => { soundOn = !soundOn; localStorage.setItem("ys_sound", soundOn ? "on" : "off");
  refreshToggles(); if (soundOn) SND.click(); };
refreshToggles();

// ---- 戦績 ----
function getRecord() {
  return { w: parseInt(localStorage.getItem("ys_win") || "0", 10),
           l: parseInt(localStorage.getItem("ys_loss") || "0", 10) };
}
function recordHtml() {
  const r = getRecord();
  return `${T(START.record)}: <b>${r.w}</b>${T(START.wins)} <span class="l"><b>${r.l}</b>${T(START.losses)}</span>`;
}

// ---- 難易度セグメント ----
function refreshDiff() {
  $("diff-title").textContent = T(START.diffTitle);
  $("diff-easy").textContent = T(START.diffEasy);
  $("diff-normal").textContent = T(START.diffNormal);
  $("diff-hard").textContent = T(START.diffHard);
  for (const b of document.querySelectorAll("#diff-seg button"))
    b.classList.toggle("sel", b.dataset.diff === difficulty);
}
for (const b of document.querySelectorAll("#diff-seg button")) {
  b.onclick = () => { difficulty = b.dataset.diff; localStorage.setItem("ys_diff", difficulty);
    refreshDiff(); SND.click(); };
}

// ---- start screen text (mode-aware) ----
function renderStart() {
  $("start-title").textContent = T(START.title);
  $("start-sub").textContent = T(START.sub);
  $("start-btn1").textContent = T(START.btn1);
  $("start-btn2").textContent = T(START.btn2);
  $("start-rule").textContent = T(START.rule);
  $("start-help-summary").textContent = T(START.helpSummary);
  $("start-help-body").innerHTML = T(START.helpBody).map(p => `<p>${p}</p>`).join("");
  $("again").textContent = T(START.again);
  $("start-record").innerHTML = recordHtml();
  refreshDiff();
}
renderStart();

// ---- tooltip ----
function showTip(el, evt) {
  if (!tipOn) return;
  const desc = el.dataset.desc; if (!desc) return;
  const tip = $("tip"); tip.textContent = desc; tip.classList.remove("hidden");
  const r = el.getBoundingClientRect();
  let x = r.left + r.width / 2 - tip.offsetWidth / 2;
  let y = r.top - tip.offsetHeight - 8;
  if (y < 6) y = r.bottom + 8;
  x = Math.max(6, Math.min(x, window.innerWidth - tip.offsetWidth - 6));
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { $("tip").classList.add("hidden"); }

// ---- start ----
for (const b of document.querySelectorAll("#start [data-first]")) {
  b.onclick = async () => {
    try { ac().resume(); } catch (e) {}   // unlock audio on first gesture
    prevHands = { human: 2, ai: 2 };
    const res = await api("/api/new",
      { human_first: b.dataset.first === "true", difficulty });
    gameId = res.game_id; show("game"); render(res.view);
  };
}
$("again").onclick = () => show("start");
$("again-inline").onclick = () => show("start");

// ---- surrender ----
$("surrender").onclick = async () => {
  if (!gameId) return;
  const res = await api("/api/act", { game_id: gameId, action: { surrender: true } });
  render(res.view);
};

// ---- board rendering ----
function handsHtml(hands) {
  let h = ""; for (let i = 0; i < 2; i++) h += `<span class="hand ${i < hands ? "" : "down"}">🖐️</span>`; return h;
}
function metaHtml(p) {
  let chips = p.buffs.map(b => `<span class="chip">${b}</span>`);
  if (p.skip_notice !== undefined) chips.unshift(`<span class="chip skip-alert">⏭️ このフェーズはスキップ（残り${p.skip_notice}）</span>`);
  if (p.stock.length) {
    const stored = p.stock.map(s => `<span class="stock-child" title="${s}">${(SKILLS[s]||{}).emoji||"❓"}<small>${s}</small></span>`).join("");
    chips.push(`<span class="stock-cluster"><span class="stock-root">📦 ストック</span><span class="stock-children">${stored}</span></span>`);
  }
  if (p.cement > 0) chips.push(`<span class="chip">🧱固定${p.cement}</span>`);
  return chips.join("");
}
function exchangeHtml(e) {
  if (!e) return `<div class="ex-empty">${mode === "kid" ? "ゲーム かいし！" : "ゲーム開始"}</div>`;
  const declarer = SIDE_JP[e.side], reactor = e.side === "human" ? "AI" : "あなた";
  const reactorCls = e.side === "human" ? "ai" : "human";
  return `
    <div class="ex-label">${UI.phaseTurn(declarer, e.phase, e.turn, mode)}</div>
    <span class="ex-side ${e.side}">${declarer}</span>
    <div class="ex-move"><span class="big ${e.decl.kind === "number" ? "number-icon" : ""}">${e.decl.emoji}</span><span class="nm">${e.decl.name}</span><span class="tb">👍${e.decl.thumb}本</span></div>
    <span class="ex-arrow">→</span>
    <div class="ex-move"><span class="big">${e.react.emoji}</span><span class="nm">${e.react.name}</span><span class="tb">👍${e.react.thumb}本</span></div>
    <span class="ex-side ${reactorCls}">${reactor}</span>`;
}
function logRowHtml(e) {
  return `<div class="log-row ${e.side}"><span class="lr-tag">${SIDE_JP[e.side]} ${e.phase}-${e.turn}</span>
    <div class="lr-body"><span class="${e.decl.kind === "number" ? "number-icon mini" : ""}">${e.decl.emoji}</span>${e.decl.name}👍${e.decl.thumb}
      <span class="lr-react">→ ${e.react.emoji}${e.react.name}👍${e.react.thumb}</span></div></div>`;
}

function flashDrop(side) {
  SND.drop();
  const panel = document.querySelector(`.player-panel.${side}`);
  if (panel) { panel.classList.remove("drop"); void panel.offsetWidth; panel.classList.add("drop"); }
}

function render(v) {
  lastView = v;
  $("surrender").textContent = "🏳️ " + T(START.surrender);
  $("surrender").classList.toggle("hidden", v.over);
  $("ai-hands").innerHTML = handsHtml(v.ai.hands);
  $("human-hands").innerHTML = handsHtml(v.human.hands);
  $("ai-meta").innerHTML = metaHtml(v.ai);
  $("human-meta").innerHTML = metaHtml(v.human);
  $("last-exchange").innerHTML = exchangeHtml(v.last_exchange);
  $("game-end").classList.toggle("hidden", !v.over);
  $("meter").classList.toggle("hidden", !meterOn);

  // 形勢メーター(あなた% / AI% を色分け表示)
  $("meter-title").textContent = T(START.meter);
  const adv = Math.max(-0.98, Math.min(0.98, v.ai_advantage || 0)); // -1..1 (AI優勢=+)
  const aiPct = Math.round((adv + 1) / 2 * 100);
  const youPct = 100 - aiPct;
  const youName = T(START.meterYou || {kid:"あなた",adult:"あなた"});
  const aiName = T(START.meterAi || {kid:"AI",adult:"AI"});
  $("mb-you").style.width = youPct + "%";
  $("mb-ai").style.width = aiPct + "%";
  $("mb-you-txt").textContent = youPct >= 12 ? `${youName} ${youPct}%` : `${youPct}%`;
  $("mb-ai-txt").textContent = aiPct >= 12 ? `${aiName} ${aiPct}%` : `${aiPct}%`;

  // 手が減ったら アニメ＋音
  if (v.ai.hands < prevHands.ai) flashDrop("ai");
  if (v.human.hands < prevHands.human) flashDrop("human");
  prevHands = { human: v.human.hands, ai: v.ai.hands };

  const list = $("log-list");
  list.dataset.empty = T(UI.emptyLog);
  let rows = (v.entries || []).map(logRowHtml).join("");
  if (v.over) rows += `<div class="log-row result ${v.human_won ? "win" : "lose"}">${
    v.human_won ? "🎉 " + T(UI.win) : "😢 " + T(UI.lose)}</div>`;
  list.innerHTML = rows; list.scrollTop = list.scrollHeight;

  if (v.over) {
    document.querySelector(".player-panel.human").classList.remove("turn-active");
    document.querySelector(".player-panel.ai").classList.remove("turn-active");
    $("human-badge").textContent = "";
    $("ai-badge").textContent = "";
    if (!v._recorded) {
      v._recorded = true;
      const key = v.human_won ? "ys_win" : "ys_loss";
      localStorage.setItem(key, String(parseInt(localStorage.getItem(key) || "0", 10) + 1));
      (v.human_won ? SND.win : SND.lose)();
    }
    $("game-end-emoji").textContent = v.human_won ? "🎉" : "😢";
    $("game-end-text").textContent = v.human_won ? T(UI.win) : T(UI.lose);
    $("game-end-record").innerHTML = recordHtml();
    $("again-inline").textContent = T(START.again);
    $("start-record").innerHTML = recordHtml();
    $("controls").innerHTML = `<div class="control-card end-note"><b>${v.human_won ? T(UI.win) : T(UI.lose)}</b><span>最後の宣言と履歴を確認できます</span></div>`;
    return;
  }

  const humanTurn = v.phase === "declare" || v.phase === "choice";
  document.querySelector(".player-panel.human").classList.toggle("turn-active", humanTurn);
  document.querySelector(".player-panel.ai").classList.toggle("turn-active", !humanTurn);
  $("human-badge").textContent = humanTurn ? T(UI.youTurn) : "";
  $("ai-badge").textContent = humanTurn ? "" : T(UI.aiTurn);

  sel = { decl: null, thumb: null };
  if (v.phase === "choice") renderChoice(v.options);
  else if (humanTurn) renderDeclare(v.options); else renderReact(v.options);
  wireOpts();
}

function optHtml(kind, dataAttrs, emoji, name, desc, extraCls = "", extraHtml = "") {
  return `<button class="opt ${extraCls}" ${dataAttrs} data-desc="${(desc||"").replace(/"/g,"&quot;")}">
    <span class="oe">${emoji}</span><span class="on">${name}</span>${extraHtml}</button>`;
}

function renderDeclare(o) {
  const numHtml = o.numbers.map(n =>
    optHtml("number", `data-kind="number" data-value="${n}" aria-label="数字${n}"`, n, "", T(SKILLS["数字"]), "number")).join("");
  const legal = new Set(o.skills);
  const skillOpt = (s, cls="", extra="") => optHtml("skill",
    `data-kind="${s === "チョイス" ? "choice" : "skill"}" data-name="${s}"`,
    (SKILLS[s]||{}).emoji||"❓", s, T(SKILLS[s]), cls, extra);
  const pick = order => order.filter(s=>legal.has(s)).map(s=>skillOpt(s)).join("");
  const normalSkills = pick(["フラッシュ","ガード","セメント","クイック","チャージ","スキップ"]);
  const normal = `<section class="declare-group normal"><div class="declare-group-head">${T(CAT.normal.label)}</div>
    <div class="normal-number-grid">${numHtml}</div><div class="normal-skill-grid">${normalSkills}</div></section>`;
  const antiCards = pick(["フェイント","ロック"]);
  const ultCards = pick(["ブースト","タイム"]);
  const alpha = ["チョイス","ドロップ","オール"].filter(s=>legal.has(s));
  const source = o.copy_source;
  const sourceIcon = source === null || source === undefined ? "" : typeof source === "number"
    ? `<span class="copy-source number-source" title="コピー元：数字${source}">${source}</span>`
    : `<span class="copy-source" title="コピー元：${source}">${(SKILLS[source]||{}).emoji||"❓"}</span>`;
  const copyTop = legal.has("コピー") ? skillOpt("コピー", "copy-opt", sourceIcon) : `<div class="ref-spacer"></div>`;
  const stockTop = legal.has("ストック") ? skillOpt("ストック") : alpha.length
    ? `<div class="stock-hub"><span>📦</span><small>ストック</small></div>` : `<div class="ref-spacer"></div>`;
  const refTop = (legal.has("コピー") || legal.has("ストック") || alpha.length) ? copyTop + stockTop : "";
  const alphaHtml = alpha.length ? `<div class="ref-branches">${alpha.map(s=>`<div class="ref-branch">${skillOpt(s,"alpha-opt")}</div>`).join("")}</div>` : "";
  const ref = (refTop || alphaHtml) ? `<section class="declare-group ref"><div class="declare-group-head">${T(CAT.ref.label)}</div>
    <div class="ref-top">${refTop}</div>${alphaHtml}</section>` : "";
  const lower = `<div class="declare-lower">${antiCards ? `<section class="declare-group anti"><div class="declare-group-head">${T(CAT.anti.label)}</div><div class="opt-grid">${antiCards}</div></section>` : ""}
    ${ultCards ? `<details class="declare-group ult"><summary class="declare-group-head">${T(CAT.ult.label)}<span>開く</span></summary><div class="opt-grid">${ultCards}</div></details>` : ""}</div>${ref}`;

  $("controls").innerHTML = `<div class="control-card">
    <div class="step-title">${T(UI.declareTitle)} <small>${T(UI.declareSub)}</small></div>
    <div class="declare-groups">${normal}${lower}</div>
    <div class="divider"></div>
    <div class="step-title">${T(UI.thumbTitle)}</div>
    <div class="thumb-row">${thumbButtons(o.thumbs)}</div>
    <button class="confirm" id="go" disabled>${T(UI.declareGo)}</button>
    <p class="hint">${T(UI.declareHint)}</p>
  </div>`;
}

function renderChoice(o) {
  const reaction = o.reaction;
  const cards = o.choices.map(s => optHtml("choice_target", `data-kind="choice_target" data-target="${s}"`,
    (SKILLS[s]||{}).emoji||"❓", s, T(SKILLS[s]))).join("");
  $("controls").innerHTML = `<div class="control-card choice-after-reaction">
    <div class="step-title">${T(UI.choiceTitle)} <small>${T(UI.choiceSub)}</small></div>
    <div class="reaction-reveal">AIの反応：<b>${reaction.emoji} ${reaction.name}</b> ／ 👍${reaction.thumb}本</div>
    <div class="opt-grid">${cards}</div>
    <button class="confirm" id="go" disabled>${T(UI.choiceGo)}</button>
  </div>`;
}

function renderReact(o) {
  const rHtml = o.reactions.map(r => {
    const extra = r === "カウンター" ? "react counter" : r === "ブロック" ? "react block ultimate" : "react";
    return optHtml("react", `data-kind="react" data-name="${r}"`, (REACTIONS[r]||{}).emoji||"❓", r, T(REACTIONS[r]), extra);
  }).join("");
  $("controls").innerHTML = `<div class="control-card">
    <div class="step-title">${T(UI.aiDeclared)}</div>
    <div class="step-title">${T(UI.reactTitle)}</div>
    <div class="opt-grid">${rHtml}</div>
    <div class="divider"></div>
    <div class="step-title">${T(UI.thumbTitle)}</div>
    <div class="thumb-row">${thumbButtons(o.thumbs)}</div>
    <button class="confirm" id="go" disabled>${T(UI.reactGo)}</button>
  </div>`;
}

function thumbButtons(thumbs) {
  return thumbs.map(t => `<button class="thumb-opt" data-thumb="${t}"><span class="te">👍</span>${t} 本</button>`).join("");
}

function wireOpts() {
  for (const b of document.querySelectorAll(".opt")) {
    b.onclick = () => { document.querySelectorAll(".opt").forEach(x => x.classList.remove("selected"));
      b.classList.add("selected"); sel.decl = { ...b.dataset }; SND.click(); updateGo(); };
    b.onmouseenter = (e) => showTip(b, e);
    b.onmouseleave = hideTip;
  }
  for (const b of document.querySelectorAll(".thumb-opt")) {
    b.onclick = () => { document.querySelectorAll(".thumb-opt").forEach(x => x.classList.remove("selected"));
      b.classList.add("selected"); sel.thumb = parseInt(b.dataset.thumb, 10); SND.click(); updateGo(); };
  }
  $("go").onclick = submit;
}
function updateGo() { $("go").disabled = !(sel.decl && (lastView.phase === "choice" || sel.thumb !== null)); }

async function submit() {
  $("go").disabled = true; hideTip();
  const d = sel.decl, action = lastView.phase === "choice" ? {} : { thumb: sel.thumb };
  if (d.kind === "number") { action.kind = "number"; action.value = parseInt(d.value, 10); }
  else if (d.kind === "choice") { action.kind = "choice"; }
  else if (d.kind === "choice_target") { action.kind = "choice_target"; action.target = d.target; }
  else if (d.kind === "react") { action.kind = "react"; action.name = d.name; }
  else { action.kind = "skill"; action.name = d.name; }
  const res = await api("/api/act", { game_id: gameId, action });
  render(res.view);
}
