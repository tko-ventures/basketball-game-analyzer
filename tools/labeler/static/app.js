/* Court Labeller — client. Vanilla JS, no build step. */
"use strict";

const $ = (s) => document.querySelector(s);
const TEAM_COLOR = { home: "--home", away: "--away", ref: "--ref", other: "--other", "": "--unset" };
const SIDES = ["home", "away"];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const S = {
  clips: [], rosters: {}, clip: null, L: null,
  cur: 0, sel: -1, armed: null, img: null, hide: false,
  drag: null, dirty: false, saveTimer: null,
};

// ---------------------------------------------------------------- utilities
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
};
const frameUrl = (idx) => `/api/frame?path=${encodeURIComponent(S.clip)}&idx=${idx}`;
const stride = () => Math.max(1, S.L.stride | 0);
const lastFrame = () => Math.max(0, (S.L.frame_count || 1) - 1);
const fr = (idx = S.cur) => S.L.frames[idx];
const frEnsure = (idx = S.cur) => (S.L.frames[idx] ||= { done: false, boxes: [] });
const boxes = () => (fr() ? fr().boxes : []);
const shotAt = (idx) => S.L.shots.find((s) => idx >= s.start && idx <= s.end) || null;
const ident = (tid) => S.L.identities[String(tid)] || null;
// meta.home / meta.away = { team, roster, kit }; roster is a data/rosters/<key>.json stem.
const side = (team) => (SIDES.includes(team) ? S.L.meta[team] || {} : {});
const rosterKeyFor = (name) => Object.keys(S.rosters).find((k) => {
  const r = S.rosters[k], n = name.toLowerCase();
  return [r.team, ...(r.aliases || []), ...Object.values(r.seasons || {}).map((v) => v.team_name)]
    .some((x) => x && x.toLowerCase() === n);
}) || "";
const roster = (team) => {
  const r = S.rosters[side(team).roster];
  return (r && r.seasons[S.L.meta.season]) ? r.seasons[S.L.meta.season].players : [];
};
const rosterName = (num, team) => (roster(team).find((p) => p.number === String(num)) || {}).name || "";
const teamTitle = (team) => side(team).team || team;
const isSampled = (i) => i % stride() === 0;
const skipShot = (i) => $("#centralOnly").checked && (shotAt(i) || {}).camera === "other";

function nextIdIn() {
  const ids = Object.keys(S.L.identities).map(Number).filter((n) => n < 1000);
  return ids.length ? Math.max(...ids) + 1 : 1;
}
function newIdentity(team = "", number = "") {
  const tid = nextIdIn();
  S.L.identities[tid] = { team, number, name: "" };
  return tid;
}
function boxLabel(b) {
  const id = ident(b.tid) || {};
  const t = { home: "H", away: "A", ref: "REF", other: "?" }[id.team] || "";
  const ok = id.reviewed ? "✓" : "";
  if (id.team === "ref") return `REF·${b.tid}${ok}`;
  return `${t}${id.number ? id.number : ""}·${b.tid}${ok}`;
}
function markDirty() {
  S.dirty = true;
  $("#saveState").textContent = "unsaved…";
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(save, 800);
}
async function save() {
  if (!S.L || !S.dirty) return;
  S.dirty = false;
  try {
    const r = await api("/api/labels", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(S.L) });
    $("#saveState").textContent = `saved ${r.saved_at.slice(11)}`;
  } catch (e) {
    S.dirty = true;
    $("#saveState").textContent = `SAVE FAILED: ${e.message}`;
  }
}

// ---------------------------------------------------------------- loading
async function init() {
  const d = await api("/api/clips");
  S.clips = d.clips; S.rosters = d.rosters;
  $("#clipSel").innerHTML = S.clips.map((c) =>
    `<option value="${c.path}">${c.path} — ${c.done} done</option>`).join("") || "<option>(no videos in data/clips or data/raw)</option>";
  $("#pipeSel").innerHTML = `<option value="">(choose tracks.json)</option>` +
    d.pipeline_outputs.map((p) => `<option>${p}</option>`).join("");
  const seasons = [...new Set(Object.values(S.rosters).flatMap((r) => Object.keys(r.seasons || {})))].sort();
  $("#seasonSel").innerHTML = seasons.map((s) => `<option>${s}</option>`).join("");
  $("#teamNames").innerHTML = Object.values(S.rosters).map((r) => `<option value="${r.team}">`).join("");
  const last = localStorage.getItem("bt.clip");
  if (S.clips.length) await openClip(S.clips.some((c) => c.path === last) ? last : S.clips[0].path);
}

async function openClip(rel) {
  await save();
  S.clip = rel;
  try { localStorage.setItem("bt.clip", rel); } catch (_) {}
  $("#clipSel").value = rel;
  const d = await api(`/api/clip?path=${encodeURIComponent(rel)}`);
  S.L = d.labels;
  const seasons = [...$("#seasonSel").options].map((o) => o.value);
  if (!S.L.meta.season && seasons.length) S.L.meta.season = seasons[seasons.length - 1];
  $("#seasonSel").value = S.L.meta.season;
  for (const t of SIDES) {
    S.L.meta[t] ||= { team: "", roster: "", kit: "" };
    $(`#${t}Team`).value = S.L.meta[t].team || "";
    $(`#${t}Kit`).value = S.L.meta[t].kit || "";
  }
  $("#strideInput").value = S.L.stride;
  S.sel = -1; S.armed = null; $("#shotProgress").textContent = "";
  const doneIdx = Object.keys(S.L.frames).filter((k) => S.L.frames[k].done).map(Number);
  await goto(doneIdx.length ? Math.max(...doneIdx) : 0);
  renderRoster();
}

function goto(idx) {
  idx = Math.min(Math.max(0, idx), lastFrame());
  S.cur = idx; S.sel = -1;
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      if (S.cur !== idx) return resolve();
      S.img = img;
      const cv = $("#cv");
      if (cv.width !== img.naturalWidth) { cv.width = img.naturalWidth; cv.height = img.naturalHeight; }
      renderAll(); prefetch(); resolve();
    };
    img.onerror = () => { $("#frameInfo").textContent = `frame ${idx}: decode failed`; resolve(); };
    img.src = frameUrl(idx);
  });
}
function prefetch() {
  let i = S.cur;
  for (let k = 0; k < 3; k++) { i = stepIdx(i, 1); new Image().src = frameUrl(i); }
}
function stepIdx(from, dir, n = 1) {
  let i = from;
  for (let k = 0; k < n; k++) {
    let j = dir > 0 ? (Math.floor(i / stride()) + 1) * stride() : (Math.ceil(i / stride()) - 1) * stride();
    let guard = 0;
    while (j >= 0 && j <= lastFrame() && skipShot(j) && guard++ < 1e6) j += dir * stride();
    if (j < 0 || j > lastFrame()) break;
    i = j;
  }
  return i;
}
function gotoShot(dir) {
  const ix = S.L.shots.indexOf(shotAt(S.cur));
  let k = ix + dir;
  while (k >= 0 && k < S.L.shots.length && $("#centralOnly").checked && S.L.shots[k].camera === "other") k += dir;
  if (k < 0 || k >= S.L.shots.length) return;
  const s = S.L.shots[k].start;
  goto(Math.ceil(s / stride()) * stride() <= S.L.shots[k].end ? Math.ceil(s / stride()) * stride() : s);
}

// ---------------------------------------------------------------- rendering
function renderAll() { renderCanvas(); renderSide(); renderTimeline(); renderBars(); }

function renderCanvas() {
  const cv = $("#cv"), g = cv.getContext("2d");
  if (!S.img) return;
  g.drawImage(S.img, 0, 0);
  if (S.hide) return;
  const lw = Math.max(2, cv.width / 700);
  g.font = `${Math.round(cv.width / 90)}px ui-monospace, Menlo, monospace`;
  boxes().forEach((b, i) => {
    const id = ident(b.tid) || {};
    const col = css(TEAM_COLOR[id.team || ""]);
    g.lineWidth = i === S.sel ? lw * 2 : lw;
    g.strokeStyle = col;
    g.setLineDash(b.occluded ? [lw * 4, lw * 3] : []);
    g.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    g.setLineDash([]);
    const t = boxLabel(b), tw = g.measureText(t).width + 8, th = cv.width / 80;
    g.fillStyle = col; g.fillRect(b.x1, b.y1 - th, tw, th);
    g.fillStyle = "#fff"; g.fillText(t, b.x1 + 4, b.y1 - th * 0.2);
    if (i === S.sel) {
      const h = handleSize();
      g.fillStyle = "#fff";
      for (const [x, y] of corners(b)) g.fillRect(x - h / 2, y - h / 2, h, h);
    }
  });
}
function renderBars() {
  const f = fr();
  const s = shotAt(S.cur);
  const secs = (S.cur / S.L.fps).toFixed(2);
  $("#frameInfo").textContent = `frame ${S.cur} / ${lastFrame()}  ·  ${secs}s  ·  ${boxes().length} boxes`;
  const st = $("#frameStatus");
  st.className = "pill" + (f && f.done ? " done" : f && (f.interp || f.prefill) ? " interp" : "");
  st.textContent = f && f.done ? "DONE" : f && f.prefill ? "prefill — review" : f && f.interp ? "interpolated" : "not labelled";
  $("#shotInfo").textContent = s ? `${s.start}–${s.end} (${s.camera})` : "—";
  document.querySelectorAll("button.cam").forEach((b) => b.classList.toggle("active", !!s && b.dataset.cam === s.camera));
  $("#doneBtn").classList.toggle("active", !!(f && f.done));
}
function renderTimeline() {
  const c = $("#timeline"), w = c.clientWidth;
  c.width = w * devicePixelRatio; c.height = 34 * devicePixelRatio;
  const g = c.getContext("2d"); g.scale(devicePixelRatio, devicePixelRatio);
  const n = lastFrame() + 1, X = (i) => (i / n) * w;
  for (const s of S.L.shots) {
    g.fillStyle = css(s.camera === "central" ? "--central" : s.camera === "other" ? "--nonc" : "--unset");
    g.fillRect(X(s.start), 0, Math.max(1, X(s.end + 1) - X(s.start) - 1), 14);
  }
  for (const [k, f] of Object.entries(S.L.frames)) {
    if (!f.done && !f.interp && !f.prefill) continue;
    g.fillStyle = f.done ? css("--ink") : f.prefill ? css("--accent") : css("--unset");
    g.fillRect(X(+k), 17, Math.max(1, w / n), 10);
  }
  g.fillStyle = css("--accent"); g.fillRect(X(S.cur) - 1, 0, 2, 34);
}
function renderSide() {
  const b = boxes()[S.sel];
  $("#selEmpty").hidden = !!b; $("#selForm").hidden = !b;
  const ids = Object.entries(S.L.identities).sort((a, b) => +a[0] - +b[0]);
  if (b) {
    const id = ident(b.tid) || {};
    $("#selTid").innerHTML = ids.map(([t, v]) =>
      `<option value="${t}">${t} ${v.team || "?"} ${v.number || ""}</option>`).join("");
    $("#selTid").value = String(b.tid);
    document.querySelectorAll(".row.team button").forEach((x) => x.classList.toggle("active", x.dataset.team === id.team));
    if (document.activeElement !== $("#selNum")) $("#selNum").value = id.number || "";
    $("#selName").textContent = rosterName(id.number, id.team);
    $("#selOcc").checked = !!b.occluded;
    $("#selNumVis").checked = !!b.num_visible;
    renderLoupe(b);
    renderStrip(b.tid);
  }
  // identities list: labelled frames per identity, optionally only those in the current shot
  const counts = {}, s = shotAt(S.cur), inShot = new Set();
  for (const [k, f] of Object.entries(S.L.frames)) for (const x of f.boxes) {
    counts[x.tid] = (counts[x.tid] || 0) + 1;
    if (s && +k >= s.start && +k <= s.end) inShot.add(String(x.tid));
  }
  const here = new Set(boxes().map((x) => String(x.tid)));
  const shown = $("#idShotOnly").checked ? ids.filter(([t]) => inShot.has(t)) : ids;
  const open = shown.filter(([, v]) => !v.reviewed).length;
  $("#idCount").textContent = `(${shown.length}, ${open} to confirm)`;
  $("#idList").innerHTML = shown.map(([t, v]) => {
    const name = rosterName(v.number, v.team);
    const cls = [String(S.armed) === t ? "armed" : "", b && String(b.tid) === t ? "sel" : "", v.reviewed ? "reviewed" : "", v.note ? "noted" : ""].join(" ");
    return `<li data-tid="${t}" class="${cls}"><span class="sw" style="background:${css(TEAM_COLOR[v.team || ""])}"></span>
      <span class="mono">${t}</span><span>${v.team || "?"} ${v.number ? "#" + v.number : ""}</span>
      <span class="muted small">${name}</span><span class="spacer"></span>
      <span class="muted small">${here.has(t) ? "● " : ""}${counts[t] || 0}</span></li>`;
  }).join("");
  renderRoster();
}
function renderRoster() {
  // Roster of the selected box's team, else home.
  const b = boxes()[S.sel], t = b && SIDES.includes((ident(b.tid) || {}).team) ? ident(b.tid).team : "home";
  $("#rosterSide").textContent = `${t}: ${teamTitle(t)}`;
  document.querySelectorAll(".row.team button").forEach((x) => {
    const k = x.dataset.team, n = "1234"[["home", "away", "ref", "other"].indexOf(k)];
    x.textContent = SIDES.includes(k) ? `${side(k).team || k} (${n})` : x.textContent;
  });
  const used = new Set(Object.values(S.L.identities).filter((v) => v.team === t).map((v) => String(v.number)));
  $("#rosterList").innerHTML = roster(t).map((p) =>
    `<li class="${used.has(String(p.number)) ? "used" : ""}"><span class="mono">${p.number ?? "?"}</span><span>${p.name}</span><span class="muted">${p.pos || ""}</span></li>`).join("")
    || `<li class="muted">No roster loaded</li>`;
  $("#rosterNums").innerHTML = roster(t).filter((p) => p.number != null).map((p) => `<option value="${p.number}">${p.name}</option>`).join("");
}
function renderLoupe(b) {
  const c = $("#loupe"), g = c.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, c.width, c.height);
  if (!S.img) return;
  const w = b.x2 - b.x1, h = b.y2 - b.y1, pad = 0.1;
  const sx = b.x1 - w * pad, sy = b.y1 - h * pad, sw = w * (1 + 2 * pad), sh = h * (1 + 2 * pad);
  const k = Math.min(c.width / sw, c.height / sh);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(S.img, sx, sy, sw, sh, (c.width - sw * k) / 2, (c.height - sh * k) / 2, sw * k, sh * k);
}

// ---------------------------------------------------------------- tracks
const shotFrames = (s) => Object.keys(S.L.frames).map(Number).filter((k) => k >= s.start && k <= s.end).sort((a, b) => a - b);
const trackFrames = (tid, s) => shotFrames(s).filter((k) => S.L.frames[k].boxes.some((x) => String(x.tid) === String(tid)));
const imgCache = new Map();
function frameImg(k) {
  const url = frameUrl(k);
  if (!imgCache.has(url)) {
    const im = new Image(); im.src = url; imgCache.set(url, im);
    if (imgCache.size > 400) imgCache.delete(imgCache.keys().next().value);
  }
  return imgCache.get(url);
}
function renderStrip(tid) {
  // Up to 8 crops of the track spread over the current shot, so a whole track is judged at once.
  const c = $("#strip"), g = c.getContext("2d"), s = shotAt(S.cur);
  g.fillStyle = "#000"; g.fillRect(0, 0, c.width, c.height);
  S.stripTid = String(tid);
  const ks = s ? trackFrames(tid, s) : [];
  const span = ks.length ? `${ks.length} frames, ${ks[0]}–${ks[ks.length - 1]}` : "";
  $("#trackInfo").textContent = span;
  $("#trackNote").textContent = (ident(tid) || {}).note || "";
  if (!ks.length) return;
  const pick = ks.length <= 8 ? ks : Array.from({ length: 8 }, (_, i) => ks[Math.round((i * (ks.length - 1)) / 7)]);
  S.stripFrames = pick;
  const w = c.width / 4, h = c.height / 2;
  pick.forEach((k, i) => {
    const b = S.L.frames[k].boxes.find((x) => String(x.tid) === String(tid)), im = frameImg(k);
    const draw = () => {
      if (S.stripTid !== String(tid)) return;
      const bw = Math.max(1, b.x2 - b.x1), bh = Math.max(1, b.y2 - b.y1), sc = Math.min(w / bw, (h - 10) / bh);
      const x = (i % 4) * w + (w - bw * sc) / 2, y = Math.floor(i / 4) * h + 10;
      g.drawImage(im, b.x1, b.y1, bw, bh, x, y, bw * sc, bh * sc);
      g.fillStyle = k === S.cur ? css("--accent") : "#aaa"; g.font = "9px ui-monospace, Menlo, monospace";
      g.fillText(String(k), (i % 4) * w + 2, Math.floor(i / 4) * h + 8);
    };
    if (im.complete && im.naturalWidth) draw(); else im.addEventListener("load", draw, { once: true });
  });
}
function confirmTrack() {
  const b = boxes()[S.sel]; if (!b) return;
  const id = ident(b.tid); id.reviewed = !id.reviewed; markDirty(); renderAll();
}
function splitTrack() {
  const b = boxes()[S.sel], s = shotAt(S.cur); if (!b || !s) return;
  const from = String(b.tid), o = ident(from) || {}, to = newIdentity(o.team || "", "");
  let n = 0;
  for (const k of shotFrames(s)) if (k >= S.cur) for (const x of S.L.frames[k].boxes) if (String(x.tid) === from) { x.tid = to; n++; }
  markDirty(); renderAll(); flash(`Split ${from}: ${n} boxes from frame ${S.cur} on are now ${to}`);
}
function deleteTrack() {
  const b = boxes()[S.sel]; if (!b) return;
  const t = String(b.tid); let n = 0;
  for (const f of Object.values(S.L.frames)) { const before = f.boxes.length; f.boxes = f.boxes.filter((x) => String(x.tid) !== t); n += before - f.boxes.length; }
  delete S.L.identities[t]; S.sel = -1; markDirty(); renderAll(); flash(`Deleted track ${t} (${n} boxes)`);
}
function mergeTrack() {
  const b = boxes()[S.sel]; if (!b) return;
  const from = String(b.tid);
  const to = S.armed != null && String(S.armed) !== from ? String(S.armed)
    : (prompt(`Merge identity ${from} into which ID? (replaced in every frame)`) || "").trim();
  if (!to || to === from || !S.L.identities[to]) return to && flash(`No identity ${to}`);
  let clash = 0;
  for (const f of Object.values(S.L.frames)) {
    const hasTo = f.boxes.some((x) => String(x.tid) === to);
    for (const x of f.boxes) if (String(x.tid) === from) { if (hasTo) clash++; x.tid = +to; }
  }
  const src = S.L.identities[from], dst = S.L.identities[to];
  if (!dst.team && src.team) dst.team = src.team;
  if (!dst.number && src.number) { dst.number = src.number; dst.name = rosterName(dst.number, dst.team); }
  delete S.L.identities[from]; S.armed = null;
  markDirty(); renderAll(); flash(`Merged ${from} → ${to}` + (clash ? ` (${clash} frames now have ${to} twice — check them)` : ""));
}
function acceptShot() {
  const s = shotAt(S.cur); if (!s) return;
  const ks = shotFrames(s).filter((k) => S.L.frames[k].boxes.length);
  const open = new Set();
  for (const k of ks) for (const x of S.L.frames[k].boxes) if (!(ident(x.tid) || {}).reviewed) open.add(String(x.tid));
  if (open.size) return flash(`${open.size} tracks in this shot not confirmed: ${[...open].slice(0, 8).join(", ")}`);
  for (const k of ks) Object.assign(S.L.frames[k], { done: true, interp: false, prefill: false });
  markDirty(); renderAll(); flash(`Shot accepted: ${ks.length} frames done`);
}
// Active labelling time per day, for the gold-set time log: gaps over 60 s count as idle.
let lastAct = 0;
function tickReview() {
  const now = Date.now();
  if (S.L && lastAct && now - lastAct < 60000) {
    const m = S.L.meta, d = new Date().toISOString().slice(0, 10);
    m.review_seconds ||= {}; m.review_seconds[d] = Math.round(((m.review_seconds[d] || 0) + (now - lastAct) / 1000) * 10) / 10;
    S.dirty = true;
    const tot = Object.values(m.review_seconds).reduce((a, v) => a + v, 0);
    $("#reviewTime").textContent = `review ${Math.floor(tot / 60)}:${String(Math.round(tot % 60)).padStart(2, "0")}`;
  }
  lastAct = now;
}
window.addEventListener("keydown", tickReview, true);
window.addEventListener("mousedown", tickReview, true);

// ---------------------------------------------------------------- geometry / mouse
const handleSize = () => { const cv = $("#cv"); return 10 * cv.width / cv.clientWidth; };
const corners = (b) => [[b.x1, b.y1], [b.x2, b.y1], [b.x1, b.y2], [b.x2, b.y2]];
function toImg(ev) {
  const cv = $("#cv"), r = cv.getBoundingClientRect();
  return [(ev.clientX - r.left) * cv.width / r.width, (ev.clientY - r.top) * cv.height / r.height];
}
function hitBox(x, y) {
  let best = -1, area = Infinity;
  boxes().forEach((b, i) => {
    if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) {
      const a = (b.x2 - b.x1) * (b.y2 - b.y1);
      if (a < area) { area = a; best = i; }
    }
  });
  return best;
}
function normBox(b) {
  const W = S.L.width || $("#cv").width, H = S.L.height || $("#cv").height;
  const x1 = Math.min(b.x1, b.x2), x2 = Math.max(b.x1, b.x2), y1 = Math.min(b.y1, b.y2), y2 = Math.max(b.y1, b.y2);
  Object.assign(b, { x1: Math.max(0, x1), y1: Math.max(0, y1), x2: Math.min(W, x2), y2: Math.min(H, y2) });
}

$("#cv").addEventListener("mousedown", (ev) => {
  if (!S.img || S.hide) return;
  const [x, y] = toImg(ev), h = handleSize();
  const b = boxes()[S.sel];
  if (b) {
    const ci = corners(b).findIndex(([cx, cy]) => Math.abs(cx - x) < h && Math.abs(cy - y) < h);
    if (ci >= 0) { S.drag = { mode: "resize", ci, b }; return; }
  }
  const hi = hitBox(x, y);
  if (hi >= 0) {
    S.sel = hi; const hb = boxes()[hi];
    S.drag = { mode: "move", b: hb, ox: x, oy: y, orig: { ...hb } };
    renderAll(); return;
  }
  const f = frEnsure();
  const nb = { tid: null, x1: x, y1: y, x2: x, y2: y, occluded: false, num_visible: false };
  f.boxes.push(nb); S.sel = f.boxes.length - 1;
  S.drag = { mode: "new", b: nb };
});
window.addEventListener("mousemove", (ev) => {
  if (!S.drag) return;
  const [x, y] = toImg(ev), d = S.drag, b = d.b;
  if (d.mode === "new") { b.x2 = x; b.y2 = y; }
  else if (d.mode === "move") {
    const dx = x - d.ox, dy = y - d.oy;
    Object.assign(b, { x1: d.orig.x1 + dx, x2: d.orig.x2 + dx, y1: d.orig.y1 + dy, y2: d.orig.y2 + dy });
  } else if (d.mode === "resize") {
    if (d.ci === 0 || d.ci === 2) b.x1 = x; else b.x2 = x;
    if (d.ci === 0 || d.ci === 1) b.y1 = y; else b.y2 = y;
  }
  renderCanvas();
});
window.addEventListener("mouseup", () => {
  const d = S.drag; if (!d) return; S.drag = null;
  const b = d.b; normBox(b);
  if (d.mode === "new") {
    if (b.x2 - b.x1 < 6 || b.y2 - b.y1 < 6) { boxes().splice(S.sel, 1); S.sel = -1; renderAll(); return; }
    let tid = S.armed;
    if (tid == null || boxes().some((o) => o !== b && String(o.tid) === String(tid))) tid = newIdentity();
    b.tid = +tid; S.armed = null;
  }
  if (d.mode !== "move" || b.x1 !== d.orig.x1 || b.y1 !== d.orig.y1) touchFrame();
  renderAll();
});
$("#timeline").addEventListener("click", (ev) => {
  const r = ev.currentTarget.getBoundingClientRect();
  const i = Math.round(((ev.clientX - r.left) / r.width) * lastFrame() / stride()) * stride();
  goto(i);
});

function touchFrame() { frEnsure(); markDirty(); }

// ---------------------------------------------------------------- actions
function setTeam(team) {
  const b = boxes()[S.sel]; if (!b) return;
  const id = ident(b.tid); id.team = team;
  id.name = rosterName(id.number, team);
  markDirty(); renderAll();
}
function deleteSel() {
  if (S.sel < 0) return;
  boxes().splice(S.sel, 1); S.sel = -1; touchFrame(); renderAll();
}
function prevLabelled(before) {
  const ks = Object.keys(S.L.frames).map(Number).filter((k) => k < before && S.L.frames[k].boxes.length).sort((a, b) => b - a);
  return ks.length ? ks[0] : null;
}
function copyPrev() {
  const p = prevLabelled(S.cur); if (p == null) return;
  const f = frEnsure(), have = new Set(f.boxes.map((b) => b.tid));
  for (const b of S.L.frames[p].boxes) if (!have.has(b.tid)) f.boxes.push({ ...b });
  touchFrame(); renderAll();
}
function interpolate() {
  const done = Object.keys(S.L.frames).map(Number).filter((k) => S.L.frames[k].done).sort((a, b) => a - b);
  const cur = S.cur;
  let A, B;
  if (fr() && fr().done) { A = done.filter((k) => k < cur).pop(); B = cur; }
  else { A = done.filter((k) => k < cur).pop(); B = done.find((k) => k > cur); }
  if (A == null || B == null) return flash("Need done frames on both sides");
  if (shotAt(A) !== shotAt(B)) return flash("Done frames are in different shots — not interpolating across a cut");
  const fa = S.L.frames[A], fb = S.L.frames[B];
  const mb = new Map(fb.boxes.map((b) => [b.tid, b]));
  let n = 0;
  for (let i = Math.ceil((A + 1) / stride()) * stride(); i < B; i += stride()) {
    const f = S.L.frames[i];
    if (f && f.done) continue;
    const t = (i - A) / (B - A), out = [];
    for (const a of fa.boxes) {
      const b = mb.get(a.tid); if (!b) continue;
      const L = (p) => a[p] + (b[p] - a[p]) * t;
      out.push({ tid: a.tid, x1: L("x1"), y1: L("y1"), x2: L("x2"), y2: L("y2"), occluded: a.occluded && b.occluded, num_visible: false });
    }
    S.L.frames[i] = { done: false, interp: true, boxes: out }; n++;
  }
  markDirty(); renderAll(); flash(`Interpolated ${n} frames between ${A} and ${B}`);
}
function toggleDone() {
  const f = frEnsure();
  if (!f.done && f.boxes.some((b) => !(ident(b.tid) || {}).team)) {
    if (!confirm("Some boxes have no team. Mark done anyway?")) return;
  }
  f.done = !f.done; if (f.done) f.interp = false;
  markDirty(); renderAll();
}
function setCamera(cam) {
  const s = shotAt(S.cur); if (!s) return;
  s.camera = cam; markDirty(); renderAll();
}
let flashT;
function flash(msg) { $("#saveState").textContent = msg; clearTimeout(flashT); flashT = setTimeout(() => ($("#saveState").textContent = S.dirty ? "unsaved…" : "saved"), 2500); }

async function detectShots() {
  if (S.L.shots.some((s) => s.camera !== "unset") && !confirm("Replace current shots (camera tags are kept where a new shot starts inside an old tagged one)?")) return;
  await api(`/api/shots?path=${encodeURIComponent(S.clip)}`, { method: "POST", body: "{}" });
  const clip = S.clip;
  const poll = async () => {
    if (clip !== S.clip) return;
    const r = await api(`/api/shots?path=${encodeURIComponent(clip)}`);
    if (r.progress == null || r.progress < 1) { $("#shotProgress").textContent = `scanning ${Math.round((r.progress || 0) * 100)}%`; return setTimeout(poll, 1000); }
    const old = S.L.shots;
    S.L.shots = r.shots.map((s) => {
      const o = old.find((x) => s.start >= x.start && s.start <= x.end);
      return { ...s, camera: o && old.length > 1 ? o.camera : "unset" };
    });
    $("#shotProgress").textContent = `${S.L.shots.length} shots`;
    markDirty(); renderAll();
  };
  poll();
}
async function prefill() {
  const p = $("#pipeSel").value; if (!p) return;
  const d = await api(`/api/pipeline?tracks=${encodeURIComponent(p)}`);
  if (d.clip_path && d.clip_path !== S.clip && !confirm(`tracks.json was produced from ${d.clip_path}, not ${S.clip}. Import anyway?`)) return;
  let nf = 0;
  for (const [k, bs] of Object.entries(d.frames)) {
    const i = +k; if (!isSampled(i) || !bs.length) continue;
    const f = S.L.frames[i]; if (f && f.boxes.length) continue;
    S.L.frames[i] = { done: false, interp: false, prefill: true, boxes: bs.map((b) => {
      const tid = 1000 + (b.track_id | 0);
      if (!S.L.identities[tid]) {
        const team = b.team || "";
        S.L.identities[tid] = { team, number: b.number || "", name: rosterName(b.number, team), source: "pipeline" };
      }
      const [x1, y1, x2, y2] = b.xyxy;
      return { tid, x1, y1, x2, y2, occluded: false, num_visible: false };
    }) };
    nf++;
  }
  markDirty(); renderAll(); flash(`Prefilled ${nf} frames`);
}


// ---------------------------------------------------------------- wiring
$("#clipSel").onchange = (e) => openClip(e.target.value);
$("#seasonSel").onchange = (e) => { S.L.meta.season = e.target.value; markDirty(); renderAll(); };
for (const t of SIDES) {
  $(`#${t}Team`).oninput = (e) => {
    const m = S.L.meta[t]; m.team = e.target.value; m.roster = rosterKeyFor(m.team);
    for (const v of Object.values(S.L.identities)) if (v.team === t) v.name = rosterName(v.number, t);
    markDirty(); renderAll();
  };
  $(`#${t}Kit`).oninput = (e) => { S.L.meta[t].kit = e.target.value; markDirty(); };
}
$("#strideInput").onchange = (e) => { S.L.stride = Math.max(1, +e.target.value | 0); markDirty(); renderAll(); };
$("#prevBtn").onclick = () => goto(stepIdx(S.cur, -1));
$("#nextBtn").onclick = () => goto(stepIdx(S.cur, 1));
$("#prevShotBtn").onclick = () => gotoShot(-1);
$("#nextShotBtn").onclick = () => gotoShot(1);
$("#centralOnly").onchange = () => renderAll();
document.querySelectorAll("button.cam").forEach((b) => (b.onclick = () => setCamera(b.dataset.cam)));
document.querySelectorAll(".row.team button").forEach((b) => (b.onclick = () => setTeam(b.dataset.team)));
$("#detectShotsBtn").onclick = detectShots;
$("#copyPrevBtn").onclick = copyPrev;
$("#interpBtn").onclick = interpolate;
$("#doneBtn").onclick = toggleDone;
$("#delBtn").onclick = deleteSel;
$("#prefillBtn").onclick = prefill;
$("#helpBtn").onclick = () => $("#help").showModal();
$("#newIdBtn").onclick = () => { const b = boxes()[S.sel]; if (!b) return; const o = ident(b.tid) || {}; b.tid = newIdentity(o.team || ""); touchFrame(); renderAll(); };
$("#mergeBtn").onclick = mergeTrack;
$("#confirmTrackBtn").onclick = confirmTrack;
$("#splitBtn").onclick = splitTrack;
$("#delTrackBtn").onclick = deleteTrack;
$("#acceptShotBtn").onclick = acceptShot;
$("#idShotOnly").onchange = () => renderAll();
$("#pruneBtn").onclick = (e) => {
  e.preventDefault();
  const used = new Set(); for (const f of Object.values(S.L.frames)) for (const x of f.boxes) used.add(String(x.tid));
  let n = 0; for (const t of Object.keys(S.L.identities)) if (!used.has(t)) { delete S.L.identities[t]; n++; }
  markDirty(); renderAll(); flash(`Removed ${n} unused identities`);
};
$("#selTid").onchange = (e) => { const b = boxes()[S.sel]; if (!b) return; b.tid = +e.target.value; touchFrame(); renderAll(); };
$("#selNum").oninput = (e) => {
  const b = boxes()[S.sel]; if (!b) return;
  const id = ident(b.tid); id.number = e.target.value.replace(/[^0-9]/g, "").slice(0, 2);
  id.name = rosterName(id.number, id.team);
  markDirty(); renderCanvas(); $("#selName").textContent = id.name;
};
$("#selNum").onchange = () => renderAll();
$("#selOcc").onchange = (e) => { const b = boxes()[S.sel]; if (b) { b.occluded = e.target.checked; touchFrame(); renderAll(); } };
$("#selNumVis").onchange = (e) => { const b = boxes()[S.sel]; if (b) { b.num_visible = e.target.checked; touchFrame(); } };
// Click a track: select it, jumping to its first frame in this shot if it is not on screen.
// Shift+click: arm it (the next box drawn gets this identity; M merges the selected box into it).
async function selectTrack(tid, frame = null) {
  const s = shotAt(S.cur), ks = s ? trackFrames(tid, s) : [];
  const k = frame ?? (boxes().some((b) => String(b.tid) === tid) ? S.cur : ks[0]);
  if (k == null) return flash(`Track ${tid} has no boxes in this shot`);
  if (k !== S.cur) await goto(k);
  S.sel = boxes().findIndex((b) => String(b.tid) === tid); S.armed = null; renderAll();
}
$("#idList").onclick = (e) => {
  const li = e.target.closest("li"); if (!li) return;
  const tid = li.dataset.tid;
  if (e.shiftKey) { S.armed = S.armed === tid ? null : tid; S.sel = -1; renderAll(); return; }
  selectTrack(tid);
};
// Click a filmstrip crop: go to that frame with the track selected (to find where a split belongs).
$("#strip").addEventListener("click", (e) => {
  const c = e.currentTarget, r = c.getBoundingClientRect();
  const col = Math.floor(((e.clientX - r.left) / r.width) * 4), row = Math.floor(((e.clientY - r.top) / r.height) * 2);
  const k = (S.stripFrames || [])[row * 4 + col];
  if (k != null) selectTrack(S.stripTid, k);
});
window.addEventListener("resize", () => S.L && renderTimeline());
window.addEventListener("beforeunload", (e) => { if (S.dirty) { save(); e.preventDefault(); } });

window.addEventListener("keydown", (e) => {
  const tag = (document.activeElement || {}).tagName;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); S.dirty = true; save(); return; }
  if (tag === "INPUT" || tag === "SELECT") { if (e.key === "Escape" || e.key === "Enter") document.activeElement.blur(); return; }
  if (!S.L) return;
  const k = e.key;
  if (k === "ArrowRight" && e.shiftKey === false) { e.preventDefault(); goto(stepIdx(S.cur, 1)); }
  else if (k === "ArrowLeft" && !e.shiftKey) { e.preventDefault(); goto(stepIdx(S.cur, -1)); }
  else if (k === "ArrowRight") { e.preventDefault(); goto(stepIdx(S.cur, 1, 10)); }
  else if (k === "ArrowLeft") { e.preventDefault(); goto(stepIdx(S.cur, -1, 10)); }
  else if (k === "ArrowDown" && e.shiftKey) { e.preventDefault(); gotoShot(1); }
  else if (k === "ArrowUp" && e.shiftKey) { e.preventDefault(); gotoShot(-1); }
  else if (k === "Tab") { e.preventDefault(); const n = boxes().length; if (n) { S.sel = (S.sel + (e.shiftKey ? n - 1 : 1)) % n; renderAll(); } }
  else if ("1234".includes(k) && k.length === 1) setTeam(["home", "away", "ref", "other"][+k - 1]);
  else if (k === "n" || k === "N") { if (S.sel >= 0) { e.preventDefault(); $("#selNum").focus(); $("#selNum").select(); } }
  else if (k === "o" || k === "O") { const b = boxes()[S.sel]; if (b) { b.occluded = !b.occluded; touchFrame(); renderAll(); } }
  else if ((k === "Delete" || k === "Backspace") && e.shiftKey) { e.preventDefault(); deleteTrack(); }
  else if (k === "Delete" || k === "Backspace") { e.preventDefault(); deleteSel(); }
  else if (k === "D" && e.shiftKey) acceptShot();
  else if (k === "k" || k === "K") confirmTrack();
  else if (k === "p" || k === "P") splitTrack();
  else if (k === "m" || k === "M") mergeTrack();
  else if (k === "c" || k === "C") copyPrev();
  else if (k === "i" || k === "I") interpolate();
  else if (k === "d" || k === "D") toggleDone();
  else if (k === "g" || k === "G") setCamera("central");
  else if (k === "x" || k === "X") setCamera("other");
  else if (k === "h" || k === "H") { S.hide = !S.hide; renderCanvas(); }
  else if (k === "Escape") { S.sel = -1; S.armed = null; renderAll(); }
  else if (k === "?") $("#help").showModal();
});

init().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p style="color:red;padding:8px">${e.message}</p>`); });
