// Mock backend for the highlights frontend (Contract 6). No dependencies.
// Run: node mock/server.mjs   (serves on http://127.0.0.1:8001)

import http from "node:http";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import crypto from "node:crypto";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SAMPLE = path.join(HERE, "sample.mp4");
const THUMB = path.join(HERE, "thumb.jpg");
const PORT = 8001;

// ---------- assets ----------

function haveFfmpeg() {
  const r = spawnSync("ffmpeg", ["-version"], { stdio: "ignore" });
  return r.status === 0;
}
const FFMPEG = haveFfmpeg();
if (!FFMPEG) console.warn("warning: ffmpeg not found; video/thumb routes will 404");

if (FFMPEG && !fs.existsSync(SAMPLE)) {
  console.log("generating sample.mp4 ...");
  spawnSync(
    "ffmpeg",
    ["-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25",
     "-f", "lavfi", "-i", "sine=frequency=440",
     "-t", "20", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-shortest", SAMPLE],
    { stdio: "ignore" },
  );
}
if (FFMPEG && fs.existsSync(SAMPLE) && !fs.existsSync(THUMB)) {
  spawnSync("ffmpeg", ["-y", "-i", SAMPLE, "-frames:v", "1", "-vf", "scale=320:180", THUMB], {
    stdio: "ignore",
  });
}

// ---------- helpers ----------

const now = () => new Date().toISOString();
const rid = () => crypto.randomBytes(6).toString("hex");
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// deterministic PRNG (mulberry32) so stats/candidates are stable across refreshes
function prng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function send(res, code, body, headers = {}) {
  const isObj = body !== null && typeof body === "object" && !Buffer.isBuffer(body);
  const data = isObj ? JSON.stringify(body) : body;
  res.writeHead(code, {
    "Content-Type": isObj ? "application/json" : headers["Content-Type"] ?? "application/octet-stream",
    ...(isObj ? {} : {}),
    ...headers,
  });
  res.end(data);
}
const jerr = (res, code, detail) => send(res, code, { detail });

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
  });
}

function serveFile(req, res, file, contentType) {
  if (!fs.existsSync(file)) return jerr(res, 404, "not found");
  const size = fs.statSync(file).size;
  const range = req.headers.range;
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (m) {
      let start = m[1] === "" ? 0 : parseInt(m[1], 10);
      let end = m[2] === "" ? size - 1 : parseInt(m[2], 10);
      if (m[1] === "" && m[2] !== "") { start = Math.max(0, size - end); end = size - 1; }
      end = Math.min(end, size - 1);
      res.writeHead(206, {
        "Content-Type": contentType,
        "Accept-Ranges": "bytes",
        "Content-Range": `bytes ${start}-${end}/${size}`,
        "Content-Length": end - start + 1,
      });
      fs.createReadStream(file, { start, end }).pipe(res);
      return;
    }
  }
  res.writeHead(200, {
    "Content-Type": contentType,
    "Accept-Ranges": "bytes",
    "Content-Length": size,
  });
  fs.createReadStream(file).pipe(res);
}

// ---------- domain ----------

const users = new Map();
function addUser(name) {
  users.set(name, { name, created_at: now() });
}
addUser("demo");
addUser("zeya");

const projects = new Map(); // id -> project

const VIDEO = { duration_s: 5400, width: 1920, height: 1080, fps: 25 };
const WINDOW = [120, 5280];
const HALVES = [
  { start: 120, end: 2880 },
  { start: 3000, end: 5280 },
];

function genCandidates(seed, n = 60) {
  const rnd = prng(seed);
  const types = [];
  const weights = [["goal", 3], ["shot", 28], ["chance", 25], ["excitement", 4]];
  const total = weights.reduce((s, w) => s + w[1], 0);
  for (let i = 0; i < n; i++) {
    let r = rnd() * total;
    let type = "excitement";
    for (const [t, w] of weights) { r -= w; if (r <= 0) { type = t; break; } }
    types.push(type);
  }
  const list = types.map((type, i) => {
    const t = Math.round((WINDOW[0] + rnd() * (WINDOW[1] - WINDOW[0])) * 10) / 10;
    const goal = type === "goal";
    const conf = goal ? 0.99 : Math.round((0.3 + rnd() * 0.6) * 100) / 100;
    const pad = goal ? 5 : 3;
    return {
      id: `c${String(i + 1).padStart(3, "0")}`,
      type,
      t,
      t_start: Math.round((t - 4) * 10) / 10,
      t_end: Math.round((t + 4) * 10) / 10,
      confidence: conf,
      signals: {
        audio_z: Math.round((rnd() * 4 + (goal ? 2 : 0)) * 10) / 10,
        motion_z: Math.round((rnd() * 4 + (goal ? 1.5 : 0)) * 10) / 10,
      },
      notes: "",
      cross_validation: goal ? "confirmed" : "pipeline_only",
      status: goal ? "confirmed" : "pending",
      clip_start: Math.round((t - pad) * 10) / 10,
      clip_end: Math.round((t + pad) * 10) / 10,
      rank: 0,
    };
  });
  list.sort((a, b) => b.confidence - a.confidence);
  list.forEach((c, i) => (c.rank = i + 1));
  return list;
}

function genStats(seed, candidates) {
  const rnd = prng(seed);
  const bin_s = 30;
  const nBins = VIDEO.duration_s / bin_s;
  const goals = candidates.filter((c) => c.type === "goal").map((c) => c.t);
  const inGap = (t) => t > HALVES[0].end && t < HALVES[1].start;
  const timeline = [];
  let mPrev = 0.4, aPrev = 0.4, ePrev = 0.3;
  for (let i = 0; i < nBins; i++) {
    const t = i * bin_s;
    if (inGap(t)) {
      timeline.push({ t, motion: 0, audio: 0, excitement: 0, events: [] });
      mPrev = aPrev = ePrev = 0.1;
      continue;
    }
    const smooth = (p) => clamp(p * 0.6 + rnd() * 0.55, 0, 1);
    let motion = smooth(mPrev), audio = smooth(aPrev), excitement = smooth(ePrev);
    const evs = [];
    for (const c of candidates) {
      if (c.t >= t && c.t < t + bin_s) evs.push({ t: c.t, type: c.type });
    }
    if (goals.some((g) => Math.abs(g - t) < bin_s)) excitement = clamp(0.8 + rnd() * 0.2, 0, 1);
    timeline.push({
      t,
      motion: Math.round(motion * 100) / 100,
      audio: Math.round(audio * 100) / 100,
      excitement: Math.round(excitement * 100) / 100,
      events: evs,
    });
    mPrev = motion; aPrev = audio; ePrev = excitement;
  }
  const events_by_type = {};
  for (const c of candidates) events_by_type[c.type] = (events_by_type[c.type] ?? 0) + 1;
  const events_per_10min = [];
  for (let t = 0; t <= VIDEO.duration_s; t += 600) {
    const inWin = (c) => c.t >= t && c.t < t + 600;
    events_per_10min.push({
      t,
      goal: candidates.filter((c) => c.type === "goal" && inWin(c)).length,
      shot: candidates.filter((c) => c.type === "shot" && inWin(c)).length,
      chance: candidates.filter((c) => c.type === "chance" && inWin(c)).length,
    });
  }
  const top_moments = [...candidates]
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 10)
    .map((c) => ({
      t: c.t,
      type: c.type,
      confidence: c.confidence,
      reason: `audio z=${c.signals.audio_z.toFixed(1)}, motion z=${c.signals.motion_z.toFixed(1)}`,
    }));
  const whistles = [120, 2880, 3000, 5280];
  for (let i = 0; i < 8; i++) {
    const h = HALVES[Math.floor(rnd() * 2)];
    whistles.push(Math.round(h.start + rnd() * (h.end - h.start)));
  }
  whistles.sort((a, b) => a - b);
  const motions = timeline.map((b) => b.motion);
  const peakIdx = motions.indexOf(Math.max(...motions));
  let quietA = 0, quietLen = 0, runStart = null, sum = 0;
  for (const b of timeline) {
    sum += b.motion;
    if (b.motion < 0.15) {
      if (runStart === null) runStart = b.t;
      if (b.t - runStart + bin_s > quietLen) { quietLen = b.t - runStart + bin_s; quietA = runStart; }
    } else runStart = null;
  }
  const loudest = timeline.reduce((a, b) => (b.audio > a.audio ? b : a));
  return {
    duration_s: VIDEO.duration_s,
    match_window: WINDOW,
    halves: HALVES,
    bin_s,
    timeline,
    events_by_type,
    events_per_10min,
    top_moments,
    whistles,
    activity: {
      mean_motion: Math.round((sum / timeline.length) * 100) / 100,
      peak_motion_t: timeline[peakIdx].t,
      loudest_t: loudest.t,
      quietest_stretch: [quietA, quietA + quietLen],
    },
    pipeline: { model: "audio_motion_lr v1", auroc_reference: 0.81, notes: "mock data" },
  };
}

const STAGES = [
  { name: "download", dur: 6000, ytOnly: true },
  { name: "probe", dur: 1000 },
  { name: "audio", dur: 3000 },
  { name: "motion", dur: 4000 },
  { name: "features", dur: 1000 },
  { name: "score", dur: 1000 },
  { name: "candidates", dur: 1000 },
  { name: "stats", dur: 1000 },
];

function stageMessage(p, stage, frac) {
  const pct = Math.round(frac * 100);
  switch (stage) {
    case "download": return `Downloading 1080p (${pct}%)`;
    case "probe": return "Probing video";
    case "audio": return `Extracting audio features ${pct}%`;
    case "motion": return `Computing motion ${pct}%`;
    case "features": return "Building features";
    case "score": return "Scoring moments";
    case "candidates": return "Selecting candidates";
    case "stats": return "Computing stats";
    default: return stage;
  }
}

function statusObj(p) {
  return {
    state: p.pipeline_state,
    stage: p.stage,
    progress: p.progress,
    stage_progress: p.stage_progress,
    message: p.message,
    error: p.error,
    started_at: p.started_at,
    updated_at: p.updated_at,
    finished_at: p.finished_at,
    pid: p.pid,
    video_path: p.video ? "/mock/match.mp4" : null,
    video: p.video,
    download:
      p.source.kind === "youtube"
        ? { format: "137+140", resolution: "1920x1080", filesize: 1.2e9 }
        : null,
  };
}

function logLine(p, line) {
  p.log.push(`[${now()}] ${line}`);
  if (p.log.length > 200) p.log.splice(0, p.log.length - 200);
}

function finishProject(p) {
  p.pipeline_state = "done";
  p.stage = "done";
  p.progress = 1;
  p.stage_progress = 1;
  p.message = "done";
  p.finished_at = now();
  p.video = { ...VIDEO };
  p.candidates = genCandidates(p.seed);
  p.stats = genStats(p.seed + 1, p.candidates);
  logLine(p, "pipeline done: 60 candidates");
}

function runSimulation(p) {
  clearInterval(p.timer);
  clearTimeout(p.timer);
  p.pipeline_state = "queued";
  p.stage = null;
  p.progress = 0;
  p.stage_progress = 0;
  p.message = "queued";
  p.error = null;
  p.started_at = now();
  p.updated_at = now();
  p.finished_at = null;
  p.pid = 1000 + Math.floor(Math.random() * 9000);
  p.video = null;
  p.candidates = [];
  p.stats = null;
  p.log = [];
  logLine(p, "pipeline queued");

  const stages = STAGES.filter((s) => !(s.ytOnly && p.source.kind !== "youtube"));
  const total = 1500 + stages.reduce((s, x) => s + x.dur, 0);
  const t0 = Date.now();
  let lastStage = null;

  p.timer = setInterval(() => {
    const el = Date.now() - t0;
    if (el < 1500) {
      p.updated_at = now();
      return;
    }
    let t = el - 1500;
    let stage = null, frac = 0, doneDur = 0;
    for (const s of stages) {
      if (t < s.dur) { stage = s; frac = t / s.dur; break; }
      t -= s.dur; doneDur += s.dur;
    }
    if (!stage) {
      clearInterval(p.timer);
      p.timer = null;
      finishProject(p);
      p.updated_at = now();
      return;
    }
    p.pipeline_state = "running";
    if (stage.name !== lastStage) {
      lastStage = stage.name;
      logLine(p, `stage: ${stage.name}`);
    }
    p.stage = stage.name;
    p.stage_progress = Math.round(frac * 100) / 100;
    p.progress = Math.round(((1500 + doneDur + frac * stage.dur) / total) * 100) / 100;
    p.message = stageMessage(p, stage.name, frac);
    p.updated_at = now();
    logLine(p, p.message);
  }, 500);
}

function failProject(p, error) {
  clearInterval(p.timer);
  clearTimeout(p.timer);
  p.timer = null;
  p.pipeline_state = "failed";
  p.error = error;
  p.message = error;
  p.finished_at = now();
  p.updated_at = now();
  logLine(p, `failed: ${error}`);
}

function makeProject(owner, { title, source }) {
  const p = {
    id: rid(),
    owner,
    title: title || "Untitled",
    created_at: now(),
    source,
    seed: crypto.randomBytes(4).readUInt32LE(0),
    pipeline_state: "queued",
    stage: null,
    progress: 0,
    stage_progress: 0,
    message: "queued",
    error: null,
    started_at: null,
    updated_at: now(),
    finished_at: null,
    pid: null,
    video: null,
    candidates: [],
    stats: null,
    log: [],
    timer: null,
    renderJobs: new Map(),
  };
  projects.set(p.id, p);
  return p;
}

function summary(p) {
  return {
    id: p.id,
    title: p.title,
    created_at: p.created_at,
    source: p.source,
    pipeline_state: p.pipeline_state,
    progress: p.progress,
    stage: p.stage,
    message: p.message,
    video: p.video,
    n_candidates: p.candidates.length,
    n_confirmed: p.candidates.filter((c) => c.status === "confirmed").length,
    thumb_url:
      p.pipeline_state === "done" && p.candidates.length
        ? `/api/projects/${p.id}/candidates/${p.candidates[0].id}/thumb.jpg`
        : null,
  };
}

// ---------- seed data ----------

{
  const d1 = makeProject("demo", {
    title: "Demo match (5qj_nsQSzvQ)",
    source: { kind: "youtube", url: "https://www.youtube.com/watch?v=5qj_nsQSzvQ" },
  });
  finishProject(d1);

  const d2 = makeProject("demo", {
    title: "Blocked download",
    source: { kind: "youtube", url: "https://www.youtube.com/watch?v=blocked123" },
  });
  failProject(
    d2,
    "YouTube blocked the automated download (Sign in to confirm you're not a bot). Upload the file instead or provide a cookies file.",
  );
  d2.stage = "download";

  const z1 = makeProject("zeya", {
    title: "Training match (upload)",
    source: { kind: "upload", filename: "training.mp4" },
  });
  finishProject(z1);

  const z2 = makeProject("zeya", {
    title: "Evening match",
    source: { kind: "youtube", url: "https://www.youtube.com/watch?v=fake456" },
  });
  runSimulation(z2);
}

// ---------- routes ----------

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, "http://x");
  const parts = u.pathname.split("/").filter(Boolean);
  console.log(`${req.method} ${u.pathname}${u.search}`);

  try {
    // users
    if (u.pathname === "/api/users" && req.method === "POST") {
      const body = JSON.parse(await readBody(req) || "{}");
      const name = String(body.name ?? "").trim();
      if (!name || name.length > 40) return jerr(res, 400, "name must be 1-40 chars");
      if (!users.has(name)) users.set(name, { name, created_at: now() });
      return send(res, 200, { name, created_at: users.get(name).created_at });
    }
    if (u.pathname === "/api/users" && req.method === "GET") {
      return send(res, 200, [...users.values()].map((x) => ({
        name: x.name,
        created_at: x.created_at,
        n_projects: [...projects.values()].filter((p) => p.owner === x.name).length,
      })));
    }

    // everything below /api/projects requires X-User
    if (parts[0] === "api" && parts[1] === "projects") {
      // <img>/<video> tags cannot set headers, so media URLs carry ?user= instead
      const user = req.headers["x-user"] || u.searchParams.get("user");
      if (!user || !users.has(user)) return jerr(res, 401, "unknown user");

      if (parts.length === 2) {
        if (req.method === "GET") {
          const list = [...projects.values()]
            .filter((p) => p.owner === user)
            .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
            .map(summary);
          return send(res, 200, list);
        }
        if (req.method === "POST") {
          const ct = String(req.headers["content-type"] ?? "");
          const raw = await readBody(req);
          let title, source;
          if (ct.includes("application/json")) {
            const body = JSON.parse(raw || "{}");
            title = body.title;
            void body.cookies_text; // accepted and ignored by the mock
            if (body.youtube_url) source = { kind: "youtube", url: body.youtube_url };
            else if (body.path) source = { kind: "path", path: body.path };
            else return jerr(res, 400, "youtube_url or path required");
          } else {
            const m = /name="title"[^]*?\r?\n\r?\n([^\r\n]*)/.exec(raw);
            title = m ? m[1].trim() : undefined;
            source = { kind: "upload", filename: "upload.mp4" };
          }
          const p = makeProject(user, { title, source });
          runSimulation(p);
          return send(res, 200, summary(p));
        }
      }

      const p = projects.get(parts[2]);
      if (!p || p.owner !== user) return jerr(res, 404, "not found");
      const rest = parts.slice(3);
      const r0 = rest[0];

      if (rest.length === 0) {
        if (req.method === "GET") return send(res, 200, { ...summary(p), pipeline: statusObj(p) });
        if (req.method === "DELETE") {
          clearInterval(p.timer);
          projects.delete(p.id);
          res.writeHead(204); res.end(); return;
        }
      }

      if (r0 === "pipeline") {
        if (rest.length === 1 && req.method === "GET")
          return send(res, 200, { ...statusObj(p), log: p.log.slice(-50) });
        if (rest[1] === "run" && req.method === "POST") {
          runSimulation(p);
          return send(res, 200, statusObj(p));
        }
        if (rest[1] === "cancel" && req.method === "POST") {
          failProject(p, "cancelled");
          return send(res, 200, statusObj(p));
        }
      }

      if (r0 === "stats" && req.method === "GET") {
        if (p.pipeline_state !== "done" || !p.stats) return jerr(res, 404, "stats not ready");
        return send(res, 200, p.stats);
      }

      if (r0 === "candidates") {
        if (rest.length === 1) {
          if (req.method === "GET") {
            const sort = u.searchParams.get("sort") ?? "confidence";
            const list = [...p.candidates];
            if (sort === "time") list.sort((a, b) => a.t - b.t);
            else list.sort((a, b) => b.confidence - a.confidence);
            return send(res, 200, list);
          }
        }
        if (rest[1] === "load" && req.method === "POST") return send(res, 200, p.candidates);
        const cid = rest[1];
        const c = p.candidates.find((x) => x.id === cid);
        if (!c) return jerr(res, 404, "not found");
        if (rest[2] === "thumb.jpg" && req.method === "GET")
          return serveFile(req, res, THUMB, "image/jpeg");
        if (rest.length === 2 && req.method === "PATCH") {
          Object.assign(c, JSON.parse((await readBody(req)) || "{}"), { id: c.id });
          return send(res, 200, c);
        }
        if (rest[2] === "reset" && req.method === "POST") {
          const pad = c.type === "goal" ? 5 : 3;
          c.clip_start = Math.round((c.t - pad) * 10) / 10;
          c.clip_end = Math.round((c.t + pad) * 10) / 10;
          return send(res, 200, c);
        }
      }

      if (r0 === "video") {
        const vinfo = () => ({
          path: "/mock/match.mp4",
          duration_s: VIDEO.duration_s,
          width: VIDEO.width,
          height: VIDEO.height,
          fps: VIDEO.fps,
          proxy_ready: true,
          registered_at: p.created_at,
        });
        if (rest.length === 1 && req.method === "GET")
          return p.video ? send(res, 200, vinfo()) : jerr(res, 404, "no video");
        if (rest.length === 1 && req.method === "POST") {
          await readBody(req);
          p.video = { ...VIDEO };
          return send(res, 200, vinfo());
        }
        if (rest[1] === "proxy" && rest.length === 2 && req.method === "POST")
          return send(res, 200, { status: "ready" });
        if (rest[1] === "proxy" && rest[2] === "status" && req.method === "GET")
          return send(res, 200, { ready: true, progress: 1 });
        if ((rest[1] === "proxy.mp4" || rest[1] === "source.mp4") && req.method === "GET")
          return serveFile(req, res, SAMPLE, "video/mp4");
      }

      if (r0 === "render") {
        if (rest.length === 1 && req.method === "POST") {
          const body = JSON.parse((await readBody(req)) || "{}");
          const job_id = rid();
          const ids = body.ids ?? p.candidates.map((c) => c.id);
          const job = {
            job_id,
            state: "queued",
            progress: 0,
            message: "queued",
            ids,
            error: null,
          };
          const t0 = Date.now();
          job.timer = setInterval(() => {
            const el = Date.now() - t0;
            if (el > 5000) {
              clearInterval(job.timer);
              job.state = "done"; job.progress = 1; job.message = "done";
            } else {
              job.state = "running";
              job.progress = Math.round((el / 5000) * 100) / 100;
              job.message = `rendering ${Math.round((el / 5000) * 100)}%`;
            }
          }, 400);
          p.renderJobs.set(job_id, job);
          return send(res, 200, { job_id });
        }
        const job = p.renderJobs.get(rest[1]);
        if (!job) return jerr(res, 404, "not found");
        if (rest.length === 2 && req.method === "GET") {
          const done = job.state === "done";
          return send(res, 200, {
            job_id: job.job_id,
            state: job.state,
            progress: job.progress,
            message: job.message,
            clips: done
              ? job.ids.map((cid) => ({
                  id: cid,
                  path: `/mock/files/${job.job_id}/clip_${cid}.mp4`,
                  url: `/api/projects/${p.id}/files/${job.job_id}/clip_${cid}.mp4`,
                  duration: 10,
                }))
              : [],
            reel_url: done ? `/api/projects/${p.id}/files/${job.job_id}/reel.mp4` : null,
            stats_url: done ? `/api/projects/${p.id}/files/${job.job_id}/stats.json` : null,
            error: job.error,
          });
        }
      }

      if (r0 === "files" && rest.length === 3 && req.method === "GET") {
        const name = rest[2];
        if (name === "stats.json") {
          if (!p.stats) return jerr(res, 404, "stats not ready");
          return send(res, 200, p.stats);
        }
        if (name.endsWith(".mp4")) return serveFile(req, res, SAMPLE, "video/mp4");
        return jerr(res, 404, "not found");
      }

      if (r0 === "project" && req.method === "GET") {
        return send(res, 200, {
          video: p.video
            ? {
                path: "/mock/match.mp4",
                duration_s: VIDEO.duration_s,
                width: VIDEO.width,
                height: VIDEO.height,
                fps: VIDEO.fps,
                proxy_ready: true,
                registered_at: p.created_at,
              }
            : null,
          candidates_version: 1,
          proxy_ready: true,
        });
      }

      return jerr(res, 404, "not found");
    }

    return jerr(res, 404, "not found");
  } catch (e) {
    return jerr(res, 500, String(e?.message ?? e));
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`mock server on http://127.0.0.1:${PORT}`);
});
