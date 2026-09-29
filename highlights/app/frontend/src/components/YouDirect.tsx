import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Loader2, Maximize, Pause, Play, RotateCcw, Square } from "lucide-react";
import { useProjectApi } from "../api";
import { fmtClock, parseClock } from "../lib/time";
import { ANGLE_COLORS } from "./DirectorCut";
import type { DirectComparison, DirectLearnResult, DirectSession, DirectSuggest } from "../types";

const card = "card p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const input =
  "bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs font-mono placeholder:text-zinc-500 focus:outline-none focus:border-amber-400";

// fast-style defaults for the plain-words diff
const FAST_DEFAULTS: Record<string, number> = {
  zone_linger: 3, smooth_median: 3, margin_cluster: 0.10, min_hold: 2,
};

function describeOverrides(best: Record<string, number>): string {
  const parts: string[] = [];
  if ("zone_linger" in best)
    parts.push(`leave a camera ${best.zone_linger} s after the ball is lost`);
  if ("smooth_median" in best)
    parts.push(best.smooth_median <= 3
      ? "react faster to crowd moves"
      : "smooth crowd moves more");
  if ("margin_cluster" in best)
    parts.push(`switch on a ${Math.round(best.margin_cluster * 100)}% edge`);
  if ("min_hold" in best)
    parts.push(`hold each shot ${best.min_hold} s minimum`);
  const changed = parts.length
    ? parts.join(" · ")
    : "your defaults already match best";
  const diffs = Object.entries(best)
    .filter(([k, v]) => FAST_DEFAULTS[k] !== v)
    .map(([k, v]) => `${k}=${v}`);
  return diffs.length ? changed : "no changes from the fast defaults won";
}

function CompareStrip({ cmp }: { cmp: DirectComparison }) {
  const W = 100 / Math.max(1, cmp.rows.length);
  return (
    <div className="flex flex-col gap-0.5">
      {(["user", "director"] as const).map((row) => (
        <div key={row} className="flex items-center gap-1">
          <span className="w-12 text-[9px] text-zinc-500 shrink-0">
            {row === "user" ? "yours" : "director"}
          </span>
          <div className="relative h-2.5 flex-1 rounded-sm overflow-hidden bg-zinc-800">
            {cmp.rows.map((r, i) => {
              const a = row === "user" ? r.user : r.director;
              return (
                <div key={i} className="absolute top-0 h-full"
                  style={{
                    left: `${i * W}%`, width: `${W}%`,
                    backgroundColor: a == null
                      ? "#3f3f46"
                      : ANGLE_COLORS[a % ANGLE_COLORS.length],
                  }} />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function YouDirect() {
  const api = useProjectApi();
  const [open, setOpen] = useState(false);
  const [sug, setSug] = useState<DirectSuggest | null>(null);
  const [fromIn, setFromIn] = useState("");
  const [toIn, setToIn] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<DirectSession[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [phase, setPhase] = useState<"idle" | "playing" | "paused" | "stopped">("idle");
  const [elapsed, setElapsed] = useState(0);
  const [choices, setChoices] = useState<Map<number, number>>(new Map());
  const [result, setResult] = useState<DirectComparison | null>(null);
  const [ready, setReady] = useState(false);
  const [learnRes, setLearnRes] = useState<DirectLearnResult | null>(null);
  const [sug3, setSug3] = useState<DirectSuggest[] | null>(null);
  const [learnBusy, setLearnBusy] = useState(false);
  const [learnMsg, setLearnMsg] = useState<string | null>(null);
  const vids = useRef<(HTMLVideoElement | null)[]>([]);
  const stage = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void api.directSessions().then((d) => setSessions(d.sessions)).catch(() => {});
  }, [api, result]);

  // after a stretch loads: seek every camera to its file offset and wait
  // for all of them to reach canplay so playback starts in sync
  useEffect(() => {
    if (!sug) return;
    setReady(false);
    let done = 0, cancelled = false;
    const tick = () => {
      done += 1;
      if (done >= sug.n_angles && !cancelled) setReady(true);
    };
    for (let i = 0; i < sug.n_angles; i++) {
      const v = vids.current[i];
      if (!v) { tick(); continue; }
      v.currentTime = sug.offsets[i] ?? 0;
      if (v.readyState >= 2) { tick(); continue; }
      const on = () => { v.removeEventListener("canplay", on); tick(); };
      v.addEventListener("canplay", on);
    }
    return () => { cancelled = true; };
  }, [sug]);

  const doLearn = () => {
    setLearnBusy(true); setLearnMsg(null); setError(null);
    void api.learnDirect(true)
      .then((r) => {
        setLearnRes(r.learn);
        const st = (r.job as { state?: string } | null)?.state;
        setLearnMsg(st === "queued" || st === "running"
          ? "re-cut queued with your style — watch the status bar above"
          : "learn saved");
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLearnBusy(false));
  };

  const load = (t?: [number, number]) => {
    setBusy(true); setError(null); setResult(null); setChoices(new Map());
    void api.directSuggest(t)
      .then(setSug)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  };
  const load3 = () => {
    setBusy(true); setError(null);
    void api.directSuggest3()
      .then((d) => setSug3(d.stretches))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  };
  const loadStretch = (s: DirectSuggest) => {
    setResult(null); setChoices(new Map());
    setFromIn(fmtClock(s.t_start)); setToIn(fmtClock(s.t_end));
    setSug(s);
  };
  const loadManual = () => {
    if (!sug) return;
    const s = fromIn.trim() ? parseClock(fromIn) : sug.match_window[0];
    const e = toIn.trim() ? parseClock(toIn) : sug.match_window[1];
    if (s == null || e == null || !(s < e)) {
      setError("bad range — use m:ss, from before to"); return;
    }
    load([s, e]);
  };

  // master clock: every 250 ms while playing, re-sync followers to angle0
  useEffect(() => {
    if (phase !== "playing" || !sug) return;
    const iv = setInterval(() => {
      const m = vids.current[0];
      if (!m) return;
      const mt = m.currentTime;
      setElapsed(mt - sug.offsets[0]);
      for (let i = 1; i < sug.n_angles; i++) {
        const v = vids.current[i];
        const target = mt - sug.offsets[0] + (sug.offsets[i] ?? 0);
        if (v && Math.abs(v.currentTime - target) > 0.3) {
          v.currentTime = target;
          if (v.paused) void v.play();   // a re-seek can stall a follower
        }
      }
      // record one choice per whole output second
      const sec = sug.t_start + Math.max(0, Math.floor(mt - sug.offsets[0]));
      if (sec >= sug.t_end) {
        stopAll();
      } else if (selected != null) {
        setChoices((c) => {
          if (c.get(sec) === selected) return c;
          const n = new Map(c); n.set(sec, selected); return n;
        });
      }
    }, 250);
    return () => clearInterval(iv);
  }, [phase, sug, selected]);

  // keyboard 1..n selects while playing
  useEffect(() => {
    if (phase !== "playing" || !sug) return;
    const h = (e: KeyboardEvent) => {
      const k = Number(e.key);
      if (k >= 1 && k <= sug.n_angles) setSelected(k - 1);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [phase, sug]);

  const start = () => {
    if (!sug) return;
    vids.current.forEach((v, i) => {
      if (v) { v.currentTime = sug.offsets[i] ?? 0; void v.play(); }
    });
    setElapsed(0); setChoices(new Map()); setSelected(null); setPhase("playing");
  };
  const pause = () => {
    vids.current.forEach((v) => v?.pause());
    setPhase("paused");
  };
  const resume = () => {
    vids.current.forEach((v) => void v?.play());
    setPhase("playing");
  };
  const stopAll = () => {
    vids.current.forEach((v) => v?.pause());
    setPhase("stopped");
  };
  const restart = () => {
    vids.current.forEach((v, i) => {
      if (v) v.currentTime = sug?.offsets[i] ?? 0;
    });
    setChoices(new Map()); setElapsed(0); setSelected(null); setPhase("idle");
  };
  const save = () => {
    if (!sug || phase !== "stopped") return;
    const ch = [...choices.entries()]
      .map(([t, angle]) => ({ t, angle }))
      .sort((a, b) => a.t - b.t);
    if (!ch.length) { setError("record at least one choice first"); return; }
    setBusy(true);
    void api.saveDirectSession(sug.t_start, sug.t_end, ch)
      .then((s) => setResult(s.comparison))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  };

  const dur = sug ? Math.max(0, sug.t_end - sug.t_start) : 0;

  return (
    // escape the tab's max-w-4xl column: full viewport width
    <div className={`${card} relative left-1/2 -translate-x-1/2 w-screen max-w-none`}>
      <button className="flex items-center gap-1.5 w-full text-left"
        onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} className="text-zinc-500" />
               : <ChevronRight size={14} className="text-zinc-500" />}
        <span className={head}>You direct (test)</span>
        {sessions.length > 0 && (
          <span className="ml-auto text-[10px] text-zinc-500">
            {sessions.length} session{sessions.length === 1 ? "" : "s"}
          </span>
        )}
      </button>
      {!open ? null : (
        <div className="mt-3 flex flex-col gap-3">
          {sessions.length >= 2 && !learnRes && (
            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-violet-700/50 bg-violet-950/40 px-3 py-2">
              <span className="text-xs text-violet-200">
                Directed {sessions.length} of 6 stretches — ready to
                analyse your directing and re-cut.
              </span>
              <button
                disabled={learnBusy}
                onClick={doLearn}
                className="text-[11px] font-semibold text-violet-300 hover:text-violet-200 border border-violet-700/60 rounded px-2 py-0.5 disabled:opacity-40">
                {learnBusy
                  ? <Loader2 size={11} className="animate-spin" />
                  : null}
                Analyse my directing &amp; re-cut
              </button>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <button disabled={busy}
              onClick={load3}
              className="text-[11px] font-semibold text-amber-300 hover:text-amber-200 border border-amber-700/60 rounded px-2 py-0.5 disabled:opacity-40">
              {busy ? <Loader2 size={11} className="animate-spin" /> : null}
              Suggest 6 stretches (2 early / 2 mid / 2 late)
            </button>
            <span className="text-[11px] text-zinc-500">or</span>
            <input className={`${input} w-16`} value={fromIn}
              placeholder={sug ? fmtClock(sug.match_window[0]) : "19:00"}
              onChange={(e) => setFromIn(e.target.value)} aria-label="from (m:ss)" />
            <span className="text-[11px] text-zinc-600">to</span>
            <input className={`${input} w-16`} value={toIn}
              placeholder={sug ? fmtClock(sug.match_window[1]) : "20:15"}
              onChange={(e) => setToIn(e.target.value)} aria-label="to (m:ss)" />
            <button disabled={busy || !sug} onClick={loadManual}
              className="text-[11px] text-sky-300 hover:text-sky-200 border border-sky-700/60 rounded px-2 py-0.5 disabled:opacity-40">
              Load range
            </button>
            {sug?.candidate && (
              <span className="text-[10px] text-zinc-500">
                busiest: {sug.candidate.type} @{fmtClock(sug.candidate.t)}
              </span>
            )}
          </div>
          {sug3 && (
            <div className="flex flex-col gap-1 text-[11px]">
              {sug3.map((s, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-10 text-zinc-500">
                    {["Early", "Mid", "Late"][i] ?? `#${i + 1}`}
                  </span>
                  <span className="font-mono text-zinc-300">
                    {fmtClock(s.t_start)}–{fmtClock(s.t_end)}
                  </span>
                  <button type="button"
                    className="text-sky-300 hover:text-sky-200 border border-sky-700/60 rounded px-2 py-0.5"
                    onClick={() => loadStretch(s)}>
                    Load
                  </button>
                </div>
              ))}
            </div>
          )}
          {error && <div className="text-xs text-red-300">{error}</div>}

          {sug && (
            <div ref={stage}
              className="flex flex-col gap-2 bg-zinc-950 rounded p-2">
              {/* program + preview: all three <video> elements stay mounted
                  always; the selected one is promoted to the big slot via
                  CSS order only, so playback never resets */}
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: sug.n_angles }, (_, i) => {
                  const big = selected === i;
                  const tile = selected != null && !big;
                  return (
                    <button key={i}
                      onClick={() => setSelected(i)}
                      className={`relative rounded overflow-hidden border-2 text-left bg-black
                        ${big ? "w-full order-first h-[70vh] border-amber-400"
                          : tile ? "w-[calc(33.333%-0.4rem)] h-[16vh] border-zinc-800 opacity-60"
                          : "w-[calc(33.333%-0.4rem)] h-[40vh] border-zinc-800"}`}>
                      <video
                        ref={(v) => { vids.current[i] = v; }}
                        src={api.angleVideoUrl(i)}
                        muted playsInline preload="auto"
                        className="w-full h-full object-contain pointer-events-none" />
                      <span className="absolute top-1 left-1 font-bold bg-zinc-950/80 rounded px-1.5"
                        style={{ color: ANGLE_COLORS[i % ANGLE_COLORS.length],
                                 fontSize: big ? "1.4rem" : "0.95rem" }}>
                        {i + 1}
                      </span>
                    </button>
                  );
                })}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {phase === "idle" && (
                  ready ? (
                    <button onClick={start}
                      className="flex items-center gap-1.5 text-sm font-semibold text-emerald-300 border border-emerald-700/60 rounded px-3 py-1.5">
                      <Play size={14} /> Play — press 1/{sug.n_angles} or tap a camera
                    </button>
                  ) : (
                    <span className="flex items-center gap-1.5 text-sm text-zinc-400 px-3 py-1.5">
                      <Loader2 size={14} className="animate-spin" /> loading cameras…
                    </span>
                  )
                )}
                {phase === "playing" && (
                  <>
                    <button onClick={pause}
                      className="flex items-center gap-1.5 text-sm font-semibold text-amber-300 border border-amber-700/60 rounded px-3 py-1.5">
                      <Pause size={14} /> Pause
                    </button>
                    <button onClick={stopAll}
                      className="flex items-center gap-1.5 text-sm font-semibold text-red-300 border border-red-700/60 rounded px-3 py-1.5">
                      <Square size={14} /> Stop
                    </button>
                  </>
                )}
                {phase === "paused" && (
                  <>
                    <button onClick={resume}
                      className="flex items-center gap-1.5 text-sm font-semibold text-emerald-300 border border-emerald-700/60 rounded px-3 py-1.5">
                      <Play size={14} /> Resume
                    </button>
                    <button onClick={stopAll}
                      className="flex items-center gap-1.5 text-sm font-semibold text-red-300 border border-red-700/60 rounded px-3 py-1.5">
                      <Square size={14} /> Stop
                    </button>
                  </>
                )}
                {phase === "stopped" && (
                  <button onClick={start}
                    className="flex items-center gap-1.5 text-sm font-semibold text-emerald-300 border border-emerald-700/60 rounded px-3 py-1.5">
                    <Play size={14} /> Watch again
                  </button>
                )}
                {(phase === "playing" || phase === "paused" || phase === "stopped") && (
                  <button onClick={restart}
                    className="flex items-center gap-1.5 text-sm text-zinc-300 border border-zinc-700 rounded px-3 py-1.5">
                    <RotateCcw size={14} /> Restart
                  </button>
                )}
                <button onClick={() => void stage.current?.requestFullscreen?.()}
                  className="flex items-center gap-1.5 text-sm text-zinc-300 border border-zinc-700 rounded px-3 py-1.5">
                  <Maximize size={14} /> Fullscreen
                </button>
                <div className="flex-1 min-w-24 h-2 rounded bg-zinc-800 overflow-hidden">
                  <div className="h-full bg-amber-400"
                    style={{ width: `${Math.min(100, elapsed / dur * 100)}%` }} />
                </div>
                <button disabled={busy || phase !== "stopped" || !choices.size} onClick={save}
                  className="text-sm font-semibold text-sky-300 border border-sky-700/60 rounded px-3 py-1.5 disabled:opacity-40">
                  Save &amp; compare
                </button>
              </div>
            </div>
          )}

          {result && (
            <div className="flex flex-col gap-2">
              <div className="text-xs text-zinc-300">
                agreement {result.agreement_pct ?? "—"}% · you switched{" "}
                {result.user_switches}× · director {result.director_switches}×
              </div>
              <CompareStrip cmp={result} />
              {Object.keys(result.disagree_by_pair).length > 0 && (
                <div className="text-[10px] text-zinc-500">
                  {Object.entries(result.disagree_by_pair).map(([k, n]) => (
                    <span key={k} className="mr-3">{k.replace("you:", "you ").replace("->dir:", " vs dir ")} · {n}s</span>
                  ))}
                </div>
              )}
              <div className="text-[10px] text-zinc-500">
                disagreed during: {Object.entries(result.disagree_by_rule)
                  .map(([r, v]) => `${r} ${v.n}/${v.of}s`).join(" · ") || "—"}
              </div>
            </div>
          )}

          {sessions.length > 0 && (
            <div className="flex flex-col gap-1">
              <div className={head}>Past sessions</div>
              {sessions.slice(0, 5).map((s) => (
                <button key={s.id}
                  onClick={() => setResult(s.comparison)}
                  className="flex items-center gap-2 text-[11px] text-zinc-400 hover:text-zinc-200 text-left">
                  <span className="font-mono">{s.id}</span>
                  <span>{fmtClock(s.t_start)}–{fmtClock(s.t_end)}</span>
                  <span className="text-amber-300/80">
                    {s.comparison.agreement_pct ?? "—"}% agree
                  </span>
                </button>
              ))}
            </div>
          )}

          {/* learn from my directing */}
          <div className="border-t border-zinc-800 pt-3 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className={head}>Learn from my directing</span>
              <span className="text-[10px] text-zinc-500">
                {sessions.length} session{sessions.length === 1 ? "" : "s"} saved
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                disabled={learnBusy || sessions.length === 0}
                onClick={doLearn}
                className="text-[11px] font-semibold text-violet-300 hover:text-violet-200 border border-violet-700/60 rounded px-2 py-0.5 disabled:opacity-40">
                {learnBusy
                  ? <Loader2 size={11} className="animate-spin" />
                  : null}
                Analyse my directing &amp; re-cut
              </button>
              {sessions.length === 0 && (
                <span className="text-[10px] text-zinc-600">
                  direct a stretch and save it first
                </span>
              )}
            </div>
            {learnRes && (
              <div className="text-xs text-zinc-300 flex flex-col gap-1">
                <div>
                  agreement {learnRes.agreement_pct_before ?? "—"}% →{" "}
                  <span className="text-amber-300 font-semibold">
                    {learnRes.agreement_pct_after ?? "—"}%
                  </span>
                </div>
                <div className="text-[10px] text-zinc-500">
                  {describeOverrides(learnRes.best)}
                </div>
                {learnRes.zone_source !== undefined && (
                  <div className="text-[10px] text-zinc-500">
                    Learned camera areas from your sessions: agreement drawn{" "}
                    {learnRes.zone_agreement?.drawn ?? "—"}% · learned{" "}
                    {learnRes.zone_agreement?.learned ?? "—"}% · none{" "}
                    {learnRes.zone_agreement?.none ?? "—"}% → using{" "}
                    {learnRes.zone_source}
                  </div>
                )}
                {learnMsg && (
                  <div className="text-[10px] text-emerald-400">{learnMsg}</div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
