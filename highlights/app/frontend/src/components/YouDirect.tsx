import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Loader2, Play, Square } from "lucide-react";
import { useProjectApi } from "../api";
import { fmtClock, parseClock } from "../lib/time";
import { ANGLE_COLORS } from "./DirectorCut";
import type { DirectComparison, DirectSession, DirectSuggest } from "../types";

const card = "rounded-lg border border-zinc-800 bg-zinc-900 p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const input =
  "bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs font-mono placeholder:text-zinc-500 focus:outline-none focus:border-amber-400";

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
  const [selected, setSelected] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [choices, setChoices] = useState<Map<number, number>>(new Map());
  const [result, setResult] = useState<DirectComparison | null>(null);
  const vids = useRef<(HTMLVideoElement | null)[]>([]);
  const t0 = useRef(0);

  useEffect(() => {
    void api.directSessions().then((d) => setSessions(d.sessions)).catch(() => {});
  }, [api, result]);

  const load = (t?: [number, number]) => {
    setBusy(true); setError(null); setResult(null); setChoices(new Map());
    void api.directSuggest(t)
      .then(setSug)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
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
    if (!playing || !sug) return;
    const iv = setInterval(() => {
      const m = vids.current[0];
      if (!m) return;
      const mt = m.currentTime;
      setElapsed(mt - sug.offsets[0]);
      for (let i = 1; i < sug.n_angles; i++) {
        const v = vids.current[i];
        if (v && Math.abs(v.currentTime - mt) > 0.3) v.currentTime = mt;
      }
      // record one choice per whole output second
      const sec = sug.t_start + Math.max(0, Math.floor(mt - sug.offsets[0]));
      if (sec < sug.t_end) {
        setChoices((c) => {
          if (c.get(sec) === selected) return c;
          const n = new Map(c); n.set(sec, selected); return n;
        });
      } else {
        stopAll();
      }
    }, 250);
    return () => clearInterval(iv);
  }, [playing, sug, selected]);

  // keyboard 1..n selects while playing
  useEffect(() => {
    if (!playing || !sug) return;
    const h = (e: KeyboardEvent) => {
      const k = Number(e.key);
      if (k >= 1 && k <= sug.n_angles) setSelected(k - 1);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [playing, sug]);

  const start = () => {
    if (!sug) return;
    vids.current.forEach((v, i) => {
      if (v) { v.currentTime = sug.offsets[i] ?? 0; void v.play(); }
    });
    t0.current = Date.now();
    setElapsed(0); setChoices(new Map()); setSelected(0); setPlaying(true);
  };
  const stopAll = () => {
    vids.current.forEach((v) => v?.pause());
    setPlaying(false);
  };
  const save = () => {
    if (!sug) return;
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
    <div className={card}>
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
          <div className="flex flex-wrap items-center gap-2">
            <button disabled={busy}
              onClick={() => load()}
              className="text-[11px] font-semibold text-amber-300 hover:text-amber-200 border border-amber-700/60 rounded px-2 py-0.5 disabled:opacity-40">
              {busy ? <Loader2 size={11} className="animate-spin" /> : null}
              Load suggested stretch
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
          {error && <div className="text-xs text-red-300">{error}</div>}

          {sug && (
            <>
              <div className="grid gap-2 md:grid-cols-3 grid-cols-1">
                {Array.from({ length: sug.n_angles }, (_, i) => (
                  <button key={i}
                    onClick={() => setSelected(i)}
                    className={`relative rounded overflow-hidden border-2 text-left ${
                      selected === i
                        ? "border-amber-400"
                        : "border-zinc-800 opacity-60"}`}>
                    <video
                      ref={(v) => { vids.current[i] = v; }}
                      src={api.angleVideoUrl(i)}
                      muted playsInline preload="auto"
                      className="w-full aspect-video bg-black pointer-events-none" />
                    <span className="absolute top-1 left-1 text-base font-bold bg-zinc-950/80 rounded px-1.5"
                      style={{ color: ANGLE_COLORS[i % ANGLE_COLORS.length] }}>
                      {i + 1}
                    </span>
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                {!playing ? (
                  <button onClick={start}
                    className="flex items-center gap-1 text-[11px] font-semibold text-emerald-300 border border-emerald-700/60 rounded px-2 py-0.5">
                    <Play size={11} /> Play — press 1/{sug.n_angles} or tap a camera
                  </button>
                ) : (
                  <button onClick={stopAll}
                    className="flex items-center gap-1 text-[11px] font-semibold text-red-300 border border-red-700/60 rounded px-2 py-0.5">
                    <Square size={11} /> Stop
                  </button>
                )}
                <div className="flex-1 h-1.5 rounded bg-zinc-800 overflow-hidden">
                  <div className="h-full bg-amber-400"
                    style={{ width: `${Math.min(100, elapsed / dur * 100)}%` }} />
                </div>
                <button disabled={busy || !choices.size} onClick={save}
                  className="text-[11px] text-sky-300 border border-sky-700/60 rounded px-2 py-0.5 disabled:opacity-40">
                  Save &amp; compare
                </button>
              </div>
            </>
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
        </div>
      )}
    </div>
  );
}
