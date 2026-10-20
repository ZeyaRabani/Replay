import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Loader2, Pause, Play, Scissors, Trash2 } from "lucide-react";
import { useProjectApi } from "../api";
import { fmtClock } from "../lib/time";
import { ANGLE_COLORS } from "./DirectorCut";
import type { DirectSuggest, ManualState } from "../types";

const card = "card p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";

export default function ManualMode() {
  const api = useProjectApi();
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState<ManualState | null>(null);
  const [sug, setSug] = useState<DirectSuggest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [cur, setCur] = useState(0);
  const [sec, setSec] = useState(0);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const vids = useRef<(HTMLVideoElement | null)[]>([]);
  const choices = useRef(new Map<number, number>());
  const dirty = useRef(new Set<number>());

  const lo = info?.match_window[0] ?? 0;
  const hi = info?.match_window[1] ?? 0;
  const dur = Math.max(0, hi - lo);

  // load state + per-angle file offsets (via directSuggest's offsets)
  const load = useCallback(() => {
    setError(null);
    void api.manualGet()
      .then((d) => {
        setInfo(d);
        choices.current = new Map(d.choices.map((c) => [c.t, c.angle]));
        dirty.current = new Set();
        const resume = d.choices.length ? d.choices[d.choices.length - 1].t + 1 : d.match_window[0];
        const prev = d.choices.filter((c) => c.t < resume).pop();
        setCur(prev ? prev.angle : 0);
        setSec(Math.min(resume, d.match_window[1]));
        return api.directSuggest([d.match_window[0], d.match_window[1]]);
      })
      .then(setSug)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [api]);

  useEffect(() => { if (open && !info) load(); }, [open, info, load]);

  const flush = useCallback(() => {
    if (!dirty.current.size) return;
    const ch = [...dirty.current].sort((a, b) => a - b)
      .map((t) => ({ t, angle: choices.current.get(t)! }));
    dirty.current = new Set();
    void api.manualPut(ch)
      .then((d) => {
        setInfo(d);
        setSavedAt(new Date().toLocaleTimeString("en-GB"));
      })
      .catch(() => ch.forEach((c) => dirty.current.add(c.t)));
  }, [api]);

  const seekAll = useCallback((shared: number) => {
    if (!sug) return;
    const base = shared - lo;
    vids.current.forEach((v, i) => {
      if (v) v.currentTime = (sug.offsets[i] ?? 0) + base;
    });
    setSec(shared);
  }, [sug, lo]);

  const mark = useCallback((t: number, a: number) => {
    choices.current.set(t, a);
    dirty.current.add(t);
  }, []);

  // master clock: every 250 ms while playing, re-sync followers to angle 0
  // and record each whole shared second with the selected camera
  useEffect(() => {
    if (!playing || !sug) return;
    const iv = setInterval(() => {
      const m = vids.current[0];
      if (!m) return;
      const mt = m.currentTime;
      const shared = mt - (sug.offsets[0] ?? 0) + lo;
      setSec(shared);
      for (let i = 1; i < sug.n_angles; i++) {
        const v = vids.current[i];
        const target = mt - (sug.offsets[0] ?? 0) + (sug.offsets[i] ?? 0);
        if (v && Math.abs(v.currentTime - target) > 0.3) {
          v.currentTime = target;
          if (v.paused) void v.play();
        }
      }
      mark(Math.floor(shared), cur);
      if (shared >= hi) setPlaying(false);
    }, 250);
    return () => clearInterval(iv);
  }, [playing, sug, lo, hi, cur, mark]);

  const play = () => {
    vids.current.forEach((v) => void v?.play());
    setPlaying(true);
  };
  const pause = useCallback(() => {
    vids.current.forEach((v) => v?.pause());
    setPlaying(false);
    flush();
  }, [flush]);

  // keyboard while open: 1..n camera, Space pause, arrows skip 5 s
  useEffect(() => {
    if (!open || !sug) return;
    const h = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") return;
      const k = Number(e.key);
      if (k >= 1 && k <= sug.n_angles) { setCur(k - 1); return; }
      if (e.key === " ") {
        e.preventDefault();
        if (playing) pause(); else play();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        const s0 = Math.floor(sec);
        for (let s = 1; s <= 5; s++) {
          if (s0 + s < hi) mark(s0 + s, cur);
        }
        seekAll(Math.min(hi, sec + 5));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        seekAll(Math.max(lo, sec - 5));
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, sug, playing, pause, sec, lo, hi, cur, mark, seekAll]);

  // autosave every 10 s while dirty; on close
  useEffect(() => {
    const iv = setInterval(flush, 10000);
    return () => clearInterval(iv);
  }, [flush]);
  useEffect(() => { if (!open) flush(); }, [open, flush]);

  const makeCut = () => {
    setBusy(true); setError(null); setNote(null);
    void api.manualPut(
        [...choices.current.keys()].sort((a, b) => a - b)
          .map((t) => ({ t, angle: choices.current.get(t)! })))
      .then(() => api.manualCut())
      .then(() => setNote(
        "Rendering your cut — about 1–1.5 h; the AI cut stays available in the cuts list"))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  };
  const startOver = () => {
    if (!window.confirm("Delete all your manual choices?")) return;
    setBusy(true); setError(null);
    void api.manualDelete()
      .then((d) => {
        setInfo(d);
        choices.current = new Map();
        dirty.current = new Set();
        setCur(0); setSec(d.match_window[0]);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  };

  const pct = info && dur > 0
    ? ((choices.current.size / dur) * 100).toFixed(1) : "0";
  const W = 100 / Math.max(1, Math.ceil(dur));

  return (
    <div className={`${card} relative left-1/2 -translate-x-1/2 w-screen max-w-none`}>
      <button className="flex items-center gap-1.5 w-full text-left"
        onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} className="text-zinc-500" />
               : <ChevronRight size={14} className="text-zinc-500" />}
        <span className={head}>Manual mode — direct the whole match yourself</span>
        {info && (
          <span className="ml-auto text-[10px] text-zinc-500">
            {pct}% chosen by you; the rest uses the AI's choice
          </span>
        )}
      </button>
      {!open ? (
        <div className="mt-2">
          <button onClick={() => setOpen(true)}
            className="text-[11px] font-semibold text-amber-300 hover:text-amber-200 border border-amber-700/60 rounded px-2 py-0.5">
            Open
          </button>
        </div>
      ) : (
        <div className="mt-3 flex flex-col gap-3">
          {error && <div className="text-xs text-red-300">{error}</div>}
          {note && <div className="text-xs text-emerald-300">{note}</div>}
          {!sug ? (
            <span className="flex items-center gap-1.5 text-sm text-zinc-400">
              <Loader2 size={14} className="animate-spin" /> loading cameras…
            </span>
          ) : (
            <div className="flex flex-col gap-2 bg-zinc-950 rounded p-2">
              {/* program + previews: all videos stay mounted; selected is
                  promoted via CSS order so playback never resets */}
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: sug.n_angles }, (_, i) => {
                  const big = cur === i;
                  return (
                    <button key={i}
                      onClick={() => setCur(i)}
                      className={`relative rounded overflow-hidden border-2 text-left bg-black
                        ${big ? "w-full order-first h-[70vh] border-amber-400"
                          : "w-[calc(33.333%-0.4rem)] h-[16vh] border-zinc-800 opacity-60"}`}>
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
              {/* whole-match timeline: chosen seconds per angle + playhead */}
              <div className="relative h-3 rounded-sm overflow-hidden bg-zinc-800 cursor-pointer"
                onClick={(e) => {
                  const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
                  const s = lo + ((e.clientX - r.left) / r.width) * dur;
                  seekAll(Math.min(hi - 0.01, Math.max(lo, s)));
                }}>
                {[...choices.current.entries()].map(([t, a]) => (
                  <div key={t} className="absolute top-0 h-full"
                    style={{
                      left: `${((t - lo) / dur) * 100}%`, width: `${W}%`,
                      backgroundColor: ANGLE_COLORS[a % ANGLE_COLORS.length],
                    }} />
                ))}
                <div className="absolute top-0 h-full w-0.5 bg-white"
                  style={{ left: `${((sec - lo) / dur) * 100}%` }} />
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {playing ? (
                  <button onClick={pause}
                    className="flex items-center gap-1.5 text-sm font-semibold text-amber-300 border border-amber-700/60 rounded px-3 py-1.5">
                    <Pause size={14} /> Pause
                  </button>
                ) : (
                  <button onClick={play}
                    className="flex items-center gap-1.5 text-sm font-semibold text-emerald-300 border border-emerald-700/60 rounded px-3 py-1.5">
                    <Play size={14} /> Play — 1/{sug.n_angles} switch, ←/→ 5 s
                  </button>
                )}
                <span className="font-mono text-xs text-zinc-400">
                  {fmtClock(sec)} / {fmtClock(hi)}
                </span>
                {savedAt && (
                  <span className="text-[10px] text-zinc-500">saved {savedAt}</span>
                )}
                <span className="flex-1" />
                <button disabled={busy || !choices.current.size}
                  onClick={makeCut}
                  className="flex items-center gap-1.5 text-sm font-semibold text-sky-300 border border-sky-700/60 rounded px-3 py-1.5 disabled:opacity-40">
                  {busy ? <Loader2 size={14} className="animate-spin" /> : <Scissors size={14} />}
                  Make my director's cut
                </button>
                <button disabled={busy}
                  onClick={startOver}
                  className="flex items-center gap-1.5 text-sm font-semibold text-red-300 border border-red-700/60 rounded px-3 py-1.5 disabled:opacity-40">
                  <Trash2 size={14} /> Start over
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
