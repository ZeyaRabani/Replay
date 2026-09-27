import { AlertTriangle, ChevronDown, ChevronRight, Loader2, Play, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useProjectApi } from "../api";
import type { AnalysisEvent, AnalysisMatchStats, AnalysisResponse, AnalysisTeamInfo, AnalysisTerritory } from "../types";

const card = "rounded-lg border border-zinc-800 bg-zinc-900 p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-2";
const thCls = "pb-1 font-medium text-left text-zinc-500";
const tdCls = "py-1 border-b border-zinc-800/50";

const mmss = (s: number): string => {
  if (!Number.isFinite(s)) return "-:--";
  const t = Math.max(0, Math.round(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const dist = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);

function Swatch({ team, size = 14 }: { team: AnalysisTeamInfo; size?: number }) {
  const isLight = (team.hex || "").toLowerCase() > "#aaaaaa";
  return (
    <span
      role="img"
      aria-label={`${team.name} shirt colour`}
      title={team.hex}
      className={`inline-block rounded shrink-0 ${isLight ? "border border-zinc-500" : "border border-zinc-700"}`}
      style={{ backgroundColor: team.hex || "#71717a", width: size, height: size }}
    />
  );
}

function PossBar({ label, a, b, teams }: { label: string; a: number; b: number; teams: AnalysisMatchStats["teams"] }) {
  const tot = a + b;
  const pctA = tot > 0 ? (a / tot) * 100 : 50;
  return (
    <div>
      <div className="flex justify-between text-[11px] text-zinc-400 mb-0.5">
        <span>{label}</span>
        <span>
          {cap(teams.A.name)} {Math.round(a)}% · {cap(teams.B.name)} {Math.round(b)}%
        </span>
      </div>
      <div className="flex h-4 rounded overflow-hidden border border-zinc-800">
        <div
          className="flex items-center justify-center text-[10px] font-semibold text-zinc-900 min-w-0"
          style={{ width: `${pctA}%`, backgroundColor: teams.A.hex || "#71717a" }}
        >
          {pctA >= 12 ? `${Math.round(a)}%` : ""}
        </div>
        <div
          className="flex items-center justify-center text-[10px] font-semibold text-zinc-900 min-w-0"
          style={{ width: `${100 - pctA}%`, backgroundColor: teams.B.hex || "#52525b" }}
        >
          {100 - pctA >= 12 ? `${Math.round(b)}%` : ""}
        </div>
      </div>
    </div>
  );
}

function ThirdsBar({ label, t }: { label: string; t: AnalysisTerritory | null }) {
  return (
    <div>
      <div className="flex justify-between text-[11px] text-zinc-400 mb-0.5">
        <span>{label}</span>
        {t ? (
          <span>
            def {Math.round(t.def * 100)}% · mid {Math.round(t.mid * 100)}% · att {Math.round(t.att * 100)}%
          </span>
        ) : (
          <span className="text-zinc-600">n/a</span>
        )}
      </div>
      <div className="flex h-4 rounded overflow-hidden border border-zinc-800 text-[10px] text-zinc-900 font-semibold">
        {(["def", "mid", "att"] as const).map((k) => {
          const v = t ? t[k] : 0;
          const bg = k === "def" ? "#3f6212" : k === "mid" ? "#3f3f46" : "#a16207";
          return (
            <div
              key={k}
              className="flex items-center justify-center min-w-0"
              style={{ width: `${v * 100}%`, backgroundColor: bg }}
              title={`${k} ${Math.round(v * 100)}%`}
            >
              {v * 100 >= 14 ? `${Math.round(v * 100)}%` : ""}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MomentumChart({ bins, teams, onSeek }: {
  bins: AnalysisMatchStats["momentum"];
  teams: AnalysisMatchStats["teams"];
  onSeek?: (t: number) => void;
}) {
  const usable = bins.filter((b) => b.value !== null);
  if (!usable.length) return null;
  const H = 72;
  return (
    <div>
      <div className="flex justify-between text-[11px] text-zinc-500 mb-1">
        <span>
          {cap(teams.A.name)} pressure
        </span>
        <span>
          {cap(teams.B.name)} pressure
        </span>
      </div>
      <div className="relative overflow-x-auto" style={{ height: H }}>
        <div className="absolute left-0 right-0 border-t border-zinc-700" style={{ top: H / 2 }} />
        <div className="flex gap-px" style={{ height: H }}>
          {bins.map((b, i) => {
            const v = b.value ?? 0;
            const mag = Math.abs(v);
            const h = Math.max(2, Math.round(mag * (H / 2 - 4)));
            const color = v > 0 ? teams.A.hex || "#71717a" : v < 0 ? teams.B.hex || "#52525b" : "#3f3f46";
            return (
              <button
                key={i}
                disabled={b.value === null || !onSeek}
                onClick={() => onSeek?.(b.t_start_out)}
                title={`${mmss(b.t_start_out)} — ${v >= 0 ? cap(teams.A.name) : cap(teams.B.name)} ${Math.round(mag * 100)}%`}
                className="flex-1 min-w-[6px] relative disabled:opacity-40"
                style={{ height: H }}
              >
                <span
                  className="absolute left-0 right-0 rounded-sm"
                  style={{
                    height: h,
                    top: v >= 0 ? H / 2 - h : H / 2,
                    backgroundColor: color,
                  }}
                />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function MatchAnalysis({ onSeek }: { onSeek?: (t: number) => void }) {
  const api = useProjectApi();
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(true);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setData(await api.analysis());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const live = data?.status?.state === "queued" || data?.status?.state === "running";
  useEffect(() => {
    if (live) setOpen(true);
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
    if (!live) return;
    timer.current = window.setInterval(() => void refresh(), 3000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [live, refresh]);

  const start = async (force = false) => {
    setBusy(true);
    try {
      await api.analyse(force);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const status = data?.status;
  const stats = data?.stats;
  const summary = data?.summary;
  const failed = status?.state === "failed";

  // collapsible header + action button (always visible)
  const header = (
    <div className={`${card} flex items-center gap-2`}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-400 hover:text-zinc-200"
        aria-expanded={open}
      >
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        Team analysis (optional)
      </button>
      <span className="flex-1" />
      {stats && !live && (
        <button
          disabled={busy}
          onClick={() => void start(true)}
          title="Re-run the analysis (forces a fresh teams pass)"
          className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 disabled:opacity-40 shrink-0"
        >
          {busy ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
          Re-analyse
        </button>
      )}
    </div>
  );

  if (!data) {
    return (
      <div className="flex flex-col gap-3">
        {header}
        <div className="text-sm text-zinc-500 flex items-center gap-2">
          {!error && <Loader2 size={14} className="animate-spin" />}
          {error ?? "Loading…"}
        </div>
      </div>
    );
  }

  if (!open) {
    return <div className="flex flex-col gap-3">{header}</div>;
  }

  // ----- running / queued -----------------------------------------
  if (live && status) {
    const pct = Math.round((status.progress ?? 0) * 100);
    return (
      <div className="flex flex-col gap-3">
        {header}
        <div className={card}>
          <div className="flex items-center gap-3 text-xs">
            <Loader2 size={14} className="animate-spin text-amber-400 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex justify-between gap-2">
                <span className="text-zinc-300 truncate">
                  {status.state === "queued"
                    ? "Waiting for another job to finish…"
                    : `${status.stage ?? "working"} — ${status.message ?? ""}`}
                </span>
                <span className="text-zinc-400 shrink-0">{pct}%</span>
              </div>
              <div className="h-1.5 bg-zinc-800 rounded mt-1.5">
                <div className="h-1.5 rounded bg-amber-400 transition-all" style={{ width: `${pct}%` }} />
              </div>
              <div className="text-[11px] text-zinc-500 mt-1">
                You can leave this page — the analysis keeps running.
              </div>
            </div>
          </div>
        </div>
        {error && <div className="text-xs text-red-300">{error}</div>}
      </div>
    );
  }

  // ----- not analysed / failed -------------------------------------
  if (!stats) {
    return (
      <div className="flex flex-col gap-3">
        {header}
        {error && <div className="text-xs text-red-300">{error}</div>}
        <div className={card}>
          <p className="text-xs text-zinc-400 mb-3">
            Runs a shirt-colour team pass over the main camera and estimates possession, territory and
            momentum per half. Confirmed shots/goals come only from events you confirm in Review.
            Estimated from footage — not official.
          </p>
          {failed && (
            <div className="mb-3 rounded border border-red-800 bg-red-950/60 text-red-200 text-xs p-2 flex items-start gap-2">
              <AlertTriangle size={13} className="mt-0.5 shrink-0" />
              <div className="flex-1">{status?.error || status?.message || "analysis failed"}</div>
            </div>
          )}
          <button
            disabled={busy}
            onClick={() => void start(failed)}
            className="flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5 text-xs disabled:opacity-40"
          >
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
            {failed ? "Retry analysis" : "Analyse match"} (~{data.estimate_min} min)
          </button>
        </div>
      </div>
    );
  }

  // ----- done -------------------------------------------------------
  const teams = stats.teams;
  const events = stats.events ?? [];
  const unreviewed = stats.unreviewed ?? [];
  const halfLabel = (i: number) => (i === 1 ? "1st half" : i === 2 ? "2nd half" : `Half ${i}`);

  const eventRow = (s: AnalysisEvent, key: number | string) => {
    const team = s.team === "A" ? teams.A : s.team === "B" ? teams.B : null;
    const isGoal = s.kind === "goal";
    return (
      <li
        key={key}
        className={`flex items-center gap-2 text-xs py-1 border-b border-zinc-800/50 ${
          isGoal ? "bg-emerald-950/30" : ""
        }`}
      >
        <button
          onClick={() => onSeek?.(s.t_out)}
          className="font-mono text-amber-300 hover:text-amber-200 shrink-0"
          aria-label={`seek to ${s.mmss}`}
        >
          {s.mmss}
        </button>
        <span className="font-mono text-[10px] text-zinc-500 shrink-0">({mmss(s.t_file)} cam)</span>
        <span className={`capitalize ${isGoal ? "text-emerald-300 font-medium" : "text-zinc-300"}`}>
          {s.kind}
        </span>
        {team && (
          <span className="flex items-center gap-1 text-zinc-400">
            <Swatch team={team} size={10} />
            {team.name}
          </span>
        )}
        {s.attribution === "low" && (
          <span className="rounded bg-zinc-800 px-1 py-0.5 text-[9px] text-zinc-500" title="team attribution uncertain">
            ~
          </span>
        )}
        <span className="ml-auto text-[10px] text-zinc-600">{halfLabel(s.half)}</span>
      </li>
    );
  };

  return (
    <div className="flex flex-col gap-3">
      {header}
      {error && <div className="text-xs text-red-300">{error}</div>}

      {/* team header + score */}
      <div className={card}>
        <div className="flex-1 flex items-center justify-center gap-3">
          <span className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
            <Swatch team={teams.A} size={16} /> {cap(teams.A.name)}
          </span>
          <span className="text-sm font-mono text-zinc-200">
            {stats.totals.A.goals} – {stats.totals.B.goals}
          </span>
          <span className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
            <Swatch team={teams.B} size={16} /> {cap(teams.B.name)}
          </span>
        </div>
        <div className="flex justify-center mt-1.5 gap-1.5 flex-wrap">
          <span className="rounded px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-400">
            score counts confirmed goals only
          </span>
          {stats.team_confidence != null && (
            <span className="rounded px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-300">
              team ID confidence {Math.round(stats.team_confidence * 100)}%
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-3 dsk:grid-cols-2">
        {/* possession */}
        <div className={card}>
          <div className={head}>Possession</div>
          <div className="flex flex-col gap-3">
            {stats.halves.map((h) => (
              <div key={h.index}>
                <PossBar
                  label={`${halfLabel(h.index)} (${mmss(h.start_out)}–${mmss(h.end - stats.lo_out)})`}
                  a={h.teams.A.possession_pct}
                  b={h.teams.B.possession_pct}
                  teams={teams}
                />
                <div className="text-[10px] text-zinc-500 mt-0.5">
                  contested {Math.round(h.contested_pct)}% · ball in play {Math.round(h.ball_visible_pct)}%
                </div>
              </div>
            ))}
            <PossBar
              label="Match"
              a={stats.totals.A.possession_pct}
              b={stats.totals.B.possession_pct}
              teams={teams}
            />
          </div>
        </div>

        {/* territory thirds */}
        <div className={card}>
          <div className={head}>Territory — thirds</div>
          <div className="flex flex-col gap-3">
            {stats.halves.map((h) => (
              <div key={h.index}>
                <div className="text-[10px] text-zinc-500 mb-1">{halfLabel(h.index)}</div>
                <ThirdsBar label={cap(teams.A.name)} t={h.teams.A.territory} />
                <div className="mt-1">
                  <ThirdsBar label={cap(teams.B.name)} t={h.teams.B.territory} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* momentum */}
      {stats.momentum.some((b) => b.value !== null) && (
        <div className={card}>
          <div className={head}>Momentum (5-min bins)</div>
          <MomentumChart bins={stats.momentum} teams={teams} onSeek={onSeek} />
          <div className="text-[10px] text-zinc-500 mt-1">
            share of ball-in-play time in {cap(teams.A.name)}'s attacking half minus {cap(teams.B.name)}'s
          </div>
        </div>
      )}

      {/* shots & goals (confirmed only) */}
      <div className={card}>
        <div className={head}>Shots &amp; goals (confirmed)</div>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800">
              <th className={thCls}>Team</th>
              <th className={thCls}>Shots</th>
              <th className={thCls}>Goals</th>
              <th className={thCls}>Attacking third</th>
            </tr>
          </thead>
          <tbody>
            {(["A", "B"] as const).map((t) => {
              const tt = stats.totals[t];
              return (
                <tr key={t} className="text-zinc-300">
                  <td className={tdCls}>
                    <span className="flex items-center gap-1.5">
                      <Swatch team={teams[t]} size={10} /> {cap(teams[t].name)}
                    </span>
                  </td>
                  <td className={tdCls}>{tt.shots}</td>
                  <td className={tdCls}>{tt.goals}</td>
                  <td className={`${tdCls} font-mono`}>{mmss(tt.attacking_third_s)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* player stats */}
      <div className={card}>
        <div className="flex items-center gap-2 mb-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            Estimated player stats
          </div>
          <span className="rounded px-1.5 py-0.5 text-[10px] bg-amber-900/50 text-amber-200">
            rough
          </span>
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800">
              <th className={thCls}>Team</th>
              <th className={thCls}>Distance (est.)</th>
              <th className={thCls}>Sprints</th>
              <th className={thCls}>Tracked player-s</th>
            </tr>
          </thead>
          <tbody>
            {(["A", "B"] as const).map((t) => (
              <tr key={t} className="text-zinc-300">
                <td className={tdCls}>
                  <span className="flex items-center gap-1.5">
                    <Swatch team={teams[t]} size={10} /> {cap(teams[t].name)}
                  </span>
                </td>
                <td className={tdCls}>{dist(stats.totals[t].distance_m_est)}</td>
                <td className={tdCls}>{stats.totals[t].sprints}</td>
                <td className={tdCls}>{stats.totals[t].tracked_player_seconds}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-[10px] text-zinc-500 mt-1.5">
          per-team totals over tracked players only
        </div>
      </div>

      {/* confirmed events */}
      {events.length > 0 && (
        <div className={card}>
          <div className={head}>Confirmed events ({events.length})</div>
          <ul className="max-h-72 overflow-auto">{events.map((s, i) => eventRow(s, i))}</ul>
        </div>
      )}

      {/* unreviewed AI chances */}
      {(stats.n_unreviewed ?? unreviewed.length) > 0 && (
        <details className={card}>
          <summary className="text-xs text-zinc-500 cursor-pointer">
            {stats.n_unreviewed ?? unreviewed.length} unreviewed AI chances (top {unreviewed.length})
          </summary>
          <ul className="mt-2 max-h-60 overflow-auto">
            {unreviewed.map((s, i) => (
              <li key={i} className="flex items-center gap-2 text-xs py-1 border-b border-zinc-800/50 text-zinc-400">
                <button
                  onClick={() => onSeek?.(s.t_out)}
                  className="font-mono text-amber-300 hover:text-amber-200 shrink-0"
                >
                  {s.mmss}
                </button>
                <span className="capitalize">{s.type}</span>
                <span className="text-zinc-600">{s.status}</span>
                <span className="ml-auto text-zinc-500">{Math.round(s.confidence * 100)}%</span>
                <span className="text-[10px] text-zinc-600">{halfLabel(s.half)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* summary */}
      {summary && (
        <div className={card}>
          <div className={head}>Summary</div>
          <p className="text-xs text-zinc-300 leading-relaxed">{summary.text}</p>
          {summary.bullets.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {summary.bullets.map((b, i) => (
                <button
                  key={i}
                  onClick={() => onSeek?.(b.t_out)}
                  className="rounded-full border border-zinc-700 bg-zinc-800 hover:bg-zinc-700 text-[11px] text-zinc-200 px-2.5 py-0.5"
                  aria-label={`seek to ${mmss(b.t_out)}`}
                >
                  <span className="font-mono text-amber-300">{mmss(b.t_out)}</span> {b.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* caveats */}
      {stats.caveats.length > 0 && (
        <div className="rounded-lg border border-amber-900/60 bg-amber-950/30 p-4">
          <div className="text-xs font-semibold text-amber-200 mb-1">
            Estimated from footage — not official
          </div>
          <ul className="list-disc list-inside text-[11px] text-amber-100/80 flex flex-col gap-0.5">
            {stats.caveats.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
