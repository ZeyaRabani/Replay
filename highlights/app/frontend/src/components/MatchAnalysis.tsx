import { AlertTriangle, Loader2, Play, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useProjectApi } from "../api";
import type { AnalysisMatchStats, AnalysisResponse, AnalysisShot, AnalysisTeamInfo } from "../types";
import { useLayout } from "../lib/layout";

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

export default function MatchAnalysis({ onSeek }: { onSeek?: (t: number) => void }) {
  const api = useProjectApi();
  const { isMobile } = useLayout();
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
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

  if (!data) {
    return (
      <div className="flex flex-col gap-3">
        <div className="text-sm text-zinc-500 flex items-center gap-2">
          {!error && <Loader2 size={14} className="animate-spin" />}
          {error ?? "Loading…"}
        </div>
      </div>
    );
  }

  const status = data.status;
  const stats = data.stats;
  const summary = data.summary;

  // ----- running / queued -----------------------------------------
  if (live && status) {
    const pct = Math.round((status.progress ?? 0) * 100);
    return (
      <div className="flex flex-col gap-3">
          <div className={card}>
            <div className={head}>Match analysis</div>
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
    const failed = status?.state === "failed";
    return (
      <div className="flex flex-col gap-3">
          {error && <div className="text-xs text-red-300">{error}</div>}
          <div className={card}>
            <div className={head}>Match analysis</div>
            <p className="text-xs text-zinc-400 mb-3">
              Runs a shirt-colour team pass over the main camera and estimates possession, shots and
              goals per half. Estimated from footage — not official.
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
  const shots = [...(stats.shots ?? [])].sort((x, y) => x.t_shared - y.t_shared);
  const halfLabel = (i: number) => (i === 1 ? "1st half" : i === 2 ? "2nd half" : `Half ${i}`);

  const shotRow = (s: AnalysisShot, key: number | string) => {
    const team = s.team === "A" ? teams.A : s.team === "B" ? teams.B : null;
    const isGoal = s.type === "goal";
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
          aria-label={`seek to ${mmss(s.t_out)}`}
        >
          {mmss(s.t_out)}
        </button>
        <span className="font-mono text-[10px] text-zinc-500 shrink-0">({mmss(s.t_file)} cam)</span>
        <span className={`capitalize ${isGoal ? "text-emerald-300 font-medium" : "text-zinc-300"}`}>
          {s.type}
        </span>
        {team && (
          <span className="flex items-center gap-1 text-zinc-400">
            <Swatch team={team} size={10} />
            {team.name}
          </span>
        )}
        <span className="ml-auto text-zinc-500">{Math.round(s.confidence * 100)}%</span>
        <span className="text-[10px] text-zinc-600">{halfLabel(s.half)}</span>
      </li>
    );
  };

  return (
    <div className="flex flex-col gap-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Match analysis
        </div>
        {error && <div className="text-xs text-red-300">{error}</div>}

        {/* team header */}
        <div className={card}>
          <div className="flex items-start justify-between">
            <div className="flex-1 flex items-center justify-center gap-3">
              <span className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                <Swatch team={teams.A} size={16} /> {cap(teams.A.name)}
              </span>
              <span className="text-xs text-zinc-500">vs</span>
              <span className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                <Swatch team={teams.B} size={16} /> {cap(teams.B.name)}
              </span>
            </div>
            <button
              disabled={busy}
              onClick={() => void start(true)}
              title="Re-run the analysis (forces a fresh teams pass)"
              className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 disabled:opacity-40 shrink-0"
            >
              {busy ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
              Re-analyse
            </button>
          </div>
          {stats.team_confidence != null && (
            <div className="flex justify-center mt-1.5">
              <span className="rounded px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-300">
                team ID confidence {Math.round(stats.team_confidence * 100)}%
              </span>
            </div>
          )}
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
                    contested {Math.round(h.contested_pct)}% · ball visible {Math.round(h.ball_visible_pct)}%
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

          {/* shots / goals */}
          <div className={card}>
            <div className={head}>Shots &amp; goals</div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-800">
                  <th className={thCls}>Team</th>
                  <th className={thCls}>Shots</th>
                  <th className={thCls}>Goals (conf.)</th>
                  <th className={thCls}>Goals (est.)</th>
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
                      <td className={tdCls}>{tt.goals_confirmed}</td>
                      <td className={tdCls}>{tt.goals_estimated}</td>
                      <td className={`${tdCls} font-mono`}>{mmss(tt.attacking_third_s)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {isMobile ? (
              <details className="mt-2">
                <summary className="text-[11px] text-zinc-500 cursor-pointer">Per half</summary>
                {stats.halves.map((h) => (
                  <div key={h.index} className="mt-1 text-[11px] text-zinc-400">
                    {halfLabel(h.index)}: {cap(teams.A.name)} {h.teams.A.shots} sh ·{" "}
                    {h.teams.A.goals_confirmed}/{h.teams.A.goals_estimated} g ·{" "}
                    {cap(teams.B.name)} {h.teams.B.shots} sh ·{" "}
                    {h.teams.B.goals_confirmed}/{h.teams.B.goals_estimated} g
                  </div>
                ))}
              </details>
            ) : (
              <table className="w-full text-[11px] mt-2">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className={thCls}>Half</th>
                    <th className={thCls}>{cap(teams.A.name)}</th>
                    <th className={thCls}>{cap(teams.B.name)}</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.halves.map((h) => (
                    <tr key={h.index} className="text-zinc-400">
                      <td className={tdCls}>{halfLabel(h.index)}</td>
                      <td className={tdCls}>
                        {h.teams.A.shots} sh · {h.teams.A.goals_confirmed} conf ·{" "}
                        {h.teams.A.goals_estimated} est · {mmss(h.teams.A.attacking_third_s)} att
                      </td>
                      <td className={tdCls}>
                        {h.teams.B.shots} sh · {h.teams.B.goals_confirmed} conf ·{" "}
                        {h.teams.B.goals_estimated} est · {mmss(h.teams.B.attacking_third_s)} att
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
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
        </div>

        {/* chances */}
        {shots.length > 0 && (
          <div className={card}>
            <div className={head}>Chances ({shots.length})</div>
            <ul className="max-h-72 overflow-auto">{shots.map((s, i) => shotRow(s, i))}</ul>
          </div>
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
