import { AlertTriangle, ChevronDown, ChevronRight, Loader2, Play, RefreshCw, Users } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useProjectApi } from "../api";
import { useLayout } from "../lib/layout";
import { fetchPlayers, fmtDist, fmtDur, invalidatePlayers, saveRoster } from "../lib/players";
import type {
  AnalysisTeamInfo,
  PlayerTeam,
  PlayerTracklet,
  PlayersResponse,
  PlayersRoster,
  RosterPlayer,
} from "../types";

const card = "rounded-lg border border-zinc-800 bg-zinc-900 p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const btnPrimary =
  "flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5 text-xs disabled:opacity-40";
const btnGhost = "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";
const selectCls = "bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs w-full min-w-0";

const mmss = (s: number): string => {
  if (!Number.isFinite(s)) return "-:--";
  const t = Math.max(0, Math.round(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const NEW = "__new__";
const TEAM_ORDER: PlayerTeam[] = ["A", "B", null];

function Swatch({ team, size = 12 }: { team: AnalysisTeamInfo | null; size?: number }) {
  const hex = team?.hex || "#71717a";
  const isLight = hex.toLowerCase() > "#aaaaaa";
  return (
    <span
      role="img"
      aria-label={team ? `${team.name} shirt colour` : "unknown team"}
      className={`inline-block rounded shrink-0 border ${isLight ? "border-zinc-500" : "border-zinc-700"}`}
      style={{ backgroundColor: hex, width: size, height: size }}
    />
  );
}

function newPlayerId(roster: PlayersRoster): string {
  let n = roster.players.length + 1;
  const ids = new Set(roster.players.map((p) => p.id));
  while (ids.has(`p${n}`)) n += 1;
  return `p${n}`;
}

/** Roster with `tids` moved to player `pid` (or unassigned when pid is null). */
function assignTracklets(roster: PlayersRoster, tids: number[], pid: string | null): PlayersRoster {
  const set = new Set(tids);
  const players = roster.players.map((p) => {
    const kept = p.tracklet_ids.filter((t) => !set.has(t));
    if (p.id === pid) return { ...p, tracklet_ids: [...kept, ...tids].sort((a, b) => a - b) };
    return { ...p, tracklet_ids: kept };
  });
  return { ...roster, players };
}

interface CardProps {
  tr: PlayerTracklet;
  owner: RosterPlayer | undefined;
  roster: PlayersRoster;
  selected: boolean;
  onToggle: () => void;
  onAssign: (pid: string | null) => void;
  onNew: (name: string) => void;
  onSeek?: (t: number) => void;
  cropUrl: (n: string) => string;
}

function TrackletCard({ tr, owner, roster, selected, onToggle, onAssign, onNew, onSeek, cropUrl }: CardProps) {
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");

  const submitNew = () => {
    const n = name.trim();
    if (n) onNew(n);
    setName("");
    setNaming(false);
  };

  return (
    <div
      className={`rounded border p-1.5 flex flex-col gap-1.5 min-w-0 ${
        selected ? "border-amber-400 bg-zinc-800" : owner ? "border-emerald-900/70 bg-zinc-900" : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <div className="flex gap-1 h-20 items-stretch">
        {tr.crops.slice(0, 3).map((c) => (
          <img
            key={c}
            src={cropUrl(c)}
            alt=""
            loading="lazy"
            className="flex-1 min-w-0 h-full object-contain bg-zinc-950 rounded"
          />
        ))}
      </div>
      <label className="flex items-center gap-1.5 text-[11px] text-zinc-300 cursor-pointer min-w-0">
        <input type="checkbox" className="accent-amber-400 shrink-0" checked={selected} onChange={onToggle} />
        <span className="font-mono text-zinc-500 shrink-0">#{tr.id}</span>
        <span className="shrink-0">{fmtDur(tr.duration_s)}</span>
        <button
          type="button"
          className={`ml-auto font-mono text-[10px] truncate ${onSeek ? "text-amber-300 hover:text-amber-200" : "text-zinc-500 cursor-default"}`}
          onClick={(e) => {
            e.preventDefault();
            onSeek?.(tr.t_start_out);
          }}
          title={onSeek ? "seek in Review" : undefined}
        >
          {mmss(tr.t_start_out)}–{mmss(tr.t_end_out)}
        </button>
      </label>
      {naming ? (
        <div className="flex gap-1 min-w-0">
          <input
            autoFocus
            className="flex-1 min-w-0 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs"
            placeholder="Player name"
            value={name}
            maxLength={60}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submitNew();
              if (e.key === "Escape") setNaming(false);
            }}
          />
          <button type="button" className={btnGhost} onClick={submitNew}>
            OK
          </button>
        </div>
      ) : (
        <select
          className={selectCls}
          aria-label={`Assign tracklet ${tr.id}`}
          value={owner?.id ?? ""}
          onChange={(e) => {
            const v = e.target.value;
            if (v === NEW) setNaming(true);
            else onAssign(v || null);
          }}
        >
          <option value="">Assign to…</option>
          {roster.players.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
          <option value={NEW}>+ New player…</option>
        </select>
      )}
    </div>
  );
}

export default function PlayerAnalysis({ onSeek }: { onSeek?: (t: number) => void }) {
  const api = useProjectApi();
  const { isMobile } = useLayout();
  const [open, setOpen] = useState(true);
  const [data, setData] = useState<PlayersResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<"name" | "table">("name");
  const [roster, setRoster] = useState<PlayersRoster>({ players: [], scorers: {} });
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [hideAssigned, setHideAssigned] = useState(false);
  const [bulkNew, setBulkNew] = useState<string | null>(null);
  const [saving, setSaving] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const timer = useRef<number | null>(null);
  const saveTimer = useRef<number | null>(null);
  const dirty = useRef(false);

  const refresh = useCallback(async () => {
    const d = await fetchPlayers(api, true);
    if (!d) {
      setError("player analysis unavailable");
      return;
    }
    setData(d);
    if (!dirty.current) setRoster(d.roster);
    setError(null);
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const live = data?.status?.state === "queued" || data?.status?.state === "running";
  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
    if (!live) return;
    timer.current = window.setInterval(() => void refresh(), 5000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [live, refresh]);

  // debounced roster PUT
  const commit = useCallback(
    (next: PlayersRoster) => {
      setRoster(next);
      dirty.current = true;
      setSaving("saving");
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
      saveTimer.current = window.setTimeout(async () => {
        try {
          const r = await saveRoster(api, next);
          dirty.current = false;
          setRoster(r.roster);
          setData((d) => (d ? { ...d, roster: r.roster, players_stats: r.players_stats } : d));
          setSaving("saved");
        } catch (e) {
          setSaving("error");
          setError(e instanceof Error ? e.message : String(e));
        }
      }, 600);
    },
    [api],
  );

  const start = async (force = false) => {
    if (force && !window.confirm("Re-run player tracking? Your player names and assignments are kept.")) return;
    setBusy(true);
    try {
      await api.analysePlayers(force);
      invalidatePlayers(api);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const ownerOf = useMemo(() => {
    const m = new Map<number, RosterPlayer>();
    for (const p of roster.players) for (const t of p.tracklet_ids) m.set(t, p);
    return m;
  }, [roster]);

  const teams = data?.teams ?? null;
  const teamInfo = (t: PlayerTeam): AnalysisTeamInfo | null => (t && teams ? teams[t] : null);
  const teamLabel = (t: PlayerTeam) => (t ? cap(teamInfo(t)?.name ?? `Team ${t}`) : "Team unknown");

  const addPlayer = (name: string, team: PlayerTeam, tids: number[]): PlayersRoster => {
    const p: RosterPlayer = { id: newPlayerId(roster), name, team, tracklet_ids: [] };
    return assignTracklets({ ...roster, players: [...roster.players, p] }, tids, p.id);
  };

  const majorityTeam = (tids: number[]): PlayerTeam => {
    const c: Record<string, number> = { A: 0, B: 0 };
    for (const t of tids) {
      const tr = data?.tracklets.find((x) => x.id === t);
      if (tr?.team) c[tr.team] += 1;
    }
    if (c.A === 0 && c.B === 0) return null;
    return c.A >= c.B ? "A" : "B";
  };

  const bulkAssign = (pid: string | null) => {
    const tids = [...selected];
    commit(assignTracklets(roster, tids, pid));
    setSelected(new Set());
  };

  const header = (
    <button
      type="button"
      className="flex items-center gap-2 w-full text-left"
      onClick={() => setOpen((o) => !o)}
      aria-expanded={open}
    >
      {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
      <Users size={14} className="text-zinc-500" />
      <span className={head}>Player analysis (optional)</span>
      {data?.tracklets.length ? (
        <span className="ml-auto text-[11px] text-zinc-500">
          {roster.players.length} players · {data.players_stats.unassigned.n_tracklets} unassigned
        </span>
      ) : null}
    </button>
  );

  if (!data) {
    return (
      <div className={card}>
        {header}
        {open && (
          <div className="text-sm text-zinc-500 flex items-center gap-2 mt-3">
            {!error && <Loader2 size={14} className="animate-spin" />}
            {error ?? "Loading…"}
          </div>
        )}
      </div>
    );
  }

  const status = data.status;
  const hasTracklets = data.tracklets.length > 0 && status?.state === "done";

  let body: React.ReactNode;
  if (live && status) {
    const pct = Math.round((status.progress ?? 0) * 100);
    body = (
      <div className="flex items-center gap-3 text-xs mt-3">
        <Loader2 size={14} className="animate-spin text-amber-400 shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex justify-between gap-2">
            <span className="text-zinc-300 truncate">
              {status.state === "queued"
                ? "Waiting for another job to finish…"
                : `${status.stage ?? "tracking"} — ${status.message ?? ""}`}
            </span>
            <span className="text-zinc-400 shrink-0">{pct}%</span>
          </div>
          <div className="h-1.5 bg-zinc-800 rounded mt-1.5">
            <div className="h-1.5 rounded bg-amber-400 transition-all" style={{ width: `${pct}%` }} />
          </div>
          <div className="text-[11px] text-zinc-500 mt-1">You can leave this page — tracking keeps running.</div>
        </div>
      </div>
    );
  } else if (!hasTracklets) {
    const failed = status?.state === "failed";
    const noTeams = !teams;
    body = (
      <div className="mt-3">
        <p className="text-xs text-zinc-400 mb-3">
          Tracks players on the main camera and gives you short clips to name. Identity is not automatic on
          this footage — you name players from the crops; stats are estimates from tracked time only.
        </p>
        {failed && (
          <div className="mb-3 rounded border border-red-800 bg-red-950/60 text-red-200 text-xs p-2 flex items-start gap-2">
            <AlertTriangle size={13} className="mt-0.5 shrink-0" />
            <div className="flex-1">{status?.error || status?.message || "player analysis failed"}</div>
          </div>
        )}
        {status?.state === "done" && data.tracklets.length === 0 && (
          <div className="mb-3 text-xs text-zinc-400">No tracklets of 6 s or longer were found.</div>
        )}
        <div className="flex items-center gap-3 flex-wrap">
          <button
            disabled={busy || noTeams}
            onClick={() => void start(failed || status?.state === "done")}
            className={btnPrimary}
            title={noTeams ? "Run Match analysis first (team colours are needed)" : undefined}
          >
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
            {failed ? "Retry" : "Find players"} (~{data.estimate_min} min)
          </button>
          {noTeams && <span className="text-[11px] text-zinc-500">Run Match analysis above first.</span>}
        </div>
      </div>
    );
  } else {
    const shown = data.tracklets.filter((t) => !hideAssigned || !ownerOf.has(t.id));
    const groups = TEAM_ORDER.map((team) => ({ team, items: shown.filter((t) => t.team === team) })).filter(
      (g) => g.items.length > 0,
    );
    const ps = data.players_stats;
    const cols = isMobile ? "grid-cols-2" : "grid-cols-3 sm:grid-cols-4 lg:grid-cols-6";
    const chip = (active: boolean) =>
      `rounded px-2 py-0.5 text-xs ${active ? "bg-amber-500 text-zinc-900 font-semibold" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`;

    body = (
      <div className="mt-3 flex flex-col gap-3 min-w-0">
        <div className="flex items-center gap-1.5 flex-wrap">
          <button className={chip(tab === "name")} onClick={() => setTab("name")}>
            Name players
          </button>
          <button className={chip(tab === "table")} onClick={() => setTab("table")}>
            Players table
          </button>
          <span className="ml-auto text-[11px] text-zinc-500">
            {saving === "saving" ? "saving…" : saving === "saved" ? "saved" : saving === "error" ? "save failed" : ""}
          </span>
          <button className={btnGhost} disabled={busy} onClick={() => void start(true)} title="Re-run tracking (roster kept)">
            <RefreshCw size={12} /> Re-run
          </button>
        </div>
        {error && <div className="text-xs text-red-300">{error}</div>}

        {tab === "name" && (
          <>
            <div className="flex items-center gap-3 flex-wrap text-xs">
              <label className="flex items-center gap-1.5 text-zinc-300">
                <input
                  type="checkbox"
                  className="accent-amber-400"
                  checked={hideAssigned}
                  onChange={(e) => setHideAssigned(e.target.checked)}
                />
                Hide assigned
              </label>
              <span className="text-zinc-500">
                Showing the {data.n_shown ?? data.tracklets.length} longest tracks of{" "}
                {data.n_tracklets_total ?? data.tracklets.length} (≥30 s) ·{" "}
                {ps.unassigned.n_tracklets} unassigned
              </span>
              {selected.size > 0 && (
                <div className="flex items-center gap-1.5 flex-wrap ml-auto bg-zinc-800/70 rounded px-2 py-1">
                  <span className="text-amber-300">{selected.size} selected →</span>
                  {bulkNew === null ? (
                    <select
                      className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs"
                      aria-label="Assign selected to"
                      value=""
                      onChange={(e) => {
                        const v = e.target.value;
                        if (v === NEW) setBulkNew("");
                        else if (v === "__none__") bulkAssign(null);
                        else if (v) bulkAssign(v);
                      }}
                    >
                      <option value="">Assign to…</option>
                      {roster.players.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name}
                        </option>
                      ))}
                      <option value={NEW}>+ New player…</option>
                      <option value="__none__">Unassign</option>
                    </select>
                  ) : (
                    <>
                      <input
                        autoFocus
                        className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs w-32"
                        placeholder="Player name"
                        maxLength={60}
                        value={bulkNew}
                        onChange={(e) => setBulkNew(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && bulkNew.trim()) {
                            const tids = [...selected];
                            commit(addPlayer(bulkNew.trim(), majorityTeam(tids), tids));
                            setSelected(new Set());
                            setBulkNew(null);
                          }
                          if (e.key === "Escape") setBulkNew(null);
                        }}
                      />
                      <button
                        className={btnGhost}
                        onClick={() => {
                          if (!bulkNew.trim()) return;
                          const tids = [...selected];
                          commit(addPlayer(bulkNew.trim(), majorityTeam(tids), tids));
                          setSelected(new Set());
                          setBulkNew(null);
                        }}
                      >
                        OK
                      </button>
                    </>
                  )}
                  <button className={btnGhost} onClick={() => setSelected(new Set())}>
                    Clear
                  </button>
                </div>
              )}
            </div>
            {groups.length === 0 && <div className="text-xs text-zinc-500">All tracklets are assigned.</div>}
            {groups.map((g) => (
              <div key={g.team ?? "none"} className="min-w-0">
                <div className="flex items-center gap-2 text-xs text-zinc-300 mb-1.5">
                  <Swatch team={teamInfo(g.team)} />
                  <span className="font-medium">{teamLabel(g.team)}</span>
                  <span className="text-zinc-500">{g.items.length}</span>
                </div>
                <div className={`grid ${cols} gap-2`}>
                  {g.items.map((tr) => (
                    <TrackletCard
                      key={tr.id}
                      tr={tr}
                      owner={ownerOf.get(tr.id)}
                      roster={roster}
                      selected={selected.has(tr.id)}
                      onToggle={() =>
                        setSelected((s) => {
                          const n = new Set(s);
                          if (n.has(tr.id)) n.delete(tr.id);
                          else n.add(tr.id);
                          return n;
                        })
                      }
                      onAssign={(pid) => commit(assignTracklets(roster, [tr.id], pid))}
                      onNew={(name) => commit(addPlayer(name, tr.team, [tr.id]))}
                      onSeek={onSeek}
                      cropUrl={api.cropUrl}
                    />
                  ))}
                </div>
              </div>
            ))}
          </>
        )}

        {tab === "table" && (
          <div className="min-w-0 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-zinc-500 text-left">
                  <th className="pb-1 font-medium">Player</th>
                  <th className="pb-1 font-medium text-right">Tracked</th>
                  <th className="pb-1 font-medium text-right">Distance (est.)</th>
                  <th className="pb-1 font-medium text-right">Sprints</th>
                  <th className="pb-1 font-medium text-right">Goals</th>
                </tr>
              </thead>
              <tbody>
                {ps.players.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-2 text-zinc-500">
                      No named players yet — assign tracklets in “Name players”.
                    </td>
                  </tr>
                )}
                {ps.players.map((p) => (
                  <tr key={p.id} className="border-t border-zinc-800/60">
                    <td className="py-1">
                      <span className="flex items-center gap-1.5 min-w-0">
                        <Swatch team={teamInfo(p.team)} size={10} />
                        <span className="truncate">{p.name}</span>
                        <span className="text-zinc-600">({p.n_tracklets})</span>
                      </span>
                    </td>
                    <td className="py-1 text-right font-mono">{fmtDur(p.tracked_s)}</td>
                    <td className="py-1 text-right font-mono">{fmtDist(p.distance_m)}</td>
                    <td className="py-1 text-right font-mono">{p.sprints}</td>
                    <td className="py-1 text-right font-mono">{p.goals}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                {(["A", "B"] as const).map((t) => (
                  <tr key={t} className="border-t border-zinc-700 text-zinc-400">
                    <td className="py-1">
                      <span className="flex items-center gap-1.5">
                        <Swatch team={teamInfo(t)} size={10} />
                        {teamLabel(t)} total ({ps.teams[t].n_players})
                      </span>
                    </td>
                    <td className="py-1 text-right font-mono">{fmtDur(ps.teams[t].tracked_s)}</td>
                    <td className="py-1 text-right font-mono">{fmtDist(ps.teams[t].distance_m)}</td>
                    <td className="py-1 text-right font-mono">{ps.teams[t].sprints}</td>
                    <td className="py-1 text-right font-mono">{ps.teams[t].goals}</td>
                  </tr>
                ))}
              </tfoot>
            </table>
            <div className="text-[11px] text-zinc-500 mt-2">{ps.unassigned.n_tracklets} tracklets unassigned</div>
            <div className="text-[10px] text-zinc-600 mt-1">{ps.caveat}</div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`${card} min-w-0`}>
      {header}
      {open && body}
    </div>
  );
}
