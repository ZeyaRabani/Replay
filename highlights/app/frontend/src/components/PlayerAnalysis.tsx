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

const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);
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

/** A displayed group card: an AI cluster or a named roster player. */
interface GroupView {
  key: string;          // group id or `p:<player id>`
  named: boolean;
  pid?: string;
  name?: string;
  team: PlayerTeam;
  tids: number[];
  crops: string[];
  duration_s: number;
  distance_m: number;
  sprints: number;
  cohesion?: number;
}

interface GroupCardProps {
  g: GroupView;
  trackletById: Map<number, PlayerTracklet>;
  selected: boolean;
  expanded: boolean;
  onToggle: () => void;
  onExpand: () => void;
  onName: (name: string) => void;
  onHide: () => void;
  onSplit: (tid: number) => void;
  cropUrl: (n: string) => string;
}

function GroupCard({ g, trackletById, selected, expanded, onToggle, onExpand, onName, onHide, onSplit, cropUrl }: GroupCardProps) {
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const submit = () => {
    const n = name.trim();
    if (n) onName(n);
    setName("");
    setNaming(false);
  };
  return (
    <div
      className={`rounded border p-1.5 flex flex-col gap-1.5 min-w-0 ${
        selected ? "border-amber-400 bg-zinc-800" : g.named ? "border-emerald-900/70 bg-zinc-900" : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <div className="flex gap-1 h-20 items-stretch">
        {g.crops.slice(0, 3).map((c) => (
          <img
            key={c}
            src={cropUrl(c)}
            alt=""
            loading="lazy"
            className="flex-1 min-w-0 h-full object-contain bg-zinc-950 rounded"
          />
        ))}
      </div>
      <div className="flex items-center gap-1.5 text-[11px] text-zinc-300 min-w-0">
        <input type="checkbox" className="accent-amber-400 shrink-0" checked={selected} onChange={onToggle} />
        <span className="font-medium truncate">{g.named ? g.name : g.key}</span>
        <span className="text-zinc-500 shrink-0">{g.tids.length}×</span>
        <button type="button" onClick={onExpand}
          className="ml-auto text-zinc-500 hover:text-zinc-300 shrink-0"
          aria-label={expanded ? "collapse group" : "open group"}>
          {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </button>
      </div>
      <div className="text-[10px] text-zinc-500 font-mono">
        {fmtDur(g.duration_s)} · {fmtDist(g.distance_m)} · {g.sprints} sp
      </div>
      {expanded && (
        <div className="flex flex-col gap-1 border-t border-zinc-800 pt-1">
          {g.tids.map((tid) => {
            const tr = trackletById.get(tid);
            return (
              <div key={tid} className="flex items-center gap-1.5 min-w-0">
                <img src={cropUrl(`${tid}_1.jpg`)} alt="" loading="lazy"
                  className="h-8 w-6 object-contain bg-zinc-950 rounded shrink-0" />
                <span className="font-mono text-[10px] text-zinc-500">#{tid}</span>
                {tr && <span className="text-[10px] text-zinc-400">{fmtDur(tr.duration_s)}</span>}
                {g.tids.length > 1 && (
                  <button type="button" onClick={() => onSplit(tid)}
                    className="ml-auto text-[10px] text-zinc-500 hover:text-amber-300 shrink-0">
                    Remove
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
      <div className="flex gap-1 min-w-0">
        {g.named ? (
          <>
            <span className="flex-1 min-w-0 truncate text-[11px] text-emerald-300/90 py-1">{g.name}</span>
            <button type="button" className={btnGhost} onClick={onHide}>Hide</button>
          </>
        ) : naming ? (
          <>
            <input
              autoFocus
              className="flex-1 min-w-0 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs"
              placeholder="Player name"
              value={name}
              maxLength={60}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submit();
                if (e.key === "Escape") setNaming(false);
              }}
            />
            <button type="button" className={btnGhost} onClick={submit}>OK</button>
          </>
        ) : (
          <>
            <button type="button" className={`${btnGhost} flex-1"`} onClick={() => setNaming(true)}>
              Name
            </button>
            <button type="button" className={btnGhost} onClick={onHide}>Hide</button>
          </>
        )}
      </div>
    </div>
  );
}

export default function PlayerAnalysis(_props: { onSeek?: (t: number) => void }) {
  const api = useProjectApi();
  const { isMobile } = useLayout();
  const [open, setOpen] = useState(true);
  const [data, setData] = useState<PlayersResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<"name" | "table">("name");
  const [roster, setRoster] = useState<PlayersRoster>({ players: [], scorers: {}, hidden_tracklet_ids: [] });
  const [selGroups, setSelGroups] = useState<Set<string>>(new Set());
  const [openGroup, setOpenGroup] = useState<string | null>(null);
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
    if (!dirty.current)
      setRoster({ ...d.roster, hidden_tracklet_ids: d.roster.hidden_tracklet_ids ?? [] });
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
          setRoster({ ...r.roster, hidden_tracklet_ids: r.roster.hidden_tracklet_ids ?? [] });
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

  const trackletById = useMemo(
    () => new Map((data?.tracklets ?? []).map((t) => [t.id, t])),
    [data],
  );
  const hiddenSet = useMemo(
    () => new Set(roster.hidden_tracklet_ids ?? []), [roster],
  );
  // displayed groups: named roster players (tracklets live on the player)
  // + AI groups' still-unassigned/unhidden tracklets
  const groupViews = useMemo((): GroupView[] => {
    const out: GroupView[] = [];
    const agg = (tids: number[]) => {
      const trs = tids
        .map((t) => trackletById.get(t))
        .filter((t): t is PlayerTracklet => !!t);
      return {
        crops: trs
          .map((t) => t.crops[1] ?? t.crops[0])
          .filter(Boolean)
          .slice(0, 6),
        duration_s: trs.reduce((a, t) => a + t.duration_s, 0),
        distance_m: trs.reduce((a, t) => a + t.distance_m, 0),
        sprints: trs.reduce((a, t) => a + t.sprints, 0),
      };
    };
    for (const p of roster.players) {
      const tids = p.tracklet_ids.filter((t) => !hiddenSet.has(t));
      if (!tids.length) continue;
      out.push({
        key: `p:${p.id}`, named: true, pid: p.id, name: p.name,
        team: p.team, tids, ...agg(tids),
      });
    }
    const assigned = new Set(roster.players.flatMap((p) => p.tracklet_ids));
    for (const g of data?.groups ?? []) {
      const tids = g.tracklet_ids.filter(
        (t) => !assigned.has(t) && !hiddenSet.has(t),
      );
      if (!tids.length) continue;
      const trs = tids
        .map((t) => trackletById.get(t))
        .filter((t): t is PlayerTracklet => !!t);
      out.push({
        key: g.id, named: false, team: g.team, tids,
        crops: g.crops
          .filter((c) => tids.some((t) => c.startsWith(`${t}_`)))
          .slice(0, 6),
        duration_s: trs.length
          ? trs.reduce((a, t) => a + t.duration_s, 0)
          : g.duration_s,
        distance_m: trs.length
          ? trs.reduce((a, t) => a + t.distance_m, 0)
          : g.distance_m,
        sprints: trs.length
          ? trs.reduce((a, t) => a + t.sprints, 0)
          : g.sprints,
        cohesion: g.cohesion,
      });
    }
    return out;
  }, [roster, hiddenSet, trackletById, data]);

  const teams = data?.teams ?? null;
  const teamInfo = (t: PlayerTeam): AnalysisTeamInfo | null => (t && teams ? teams[t] : null);
  const teamLabel = (t: PlayerTeam) => (t ? cap(teamInfo(t)?.name ?? `Team ${t}`) : "Team unknown");

  const addPlayer = (name: string, team: PlayerTeam, tids: number[]): PlayersRoster => {
    const p: RosterPlayer = { id: newPlayerId(roster), name, team, tracklet_ids: [] };
    return assignTracklets({ ...roster, players: [...roster.players, p] }, tids, p.id);
  };

  const mergeSelected = () => {
    const sel = groupViews.filter((v) => selGroups.has(v.key));
    if (sel.length < 2) return;
    const named = sel.find((v) => v.named);
    const tids = sel.flatMap((v) => v.tids);
    if (named?.pid) {
      commit(assignTracklets(roster, tids, named.pid));
    } else {
      commit(addPlayer(
        `Player ${newPlayerId(roster).slice(1)}`, sel[0].team, tids));
    }
    setSelGroups(new Set());
  };

  const hideGroup = (v: GroupView) => {
    const drop = new Set(v.tids);
    const players = roster.players.map((p) => ({
      ...p,
      tracklet_ids: p.tracklet_ids.filter((t) => !drop.has(t)),
    }));
    commit({
      ...roster, players,
      hidden_tracklet_ids: [...hiddenSet, ...v.tids.filter((t) => !hiddenSet.has(t))],
    });
  };

  const splitOut = (v: GroupView, tid: number) => {
    // the removed tracklet becomes its own (auto-named) group
    commit(addPlayer(`Player ${newPlayerId(roster).slice(1)}`, v.team, [tid]));
  };

  const rebuildGroups = async () => {
    setBusy(true);
    try {
      await api.rebuildGroups();
      invalidatePlayers(api);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
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
    const teamGroups = TEAM_ORDER.map((team) => ({
      team,
      items: groupViews.filter((v) => v.team === team),
    })).filter((g) => g.items.length > 0);
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
              <span className="text-zinc-500">
                Groups are the AI's best guess from shirt colours and timing —
                merge, split or hide to correct them.
              </span>
              {selGroups.size > 0 && (
                <div className="flex items-center gap-1.5 flex-wrap ml-auto bg-zinc-800/70 rounded px-2 py-1">
                  <span className="text-amber-300">{selGroups.size} selected →</span>
                  <button
                    className={btnGhost}
                    disabled={selGroups.size < 2}
                    onClick={mergeSelected}
                  >
                    Merge
                  </button>
                  <button className={btnGhost} onClick={() => setSelGroups(new Set())}>
                    Clear
                  </button>
                </div>
              )}
              <button
                className={`${btnGhost} ${selGroups.size ? "" : "ml-auto"}`}
                disabled={busy}
                onClick={() => void rebuildGroups()}
                title="Re-cluster tracklets into groups (roster kept)"
              >
                <RefreshCw size={12} /> Re-group
              </button>
            </div>
            {teamGroups.length === 0 && (
              <div className="text-xs text-zinc-500">No groups to show.</div>
            )}
            {teamGroups.map((g) => (
              <div key={g.team ?? "none"} className="min-w-0">
                <div className="flex items-center gap-2 text-xs text-zinc-300 mb-1.5">
                  <Swatch team={teamInfo(g.team)} />
                  <span className="font-medium">{teamLabel(g.team)}</span>
                  <span className="text-zinc-500">{g.items.length} groups</span>
                </div>
                <div className={`grid ${cols} gap-2`}>
                  {g.items.map((v) => (
                    <GroupCard
                      key={v.key}
                      g={v}
                      trackletById={trackletById}
                      selected={selGroups.has(v.key)}
                      expanded={openGroup === v.key}
                      onToggle={() =>
                        setSelGroups((s) => {
                          const n = new Set(s);
                          if (n.has(v.key)) n.delete(v.key);
                          else n.add(v.key);
                          return n;
                        })
                      }
                      onExpand={() =>
                        setOpenGroup((k) => (k === v.key ? null : v.key))
                      }
                      onName={(name) =>
                        commit(addPlayer(name, v.team, v.tids))
                      }
                      onHide={() => hideGroup(v)}
                      onSplit={(tid) => splitOut(v, tid)}
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
