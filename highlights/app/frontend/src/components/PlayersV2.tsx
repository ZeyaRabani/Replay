import { AlertTriangle, Loader2, Play } from "lucide-react";
import { useState } from "react";
import { fmtDur } from "../lib/players";
import { fmtClock } from "../lib/time";
import type { AnalysisStatus, AnalysisTeamInfo, PlayerTeam, PlayersRoster, PlayersV2Group, PlayersV2Track, PlayersV2Tracks } from "../types";
import Swatch from "./Swatch";

const btnPrimary =
  "flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5 text-xs disabled:opacity-40";
const btnGhost = "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";
const TEAM_ORDER: PlayerTeam[] = ["A", "B", null];

export const isLive = (s: AnalysisStatus | null | undefined) => s?.state === "queued" || s?.state === "running";

/** Run button + live status for the multi-camera player pass. */
export function PlayersV2Bar({ v2, status, missing, busy, onRun, onRebuild }: {
  v2: PlayersV2Tracks | null;
  status: AnalysisStatus | null;
  /** angle indexes without a solved calibration */
  missing: number[];
  busy: boolean;
  onRun: () => void;
  onRebuild?: () => void;
}) {
  if (isLive(status) && status) {
    const pct = Math.round((status.progress ?? 0) * 100);
    return (
      <div className="flex items-center gap-3 text-xs">
        <Loader2 size={14} className="animate-spin text-amber-400 shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex justify-between gap-2">
            <span className="text-zinc-300 truncate">
              {status.state === "queued" ? "Players v2 queued…" : `Players v2: ${status.stage ?? "tracking"} — ${status.message ?? ""}`}
            </span>
            <span className="text-zinc-400 shrink-0">{pct}%</span>
          </div>
          <div className="h-1.5 bg-zinc-800 rounded mt-1.5">
            <div className="h-1.5 rounded bg-amber-400 transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
    );
  }
  const has = !!v2?.tracks.length;
  return (
    <div className="flex flex-col gap-1.5">
      {status?.state === "failed" && (
        <div className="rounded border border-red-800 bg-red-950/60 text-red-200 text-xs p-2 flex items-start gap-2">
          <AlertTriangle size={13} className="mt-0.5 shrink-0" />
          <div className="flex-1">{status.error || status.message || "players v2 failed"}</div>
        </div>
      )}
      <div className="flex items-center gap-3 flex-wrap text-[11px]">
        <button type="button" className={has ? btnGhost : btnPrimary} disabled={busy || missing.length > 0} onClick={onRun}
          title={missing.length ? "Calibrate every camera first" : "Fuse all cameras into pitch-space tracks"}>
          {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
          {has ? "Re-run players v2" : "Run players v2"}
        </button>
        {missing.length > 0 && (
          <span className="text-zinc-500">
            Calibrate angle {missing.map((i) => i + 1).join(", ")} in Radar replay first.
          </span>
        )}
        {has && onRebuild && (
          <button type="button" className={btnGhost} disabled={busy} onClick={onRebuild}
            title="Re-cluster tracks into player groups">
            Rebuild groups
          </button>
        )}
        {has && v2 && (
          <span className="text-zinc-500">
            {v2.groups?.length
              ? `${v2.groups.length} players grouped from ${v2.summary.n_tracks} tracks`
              : `${v2.summary.n_tracks} tracks`} · median {v2.summary.median_visible} visible · mean track {fmtDur(v2.summary.mean_len_s)}
          </span>
        )}
      </div>
    </div>
  );
}

function V2Card({ label, name, crops, mins, distM, sprints, start, end, members, cropSrc, onName, onHide }: {
  label: string; name: string | null; crops: string[];
  mins: number; distM: number; sprints: number; start: number; end: number;
  /** fused-track count behind a group card */
  members?: number | null;
  cropSrc: (u: string) => string;
  onName: (n: string) => void; onHide: () => void;
}) {
  const [naming, setNaming] = useState(false);
  const [text, setText] = useState("");
  const submit = () => {
    const n = text.trim();
    if (n) onName(n);
    setText("");
    setNaming(false);
  };
  return (
    <div className={`rounded border p-1.5 flex flex-col gap-1.5 min-w-0 ${
      name ? "border-emerald-900/70 bg-zinc-900" : "border-zinc-800 bg-zinc-900"}`}>
      <div className="flex gap-1">
        {crops.slice(0, 4).map((c) => (
          <img key={c} src={cropSrc(c)} alt="" loading="lazy"
            className="w-12 h-[120px] object-contain bg-zinc-950 rounded"
            style={{ imageRendering: "auto" }} />
        ))}
        {!crops.length && <div className="w-12 h-[120px] bg-zinc-950 rounded" />}
      </div>
      <div className="flex items-baseline gap-1.5 text-[11px] min-w-0">
        <span className={`font-medium truncate ${name ? "text-emerald-300/90" : "text-zinc-300"}`}>{name ?? label}</span>
        <span className="ml-auto text-zinc-500 font-mono shrink-0">{fmtClock(start)}–{fmtClock(end)}</span>
      </div>
      <div className="grid grid-cols-3 gap-1 text-center">
        {([[mins.toFixed(1), "min"], [(distM / 1000).toFixed(1), "km"], [String(sprints), "sprints"]] as const).map(([v, l]) => (
          <div key={l} className="rounded bg-zinc-950/60 py-0.5">
            <div className="font-mono text-[11px] text-zinc-200">{v}</div>
            <div className="text-[9px] uppercase tracking-wide text-zinc-500">{l}</div>
          </div>
        ))}
      </div>
      {members != null && (
        <div className="text-[10px] text-zinc-500">made of {members} tracks</div>
      )}
      <div className="flex gap-1 min-w-0">
        {naming ? (
          <>
            <input autoFocus placeholder="Player name" value={text} maxLength={60}
              className="flex-1 min-w-0 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submit();
                if (e.key === "Escape") setNaming(false);
              }} />
            <button type="button" className={btnGhost} onClick={submit}>OK</button>
          </>
        ) : (
          <>
            {!name && <button type="button" className={`${btnGhost} flex-1 justify-center`} onClick={() => setNaming(true)}>Name</button>}
            <button type="button" className={`${btnGhost} ${name ? "ml-auto" : ""}`} onClick={onHide}>Hide</button>
          </>
        )}
      </div>
    </div>
  );
}

/** One card per player group (default) or per fused track, by team. */
export function PlayersV2Grid({ tracks, groups, nMerged, roster, cols, teamInfo, teamLabel, cropSrc, onName, onHide }: {
  tracks: PlayersV2Track[];
  groups?: PlayersV2Group[];
  /** super-track count after duplicate merging (for the heading) */
  nMerged?: number | null;
  roster: PlayersRoster;
  cols: string;
  teamInfo: (t: PlayerTeam) => AnalysisTeamInfo | null;
  teamLabel: (t: PlayerTeam) => string;
  cropSrc: (u: string) => string;
  onName: (t: { ids: number[]; team: PlayerTeam }, name: string) => void;
  onHide: (ids: number[]) => void;
}) {
  const hidden = new Set(roster.hidden_tracklet_ids ?? []);
  const nameOf = new Map<number, string>();
  for (const p of roster.players) for (const id of p.tracklet_ids) nameOf.set(id, p.name);
  const nameOfPlayer = new Map(roster.players.map((p) => [p.id, p.name]));
  const [raw, setRaw] = useState(false);

  const shownGroups = (groups ?? []).filter(
    (g) => g.track_ids.some((t) => !hidden.has(t)));
  const shown = tracks.filter((t) => !hidden.has(t.id));
  const nHidden = tracks.length - shown.length;
  const useGroups = !raw && shownGroups.length > 0;

  const heading = `${shownGroups.length} player groups from ${tracks.length} tracks` +
    (nMerged ? ` (${nMerged} after merging duplicates)` : "");
  const toggle = groups?.length ? (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="text-zinc-500">{heading}</span>
      <button type="button" className={btnGhost} onClick={() => setRaw((r) => !r)}>
        {raw ? `Show groups (${shownGroups.length})` : `Show raw tracks (${tracks.length})`}
      </button>
    </div>
  ) : null;

  if (useGroups) {
    const byTeam = TEAM_ORDER.map((team) => ({
      team,
      items: shownGroups.filter((g) => g.team === team)
        .sort((a, b) => b.minutes - a.minutes),
    })).filter((g) => g.items.length > 0);
    const groupCard = (grp: PlayersV2Group) => (
      <V2Card key={grp.id} label={`Player ${grp.id}`}
        name={(grp.player_id && nameOfPlayer.get(grp.player_id)) ?? nameOf.get(grp.track_ids[0]) ?? null}
        crops={grp.crops} mins={grp.minutes} distM={grp.dist_m}
        sprints={grp.sprints} start={grp.start} end={grp.end}
        members={grp.n_members ?? grp.track_ids.length}
        cropSrc={cropSrc}
        onName={(n) => onName({ ids: grp.track_ids, team: grp.team }, n)}
        onHide={() => onHide(grp.track_ids)} />
    );
    return (
      <div className="flex flex-col gap-3 min-w-0">
        {toggle}
        {byTeam.map((g) => {
          const top = g.items.slice(0, 11);
          const rest = g.items.slice(11);
          return (
            <div key={g.team ?? "none"} className="min-w-0">
              <div className="flex items-center gap-2 text-xs text-zinc-300 mb-1.5">
                <Swatch team={teamInfo(g.team)} />
                <span className="font-medium">{teamLabel(g.team)}</span>
                <span className="text-zinc-500">{g.items.length} groups</span>
              </div>
              <div className={`grid ${cols} gap-2`}>
                {top.map(groupCard)}
              </div>
              {rest.length > 0 && (
                <details className="mt-1.5">
                  <summary className="text-[11px] text-zinc-500 cursor-pointer hover:text-zinc-300">
                    {rest.length} more fragments (short appearances)
                  </summary>
                  <div className={`grid ${cols} gap-2 mt-2`}>
                    {rest.map(groupCard)}
                  </div>
                </details>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  const byTeam = TEAM_ORDER.map((team) => ({
    team,
    items: shown.filter((t) => t.team === team).sort((a, b) => (b.end - b.start) - (a.end - a.start)),
  })).filter((g) => g.items.length > 0);
  return (
    <div className="flex flex-col gap-3 min-w-0">
      {toggle}
      {byTeam.map((g) => (
        <div key={g.team ?? "none"} className="min-w-0">
          <div className="flex items-center gap-2 text-xs text-zinc-300 mb-1.5">
            <Swatch team={teamInfo(g.team)} />
            <span className="font-medium">{teamLabel(g.team)}</span>
            <span className="text-zinc-500">{g.items.length} tracks</span>
          </div>
          <div className={`grid ${cols} gap-2`}>
            {g.items.map((tr) => (
              <V2Card key={tr.id} label={`Track ${tr.id}`}
                name={nameOf.get(tr.id) ?? null}
                crops={tr.crops} mins={Math.max(0, tr.end - tr.start) / 60}
                distM={tr.dist_m} sprints={tr.sprints}
                start={tr.start} end={tr.end} cropSrc={cropSrc}
                onName={(n) => onName({ ids: [tr.id], team: tr.team }, n)}
                onHide={() => onHide([tr.id])} />
            ))}
          </div>
        </div>
      ))}
      {!byTeam.length && <div className="text-xs text-zinc-500">No tracks to show.</div>}
      {nHidden > 0 && <div className="text-[11px] text-zinc-500">{nHidden} tracks hidden</div>}
    </div>
  );
}
