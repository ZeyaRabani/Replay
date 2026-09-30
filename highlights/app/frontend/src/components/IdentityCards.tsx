import { Download, Film, Loader2, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { fmtClock } from "../lib/time";
import { fmtDur, fmtSpeed, identityHex } from "../lib/players";
import type { AnalysisTeamInfo, PlayerIdentities, PlayerIdentity, PlayerReel } from "../types";
import Swatch from "./Swatch";

const btnGhost = "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";

export interface ReelApi {
  get: (iid: string) => Promise<PlayerReel>;
  make: (iid: string) => Promise<PlayerReel>;
  fileUrl: (u: string) => string;
}

const reelLive = (r: PlayerReel | null) => r?.status.state === "queued" || r?.status.state === "running";

/** "Make player reel" + progress + download for one identity. */
function ReelControl({ iid, reelApi }: { iid: string; reelApi: ReelApi }) {
  const [reel, setReel] = useState<PlayerReel | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [kick, setKick] = useState(0);
  const timer = useRef<number | null>(null);
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const r = await reelApi.get(iid);
        if (!alive) return;
        setReel(r);
        if (reelLive(r)) timer.current = window.setTimeout(() => void poll(), 1500);
      } catch {
        if (alive) setReel(null);
      }
    };
    void poll();
    return () => {
      alive = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [iid, reelApi, kick]);
  const make = async () => {
    setErr(null);
    try {
      setReel(await reelApi.make(iid));
      setKick((k) => k + 1);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };
  const live = reelLive(reel);
  const st = reel?.status;
  return (
    <div className="flex items-center gap-1.5 text-[10px] min-w-0" data-reel={iid}>
      <button type="button" className={btnGhost} disabled={live} onClick={() => void make()}
        title="Cut this player's fastest runs + confirmed shots/goals they were involved in from the match cut">
        {live ? <Loader2 size={12} className="animate-spin" /> : <Film size={12} />}
        {reel?.url ? "Remake reel" : "Make player reel"}
      </button>
      {live && (
        <span className="flex-1 min-w-0 flex items-center gap-1">
          <span className="flex-1 h-1 bg-zinc-800 rounded overflow-hidden">
            <span className="block h-full bg-emerald-500" style={{ width: `${Math.round((st?.progress ?? 0) * 100)}%` }} />
          </span>
          <span className="text-zinc-500 truncate">{st?.message ?? "queued"}</span>
        </span>
      )}
      {!live && reel?.url && (
        <a className="flex items-center gap-1 text-emerald-400 hover:underline" href={reelApi.fileUrl(reel.url)} download>
          <Download size={12} />
          reel.mp4 · {st?.n_clips ?? reel.manifest?.items.length} clips · {fmtDur(st?.reel_s ?? reel.manifest?.reel_s ?? 0)}
        </a>
      )}
      {!live && st?.state === "failed" && <span className="text-red-400 truncate" title={st.error ?? ""}>{st.error ?? "failed"}</span>}
      {err && <span className="text-red-400 truncate" title={err}>{err}</span>}
    </div>
  );
}

function IdentityCard({ ident, lo, teamHex, cropSrc, onName, reelApi }: {
  ident: PlayerIdentity;
  /** match start on the shared timeline (for the first/last clock) */
  lo: number;
  teamHex: string;
  cropSrc: (u: string) => string;
  onName: (name: string | null) => Promise<void>;
  reelApi?: ReelApi;
}) {
  const [text, setText] = useState(ident.name ?? "");
  const [state, setState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const save = async () => {
    const n = text.trim();
    if (n === (ident.name ?? "")) return;
    setState("saving");
    try {
      await onName(n || null);
      setState("saved");
    } catch {
      setState("error");
    }
  };
  const stats: [string, string][] = [
    [(ident.distance_m / 1000).toFixed(2), "km"],
    [String(ident.sprints), "sprints"],
    [fmtSpeed(ident.top_speed_ms).replace(" km/h", ""), "top km/h"],
    [`${Math.round(ident.coverage_pct)}%`, "tracked"],
  ];
  return (
    <div data-identity={ident.id}
      className="card p-1.5 flex flex-col gap-1.5 min-w-0 border-l-4"
      style={{ borderLeftColor: teamHex }}>
      <div className="flex gap-1 overflow-hidden">
        {ident.crops.slice(0, 4).map((c) => (
          <img key={c} src={cropSrc(c)} alt="" loading="lazy"
            className="w-12 h-[120px] object-contain bg-zinc-950 rounded shrink-0" />
        ))}
        {!ident.crops.length && <div className="w-12 h-[120px] bg-zinc-950 rounded" />}
      </div>
      <div className="flex items-center gap-1.5 text-[11px] min-w-0">
        <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0 border border-zinc-900"
          style={{ backgroundColor: identityHex(ident.id) }} title="radar colour" />
        <span className="font-mono text-zinc-400 shrink-0">{ident.id}</span>
        {ident.role === "gk" && <span className="text-[9px] uppercase text-sky-300/80">gk?</span>}
        <span className="ml-auto text-zinc-500 font-mono shrink-0">
          {fmtClock(ident.first_s - lo)}–{fmtClock(ident.last_s - lo)}
        </span>
      </div>
      <input aria-label={`name for ${ident.id}`} placeholder="Player name" value={text} maxLength={60}
        className={`w-full min-w-0 bg-zinc-800 border rounded px-1.5 py-1 text-xs ${
          state === "error" ? "border-red-700" : ident.name ? "border-emerald-700" : "border-zinc-700"}`}
        onChange={(e) => { setText(e.target.value); setState("idle"); }}
        onBlur={() => void save()}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
          if (e.key === "Escape") { setText(ident.name ?? ""); e.currentTarget.blur(); }
        }} />
      <div className="grid grid-cols-4 gap-1 text-center">
        {stats.map(([v, l]) => (
          <div key={l} className="rounded bg-zinc-950/60 py-0.5">
            <div className="font-mono text-[11px] text-zinc-200">{v}</div>
            <div className="text-[9px] uppercase tracking-wide text-zinc-500">{l}</div>
          </div>
        ))}
      </div>
      <div className="text-[10px] text-zinc-500 flex justify-between">
        <span>{fmtDur(ident.coverage_s)} on camera · {ident.track_ids.length} tracks</span>
        <span>{state === "saving" ? "saving…" : state === "saved" ? "saved" : state === "error" ? "save failed" : ""}</span>
      </div>
      {reelApi && <ReelControl iid={ident.id} reelApi={reelApi} />}
    </div>
  );
}

/** Whole-match player identities (identities.json), one card each, by team. */
export default function IdentityCards({ doc, cols, busy, teamInfo, teamLabel, cropSrc, onName, onRebuild, reelApi }: {
  doc: PlayerIdentities;
  cols: string;
  busy?: boolean;
  teamInfo: (t: "A" | "B") => AnalysisTeamInfo | null;
  teamLabel: (t: "A" | "B") => string;
  cropSrc: (u: string) => string;
  onName: (iid: string, name: string | null) => Promise<void>;
  onRebuild?: () => void;
  reelApi?: ReelApi;
}) {
  const lo = doc.window?.[0] ?? 0;
  const q = doc.quality;
  return (
    <div className="flex flex-col gap-3 min-w-0" data-identity-cards>
      <div className="flex items-center gap-2 text-[11px] flex-wrap">
        <span className="text-zinc-400">
          {doc.identities.length} whole-match players · mean {Math.round(q.mean_coverage_pct ?? 0)}% of the match tracked
          · {q.n_unassigned ?? doc.unassigned_track_ids.length} fragments unassigned
        </span>
        {onRebuild && (
          <button type="button" className={`${btnGhost} ml-auto`} disabled={busy} onClick={onRebuild}
            title="Re-link identities from the saved tracks (names are kept)">
            {busy ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
            Re-link
          </button>
        )}
      </div>
      {(["A", "B"] as const).map((team) => {
        const items = doc.identities.filter((i) => i.team === team);
        if (!items.length) return null;
        const hex = doc.teams[team]?.hex || teamInfo(team)?.hex || "#71717a";
        return (
          <div key={team} className="min-w-0">
            <div className="flex items-center gap-2 text-xs text-zinc-300 mb-1.5">
              <Swatch team={teamInfo(team)} />
              <span className="font-medium">{teamLabel(team)}</span>
              <span className="text-zinc-500">{items.length} players</span>
            </div>
            <div className={`grid ${cols} gap-2`}>
              {items.map((i) => (
                <IdentityCard key={`${i.id}:${i.track_ids[0] ?? ""}`} ident={i} lo={lo} teamHex={hex}
                  cropSrc={cropSrc} onName={(n) => onName(i.id, n)} reelApi={reelApi} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
