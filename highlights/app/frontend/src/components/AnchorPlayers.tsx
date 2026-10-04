import { Crosshair, RefreshCw, Save } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useProjectApi } from "../api";
import { IDENTITIES_CHANGED_EVENT } from "../lib/players";
import type { AnchorClick, PlayerAnchors, PlayerIdentities } from "../types";

const btnGhost =
  "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";
const TEAM_HEX: Record<string, string> = { A: "#86efac", B: "#fdba74" };
const HIT_PX = 14;

/** team key -> css hex: teams.json colour when present, else fallback. */
type HexFor = (t: string) => string;

/** mm:ss (optionally h:mm:ss) -> seconds, null when unparseable. */
function parseClock(s: string): number | null {
  const m = /^\s*(?:(\d+):)?([0-5]?\d):([0-5]?\d)\s*$/.exec(s);
  if (!m) return null;
  return (Number(m[1] || 0) * 3600) + Number(m[2]) * 60 + Number(m[3]);
}

const mmss = (t: number) => {
  const s = Math.max(0, Math.round(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

/** One camera still filling the canvas; clicks in [0,1] image coords. */
function Still({ src, markers, hexFor, onPick, onDelete }: {
  src: string;
  markers: AnchorClick[];
  hexFor: HexFor;
  onPick: (fx: number, fy: number) => void;
  onDelete: (id: string) => void;
}) {
  const cvRef = useRef<HTMLCanvasElement>(null);
  const img = useRef<HTMLImageElement | null>(null);
  const [ready, setReady] = useState(0);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
    const im = new Image();
    im.onload = () => { img.current = im; setReady((n) => n + 1); };
    im.onerror = () => { img.current = null; setFailed(true); };
    im.src = src;
  }, [src]);

  useEffect(() => {
    const cv = cvRef.current;
    if (!cv) return;
    const w = cv.clientWidth || 800;
    const im = img.current;
    const aspect = im ? im.naturalHeight / im.naturalWidth : 9 / 16;
    cv.width = w;
    cv.height = Math.round(w * aspect);
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, cv.height);
    if (im) ctx.drawImage(im, 0, 0, w, cv.height);
    for (const m of markers) {
      const x = m.fx * w, y = m.fy * cv.height;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = hexFor(m.team);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#09090b";
      ctx.stroke();
      ctx.font = "bold 13px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      const short = m.label.length > 10 ? `${m.label.slice(0, 9)}…` : m.label;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#09090b";
      ctx.strokeText(short, x, y - 8);
      ctx.fillStyle = hexFor(m.team);
      ctx.fillText(short, x, y - 8);
    }
  }, [markers, hexFor, ready]);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const hit = markers.find(
      (m) => Math.hypot(m.fx * r.width - px, m.fy * r.height - py) < HIT_PX);
    if (hit) onDelete(hit.id);
    else onPick(px / r.width, py / r.height);
  };

  return (
    <div className="relative">
      <canvas ref={cvRef} onClick={onClick}
        className="w-full rounded cursor-crosshair"
        aria-label="camera still — click a player to anchor them" />
      {failed && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-zinc-500">
          Still unavailable for this camera/time.
        </div>
      )}
    </div>
  );
}

export default function AnchorPlayers({ onDoc }: {
  onDoc: (d: PlayerIdentities) => void;
}) {
  const api = useProjectApi();
  const [doc, setDoc] = useState<PlayerAnchors | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [tab, setTab] = useState("start");
  const [times, setTimes] = useState<Record<string, string>>({});
  const [pend, setPend] = useState<{ angle: number; fx: number; fy: number } | null>(null);
  const [team, setTeam] = useState<"A" | "B">("A");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void api.anchors().then((d) => {
      setDoc(d);
      const lo = d.window?.[0] ?? 0;
      setTimes(Object.fromEntries(d.doc.moments.map(
        (m) => [m.id, mmss(m.t - lo)])));
    }).catch(() => setDoc(null));
    void api.multiangle().then((m) => setLabels(
      m.angles.map((a) => a.label || `Camera ${a.index + 1}`)))
      .catch(() => setLabels([]));
  }, [api]);

  useEffect(() => { if (pend) nameRef.current?.focus(); }, [pend]);

  if (!doc) return null;
  const lo = doc.window?.[0] ?? 0;
  const moments = doc.doc.moments;
  const moment = moments.find((m) => m.id === tab) ?? moments[0];
  const clicks = doc.doc.clicks;
  const forMoment = (a: number) =>
    clicks.filter((c) => c.moment === moment?.id && c.angle === a);

  const momentT = () => {
    if (!moment) return lo;
    const v = parseClock(times[moment.id] ?? "");
    return v == null ? moment.t : lo + v;
  };

  const addClick = (angle: number, fx: number, fy: number) => {
    const label = name.trim().slice(0, 40);
    if (!moment || !label) return;
    setDoc({
      ...doc,
      doc: {
        ...doc.doc,
        clicks: [...clicks, {
          id: `${Date.now().toString(36)}-${clicks.length}`,
          moment: moment.id, angle,
          fx: Math.min(1, Math.max(0, fx)),
          fy: Math.min(1, Math.max(0, fy)),
          team, label,
        }],
      },
    });
    setPend(null);
    setName("");
  };

  const save = async () => {
    setBusy(true);
    setErr(null);
    try {
      const momentsOut = moments.map((m) => ({
        id: m.id,
        t: parseClock(times[m.id] ?? "") != null
          ? lo + (parseClock(times[m.id] ?? "") ?? 0) : m.t,
      }));
      const d = await api.putAnchors({
        moments: momentsOut,
        clicks: clicks.map((c) => ({
          id: c.id, moment: c.moment, angle: c.angle, fx: c.fx, fy: c.fy,
          team: c.team, label: c.label,
        })),
      });
      setDoc(d);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const relink = async () => {
    if (!window.confirm(
      "Re-link player identities using these anchors? Manual merge/split "
      + "edits will be discarded.")) return;
    setBusy(true);
    setErr(null);
    try {
      const d = await api.relinkAnchors();
      onDoc(d);
      window.dispatchEvent(new Event(IDENTITIES_CHANGED_EVENT));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const teamName = (t: string) =>
    doc.teams?.[t]?.name || (t === "A" ? "Team A" : "Team B");
  const hexFor = (t: string) => doc.teams?.[t]?.hex || TEAM_HEX[t] || "#e4e4e7";

  return (
    <div className="rounded border border-zinc-800 p-3 mt-3 flex flex-col gap-2 min-w-0">
      <div className="flex items-center gap-2 flex-wrap">
        <Crosshair size={13} className="text-zinc-500" />
        <span className="text-xs font-semibold text-zinc-300">
          Anchor players (start / middle / end)
        </span>
        <span className="text-[11px] text-zinc-500">
          — click every player on each team, give their name
        </span>
      </div>
      <div className="flex items-center gap-1.5 flex-wrap">
        {moments.map((m) => (
          <button key={m.id} type="button"
            className={`rounded px-2 py-0.5 text-xs ${m.id === moment?.id
              ? "bg-amber-500 text-zinc-950 font-semibold"
              : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`}
            onClick={() => { setTab(m.id); setPend(null); }}>
            {m.id === "start" ? "Start" : m.id === "mid" ? "Middle" : "End"}
          </button>
        ))}
        <label className="flex items-center gap-1 text-[11px] text-zinc-500 ml-1">
          at
          <input
            className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-xs w-14 text-center font-mono"
            value={moment ? (times[moment.id] ?? "") : ""}
            onChange={(e) => moment && setTimes(
              (tt) => ({ ...tt, [moment.id]: e.target.value }))}
            aria-label="moment time (mm:ss)" />
        </label>
        <span className="ml-auto flex items-center gap-1.5">
          <button type="button" className={btnGhost} disabled={busy}
            onClick={() => void save()}>
            <Save size={12} /> Save anchors
          </button>
          <button type="button" className={btnGhost} disabled={busy}
            onClick={() => void relink()}>
            <RefreshCw size={12} /> Re-link players with anchors
          </button>
        </span>
      </div>
      {err && <div className="text-xs text-red-300">{err}</div>}
      {Array.from({ length: doc.n_angles }, (_, a) => (
        <div key={a} className="relative">
          <div className="text-[11px] text-zinc-500 mb-1">
            {labels[a] ?? `Camera ${a + 1}`}
          </div>
          <Still
            src={api.angleFrameUrl(a, momentT() - (doc.offsets[a] ?? 0), 720)}
            markers={forMoment(a)}
            hexFor={hexFor}
            onPick={(fx, fy) => { setPend({ angle: a, fx, fy }); setName(""); }}
            onDelete={(id) => setDoc({
              ...doc,
              doc: { ...doc.doc, clicks: clicks.filter((c) => c.id !== id) },
            })} />
          {pend && pend.angle === a && (
            <div className="absolute top-6 left-2 z-10 rounded bg-zinc-900/95 border border-zinc-700 p-2 flex items-center gap-1.5">
              {(["A", "B"] as const).map((t) => (
                <button key={t} type="button"
                  className="rounded px-2 py-0.5 text-xs font-semibold"
                  style={{
                    background: team === t ? hexFor(t) : "#27272a",
                    color: team === t ? "#09090b" : "#d4d4d8",
                  }}
                  onClick={() => setTeam(t)}>
                  {teamName(t)}
                </button>
              ))}
              <input ref={nameRef} value={name} placeholder="Player name"
                list={`anchor-names-${team}`} maxLength={40}
                className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-xs w-36"
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") addClick(a, pend.fx, pend.fy);
                  if (e.key === "Escape") setPend(null);
                }}
                aria-label="player name" />
              <datalist id={`anchor-names-${team}`}>
                {[...new Set(clicks.filter((c) => c.team === team)
                  .map((c) => c.label))].map((l) => (
                  <option key={l} value={l} />
                ))}
              </datalist>
              <button type="button" className={btnGhost}
                disabled={!name.trim()}
                onClick={() => addClick(a, pend.fx, pend.fy)}>
                OK
              </button>
              <button type="button" className={btnGhost}
                onClick={() => setPend(null)}>
                Cancel
              </button>
            </div>
          )}
        </div>
      ))}
      {clicks.filter((c) => c.moment === moment?.id).length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
          {clicks.filter((c) => c.moment === moment?.id).map((c) => (
            <span key={c.id} title={`${teamName(c.team)} — ${c.label}`}>
              <span style={{ color: hexFor(c.team) }}>{c.label}</span>
              <span className="text-zinc-500"> ·{labels[c.angle] ?? `Cam ${c.angle + 1}`} </span>
              {c.track_id != null ? (
                <span className="text-zinc-400">
                  → track {c.track_id} ({c.dist_m?.toFixed(1)} m)
                </span>
              ) : c.note ? (
                <span className="text-amber-400">{c.note}</span>
              ) : (
                <span className="text-zinc-500">unsaved</span>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
