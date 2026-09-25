import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useProjectApi } from "../api";
import type { DirectorFull, DirectorSegment, MultiangleInfo } from "../types";

export const ANGLE_COLORS = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399"];

const RULE_COLORS: Record<string, string> = {
  event: "#f87171",
  ball: "#10b981",
  cluster: "#0ea5e9",
  hold: "#71717a",
  coverage: "#64748b",
  start: "#94a3b8",
};

const RULE_LABELS: Record<string, string> = {
  event: "Event (shot/goal)",
  ball: "Ball",
  cluster: "Player cluster",
  hold: "Hold",
  coverage: "Coverage",
  start: "Start",
};

const RULE_ORDER = ["event", "ball", "cluster", "hold", "coverage", "start"];

function ratioKeys(ratios: Record<string, number>): string[] {
  const known = RULE_ORDER.filter((k) => (ratios[k] ?? 0) > 0);
  const rest = Object.keys(ratios).filter(
    (k) => !RULE_ORDER.includes(k) && ratios[k] > 0);
  return [...known, ...rest];
}

const fmtT = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const fmtS = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

const card = "rounded-lg border border-zinc-800 bg-zinc-900 p-4";
const input =
  "bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs font-mono placeholder:text-zinc-500 focus:outline-none focus:border-amber-400";

interface Props {
  onSeek?: (t: number) => void;
}

export default function DirectorCut({ onSeek }: Props) {
  const api = useProjectApi();
  const [info, setInfo] = useState<MultiangleInfo | null>(null);
  const [director, setDirector] = useState<DirectorFull | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hoveredSeg, setHoveredSeg] = useState<DirectorSegment | null>(null);
  const [offsetsOpen, setOffsetsOpen] = useState(false);
  const [offsets, setOffsets] = useState<string[]>([]);
  const [offBusy, setOffBusy] = useState(false);
  const [offErr, setOffErr] = useState<string | null>(null);
  const [recutBusy, setRecutBusy] = useState(false);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const i = await api.multiangle();
      setInfo(i);
      if (i.director) {
        try {
          setDirector(await api.multiangleDirector());
        } catch {
          /* director.json not ready yet */
        }
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const live = info?.status?.state === "queued" || info?.status?.state === "running";
  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
    if (!live) return;
    timer.current = window.setInterval(() => void refresh(), 3000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [live, refresh]);

  // prefill offsets from sync info
  useEffect(() => {
    if (info?.sync?.offsets && offsets.length === 0)
      setOffsets(info.sync.offsets.map((o) => String(o)));
  }, [info?.sync?.offsets, offsets.length]);

  const needsInput = info?.status?.state === "needs_input";
  const nAngles = info?.angles.length ?? 0;
  const segs = director?.segments ?? [];
  const total = segs.length ? segs[segs.length - 1].t_end : 0;
  const angleLabel = (i: number) => info?.angles[i]?.label ?? `Angle ${i + 1}`;

  const doRecut = async () => {
    setRecutBusy(true);
    try {
      const next = (info?.cut_style ?? "normal") === "fast" ? "normal" : "fast";
      await api.recut(next);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecutBusy(false);
    }
  };

  const submitOffsets = async () => {
    const vals = Array.from({ length: nAngles }, (_, i) => parseFloat(offsets[i] ?? ""));
    if (vals.some((v) => Number.isNaN(v))) {
      setOffErr("enter a number for every angle");
      return;
    }
    setOffBusy(true);
    setOffErr(null);
    try {
      await api.putOffsets(vals);
      setOffsetsOpen(false);
      await refresh();
    } catch (e) {
      setOffErr(e instanceof Error ? e.message : String(e));
    } finally {
      setOffBusy(false);
    }
  };

  const offsetsForm = (
    <div className="flex flex-col gap-2">
      <div className="flex items-end gap-2 flex-wrap">
        {Array.from({ length: nAngles }, (_, i) => (
          <label key={i} className="flex flex-col gap-0.5">
            <span className="text-[10px] text-zinc-500">
              a{i} · {angleLabel(i)}
              {i === 0 ? " (reference)" : ""}
            </span>
            <input
              type="number"
              step={0.1}
              className={`${input} w-20`}
              value={i === 0 ? "0" : (offsets[i] ?? "")}
              disabled={i === 0}
              onChange={(e) =>
                setOffsets((o) => {
                  const next = [...o];
                  while (next.length < nAngles) next.push("");
                  next[i] = e.target.value;
                  return next;
                })
              }
            />
          </label>
        ))}
        <button
          disabled={offBusy}
          onClick={() => void submitOffsets()}
          className="flex items-center gap-1 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5 text-xs disabled:opacity-40"
        >
          {offBusy ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
          Re-run from sync
        </button>
      </div>
      {offErr && <div className="text-[11px] text-red-300">{offErr}</div>}
    </div>
  );

  return (
    <div className="flex-1 min-h-0 overflow-auto p-4">
      <div className="max-w-4xl mx-auto flex flex-col gap-3">
        {error && <div className="text-xs text-red-300">{error}</div>}
        {!info ? (
          <div className="text-sm text-zinc-500 flex items-center gap-2">
            <Loader2 size={14} className="animate-spin" /> Loading…
          </div>
        ) : (
          <>
            {/* angles */}
            <div className={card}>
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-2">Angles</div>
              <ul className="flex flex-col gap-1.5">
                {info.angles.map((a) => (
                  <li key={a.index} className="flex items-center gap-2 text-xs">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: ANGLE_COLORS[a.index % ANGLE_COLORS.length] }}
                    />
                    <span className="font-medium text-zinc-200">
                      a{a.index} · {a.label}
                    </span>
                    <span className="text-zinc-500 truncate">
                      {a.url ?? a.filename ?? ""}
                    </span>
                    {a.duration != null && <span className="text-zinc-400 font-mono">{fmtT(a.duration)}</span>}
                    {a.status && (
                      <span className="ml-auto rounded px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-300">
                        {a.status}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>

            {/* sync */}
            <div className={card}>
              <div className="flex items-center gap-2 mb-2">
                <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Sync</div>
                {info.sync && (
                  <span className="rounded px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-300">
                    {info.sync.method}
                  </span>
                )}
              </div>
              {needsInput && (
                <div className="mb-3 rounded border border-orange-800 bg-orange-950/60 text-orange-200 text-xs p-2 flex items-start gap-2">
                  <AlertTriangle size={13} className="mt-0.5 shrink-0" />
                  <div className="flex-1">{info.status?.message}</div>
                </div>
              )}
              {!info.sync ? (
                <div className="text-xs text-zinc-500">Sync not available yet.</div>
              ) : (
                <>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-zinc-500 border-b border-zinc-800">
                        <th className="pb-1 font-medium">Pair</th>
                        <th className="pb-1 font-medium">Offset (s)</th>
                        <th className="pb-1 font-medium">PNR</th>
                        <th className="pb-1 font-medium" title="second peak / main peak">r2</th>
                        <th className="pb-1 font-medium"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {info.sync.pairs.map((pr) => (
                        <tr key={`${pr.a}-${pr.b}`} className="border-b border-zinc-800/50">
                          <td className="py-1">
                            a{pr.a} → a{pr.b}
                            <span className="text-zinc-500"> · {angleLabel(pr.b)}</span>
                          </td>
                          <td className="py-1 font-mono">{pr.offset.toFixed(2)}</td>
                          <td className="py-1 font-mono">{pr.pnr.toFixed(1)}</td>
                          <td className="py-1 font-mono">{pr.r2.toFixed(2)}</td>
                          <td className="py-1">
                            <span
                              className={`rounded px-1.5 py-0.5 text-[10px] ${
                                pr.confident
                                  ? "bg-emerald-900/60 text-emerald-200"
                                  : "bg-orange-900/60 text-orange-200"
                              }`}
                            >
                              {pr.confident ? "confident" : "low"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="mt-2 text-[11px] text-zinc-500">
                    {info.sync.triangle_residual_s != null &&
                      `triangle residual ${info.sync.triangle_residual_s.toFixed(2)} s`}
                    {info.sync.needs_manual.length > 0 &&
                      ` · needs manual: ${info.sync.needs_manual.map((i) => `a${i}`).join(", ")}`}
                  </div>
                  {info.sync.confidence_note && (
                    <div className="mt-1 text-[11px] text-zinc-500">
                      {info.sync.confidence_note}
                    </div>
                  )}
                  <div className="mt-3">
                    {needsInput ? (
                      offsetsForm
                    ) : offsetsOpen ? (
                      offsetsForm
                    ) : (
                      <button
                        className="text-xs text-amber-300 hover:text-amber-200 underline"
                        onClick={() => setOffsetsOpen(true)}
                      >
                        Override offsets
                      </button>
                    )}
                  </div>
                </>
              )}
            </div>

            {/* rule ratios */}
            <div className={card}>
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                  Director decisions
                </div>
                <div className="flex items-center gap-2">
                  <span className="rounded px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-300">
                    {(info.cut_style ?? info.director?.style ?? "normal") === "fast"
                      ? "fast cuts" : "normal cuts"}
                  </span>
                  <button
                    disabled={live || info.sources_purged || recutBusy}
                    title={info.sources_purged ? "angle sources deleted — cannot re-cut" : undefined}
                    onClick={() => void doRecut()}
                    className="text-[11px] text-amber-300 hover:text-amber-200 border border-zinc-700 rounded px-2 py-0.5 disabled:opacity-40 disabled:hover:text-amber-300"
                  >
                    {recutBusy ? <Loader2 size={11} className="animate-spin" /> : null}
                    {(info.cut_style ?? "normal") === "fast"
                      ? "Re-cut as Normal" : "Re-cut as Fast"}
                  </button>
                </div>
              </div>
              {!info.director ? (
                <div className="text-xs text-zinc-500">Director decisions not available yet.</div>
              ) : (
                <>
                  <div className="h-4 rounded overflow-hidden flex bg-zinc-800">
                    {ratioKeys(info.director.ratios).map((r) => {
                      const v = info.director!.ratios[r] ?? 0;
                      return (
                        <div
                          key={r}
                          className="h-full"
                          style={{ width: `${v * 100}%`, backgroundColor: RULE_COLORS[r] ?? "#52525b" }}
                          title={`${RULE_LABELS[r] ?? r}: ${(v * 100).toFixed(1)}%`}
                        />
                      );
                    })}
                  </div>
                  <div className="mt-2 flex items-center gap-3 text-[11px] text-zinc-400 flex-wrap">
                    {ratioKeys(info.director.ratios).map((r) => (
                      <span key={r} className="flex items-center gap-1">
                        <span
                          className="inline-block w-2 h-2 rounded-sm"
                          style={{ backgroundColor: RULE_COLORS[r] ?? "#52525b" }}
                        />
                        {RULE_LABELS[r] ?? r} {((info.director!.ratios[r] ?? 0) * 100).toFixed(1)}%
                      </span>
                    ))}
                    <span className="ml-auto">
                      {info.director.n_cuts} cuts · mean hold {info.director.mean_hold_s.toFixed(1)} s
                    </span>
                  </div>

                  {/* angle share */}
                  <div className="mt-4 h-4 rounded overflow-hidden flex bg-zinc-800">
                    {Object.entries(info.director.angle_share).map(([k, v]) =>
                      v > 0 ? (
                        <div
                          key={k}
                          className="h-full"
                          style={{
                            width: `${v * 100}%`,
                            backgroundColor: ANGLE_COLORS[Number(k) % ANGLE_COLORS.length],
                          }}
                          title={`${angleLabel(Number(k))}: ${(v * 100).toFixed(1)}%`}
                        />
                      ) : null,
                    )}
                  </div>
                  <div className="mt-2 flex items-center gap-3 text-[11px] text-zinc-400 flex-wrap">
                    {Object.keys(info.director.angle_share).map((k) => (
                      <span key={k} className="flex items-center gap-1">
                        <span
                          className="inline-block w-2 h-2 rounded-sm"
                          style={{ backgroundColor: ANGLE_COLORS[Number(k) % ANGLE_COLORS.length] }}
                        />
                        {angleLabel(Number(k))} {((info.director!.angle_share[k] ?? 0) * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* cut timeline */}
            <div className={card}>
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-2">
                Cut timeline
              </div>
              {segs.length === 0 || total <= 0 ? (
                <div className="text-xs text-zinc-500">Cut timeline not available yet.</div>
              ) : (
                <>
                  <div className="relative h-8 rounded bg-zinc-800 overflow-hidden">
                    {segs.map((s, i) => (
                      <div
                        key={i}
                        className="absolute top-0 h-full cursor-pointer hover:brightness-125"
                        style={{
                          left: `${(s.t_start / total) * 100}%`,
                          width: `${((s.t_end - s.t_start) / total) * 100}%`,
                          backgroundColor: ANGLE_COLORS[s.angle % ANGLE_COLORS.length],
                        }}
                        title={`${fmtS(s.t_start)}–${fmtS(s.t_end)} · ${angleLabel(s.angle)} · ${s.rule}`}
                        onMouseEnter={() => setHoveredSeg(s)}
                        onMouseLeave={() => setHoveredSeg((h) => (h === s ? null : h))}
                        onClick={() => onSeek?.(s.t_start)}
                      />
                    ))}
                  </div>
                  <div className="mt-1.5 text-[11px] text-zinc-400 h-4">
                    {hoveredSeg ? (
                      <span>
                        {fmtS(hoveredSeg.t_start)}–{fmtS(hoveredSeg.t_end)} · {angleLabel(hoveredSeg.angle)} ·{" "}
                        {hoveredSeg.rule} · score {hoveredSeg.score.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-zinc-600">hover a segment · click to seek</span>
                    )}
                  </div>
                </>
              )}
            </div>

            {/* score */}
            <div className={card}>
              <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-2">Score</div>
              <div className="text-lg font-semibold text-zinc-100">
                {info.score.home.label} {info.score.home.goals} – {info.score.away.goals}{" "}
                {info.score.away.label}
              </div>
              {(info.score.unassigned ?? 0) > 0 && (
                <div className="text-xs text-zinc-400 mt-1">unassigned: {info.score.unassigned}</div>
              )}
              <div className="text-[11px] text-zinc-500 mt-1">
                from confirmed goals with team set — set Home/Away on goal candidates in Review
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
