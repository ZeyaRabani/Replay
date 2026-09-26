import { Activity, Flag, Loader2, Timer, Volume2, Waves } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useProjectApi } from "../api";
import { TYPE_COLORS, TYPE_LABEL } from "../components/Timeline";
import { fmtClock } from "../lib/time";
import type { MatchStats, Stats, TopMoment } from "../types";

const fmt = fmtClock;
const pct = (v: number) => `${Math.round(v * 100)}%`;

const SERIES = [
  { key: "motion", color: "#38bdf8", label: "Motion" },
  { key: "audio", color: "#a78bfa", label: "Audio" },
  { key: "excitement", color: "#fbbf24", label: "Excitement" },
] as const;

const TYPE_ORDER = [
  "goal", "shot", "goalmouth", "crowd", "attack",
  "chance", "excitement", "other",
];

interface Props {
  onSeek: (t: number) => void;
}

interface ChartClickState {
  activeTooltipIndex?: number | string | null;
  activeLabel?: string | number | null;
}

const tooltipStyle = {
  contentStyle: { background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6, fontSize: 12 },
  labelStyle: { color: "#e4e4e7" },
  itemStyle: { color: "#d4d4d8" },
};

function Card({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-lg border border-zinc-800 bg-zinc-900 p-3 flex flex-col ${className}`}>
      <div className="text-xs font-semibold text-zinc-300 mb-2">{title}</div>
      {children}
    </div>
  );
}

function Kpi({ icon, label, value, sub, onClick }: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  onClick?: () => void;
}) {
  const Tag = onClick ? "button" : "div";
  return (
    <Tag
      onClick={onClick}
      className={`rounded-lg border border-zinc-800 bg-zinc-900 p-3 flex items-start gap-3 text-left ${
        onClick ? "hover:border-amber-400/60 cursor-pointer" : ""
      }`}
    >
      <div className="text-amber-400 mt-0.5">{icon}</div>
      <div className="min-w-0">
        <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
        <div className="text-lg font-semibold leading-tight">{value}</div>
        {sub && <div className="text-[11px] text-zinc-500 truncate">{sub}</div>}
      </div>
    </Tag>
  );
}

function StatRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 py-1.5 border-b border-zinc-800/60 last:border-b-0">
      <span className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</span>
      <span className="text-sm font-semibold text-zinc-200">{children}</span>
    </div>
  );
}

function MatchStatsCard({ ms, onSeek }: { ms: MatchStats; onSeek: (t: number) => void }) {
  const near = ms.territory.near_goal_pct;
  const far = ms.territory.far_goal_pct;
  return (
    <Card title="Match stats">
      <div className="grid grid-cols-1 dsk:md:grid-cols-2 gap-x-8">
        <div>
          <StatRow label="Goals">{ms.goals}</StatRow>
          <StatRow label="Shots on goal">{ms.shots_on_goal}</StatRow>
          <StatRow label="Goalmouth actions">{ms.goalmouth_actions}</StatRow>
          <StatRow label="Attacks">{ms.attacks}</StatRow>
          <StatRow label="Crowd reactions">{ms.crowd_reactions}</StatRow>
          <StatRow label="Big moments">{ms.big_moments}</StatRow>
        </div>
        <div>
          <div className="py-1.5 border-b border-zinc-800/60">
            <div className="flex items-center justify-between gap-4 mb-1">
              <span className="text-[11px] uppercase tracking-wide text-zinc-500">Territory</span>
              <span className="text-[11px] text-zinc-400">
                near goal {Math.round(near)}% · far {Math.round(far)}%
              </span>
            </div>
            <div className="flex h-2 rounded overflow-hidden">
              <div className="bg-amber-400" style={{ width: `${near}%` }} />
              <div className="bg-zinc-600" style={{ width: `${far}%` }} />
            </div>
          </div>
          <StatRow label="High-intensity play">{Math.round(ms.tempo.high_intensity_pct)}%</StatRow>
          <StatRow label="Estimated stoppages">{Math.round(ms.stoppages.estimated_stoppage_pct)}%</StatRow>
          <StatRow label="Whistles">{ms.stoppages.whistles}</StatRow>
          <StatRow label="Peak minute">
            <button
              onClick={() => onSeek(ms.peak_minute.t)}
              className="text-amber-400 hover:underline"
            >
              {fmt(ms.peak_minute.t)} ({ms.peak_minute.events} events)
            </button>
          </StatRow>
        </div>
      </div>
      {ms.halves.length > 0 && (
        <table className="w-full text-[11px] text-zinc-400 mt-2">
          <thead>
            <tr className="text-zinc-500 text-left">
              <th className="py-0.5 font-normal">Half</th>
              <th className="py-0.5 font-normal">Goals</th>
              <th className="py-0.5 font-normal">Shots</th>
              <th className="py-0.5 font-normal">Attacks</th>
              <th className="py-0.5 font-normal">Motion</th>
            </tr>
          </thead>
          <tbody>
            {ms.halves.map((h, i) => (
              <tr key={i} className="border-t border-zinc-800/60">
                <td className="py-0.5">{fmt(h.start)}–{fmt(h.end)}</td>
                <td>{h.goals}</td>
                <td>{h.shots_on_goal}</td>
                <td>{h.attacks}</td>
                <td>{Math.round(h.mean_motion_pct)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="text-[10px] text-zinc-600 mt-2">
        estimated from motion/audio — single camera
      </div>
    </Card>
  );
}

export default function ProjectStats({ onSeek }: Props) {
  const api = useProjectApi();
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .stats()
      .then(setStats)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [api]);

  const byType = useMemo(() => {
    if (!stats) return [];
    return TYPE_ORDER.filter((k) => (stats.events_by_type[k] ?? 0) > 0).map((k) => ({
      name: k,
      value: stats.events_by_type[k],
    }));
  }, [stats]);

  const per10 = useMemo(
    () => (stats ? stats.events_per_10min.map((r) => ({ ...r, label: fmt(r.t) })) : []),
    [stats],
  );
  const per10Keys = useMemo(() => {
    const ks = new Set<string>();
    stats?.events_per_10min.forEach((r) => Object.keys(r).forEach((k) => k !== "t" && ks.add(k)));
    return TYPE_ORDER.filter((k) => ks.has(k));
  }, [stats]);

  if (error)
    return (
      <div className="p-6 text-sm text-zinc-400">
        Stats not available: <span className="text-red-300">{error}</span>
      </div>
    );
  if (!stats)
    return (
      <div className="p-6 text-sm text-zinc-500 flex items-center gap-2">
        <Loader2 size={14} className="animate-spin" /> Loading stats…
      </div>
    );

  const seekFromChart = (s: ChartClickState) => {
    if (s.activeLabel !== undefined && s.activeLabel !== null) {
      const t = Number(s.activeLabel);
      if (!Number.isNaN(t)) onSeek(t);
      return;
    }
    if (s.activeTooltipIndex !== undefined && s.activeTooltipIndex !== null) {
      const row = stats.timeline[Number(s.activeTooltipIndex)];
      if (row) onSeek(row.t);
    }
  };

  const ms = stats.match_stats;
  const halfGap =
    stats.halves.length >= 2 ? { from: stats.halves[0].end, to: stats.halves[1].start } : null;
  const totalEvents = byType.reduce((s, b) => s + b.value, 0);

  return (
    <div className="flex-1 min-h-0 overflow-auto p-4">
      <div className="max-w-7xl mx-auto flex flex-col gap-4">
        {ms && <MatchStatsCard ms={ms} onSeek={onSeek} />}
        <div className="grid grid-cols-2 dsk:md:grid-cols-3 dsk:xl:grid-cols-6 gap-3">
          <Kpi icon={<Activity size={18} />} label="Mean motion" value={pct(stats.activity.mean_motion)} />
          <Kpi
            icon={<Waves size={18} />}
            label="Peak motion"
            value={fmt(stats.activity.peak_motion_t)}
            sub="click to seek"
            onClick={() => onSeek(stats.activity.peak_motion_t)}
          />
          <Kpi
            icon={<Volume2 size={18} />}
            label="Loudest moment"
            value={fmt(stats.activity.loudest_t)}
            sub="click to seek"
            onClick={() => onSeek(stats.activity.loudest_t)}
          />
          <Kpi
            icon={<Timer size={18} />}
            label="Halves detected"
            value={String(stats.halves.length)}
            sub={stats.halves.map((h) => `${fmt(h.start)}–${fmt(h.end)}`).join(" · ") || "none"}
          />
          <Kpi icon={<Flag size={18} />} label="Whistles" value={String(stats.whistles.length)} />
          <Kpi
            icon={<Timer size={18} />}
            label="Quietest stretch"
            value={`${fmt(stats.activity.quietest_stretch[0])}–${fmt(stats.activity.quietest_stretch[1])}`}
            onClick={() => onSeek(stats.activity.quietest_stretch[0])}
            sub="click to seek"
          />
        </div>

        <Card title={`Match timeline (${stats.bin_s}s bins) — click to seek in Review`}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={stats.timeline} onClick={seekFromChart} margin={{ left: -20, right: 10, top: 10 }}>
                <defs>
                  {SERIES.map((s) => (
                    <linearGradient key={s.key} id={`g-${s.key}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={s.color} stopOpacity={0.5} />
                      <stop offset="100%" stopColor={s.color} stopOpacity={0.02} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid stroke="#27272a" vertical={false} />
                <XAxis
                  dataKey="t"
                  type="number"
                  domain={[0, stats.duration_s]}
                  tickFormatter={fmt}
                  stroke="#71717a"
                  fontSize={10}
                  tickCount={12}
                />
                <YAxis domain={[0, 1]} stroke="#71717a" fontSize={10} tickFormatter={pct} />
                <Tooltip
                  {...tooltipStyle}
                  labelFormatter={(v) => fmt(Number(v))}
                  formatter={(v, name) => [typeof v === "number" ? v.toFixed(2) : String(v), String(name)]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {halfGap && (
                  <ReferenceArea
                    x1={halfGap.from}
                    x2={halfGap.to}
                    fill="#f4f4f5"
                    fillOpacity={0.08}
                    stroke="#71717a"
                    strokeDasharray="3 3"
                    label={{ value: "half-time", fill: "#a1a1aa", fontSize: 10, position: "insideTop" }}
                  />
                )}
                {stats.halves.map((h, i) => (
                  <ReferenceLine key={i} x={h.start} stroke="#52525b" strokeDasharray="2 2" />
                ))}
                {SERIES.map((s) => (
                  <Area
                    key={s.key}
                    type="monotone"
                    dataKey={s.key}
                    name={s.label}
                    stroke={s.color}
                    fill={`url(#g-${s.key})`}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                {stats.top_moments.map((m, i) => (
                  <ReferenceDot
                    key={i}
                    x={m.t}
                    y={Math.min(1, 0.92 + (i % 3) * 0.03)}
                    r={5}
                    fill={TYPE_COLORS[m.type] ?? "#9ca3af"}
                    stroke="#18181b"
                    onClick={() => onSeek(m.t)}
                    className="cursor-pointer"
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-3 text-[11px] text-zinc-400 mt-1 px-1">
            {TYPE_ORDER.filter((k) => stats.top_moments.some((m) => m.type === k)).map((k) => (
              <span key={k} className="flex items-center gap-1">
                <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: TYPE_COLORS[k] }} />
                {(TYPE_LABEL as Record<string, string>)[k] ?? k}
              </span>
            ))}
            <span className="ml-auto">
              match window {fmt(stats.match_window[0])}–{fmt(stats.match_window[1])}
            </span>
          </div>
        </Card>

        <div className="grid grid-cols-1 dsk:lg:grid-cols-3 gap-4">
          <Card title="Events per 10 min" className="dsk:lg:col-span-2">
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={per10}
                  margin={{ left: -25, right: 10, top: 10 }}
                  onClick={(s: ChartClickState) => {
                    const idx = s.activeTooltipIndex;
                    if (idx === undefined || idx === null) return;
                    const row = per10[Number(idx)];
                    if (row) onSeek(row.t);
                  }}
                >
                  <CartesianGrid stroke="#27272a" vertical={false} />
                  <XAxis dataKey="label" stroke="#71717a" fontSize={10} />
                  <YAxis allowDecimals={false} stroke="#71717a" fontSize={10} />
                  <Tooltip {...tooltipStyle} cursor={{ fill: "#27272a" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {per10Keys.map((k) => (
                    <Bar key={k} dataKey={k} stackId="a" fill={TYPE_COLORS[k] ?? "#9ca3af"} isAnimationActive={false} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title={`Events by type (${totalEvents})`}>
            <div className="h-56 flex items-center">
              <ResponsiveContainer width="50%" height="100%">
                <PieChart>
                  <Pie
                    data={byType}
                    dataKey="value"
                    nameKey="name"
                    innerRadius="55%"
                    outerRadius="90%"
                    stroke="#18181b"
                    isAnimationActive={false}
                  >
                    {byType.map((b) => (
                      <Cell key={b.name} fill={TYPE_COLORS[b.name] ?? "#9ca3af"} />
                    ))}
                  </Pie>
                  <Tooltip {...tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
              <ul className="flex-1 flex flex-col gap-1.5 text-xs">
                {byType.map((b) => (
                  <li key={b.name} className="flex items-center gap-2">
                    <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: TYPE_COLORS[b.name] }} />
                    <span className="flex-1 text-zinc-300">{(TYPE_LABEL as Record<string, string>)[b.name] ?? b.name}</span>
                    <b>{b.value}</b>
                    <span className="text-zinc-500 w-9 text-right">{totalEvents ? pct(b.value / totalEvents) : "–"}</span>
                  </li>
                ))}
              </ul>
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 dsk:lg:grid-cols-3 gap-4">
          <Card title="Top moments" className="dsk:lg:col-span-2">
            <ol className="divide-y divide-zinc-800">
              {stats.top_moments.map((m: TopMoment, i) => (
                <li key={i}>
                  <button
                    onClick={() => onSeek(m.t)}
                    className="w-full flex items-center gap-3 py-1.5 px-1 text-left hover:bg-zinc-800 rounded"
                  >
                    <span className="text-zinc-500 text-xs w-5">{i + 1}</span>
                    <span
                      className="text-[10px] uppercase font-semibold rounded px-1.5 py-0.5 text-zinc-900 w-16 text-center"
                      style={{ background: TYPE_COLORS[m.type] ?? "#9ca3af" }}
                    >
                      {(TYPE_LABEL as Record<string, string>)[m.type] ?? m.type}
                    </span>
                    <span className="font-mono text-sm w-14">{fmt(m.t)}</span>
                    <span className="flex-1 text-xs text-zinc-400 truncate">{m.reason}</span>
                    <span className="text-xs text-amber-300 font-semibold">{pct(m.confidence)}</span>
                  </button>
                </li>
              ))}
            </ol>
          </Card>
          <Card title="Pipeline">
            <dl className="text-xs grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-zinc-300">
              <dt className="text-zinc-500">Model</dt>
              <dd>{stats.pipeline.model}</dd>
              <dt className="text-zinc-500">Reference AUROC</dt>
              <dd>{stats.pipeline.auroc_reference != null ? stats.pipeline.auroc_reference.toFixed(2) : "—"}</dd>
              <dt className="text-zinc-500">Duration</dt>
              <dd>{fmt(stats.duration_s)}</dd>
              <dt className="text-zinc-500">Whistles</dt>
              <dd className="flex flex-wrap gap-1">
                {stats.whistles.map((w, i) => (
                  <button
                    key={i}
                    onClick={() => onSeek(w)}
                    className="font-mono bg-zinc-800 hover:bg-zinc-700 rounded px-1"
                  >
                    {fmt(w)}
                  </button>
                ))}
              </dd>
              {stats.pipeline.notes && (
                <>
                  <dt className="text-zinc-500">Notes</dt>
                  <dd className="text-zinc-400">{stats.pipeline.notes}</dd>
                </>
              )}
            </dl>
          </Card>
        </div>
      </div>
    </div>
  );
}
