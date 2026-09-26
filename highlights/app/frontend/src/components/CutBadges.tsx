import type { CutInfo } from "../types";

function fmtT(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = `${m}`.padStart(2, "0");
  const ss = `${sec}`.padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

/** "zones · 2/3 angles · 85%" + "window 29:00–1:34:00" badges for a
 * multi-angle director cut. Renders nothing for single-camera
 * projects (cut_info null/undefined). */
export default function CutBadges({ info }: { info?: CutInfo | null }) {
  if (!info) return null;
  const cls = "rounded px-1.5 py-0.5 text-[10px]";
  return (
    <>
      {info.zones_angles > 0 ? (
        <span
          className={`${cls} bg-emerald-900/60 text-emerald-300`}
          title={`zones drawn on ${info.zones_angles} of ${info.zones_total} angles`}
        >
          zones · {info.zones_angles}/{info.zones_total} angles
          {info.zone_share != null &&
            ` · ${Math.round(info.zone_share * 100)}%`}
        </span>
      ) : (
        <span className={`${cls} bg-zinc-800 text-zinc-500`}>no zones</span>
      )}
      {info.window_set && info.window && (
        <span
          className={`${cls} bg-sky-900/60 text-sky-300`}
          title="match window used for this cut"
        >
          window {fmtT(info.window[0])}–{fmtT(info.window[1])}
        </span>
      )}
    </>
  );
}
