/** Live↔output time mapping for the director cut, mirroring
 *  highlights/multiangle/timemap.py. `live` is time on the shared clock
 *  relative to the cut start (0 = first second of the cut); `out` is the
 *  position inside match.mp4, shifted by inserted slow-mo replays. */
import type { ReplayInfo } from "../types";

const dur = (r: ReplayInfo) => (r.t_src_end - r.t_src_start) / (r.speed || 1);

/** Map a live-time point to output time (after replays inserted at/before it). */
export function toOut(live: number, replays: ReplayInfo[] = []): number {
  return live + replays.reduce((s, r) => s + (r.t_live_at <= live ? dur(r) : 0), 0);
}

/** Map output time back to live time; a point inside a replay maps to the
 *  replay's (frozen) live time, matching from_output_time_with_replays. */
export function fromOut(o: number, replays: ReplayInfo[] = []): number {
  let shift = 0;
  const rs = [...replays].sort((a, b) => a.t_out_start - b.t_out_start);
  for (const r of rs) {
    if (o < r.t_out_start) break;
    if (o < r.t_out_end) return r.t_live_at;
    shift += r.t_out_end - r.t_out_start;
  }
  return o - shift;
}

/** True when the output time falls inside an inserted replay segment. */
export function inReplay(o: number, replays: ReplayInfo[] = []): boolean {
  return replays.some((r) => o >= r.t_out_start && o < r.t_out_end);
}
