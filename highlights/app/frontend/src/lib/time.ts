/** h:mm:ss clock helpers shared by cards and stats pages. */

export function fmtClock(t: number): string {
  const s = Math.max(0, t);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

/** Parse "h:mm:ss", "m:ss", "m:ss.s" or a bare number of seconds; null on garbage. */
export function parseClock(s: string): number | null {
  const v = s.trim();
  if (v === "") return null;
  if (!v.includes(":")) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  const parts = v.split(":");
  if (parts.length === 2) {
    const m = Number(parts[0]);
    const sec = Number(parts[1]);
    if (!Number.isFinite(m) || !Number.isFinite(sec)) return null;
    return m * 60 + sec;
  }
  if (parts.length === 3) {
    const h = Number(parts[0]);
    const m = Number(parts[1]);
    const sec = Number(parts[2]);
    if (!Number.isFinite(h) || !Number.isFinite(m) || !Number.isFinite(sec)) return null;
    return h * 3600 + m * 60 + sec;
  }
  return null;
}
