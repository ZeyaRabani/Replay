/** m:ss clock helpers shared by cards and stats pages. */

export function fmtClock(t: number): string {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}

/** Parse "m:ss", "m:ss.s" or a bare number of seconds; null on garbage. */
export function parseClock(s: string): number | null {
  const v = s.trim();
  if (v === "") return null;
  if (!v.includes(":")) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  const parts = v.split(":");
  if (parts.length !== 2) return null;
  const m = Number(parts[0]);
  const sec = Number(parts[1]);
  if (!Number.isFinite(m) || !Number.isFinite(sec)) return null;
  return m * 60 + sec;
}
