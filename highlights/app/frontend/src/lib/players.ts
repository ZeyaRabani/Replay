import type { ProjectApi } from "../api";
import type { PlayersResponse, PlayersRoster } from "../types";

/** Shared, per-project cache of GET /analysis/players so the Review list
 *  (one ScorerSelect per candidate card) fetches the roster once. */
const cache = new Map<string, Promise<PlayersResponse | null>>();

export function fetchPlayers(api: ProjectApi, fresh = false): Promise<PlayersResponse | null> {
  if (fresh) cache.delete(api.id);
  let p = cache.get(api.id);
  if (!p) {
    // 404 (single-camera project) or any other failure -> null, cached too
    p = api.players().catch(() => null);
    cache.set(api.id, p);
  }
  return p;
}

export function invalidatePlayers(api: ProjectApi): void {
  cache.delete(api.id);
}

export async function saveRoster(api: ProjectApi, roster: PlayersRoster) {
  const r = await api.putRoster(roster);
  invalidatePlayers(api);
  return r;
}

export const fmtDur = (s: number): string => {
  const t = Math.max(0, Math.round(s));
  const m = Math.floor(t / 60);
  return m ? `${m}m ${String(t % 60).padStart(2, "0")}s` : `${t}s`;
};

export const fmtDist = (m: number): string => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);
