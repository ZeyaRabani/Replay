import { useEffect, useState } from "react";
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

/** Fired after an identity name is saved or identities are re-linked so
 *  other views (radar) can refetch. */
export const IDENTITIES_CHANGED_EVENT = "hl:identities-changed";

/** Opt-in per-player identity cards (experimental, off by default).
 *  localStorage-backed; a window event keeps every mounted view
 *  (cards, radar, 3D replay) in sync when it changes. */
export const SHOW_IDS_KEY = "replay.showPlayerIdentities";
export const PLAYER_IDS_EVENT = "hl:player-identities-toggle";

export function showPlayerIdentities(): boolean {
  try {
    return localStorage.getItem(SHOW_IDS_KEY) === "1";
  } catch {
    return false;
  }
}

export function setShowPlayerIdentities(on: boolean): void {
  try {
    localStorage.setItem(SHOW_IDS_KEY, on ? "1" : "0");
  } catch {
    /* private mode */
  }
  window.dispatchEvent(new Event(PLAYER_IDS_EVENT));
}

export function usePlayerIdentities(): boolean {
  const [on, setOn] = useState(showPlayerIdentities);
  useEffect(() => {
    const h = () => setOn(showPlayerIdentities());
    window.addEventListener(PLAYER_IDS_EVENT, h);
    return () => window.removeEventListener(PLAYER_IDS_EVENT, h);
  }, []);
  return on;
}

const IDENTITY_PALETTE = [
  "#ef4444", "#3b82f6", "#eab308", "#a855f7", "#14b8a6", "#f97316",
  "#ec4899", "#84cc16", "#06b6d4", "#f43f5e", "#8b5cf6", "#22c55e",
  "#0ea5e9", "#d946ef", "#facc15", "#10b981", "#fb7185", "#6366f1",
  "#f59e0b", "#2dd4bf", "#c084fc", "#4ade80",
];

/** Stable per-identity colour: A1..A11 then B1..B11 walk the palette. */
export function identityHex(iid: string): string {
  const team = iid[0] === "B" ? 1 : 0;
  const n = Math.max(1, Number(iid.slice(1)) || 1);
  return IDENTITY_PALETTE[(team * 11 + n - 1) % IDENTITY_PALETTE.length];
}

export const fmtSpeed = (ms: number): string => `${(ms * 3.6).toFixed(1)} km/h`;
