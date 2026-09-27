import { useEffect, useState } from "react";
import { useProjectApi } from "../api";
import { fetchPlayers, saveRoster } from "../lib/players";
import type { PlayersRoster } from "../types";

/** "Scorer" picker for a review candidate; renders nothing until a roster
 *  with at least one named player exists (multi-angle projects only). */
export default function ScorerSelect({ candidateId }: { candidateId: string }) {
  const api = useProjectApi();
  const [roster, setRoster] = useState<PlayersRoster | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    void fetchPlayers(api).then((d) => {
      if (alive) setRoster(d?.roster ?? null);
    });
    return () => {
      alive = false;
    };
  }, [api]);

  if (!roster || roster.players.length === 0) return null;
  const value = roster.scorers[candidateId] ?? "";

  const change = async (pid: string) => {
    const scorers = { ...roster.scorers };
    if (pid) scorers[candidateId] = pid;
    else delete scorers[candidateId];
    const next = { ...roster, scorers };
    setRoster(next);
    setBusy(true);
    try {
      const r = await saveRoster(api, next);
      setRoster(r.roster);
    } catch {
      setRoster(roster);
    } finally {
      setBusy(false);
    }
  };

  return (
    <select
      className="bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-[10px] max-w-[9rem]"
      title="Scorer (from Player analysis)"
      aria-label="Scorer"
      value={value}
      disabled={busy}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => void change(e.target.value)}
    >
      <option value="">Scorer…</option>
      {roster.players.map((p) => (
        <option key={p.id} value={p.id}>
          {p.name}
        </option>
      ))}
    </select>
  );
}
