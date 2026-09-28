import type { AnalysisTeamInfo } from "../types";

export default function Swatch({ team, size = 12 }: { team: AnalysisTeamInfo | null; size?: number }) {
  const hex = team?.hex || "#71717a";
  const isLight = hex.toLowerCase() > "#aaaaaa";
  return (
    <span
      role="img"
      aria-label={team ? `${team.name} shirt colour` : "unknown team"}
      className={`inline-block rounded shrink-0 border ${isLight ? "border-zinc-500" : "border-zinc-700"}`}
      style={{ backgroundColor: hex, width: size, height: size }}
    />
  );
}
