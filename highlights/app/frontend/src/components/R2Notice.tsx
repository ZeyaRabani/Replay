import { Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { mediaUrl, r2Api } from "../api";
import type { R2Current } from "../types";

/** Slim bar showing which match currently holds the fast-download slot. */
export default function R2Notice() {
  const [cur, setCur] = useState<R2Current | null>(null);

  useEffect(() => {
    let stop = false;
    const poll = () => {
      r2Api.current()
        .then((c) => !stop && setCur(c))
        .catch(() => undefined);
    };
    poll();
    const t = window.setInterval(poll, 15000);
    return () => { stop = true; window.clearInterval(t); };
  }, []);

  if (!cur || cur.state === "off" || cur.state === "none") return null;
  const cls = "bg-zinc-900/90 border-b border-zinc-800 text-[12px] text-zinc-300 px-4 py-1.5 flex items-center gap-2";
  if (cur.state === "ready")
    return (
      <div className={cls}>
        <Zap size={13} className="text-amber-300" />
        Fast download ready: <span className="font-semibold">{cur.title}</span>
        <a className="text-amber-300 hover:text-amber-200 ml-1" href={mediaUrl(`/api/projects/${cur.pid}/download`)}>— Download</a>
        <span className="text-zinc-500 ml-2 mob:hidden">
          Delete the match after downloading and the next one gets the fast download.
        </span>
      </div>
    );
  return (
    <div className={cls}>
      <Zap size={13} className="text-zinc-500" />
      Preparing fast download: {cur.title} {Math.round(cur.progress * 100)}%
    </div>
  );
}
