import { Clapperboard, FileUp, Loader2, Upload, Video } from "lucide-react";
import { useState } from "react";

interface Props {
  onLoadVideo: (path: string) => Promise<void>;
  onLoadCandidatesPath: (path: string) => Promise<void>;
  onLoadCandidatesFile: (file: File) => Promise<void>;
  onBuildProxy: () => Promise<void>;
  proxyProgress: number | null;
  busy: boolean;
}

export default function Navbar(props: Props) {
  const [videoPath, setVideoPath] = useState("");
  const [candPath, setCandPath] = useState("");

  const input =
    "bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs w-64 placeholder:text-zinc-500";
  const btn =
    "flex items-center gap-1 bg-zinc-700 hover:bg-zinc-600 disabled:opacity-40 rounded px-2.5 py-1 text-xs font-medium";

  return (
    <nav className="flex items-center gap-3 px-4 py-2 bg-zinc-900 border-b border-zinc-800 flex-wrap">
      <span className="flex items-center gap-2 font-semibold text-sm mr-2">
        <Clapperboard size={18} className="text-amber-400" /> Replay Highlights
      </span>
      <input
        className={input}
        placeholder="/abs/path/match.mp4"
        value={videoPath}
        onChange={(e) => setVideoPath(e.target.value)}
      />
      <button className={btn} disabled={props.busy || !videoPath} onClick={() => props.onLoadVideo(videoPath)}>
        <Video size={14} /> Load video
      </button>
      <input
        className={input}
        placeholder="/abs/path/candidates.json"
        value={candPath}
        onChange={(e) => setCandPath(e.target.value)}
      />
      <button
        className={btn}
        disabled={props.busy || !candPath}
        onClick={() => props.onLoadCandidatesPath(candPath)}
      >
        <FileUp size={14} /> Load candidates
      </button>
      <label className={`${btn} cursor-pointer`}>
        <Upload size={14} /> Upload
        <input
          type="file"
          accept=".json"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && props.onLoadCandidatesFile(e.target.files[0])}
        />
      </label>
      <button className={btn} disabled={props.busy} onClick={props.onBuildProxy}>
        {props.proxyProgress !== null && props.proxyProgress < 1 ? (
          <>
            <Loader2 size={14} className="animate-spin" /> {Math.round(props.proxyProgress * 100)}%
          </>
        ) : (
          "Build proxy"
        )}
      </button>
    </nav>
  );
}
