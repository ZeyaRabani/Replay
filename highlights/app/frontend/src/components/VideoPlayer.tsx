import { Flag, Play } from "lucide-react";
import { forwardRef, useEffect, useRef, useState } from "react";
import type { Candidate } from "../types";

interface Props {
  src: string;
  /** changes when a different video is registered (vs a source->proxy swap) */
  resetKey: number | undefined;
  selected: Candidate | null;
  onTime: (t: number) => void;
  onSetIn: (t: number) => void;
  onSetOut: (t: number) => void;
}

const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

const VideoPlayer = forwardRef<HTMLVideoElement, Props>(function VideoPlayer(props, ref) {
  const [time, setTime] = useState(0);
  const clipEnd = useRef<number | null>(null);
  const vref = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const v = vref.current;
    if (!v) return;
    const onTime = () => {
      setTime(v.currentTime);
      props.onTime(v.currentTime);
      if (clipEnd.current !== null && v.currentTime >= clipEnd.current) {
        v.pause();
        clipEnd.current = null;
      }
    };
    v.addEventListener("timeupdate", onTime);
    return () => v.removeEventListener("timeupdate", onTime);
  }, [props]);

  const setRefs = (el: HTMLVideoElement | null) => {
    vref.current = el;
    if (typeof ref === "function") ref(el);
    else if (ref) ref.current = el;
  };

  // On src change: reload explicitly. Preserve the playhead only for a
  // source->proxy swap of the same video; a new video (resetKey changed)
  // restarts at 0.
  const lastSrc = useRef<string | null>(null);
  const lastReset = useRef<number | undefined>(undefined);
  useEffect(() => {
    const v = vref.current;
    if (!v) return;
    const srcChanged = lastSrc.current !== null && lastSrc.current !== props.src;
    const videoChanged = lastReset.current !== props.resetKey;
    if (srcChanged) {
      if (videoChanged) {
        v.load();
      } else {
        const t = time;
        const resume = !v.paused;
        const onMeta = () => {
          v.currentTime = Math.min(t, v.duration || t);
          if (resume) void v.play();
        };
        v.addEventListener("loadedmetadata", onMeta, { once: true });
        v.load();
      }
    }
    lastSrc.current = props.src;
    lastReset.current = props.resetKey;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.src, props.resetKey]);

  const btn =
    "flex items-center gap-1 bg-zinc-700 hover:bg-zinc-600 disabled:opacity-40 rounded px-2 py-1 text-xs";

  return (
    <div className="bg-black rounded-lg overflow-hidden">
      <video ref={setRefs} src={props.src} controls className="w-full aspect-video bg-black" />
      <div className="flex items-center gap-2 px-3 py-2 bg-zinc-900">
        <span className="text-xs font-mono text-zinc-300">{fmt(time)}</span>
        <span className="flex-1" />
        <button className={btn} disabled={!props.selected} onClick={() => props.onSetIn(time)}>
          <Flag size={12} /> Set IN
        </button>
        <button className={btn} disabled={!props.selected} onClick={() => props.onSetOut(time)}>
          <Flag size={12} className="rotate-180" /> Set OUT
        </button>
        <button
          className={btn}
          disabled={!props.selected}
          onClick={() => {
            const v = vref.current;
            const c = props.selected;
            if (!v || !c) return;
            v.currentTime = c.clip_start;
            clipEnd.current = c.clip_end;
            void v.play();
          }}
        >
          <Play size={12} /> Play clip
        </button>
      </div>
    </div>
  );
});

export default VideoPlayer;
