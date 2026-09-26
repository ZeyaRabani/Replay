import { Check, ChevronUp, Download, Loader2, RotateCcw, Volume2, VolumeX, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ProjectApiContext, projectApi, useProjectApi } from "../api";
import { TYPE_LABEL } from "../components/Timeline";
import { fmtClock } from "../lib/time";
import type { Candidate, RenderJob, Status, VideoInfo } from "../types";

const STATUS_STYLE: Record<string, string> = {
  pending: "bg-zinc-700 text-zinc-300",
  confirmed: "bg-emerald-700 text-emerald-100",
  rejected: "bg-red-800 text-red-100",
};

interface DragState {
  x0: number;
  y0: number;
  dx: number;
  decided: boolean;
}

function SwipeCard({
  c,
  index,
  total,
  active,
  videoSrc,
  onDecision,
  onReset,
}: {
  c: Candidate;
  index: number;
  total: number;
  active: boolean;
  videoSrc?: string;
  onDecision: (c: Candidate, status: Status) => void;
  onReset: (c: Candidate) => void;
}) {
  const vidRef = useRef<HTMLVideoElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const drag = useRef<DragState | null>(null);
  const [dx, setDx] = useState(0);
  const [muted, setMuted] = useState(true);

  // only the active card plays; keep the clip looping
  useEffect(() => {
    const el = vidRef.current;
    if (!el) return;
    if (active) {
      el.currentTime = c.clip_start;
      void el.play().catch(() => undefined);
    } else {
      el.pause();
    }
  }, [active, c.clip_start]);

  const onTime = () => {
    const el = vidRef.current;
    if (el && el.currentTime >= c.clip_end) el.currentTime = c.clip_start;
  };

  const onDown = (e: React.PointerEvent) => {
    drag.current = { x0: e.clientX, y0: e.clientY, dx: 0, decided: false };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const ndx = e.clientX - d.x0;
    const ndy = e.clientY - d.y0;
    if (!d.decided && Math.abs(ndx) > 10 && Math.abs(ndx) > Math.abs(ndy) * 1.5) d.decided = true;
    if (d.decided) {
      d.dx = ndx;
      setDx(ndx);
    }
  };
  const onUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d?.decided && Math.abs(d.dx) > 80) {
      onDecision(c, d.dx > 0 ? "confirmed" : "rejected");
    }
    setDx(0);
  };

  const frac = Math.min(1, Math.abs(dx) / 160);
  const stamp = dx > 10 ? "CONFIRM" : dx < -10 ? "REJECT" : null;

  return (
    <div
      ref={cardRef}
      data-card-index={index}
      className="relative h-[100dvh] w-full snap-start snap-always bg-black overflow-hidden"
    >
      <div
        className="absolute inset-0 dsk:max-w-[480px] dsk:mx-auto dsk:left-0 dsk:right-0"
        style={{ transform: dx ? `translateX(${dx}px)` : undefined, transition: drag.current ? "none" : "transform .2s" }}
      >
        {videoSrc && (
          <video
            ref={vidRef}
            src={videoSrc}
            className="absolute inset-0 w-full h-full object-contain"
            playsInline
            preload="metadata"
            muted={muted}
            onTimeUpdate={onTime}
          />
        )}
        {/* drag tint + stamp */}
        {dx !== 0 && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{ backgroundColor: dx > 0 ? `rgba(16,185,129,${frac * 0.35})` : `rgba(239,68,68,${frac * 0.35})` }}
          />
        )}
        {stamp && (
          <div
            className="absolute top-1/4 left-1/2 -translate-x-1/2 pointer-events-none font-black tracking-widest border-4 rounded-lg px-4 py-1"
            style={{
              fontSize: 28 + frac * 14,
              color: dx > 0 ? "#34d399" : "#f87171",
              borderColor: dx > 0 ? "#34d399" : "#f87171",
              opacity: 0.4 + frac * 0.6,
              transform: `translateX(-50%) rotate(-10deg)`,
            }}
          >
            {stamp}
          </div>
        )}
        {/* gesture/tap layer */}
        <div
          className="absolute inset-0 touch-pan-y cursor-grab"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onClick={() => setMuted((m) => !m)}
        />
        {/* top overlay */}
        <div className="absolute top-0 inset-x-0 p-4 pt-14 flex flex-col gap-1 bg-gradient-to-b from-black/70 to-transparent pointer-events-none">
          <div className="flex items-center justify-between text-xs text-zinc-300">
            <span className="font-mono">{index + 1} / {total}</span>
            <span className="font-mono">{fmtClock(c.clip_start)}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-amber-300">
              {TYPE_LABEL[c.type] ?? c.type}
              <span className="text-xs text-zinc-300 font-normal"> · {(c.confidence * 100).toFixed(0)}%</span>
            </span>
            <span className={`rounded px-1.5 py-0.5 text-[10px] uppercase ${STATUS_STYLE[c.status]}`}>{c.status}</span>
          </div>
        </div>
        {/* bottom overlay */}
        <div className="absolute bottom-0 inset-x-0 p-5 pb-8 flex items-end justify-center gap-5 bg-gradient-to-t from-black/80 to-transparent">
          <button
            className="w-16 h-16 rounded-full bg-red-600/90 hover:bg-red-500 text-white flex items-center justify-center shadow-lg"
            title="Reject"
            onClick={(e) => { e.stopPropagation(); onDecision(c, "rejected"); }}
          >
            <X size={30} />
          </button>
          {c.status !== "pending" && (
            <button
              className="w-10 h-10 rounded-full bg-zinc-700/80 hover:bg-zinc-600 text-zinc-200 flex items-center justify-center"
              title="Reset to pending"
              onClick={(e) => { e.stopPropagation(); onReset(c); }}
            >
              <RotateCcw size={16} />
            </button>
          )}
          <button
            className="w-16 h-16 rounded-full bg-emerald-600/90 hover:bg-emerald-500 text-white flex items-center justify-center shadow-lg"
            title="Confirm"
            onClick={(e) => { e.stopPropagation(); onDecision(c, "confirmed"); }}
          >
            <Check size={30} />
          </button>
        </div>
        {/* mute indicator */}
        <div className="absolute top-16 right-4 pointer-events-none text-zinc-300/80">
          {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
        </div>
      </div>
    </div>
  );
}

export default function SwipeReview() {
  const { id = "" } = useParams();
  const api = useMemo(() => projectApi(id), [id]);
  return (
    <ProjectApiContext.Provider value={api}>
      <SwipeReviewInner />
    </ProjectApiContext.Provider>
  );
}

function SwipeReviewInner() {
  const api = useProjectApi();
  const navigate = useNavigate();
  const [video, setVideo] = useState<VideoInfo | null>(null);
  const [proxyReady, setProxyReady] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [active, setActive] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<RenderJob | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const jobTimer = useRef<number | null>(null);

  const showError = (msg: string) => {
    setError(msg);
    window.setTimeout(() => setError(null), 5000);
  };

  useEffect(() => {
    void (async () => {
      try {
        const p = await api.project();
        setVideo(p.video);
        setProxyReady(p.proxy_ready);
      } catch (e) {
        showError(e instanceof Error ? e.message : String(e));
      }
      try {
        setCandidates(await api.listCandidates("time"));
      } catch {
        /* none yet */
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // track which card is in view
  useEffect(() => {
    const root = listRef.current;
    if (!root) return;
    const obs = new IntersectionObserver(
      (entries) => {
        for (const en of entries) {
          if (en.isIntersecting) {
            const i = Number((en.target as HTMLElement).dataset.cardIndex);
            if (!Number.isNaN(i)) setActive(i);
          }
        }
      },
      { root, threshold: 0.6 },
    );
    for (const el of root.querySelectorAll("[data-card-index]")) obs.observe(el);
    return () => obs.disconnect();
  }, [candidates.length]);

  const scrollTo = useCallback((i: number) => {
    const root = listRef.current;
    const el = root?.querySelector(`[data-card-index="${i}"]`);
    el?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const decide = useCallback(
    (c: Candidate, status: Status) => {
      // optimistic; revert on failure
      setCandidates((cs) => cs.map((x) => (x.id === c.id ? { ...x, status } : x)));
      api
        .patchCandidate(c.id, { status })
        .then((upd) => setCandidates((cs) => cs.map((x) => (x.id === c.id ? upd : x))))
        .catch((e) => {
          setCandidates((cs) => cs.map((x) => (x.id === c.id ? c : x)));
          showError(e instanceof Error ? e.message : String(e));
        });
      const i = candidates.findIndex((x) => x.id === c.id);
      if (i >= 0 && i < candidates.length - 1) {
        window.setTimeout(() => scrollTo(i + 1), 250);
      } else {
        window.setTimeout(() => scrollTo(candidates.length), 250); // end card
      }
    },
    [api, candidates, scrollTo],
  );

  const reset = useCallback(
    (c: Candidate) => {
      api
        .resetCandidate(c.id)
        .then((upd) => setCandidates((cs) => cs.map((x) => (x.id === c.id ? upd : x))))
        .catch((e) => showError(e instanceof Error ? e.message : String(e)));
    },
    [api],
  );

  // keyboard: up/down navigate, left reject, right confirm, space play/pause
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        scrollTo(Math.min(active + 1, candidates.length));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        scrollTo(Math.max(active - 1, 0));
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const c = candidates[active];
        if (c) {
          e.preventDefault();
          decide(c, e.key === "ArrowRight" ? "confirmed" : "rejected");
        }
      } else if (e.key === " ") {
        e.preventDefault();
        const v = document.querySelector<HTMLVideoElement>(`[data-card-index="${active}"] video`);
        if (v) (v.paused ? v.play() : v.pause());
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, candidates, decide, scrollTo]);

  useEffect(() => () => {
    if (jobTimer.current) window.clearInterval(jobTimer.current);
  }, []);

  const startRender = async () => {
    try {
      const { job_id } = await api.startRender({ overlay: true, reencode: false });
      setJob(null);
      if (jobTimer.current) window.clearInterval(jobTimer.current);
      jobTimer.current = window.setInterval(async () => {
        try {
          const j = await api.renderJob(job_id);
          setJob(j);
          if (j.state === "done" || j.state === "error") {
            if (jobTimer.current) window.clearInterval(jobTimer.current);
            jobTimer.current = null;
          }
        } catch {
          /* transient */
        }
      }, 1000);
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
    }
  };

  const videoSrc = video
    ? api.videoUrl(
        localStorage.getItem("replay.quality") === "hd" || !proxyReady ? "source" : "proxy",
        String(video.registered_at),
      )
    : undefined;

  const nConf = candidates.filter((c) => c.status === "confirmed").length;
  const nRej = candidates.filter((c) => c.status === "rejected").length;
  const nPend = candidates.length - nConf - nRej;
  const rendering = job && (job.state === "queued" || job.state === "running");

  return (
    <div className="h-[100dvh] bg-black text-zinc-100 overflow-hidden">
      {/* close */}
      <Link
        to={`/projects/${api.id}`}
        className="absolute top-3 left-3 z-20 w-10 h-10 rounded-full bg-zinc-800/80 hover:bg-zinc-700 flex items-center justify-center"
        title="Back to project"
      >
        <X size={18} />
      </Link>
      {error && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 bg-red-900/90 text-red-100 text-sm rounded px-3 py-1.5">
          {error}
        </div>
      )}
      <div ref={listRef} className="h-full overflow-y-scroll" style={{ scrollSnapType: "y mandatory" }}>
        {candidates.map((c, i) => (
          <SwipeCard
            key={c.id}
            c={c}
            index={i}
            total={candidates.length}
            active={i === active}
            videoSrc={videoSrc}
            onDecision={decide}
            onReset={reset}
          />
        ))}
        {candidates.length === 0 && (
          <div className="h-[100dvh] snap-start flex items-center justify-center text-zinc-500 text-sm">
            No candidates to review
          </div>
        )}
        {/* end card */}
        <div data-card-index={candidates.length} className="h-[100dvh] snap-start flex flex-col items-center justify-center gap-5 px-8 text-center">
          <div className="text-xl font-semibold">All reviewed</div>
          <div className="text-sm text-zinc-400">
            <b className="text-emerald-400">{nConf}</b> confirmed ·{" "}
            <b className="text-red-400">{nRej}</b> rejected ·{" "}
            <b className="text-zinc-300">{nPend}</b> pending
          </div>
          {!job && (
            <button
              onClick={() => void startRender()}
              disabled={nConf === 0}
              className="bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-zinc-900 font-semibold rounded px-5 py-2.5 text-sm"
            >
              Render &amp; download
            </button>
          )}
          {rendering && (
            <div className="flex items-center gap-3 text-sm text-zinc-300">
              <Loader2 size={15} className="animate-spin" />
              {job.message} · {Math.round(job.progress * 100)}%
            </div>
          )}
          {job?.state === "error" && (
            <div className="text-sm text-red-400">{job.error ?? "render failed"}</div>
          )}
          {job?.state === "done" && job.reel_url && (
            <a
              className="flex items-center gap-1.5 text-amber-400 hover:underline text-sm"
              href={api.fileUrl(job.reel_url)}
            >
              <Download size={14} /> Download reel
            </a>
          )}
          <button
            onClick={() => scrollTo(0)}
            className="flex items-center gap-1.5 text-sm text-zinc-300 hover:text-amber-300 border border-zinc-700 rounded px-4 py-2"
          >
            <ChevronUp size={15} /> Back to top
          </button>
          <button
            onClick={() => navigate(`/projects/${api.id}`)}
            className="text-sm text-zinc-500 hover:text-zinc-300"
          >
            Exit
          </button>
        </div>
      </div>
    </div>
  );
}
