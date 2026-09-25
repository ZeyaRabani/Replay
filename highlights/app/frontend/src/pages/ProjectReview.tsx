import { useCallback, useEffect, useRef, useState } from "react";
import { useProjectApi } from "../api";
import CandidateList from "../components/CandidateList";
import Navbar from "../components/Navbar";
import RenderBar from "../components/RenderBar";
import Timeline from "../components/Timeline";
import VideoPlayer from "../components/VideoPlayer";
import type { Candidate, Team, VideoInfo } from "../types";

interface Props {
  /** seek request from another tab (e.g. Stats); {t, n} so repeated seeks to the same t still fire */
  seekRequest: { t: number; n: number } | null;
}

export default function ProjectReview({ seekRequest }: Props) {
  const api = useProjectApi();
  const [video, setVideo] = useState<VideoInfo | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [sort, setSort] = useState<"confidence" | "time">("confidence");
  const [playhead, setPlayhead] = useState(0);
  const [proxyReady, setProxyReady] = useState(false);
  const [proxyProgress, setProxyProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [candVersion, setCandVersion] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const proxyTimer = useRef<number | null>(null);

  const showError = (msg: string) => {
    setError(msg);
    window.setTimeout(() => setError(null), 8000);
  };

  const run = useCallback(async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void run(async () => {
      try {
        const p = await api.project();
        setVideo(p.video);
        setProxyReady(p.proxy_ready);
        setCandVersion(p.candidates_version);
      } catch {
        /* not ready yet */
      }
      try {
        setCandidates(await api.listCandidates("confidence"));
      } catch {
        /* none yet */
      }
    });
  }, [run]);

  useEffect(() => {
    api
      .listCandidates(sort)
      .then(setCandidates)
      .catch(() => undefined);
  }, [sort]);

  const seek = (t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t;
    setPlayhead(t);
  };

  useEffect(() => {
    if (!seekRequest) return;
    const el = videoRef.current;
    if (!el) return;
    const apply = () => {
      el.currentTime = seekRequest.t;
      setPlayhead(seekRequest.t);
    };
    if (el.readyState >= 1) apply();
    else el.addEventListener("loadedmetadata", apply, { once: true });
    return () => el.removeEventListener("loadedmetadata", apply);
  }, [seekRequest, video]);

  const patch = async (id: string, p: Partial<Candidate> & { team?: Team }): Promise<boolean> => {
    try {
      const updated = await api.patchCandidate(id, p);
      setCandidates((cs) => cs.map((c) => (c.id === id ? updated : c)));
      if (selected?.id === id) setSelected(updated);
      return true;
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
      return false;
    }
  };

  const buildProxy = () =>
    run(async () => {
      const r = await api.buildProxy();
      if (r.status === "ready") {
        setProxyReady(true);
        setProxyProgress(1);
        return;
      }
      setProxyProgress(0);
      if (proxyTimer.current) window.clearInterval(proxyTimer.current);
      proxyTimer.current = window.setInterval(async () => {
        try {
          const s = await api.proxyStatus();
          setProxyProgress(s.progress);
          if (s.ready) {
            setProxyReady(true);
            if (proxyTimer.current) window.clearInterval(proxyTimer.current);
          }
        } catch {
          /* transient poll failure */
        }
      }, 1000);
    });

  const videoSrc = video ? api.videoUrl(proxyReady ? "proxy" : "source", String(video.registered_at)) : undefined;

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <Navbar
        busy={busy}
        proxyProgress={proxyProgress}
        onLoadVideo={async (p) =>
          run(async () => {
            const v = await api.registerVideo(p);
            setVideo(v);
            setProxyReady(v.proxy_ready);
            setProxyProgress(null);
            const cs = await api.listCandidates(sort);
            setCandidates(cs);
            if (selected) setSelected(cs.find((c) => c.id === selected.id) ?? null);
          })
        }
        onLoadCandidatesPath={async (p) =>
          run(async () => {
            await api.loadCandidatesPath(p);
            const proj = await api.project();
            setCandVersion(proj.candidates_version);
            setCandidates(await api.listCandidates(sort));
          })
        }
        onLoadCandidatesFile={async (f) =>
          run(async () => {
            await api.loadCandidatesFile(f);
            const proj = await api.project();
            setCandVersion(proj.candidates_version);
            setCandidates(await api.listCandidates(sort));
          })
        }
        onBuildProxy={buildProxy}
      />
      {error && (
        <div className="bg-red-900/80 text-red-100 text-sm px-4 py-2 border-b border-red-700">{error}</div>
      )}
      <div className="flex flex-1 min-h-0 gap-3 p-3">
        <div className="flex flex-col gap-3 w-[62%] min-w-0">
          {video && videoSrc ? (
            <VideoPlayer
              ref={videoRef}
              src={videoSrc}
              resetKey={video.registered_at}
              selected={selected}
              onTime={setPlayhead}
              onSetIn={(t) => selected && patch(selected.id, { clip_start: t })}
              onSetOut={(t) => selected && patch(selected.id, { clip_end: t })}
            />
          ) : (
            <div className="aspect-video bg-zinc-900 rounded-lg flex items-center justify-center text-zinc-500 text-sm">
              Load a video to begin
            </div>
          )}
          <Timeline
            duration={video?.duration_s ?? 0}
            candidates={candidates}
            selectedId={selected?.id ?? null}
            playhead={playhead}
            onSeek={seek}
            onSelect={(c) => {
              setSelected(c);
              seek(c.t);
            }}
          />
        </div>
        <div className="flex-1 min-w-0 min-h-0">
          <CandidateList
            candidates={candidates}
            selectedId={selected?.id ?? null}
            thumbV={video ? `${video.registered_at}-${candVersion}` : undefined}
            sort={sort}
            onSort={setSort}
            onSelect={(c) => {
              setSelected(c);
              seek(c.t);
            }}
            onPatch={patch}
            onReset={async (id) => {
              try {
                const updated = await api.resetCandidate(id);
                setCandidates((cs) => cs.map((c) => (c.id === id ? updated : c)));
                if (selected?.id === id) setSelected(updated);
              } catch (e) {
                showError(e instanceof Error ? e.message : String(e));
              }
            }}
          />
        </div>
      </div>
      <RenderBar
        candidates={candidates}
        onError={showError}
        resetKey={video ? `${video.registered_at}-${candVersion}` : ""}
      />
    </div>
  );
}
