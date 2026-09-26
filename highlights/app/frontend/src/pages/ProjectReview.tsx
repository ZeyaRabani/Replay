import { useCallback, useEffect, useRef, useState } from "react";
import { useProjectApi } from "../api";
import CandidateList from "../components/CandidateList";
import Navbar from "../components/Navbar";
import RenderBar from "../components/RenderBar";
import Timeline from "../components/Timeline";
import VideoPlayer from "../components/VideoPlayer";
import { fmtClock, parseClock } from "../lib/time";
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
  const [quality, setQuality] = useState<"fast" | "hd">(
    () => (localStorage.getItem("replay.quality") === "hd" ? "hd" : "fast"));
  const [isMultiangle, setIsMultiangle] = useState(false);
  const [cutLabel, setCutLabel] = useState<string | null>(null);
  const [win, setWin] = useState<[number, number] | null>(null);
  const [winEdit, setWinEdit] = useState(false);
  const [winIn, setWinIn] = useState("");
  const [winOut, setWinOut] = useState("");
  const [trimProg, setTrimProg] = useState<number | null>(null);
  const trimTimer = useRef<number | null>(null);
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
        setIsMultiangle(p.mode === "multiangle");
        if (p.mode === "multiangle") {
          api.listCuts()
            .then((cl) => {
              const act = cl.cuts.find((c) => c.id === cl.active);
              setCutLabel(act?.label ?? null);
            })
            .catch(() => undefined);
        }
        try {
          const mw = await api.getMatchWindow();
          setWin(mw.match_window);
        } catch {
          /* no window yet */
        }
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

  const videoSrc = video
    ? api.videoUrl(quality === "hd" || !proxyReady ? "source" : "proxy",
                   String(video.registered_at))
    : undefined;

  const setQ = (q: "fast" | "hd") => {
    setQuality(q);
    localStorage.setItem("replay.quality", q);
  };

  const saveWindow = async () => {
    const s = parseClock(winIn);
    const e = parseClock(winOut);
    if (s === null || e === null || s >= e) {
      showError("invalid window (need m:ss start < end)");
      return;
    }
    try {
      const r = await api.putMatchWindow(s, e);
      setWin(r.match_window);
      setWinEdit(false);
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err));
    }
  };

  const downloadTrimmed = async () => {
    if (!win || !video) return;
    const [s, e] = win;
    try {
      const st = await api.startTrim(s, e);
      if (st.ready) {
        window.open(api.trimmedUrl(s, e, String(video.registered_at)), "_blank");
        return;
      }
      setTrimProg(0);
      if (trimTimer.current) window.clearInterval(trimTimer.current);
      trimTimer.current = window.setInterval(async () => {
        try {
          const r = await api.trimStatus(s, e);
          setTrimProg(r.progress);
          if (r.ready) {
            if (trimTimer.current) window.clearInterval(trimTimer.current);
            setTrimProg(null);
            window.open(api.trimmedUrl(s, e, String(video.registered_at)), "_blank");
          }
        } catch {
          /* transient */
        }
      }, 1000);
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err));
    }
  };

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
      <div className="flex flex-1 min-h-0 gap-3 p-3 mob:flex-col mob:overflow-y-auto">
        <div className="flex flex-col gap-3 dsk:w-[62%] mob:w-full min-w-0">
          {video && videoSrc ? (
            <div className="flex items-center gap-2 justify-end mob:flex-col mob:items-stretch">
              <div className="flex rounded overflow-hidden border border-zinc-700 text-[11px] mob:w-full [&>button]:mob:flex-1">
                <button
                  disabled={!proxyReady}
                  onClick={() => setQ("fast")}
                  className={`px-2 py-0.5 ${quality === "fast" ? "bg-amber-500 text-zinc-900 font-semibold" : "bg-zinc-800 text-zinc-400"} disabled:opacity-40`}
                >
                  Fast
                </button>
                <button
                  onClick={() => setQ("hd")}
                  className={`px-2 py-0.5 ${quality === "hd" || !proxyReady ? "bg-amber-500 text-zinc-900 font-semibold" : "bg-zinc-800 text-zinc-400"}`}
                >
                  HD
                </button>
              </div>
              {win && (
                <span className="flex items-center gap-1 text-[11px] text-zinc-400">
                  {winEdit ? (
                    <>
                      <input
                        className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 font-mono text-[11px]"
                        value={winIn}
                        onChange={(e) => setWinIn(e.target.value)}
                        placeholder="0:00"
                      />
                      –
                      <input
                        className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 font-mono text-[11px]"
                        value={winOut}
                        onChange={(e) => setWinOut(e.target.value)}
                        placeholder="0:00"
                      />
                      <button
                        className="text-zinc-400 hover:text-amber-300 border border-zinc-700 rounded px-1.5 py-0.5"
                        title="Set start from playhead"
                        onClick={() => setWinIn(fmtClock(playhead))}
                      >
                        ▶in
                      </button>
                      <button
                        className="text-zinc-400 hover:text-amber-300 border border-zinc-700 rounded px-1.5 py-0.5"
                        title="Set end from playhead"
                        onClick={() => setWinOut(fmtClock(playhead))}
                      >
                        ▶out
                      </button>
                      <button
                        className="text-amber-300 hover:text-amber-200 border border-zinc-700 rounded px-2 py-0.5"
                        onClick={() => void saveWindow()}
                      >
                        Save
                      </button>
                    </>
                  ) : (
                    <>
                      window {fmtClock(win[0])}–{fmtClock(win[1])}
                      <button
                        className="text-amber-300 hover:text-amber-200 underline"
                        onClick={() => {
                          setWinIn(fmtClock(win[0]));
                          setWinOut(fmtClock(win[1]));
                          setWinEdit(true);
                        }}
                      >
                        Edit
                      </button>
                    </>
                  )}
                </span>
              )}
              {isMultiangle && (
                <a
                  href={api.videoUrl("source", String(video.registered_at))}
                  download
                  className="text-[11px] text-amber-300 hover:text-amber-200 border border-zinc-700 rounded px-2 py-0.5 mob:text-center"
                >
                  Download director cut<span className="mob:hidden"> (full match, MP4)</span>
                </a>
              )}
              {isMultiangle && cutLabel && (
                <span className="rounded px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-400 border border-zinc-700">
                  {cutLabel}
                </span>
              )}
              {win && (
                <button
                  onClick={() => void downloadTrimmed()}
                  disabled={trimProg !== null}
                  className="text-[11px] text-amber-300 hover:text-amber-200 border border-zinc-700 rounded px-2 py-0.5 disabled:opacity-40"
                >
                  {trimProg !== null
                    ? `Trimming… ${Math.round(trimProg * 100)}%`
                    : <>Download trimmed match<span className="mob:hidden"> (MP4)</span></>}
                </button>
              )}
            </div>
          ) : null}
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
