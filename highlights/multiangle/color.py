"""HDR source detection and the SDR tonemap/tag filter chains used by
every user-visible encode that reads a raw camera source.

21/8 a2 is VP9 yuv420p10le smpte2084/bt2020 (PQ HDR). Scaling to
yuv420p alone keeps PQ code values and libx264 inherits the PQ tags,
so concat-copied outputs end up tagged PQ and players show washed-out
grey. TONEMAP_VF maps linear->bt709 with mobius at npl=203 (measured
to match YouTube's SDR proxy of the same frame), SDR_TAG_VF forces
bt709 stream-side metadata on every chain, and SDR_OUT_ARGS tag the
container/codec so the first segment can't leak the source's tags
through `concat -c copy`.
"""
import functools
import subprocess

TONEMAP_VF = ("zscale=t=linear:npl=203,format=gbrpf32le,"
              "zscale=p=bt709,tonemap=tonemap=mobius:desat=0,"
              "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
SDR_TAG_VF = ("setparams=color_primaries=bt709:color_trc=bt709:"
              "colorspace=bt709:range=tv")
SDR_OUT_ARGS = ["-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-color_range", "tv"]


@functools.lru_cache(maxsize=64)
def is_hdr(video: str) -> bool:
    """True when v:0 is PQ or HLG (HDR transfer that needs tonemapping).
    Cached per path."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=False)
    except OSError:
        return False
    if proc.returncode != 0:
        return False
    return proc.stdout.strip() in ("smpte2084", "arib-std-b67")
