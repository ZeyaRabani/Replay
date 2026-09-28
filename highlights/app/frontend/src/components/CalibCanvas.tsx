import { useEffect, useRef, useState } from "react";
import type { CalibPoint } from "../types";

// image occupies the middle half of each axis, so clicks may land off-image:
// normalised coords span [-0.5, 1.5]
const IMG_FRAC = 0.5;
const OFF = (1 - IMG_FRAC) / 2;
const HIT_PX = 12;

interface Props {
  src: string;
  pts: CalibPoint[];
  /** landmark name -> 1-based number and short label */
  num: (name: string) => number;
  label: (name: string) => string;
  sel: string | null;
  /** `click` = placed by a click (vs. dragged) */
  onPlace: (name: string, fx: number, fy: number, click: boolean) => void;
  onSelect: (name: string) => void;
}

/** Padded still with numbered, draggable landmark markers. */
export default function CalibCanvas({ src, pts, num, label, sel, onPlace, onSelect }: Props) {
  const cvRef = useRef<HTMLCanvasElement>(null);
  const img = useRef<HTMLImageElement | null>(null);
  const drag = useRef<string | null>(null);
  const [ready, setReady] = useState(0);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
    const im = new Image();
    im.onload = () => { img.current = im; setReady((n) => n + 1); };
    im.onerror = () => setFailed(true);
    im.src = src;
  }, [src]);

  useEffect(() => {
    const cv = cvRef.current;
    const im = img.current;
    if (!cv) return;
    const W = cv.clientWidth || 800;
    const aspect = im ? im.naturalHeight / im.naturalWidth : 9 / 16;
    const iw = W * IMG_FRAC, ih = iw * aspect;
    const ox = W * OFF, oy = ih / 2;
    cv.width = W;
    cv.height = Math.round(2 * ih);
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#18181b";
    ctx.fillRect(0, 0, W, cv.height);
    if (im) ctx.drawImage(im, ox, oy, iw, ih);
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, iw, ih);
    const SX = (fx: number) => ox + fx * iw, SY = (fy: number) => oy + fy * ih;
    ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
    for (const p of pts) {
      const x = SX(p.fx), y = SY(p.fy), on = p.name === sel;
      const txt = label(p.name);
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(9,9,11,0.75)";
      ctx.fillRect(x + 11, y - 8, tw + 8, 16);
      ctx.fillStyle = on ? "#fde68a" : "#e4e4e7";
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(txt, x + 15, y);
      ctx.beginPath();
      ctx.arc(x, y, on ? 9 : 8, 0, Math.PI * 2);
      ctx.fillStyle = on ? "#fde68a" : "#fbbf24";
      ctx.fill();
      ctx.lineWidth = on ? 2.5 : 1.5;
      ctx.strokeStyle = on ? "#fafafa" : "#18181b";
      ctx.stroke();
      ctx.fillStyle = "#18181b";
      ctx.textAlign = "center";
      ctx.font = "bold 9px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(String(num(p.name)), x, y + 0.5);
      ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
    }
  }, [pts, sel, ready, num, label]);

  const pos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    return {
      px, py, r,
      fx: (px / r.width - OFF) / IMG_FRAC,
      fy: (py / r.height - OFF) / IMG_FRAC,
    };
  };
  const hit = (px: number, py: number, r: DOMRect) =>
    pts.find((p) => Math.hypot((p.fx * IMG_FRAC + OFF) * r.width - px,
                               (p.fy * IMG_FRAC + OFF) * r.height - py) < HIT_PX);

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { px, py, r } = pos(e);
    const h = hit(px, py, r);
    if (!h) return;
    drag.current = h.name;
    onSelect(h.name);
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag.current) return;
    const { fx, fy } = pos(e);
    onPlace(drag.current, fx, fy, false);
  };
  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag.current) { drag.current = null; return; }
    if (!sel) return;
    const { fx, fy } = pos(e);
    onPlace(sel, fx, fy, true);
  };

  return (
    <div className="relative min-w-0 flex-1">
      <canvas
        ref={cvRef}
        className={`w-full rounded touch-none ${sel ? "cursor-crosshair" : "cursor-default"}`}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        aria-label="camera still — click to place the selected landmark"
      />
      {failed && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-zinc-500">
          Still unavailable for this angle/time.
        </div>
      )}
    </div>
  );
}
