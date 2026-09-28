/** Pitch geometry in landmark coords: x along the length (0..len_m),
 *  y from the near touchline (0) to the far one (wid_m). */
import type { PitchDims } from "../types";

export type PitchShape =
  | { k: "poly"; pts: [number, number][]; closed?: boolean; faint?: boolean }
  | { k: "dot"; x: number; y: number };

export const DEFAULT_PITCH: PitchDims = { len_m: 100, wid_m: 64 };

function circle(cx: number, cy: number, r: number, a0 = 0, a1 = Math.PI * 2, n = 48): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n;
    out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return out;
}

const rect = (x: number, y: number, w: number, h: number): Extract<PitchShape, { k: "poly" }> => ({
  k: "poly", closed: true, pts: [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
});

/** Standard markings, shrunk proportionally on small (grassroots) pitches. */
export function pitchShapes({ len_m: L, wid_m: W }: PitchDims): PitchShape[] {
  const boxW = Math.min(40.32, W * 0.63), boxD = Math.min(16.5, L * 0.165);
  const sixW = Math.min(18.32, W * 0.29), sixD = Math.min(5.5, L * 0.055);
  const pen = Math.min(11, L * 0.11), R = Math.min(9.15, W * 0.143);
  const goalW = Math.min(7.32, W * 0.12), cy = W / 2;
  const th = Math.acos(Math.min(1, (boxD - pen) / R));
  return [
    rect(0, 0, L, W),
    { k: "poly", pts: [[L / 2, 0], [L / 2, W]] },
    { k: "poly", pts: circle(L / 2, cy, R) },
    { k: "dot", x: L / 2, y: cy },
    rect(0, cy - boxW / 2, boxD, boxW),
    rect(L - boxD, cy - boxW / 2, boxD, boxW),
    rect(0, cy - sixW / 2, sixD, sixW),
    rect(L - sixD, cy - sixW / 2, sixD, sixW),
    { k: "dot", x: pen, y: cy },
    { k: "dot", x: L - pen, y: cy },
    { k: "poly", pts: circle(pen, cy, R, -th, th, 16) },
    { k: "poly", pts: circle(L - pen, cy, R, Math.PI - th, Math.PI + th, 16) },
    { k: "poly", pts: circle(0, 0, 1, 0, Math.PI / 2, 6) },
    { k: "poly", pts: circle(L, 0, 1, Math.PI / 2, Math.PI, 6) },
    { k: "poly", pts: circle(L, W, 1, Math.PI, Math.PI * 1.5, 6) },
    { k: "poly", pts: circle(0, W, 1, Math.PI * 1.5, Math.PI * 2, 6) },
    { ...rect(-2, cy - goalW / 2, 2, goalW), faint: true },
    { ...rect(L, cy - goalW / 2, 2, goalW), faint: true },
  ];
}

export interface PitchView {
  X: (x: number) => number;
  Y: (y: number) => number;
  s: number; // px per metre
}

/** Fit the pitch into w x h with `pad` px margin; near touchline at the bottom. */
export function pitchView(w: number, h: number, pad: number, p: PitchDims): PitchView {
  const s = Math.min((w - 2 * pad) / p.len_m, (h - 2 * pad) / p.wid_m);
  const ox = (w - s * p.len_m) / 2, oy = (h - s * p.wid_m) / 2;
  return { X: (x) => ox + x * s, Y: (y) => oy + (p.wid_m - y) * s, s };
}

export function drawPitch(ctx: CanvasRenderingContext2D, w: number, h: number, v: PitchView, p: PitchDims) {
  ctx.fillStyle = "#0f3a22";
  ctx.fillRect(0, 0, w, h);
  const bands = 12, bw = p.len_m / bands;
  for (let i = 0; i < bands; i++) {
    ctx.fillStyle = i % 2 ? "#1a5f36" : "#1d6a3c";
    ctx.fillRect(v.X(i * bw), v.Y(p.wid_m), bw * v.s + 0.5, p.wid_m * v.s);
  }
  ctx.lineWidth = Math.max(1, v.s * 0.15);
  ctx.lineJoin = "round";
  for (const sh of pitchShapes(p)) {
    if (sh.k === "dot") {
      ctx.beginPath();
      ctx.arc(v.X(sh.x), v.Y(sh.y), Math.max(1.5, v.s * 0.3), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fill();
      continue;
    }
    ctx.beginPath();
    sh.pts.forEach(([x, y], i) => (i ? ctx.lineTo(v.X(x), v.Y(y)) : ctx.moveTo(v.X(x), v.Y(y))));
    if (sh.closed) ctx.closePath();
    ctx.strokeStyle = sh.faint ? "rgba(255,255,255,0.45)" : "rgba(255,255,255,0.8)";
    ctx.stroke();
  }
}
