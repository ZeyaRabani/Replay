/** 3x3 homography from 4 point correspondences (DLT, 8x8 Gaussian
 *  elimination). src -> dst, points as [x, y]. */

export type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

function solve8(A: number[][], b: number[]): number[] | null {
  // Gaussian elimination with partial pivoting
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

/** Homography mapping the 4 src points onto the 4 dst points (order
 *  matters and must correspond). Returns null when degenerate. */
export function homography(
  src: [number, number][],
  dst: [number, number][],
): Mat3 | null {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const h = solve8(A, b);
  if (!h) return null;
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

export function applyH(H: Mat3, x: number, y: number): [number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  if (Math.abs(w) < 1e-12) return [NaN, NaN];
  return [(H[0] * x + H[1] * y + H[2]) / w,
          (H[3] * x + H[4] * y + H[5]) / w];
}
