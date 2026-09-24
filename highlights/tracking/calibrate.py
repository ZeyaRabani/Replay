"""Single-camera ground-plane calibration from player boxes.

The pitch has no usable standard markings for a keypoint model, so we
self-calibrate: for a pinhole camera at height Hc above a flat pitch, the
pixel height of a standing player of height Hp whose feet are at image row v is

    h_px = (Hp / Hc) * (v - v_h(u)),      v_h(u) = a + b*u   (rolled horizon)

which is linear in the unknowns. A robust fit over all (stabilised) detections
gives the horizon line and Hc (kept as a diagnostic). On this footage the
foot-row depth Z = f*Hc/(v - v_h) is unreliable at the far end (horizon only
~20 px above the far players, and the local slope k drifts with u), so player
depth is taken from box height instead, Z = f*Hp/h_px, which has a consistent
scale across the pitch; lateral offset X = (u - u0)*Z/f. The focal length f is
assumed (phone main camera, ~950 px at 1280 wide). World frame: origin at the
near-goal centre (foot-row depth, reliable there), x toward the far goal, y
across. Pitch length is estimated from the far-goal pixel via foot-row depth
unless overridden.
"""

import argparse
import json

import numpy as np

PLAYER_HEIGHT_M = 1.75
IMG_W, IMG_H = 1280, 576


def to_ref(pts, H):
    """Apply 3x3 homography to Nx2 pixel points."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def load_tracks(path):
    d = np.load(path)
    det, frames = d["det"], d["frames"]
    Hs = frames[:, 4:13].reshape(-1, 3, 3).astype(np.float64)
    fidx = det[:, 0].astype(int)
    feet = np.c_[(det[:, 3] + det[:, 5]) / 2, det[:, 6]]
    heads = np.c_[(det[:, 3] + det[:, 5]) / 2, det[:, 4]]
    feet_ref = np.empty_like(feet)
    heads_ref = np.empty_like(heads)
    for fi in np.unique(fidx):
        m = fidx == fi
        feet_ref[m] = to_ref(feet[m], Hs[fi])
        heads_ref[m] = to_ref(heads[m], Hs[fi])
    return det, frames, Hs, feet_ref, heads_ref


def fit_horizon(feet_ref, heads_ref, det, iters=30):
    """Robust linear fit h = c0 + c1*v + c2*u -> (a, b, k)."""
    x1, y1, x2, y2 = det[:, 3], det[:, 4], det[:, 5], det[:, 6]
    h = feet_ref[:, 1] - heads_ref[:, 1]
    w = x2 - x1
    ok = (x1 > 4) & (y1 > 4) & (x2 < IMG_W - 4) & (y2 < IMG_H - 4)
    ok &= (h > 12) & (h / np.maximum(w, 1) > 1.6) & (h / np.maximum(w, 1) < 4.5)
    ok &= det[:, 7] > 0.5
    u, v = feet_ref[ok, 0], feet_ref[ok, 1]
    hh = h[ok]
    A = np.c_[np.ones_like(v), v, u]
    wts = np.ones_like(hh)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A * wts[:, None], hh * wts, rcond=None)
        r = hh - A @ c
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        wts = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * s))
    inl = np.abs(r) < 3 * s
    c0, c1, c2 = c
    k = c1
    a = -c0 / c1
    b = -c2 / c1
    return dict(a=float(a), b=float(b), k=float(k), n_used=int(ok.sum()),
                n_inliers=int(inl.sum()), resid_px=float(s))


class GroundModel:
    def __init__(self, a, b, k, f, u0=IMG_W / 2, player_h=PLAYER_HEIGHT_M):
        self.a, self.b, self.k, self.f, self.u0 = a, b, k, f, u0
        self.cam_h = player_h / k
        self.origin = np.zeros(2)
        self.R = np.eye(2)

    def horizon_v(self, u):
        return self.a + self.b * u

    def cam_xz_ground(self, pts):
        """Ref-frame ground pixels -> camera-centred ground (X right, Z depth)
        via the horizon (foot-row) model. Reliable only for the near half."""
        pts = np.atleast_2d(pts)
        u, v = pts[:, 0], pts[:, 1]
        dv = v - self.horizon_v(u)
        dv = np.where(dv < 1.0, np.nan, dv)
        Z = self.f * self.cam_h / dv
        X = (u - self.u0) * Z / self.f
        return np.c_[X, Z]

    def cam_xz(self, feet, heads, player_h=PLAYER_HEIGHT_M):
        """Ref-frame foot + head pixels -> camera-centred ground via box height."""
        feet = np.atleast_2d(feet)
        heads = np.atleast_2d(heads)
        h = feet[:, 1] - heads[:, 1]
        h = np.where(h < 6, np.nan, h)
        Z = self.f * player_h / h
        X = (feet[:, 0] - self.u0) * Z / self.f
        return np.c_[X, Z]

    def set_world(self, near_goal_px, far_goal_px, length=None):
        g0 = self.cam_xz_ground(near_goal_px)[0]
        g1 = self.cam_xz_ground(far_goal_px)[0]
        d = g1 - g0
        ax = d / np.linalg.norm(d)
        # world x along goal->goal, y to the left of x
        self.R = np.array([[ax[0], ax[1]], [-ax[1], ax[0]]])
        self.origin = g0
        self.length = float(length or np.linalg.norm(d))

    def world(self, feet, heads):
        return (self.cam_xz(feet, heads) - self.origin) @ self.R.T

    def to_json(self):
        return dict(a=self.a, b=self.b, k=self.k, f=self.f, u0=self.u0,
                    cam_h=self.cam_h, origin=self.origin.tolist(),
                    R=self.R.tolist(), length=getattr(self, "length", None))

    @classmethod
    def from_json(cls, d):
        g = cls(d["a"], d["b"], d["k"], d["f"], d["u0"])
        g.origin = np.array(d["origin"])
        g.R = np.array(d["R"])
        g.length = d.get("length")
        return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focal", type=float, default=950.0)
    ap.add_argument("--near-goal", type=float, nargs=2, required=True,
                    help="ref-frame pixel of the near goal centre on the ground")
    ap.add_argument("--far-goal", type=float, nargs=2, required=True,
                    help="ref-frame pixel of the far goal centre on the ground")
    ap.add_argument("--length", type=float, default=None,
                    help="override pitch length (m) instead of far-goal depth")
    args = ap.parse_args()

    det, _frames, _Hs, feet_ref, heads_ref = load_tracks(args.tracks)
    fit = fit_horizon(feet_ref, heads_ref, det)
    g = GroundModel(fit["a"], fit["b"], fit["k"], args.focal)
    g.set_world(np.array(args.near_goal), np.array(args.far_goal), args.length)
    out = dict(fit=fit, model=g.to_json(),
               near_goal_px=args.near_goal, far_goal_px=args.far_goal)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
