"""
地面高度采样 — 双线性插值 + 最近邻回退。
"""

import numpy as np
from numba import njit


def sample_terrain(mesh, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """在 (xp, yp) 处采样地面高度。"""
    pts = np.asarray(mesh.points, dtype=np.float32)
    if pts.shape[1] < 3:
        return np.zeros_like(xp, dtype=np.float32)
    try:
        dims = mesh.dimensions
        nx, ny = dims[0], dims[1]
        xs = pts[:nx, 0]
        ys = pts[::nx, 1][:ny]
        z2d = np.ascontiguousarray(
            pts[:, 2].reshape((ny, nx), order="F").astype(np.float32)
        )
        return _bilinear_interp_terrain(
            xs, ys, z2d,
            np.asarray(xp, dtype=np.float32),
            np.asarray(yp, dtype=np.float32),
        )
    except Exception:
        return _nearest_terrain(
            pts,
            np.asarray(xp, dtype=np.float32),
            np.asarray(yp, dtype=np.float32),
        )


@njit
def _bilinear_interp_terrain(xs, ys, z2d, xp, yp):
    nx = z2d.shape[1]; ny = z2d.shape[0]
    r = np.empty(len(xp), dtype=np.float32)
    for i in range(len(xp)):
        x, y = xp[i], yp[i]
        ix = max(0, min(np.searchsorted(xs, x) - 1, nx - 2))
        iy = max(0, min(np.searchsorted(ys, y) - 1, ny - 2))
        x1, x2 = xs[ix], xs[ix + 1]
        y1, y2 = ys[iy], ys[iy + 1]
        dx = (x - x1) / (x2 - x1) if x2 != x1 else 0.5
        dy = (y - y1) / (y2 - y1) if y2 != y1 else 0.5
        r[i] = (
            z2d[iy, ix] * (1.0 - dx) * (1.0 - dy) +
            z2d[iy, ix + 1] * dx * (1.0 - dy) +
            z2d[iy + 1, ix] * (1.0 - dx) * dy +
            z2d[iy + 1, ix + 1] * dx * dy
        )
    return r


@njit
def _nearest_terrain(pts, xp, yp):
    r = np.empty(len(xp), dtype=np.float32)
    for i in range(len(xp)):
        x, y = xp[i], yp[i]
        best = 1e30; best_z = 0.0
        for j in range(len(pts)):
            d = (pts[j, 0] - x) ** 2 + (pts[j, 1] - y) ** 2
            if d < best:
                best = d; best_z = pts[j, 2]
        r[i] = best_z
    return r
