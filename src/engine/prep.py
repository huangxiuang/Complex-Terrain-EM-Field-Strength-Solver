"""
Preprocessing 模块 — 网格建立、材料映射、DMFT、初始场。
"""

import numpy as np
from scipy.spatial import cKDTree
from src.core.context import Context
from src.physics.terrain import sample_terrain
from src.physics.materials import resolve_material, get_material, is_conductor
from src.physics.dmft import build_dmft
from src.antenna_types import make_initial_field
from numba import njit


def run(ctx: Context) -> Context:
    cfg = ctx.config
    scene = ctx.scene

    # ── 1. 高度网格 ──
    z_rx_max = max(p[2] for p in ctx.rx_points) if ctx.rx_points else cfg.antenna_pos[2]
    z_max = max(z_rx_max, cfg.antenna_pos[2]) + cfg.z_pad_above
    dz = z_max / (cfg.n_z - 1)
    z_vals = np.linspace(0, z_max, cfg.n_z)
    ctx.z_vals = z_vals
    ctx.dz = dz

    # ── 2. 径向网格 — 为所有 rx 中最大距离建网格 ──
    rt_all = [_cart_to_cyl(rx, cfg.antenna_pos)[0] for rx in ctx.rx_points]
    rt = max(rt_all) if rt_all else 3.0
    r0 = cfg.r0_factor * cfg.wavelength
    dr = cfg.dr_factor * cfg.wavelength
    nr = max(int((rt - r0) / dr) + 1, 2)
    r_vals = np.linspace(r0, rt, nr)
    ctx.r_vals = r_vals
    ctx.dr = dr

    # 高度网格精度检查：dz 过大 → 可表示的垂直波数谱受限，大角度绕射丢失
    lam = cfg.wavelength
    max_angle = np.degrees(np.arcsin(min(1.0, np.pi / (dz * cfg.k0 * cfg.n_atm))))
    if dz > lam / 2:
        ctx.warnings.append(
            f"高度网格过粗：dz={dz:.3f} m（λ/2={lam/2:.3f} m），"
            f"传播角度谱受限至 ±{max_angle:.1f}°，大角度绕射/垂直干涉精度下降。"
            f"建议增大 N_Z 或减小高度留白。"
        )

    # ── 3. 材料映射 ──
    phi_vals = np.linspace(0, 2 * np.pi, cfg.n_phi, endpoint=False)
    ctx.phi_vals = phi_vals

    zg, zt, ng, cg, nl, cl = _extract_material_maps(
        cfg, scene, r_vals, phi_vals, nr
    )
    ctx.z_grd = zg
    ctx.z_top = zt
    ctx.n_grd = ng
    ctx.c_grd = cg
    ctx.n_lay = nl
    ctx.c_lay = cl

    # ── 4. DMFT ──
    ctx.dmft = build_dmft(cfg.frequency, cfg.k0, cfg.n_z, z_max, scene)

    # ── 5. 初始场 ──
    ant_cfg = cfg.antenna_config()
    ctx.u_init = make_initial_field(
        ant_cfg, z_vals, cfg.n_phi, cfg.antenna_pos[2], cfg.k0, r0
    )

    if ctx.progress_cb is not None:
        ctx.progress_cb("预处理", 1, 1)
    return ctx


def _cart_to_cyl(rx, antenna):
    dx, dy = rx[0] - antenna[0], rx[1] - antenna[1]
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    if phi < 0:
        phi += 2 * np.pi
    return r, phi


def _extract_material_maps(cfg, scene, r_vals, phi_vals, nr):
    n_phi = cfg.n_phi
    n_atm = cfg.n_atm
    freq = cfg.frequency
    antenna = np.array(cfg.antenna_pos, dtype=np.float32)

    zg = np.zeros((n_phi, nr), dtype=np.float32)
    zt = np.zeros((n_phi, nr), dtype=np.float32)
    ng = np.full((n_phi, nr), n_atm, dtype=np.complex64)
    cg = np.zeros((n_phi, nr), dtype=bool)
    nl = np.full((n_phi, nr), n_atm, dtype=np.complex64)
    cl = np.zeros((n_phi, nr), dtype=bool)

    # 预计算不规则材质层的 XY 精确覆盖（三角单元投影 + KDTree 候选检索）
    layer_covers = {}
    for nm_key, o in scene.items():
        extra = o.get("extra") or {}
        if not extra.get("is_material_layer"):
            continue
        cover = _build_layer_cover(o.get("mesh"))
        if cover is not None:
            layer_covers[nm_key] = cover

    for pi, phi in enumerate(phi_vals):
        c, s = np.cos(phi), np.sin(phi)
        xp = antenna[0] + r_vals * c
        yp = antenna[1] + r_vals * s

        # 地面 — use original Z array for accuracy
        t = scene.get("terrain")
        if t is not None:
            gz = _fast_sample_terrain(t, xp, yp)
            zg[pi, :] = gz; zt[pi, :] = gz
            m = get_material(t)
            if m is not None:
                nc, cond = resolve_material(freq, m)
                ng[pi, :] = nc; nl[pi, :] = nc
                cg[pi, :] = cond; cl[pi, :] = cond

        # ── 墙障碍物：按 phi + 最小 3 步径向段 ──
        for nm_key, o in scene.items():
            extra = o.get("extra") or {}
            if extra.get("obstacle_type") != "wall":
                continue
            msh = o.get("mesh")
            if msh is None: continue
            wb = msh.bounds
            wm = get_material(o)
            if wm is not None:
                wnc, wcond = resolve_material(freq, wm)
            else:
                wnc, wcond = np.complex64(n_atm), True
            inside = ((xp >= wb[0]) & (xp <= wb[1]) &
                      (yp >= wb[2]) & (yp <= wb[3]))
            if not inside.any():
                continue
            idx_list = np.where(inside)[0]
            if len(idx_list) < 3:
                mid = idx_list[len(idx_list) // 2]
                idx_list = np.arange(max(0, mid - 1), min(nr, mid + 2))
            zg[pi, idx_list] = np.maximum(zg[pi, idx_list], wb[5])
            zt[pi, idx_list] = np.maximum(zt[pi, idx_list], wb[5])
            ng[pi, idx_list] = wnc; nl[pi, idx_list] = wnc
            cg[pi, idx_list] = wcond; cl[pi, idx_list] = wcond

        # 其他障碍物
        for nm_key, o in scene.items():
            if nm_key in ("terrain", "antenna") or "_clip" in nm_key:
                continue
            if nm_key.startswith("layer_"):
                continue
            extra = o.get("extra") or {}
            if extra.get("obstacle_type") == "wall":
                continue
            if extra.get("is_material_layer"):
                continue
            if o.get("type") != "mesh": continue
            # 只处理有材料但非材质层的对象
            extra_o = o.get("extra") or {}
            if extra_o.get("is_material_layer"):
                continue
            if extra_o.get("obstacle_type") not in ("wall", "building", "obstacle"):
                continue
            msh = o.get("mesh")
            if msh is None: continue
            b = msh.bounds
            inside = ((xp >= b[0]) & (xp <= b[1]) &
                      (yp >= b[2]) & (yp <= b[3]))
            if inside.any():
                zg[pi, inside] = np.maximum(zg[pi, inside], b[5])
                zt[pi, inside] = np.maximum(zt[pi, inside], b[5])
                m = get_material(o)
                if m is not None:
                    nc, cond = resolve_material(freq, m)
                    ng[pi, inside] = nc; cg[pi, inside] = cond
                    nl[pi, inside] = nc; cl[pi, inside] = cond

        # 材质层（沙地/草地/森林冠层）— 作为裁剪层叠加在地形上
        for nm_key, o in scene.items():
            extra = o.get("extra") or {}
            if not extra.get("is_material_layer"):
                continue
            msh = o.get("mesh")
            if msh is None: continue
            m = extra.get("material")
            if m is None: continue
            thick = m.get("thickness_cm", 0.0) / 100.0
            if thick <= 0: continue
            # 精确 XY 覆盖（网格单元投影），替代会严重夸大范围的矩形 bounds
            cover = layer_covers.get(nm_key)
            if cover is not None:
                inside = _points_covered(cover, xp, yp)
            else:
                b = msh.bounds
                inside = ((xp >= b[0]) & (xp <= b[1]) &
                          (yp >= b[2]) & (yp <= b[3]))
            if not inside.any(): continue
            zt[pi, inside] = np.maximum(zt[pi, inside], zg[pi, inside] + thick)
            nc, cond = resolve_material(freq, m)
            nl[pi, inside] = nc; cl[pi, inside] = cond

        # 裁剪图层
        clips = [(k, o) for k, o in scene.items() if "_clip" in k]
        clips.sort(key=lambda x: _clip_counter(x[0]))
        for nm_key, o in clips:
            msh = o.get("mesh")
            if msh is None: continue
            extra = o.get("extra", {})
            polys = extra.get("polygons", [])
            m = extra.get("material")
            thick = m.get("thickness_cm", 0.0) / 100.0 if m else 0.0
            if polys:
                for ri in range(len(xp)):
                    if any(_point_in_poly(xp[ri], yp[ri], p) for p in polys):
                        zt[pi, ri] = zg[pi, ri] + thick
                        if m is not None:
                            nc, cond = resolve_material(freq, m)
                            nl[pi, ri] = nc; cl[pi, ri] = cond
            else:
                b = msh.bounds
                inside = ((xp >= b[0]) & (xp <= b[1]) &
                          (yp >= b[2]) & (yp <= b[3]))
                if inside.any():
                    zt[pi, inside] = zg[pi, inside] + thick
                    if m is not None:
                        nc, cond = resolve_material(freq, m)
                        nl[pi, inside] = nc; cl[pi, inside] = cond

    return zg, zt, ng, cg, nl, cl


def _fast_sample_terrain(t, xp, yp):
    """直接读 original_z 数组做双线性插值，越界点钳位到边缘。"""
    extra = t.get("extra") or {}
    Z = extra.get("original_z")
    X = extra.get("X")
    Y = extra.get("Y")
    if Z is None or X is None:
        return sample_terrain(t["mesh"], xp, yp)
    xs = X[0, :]; ys = Y[:, 0]
    nx, ny = len(xs), len(ys)
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    r = np.empty(len(xp), dtype=np.float32)
    for i in range(len(xp)):
        x = max(x_min, min(xp[i], x_max))
        y = max(y_min, min(yp[i], y_max))
        ix = max(0, min(np.searchsorted(xs, x) - 1, nx - 2))
        iy = max(0, min(np.searchsorted(ys, y) - 1, ny - 2))
        x1, x2 = xs[ix], xs[ix + 1]; y1, y2 = ys[iy], ys[iy + 1]
        dx = (x - x1) / (x2 - x1) if x2 != x1 else 0.5
        dy = (y - y1) / (y2 - y1) if y2 != y1 else 0.5
        dx = max(0.0, min(dx, 1.0)); dy = max(0.0, min(dy, 1.0))
        r[i] = (Z[iy, ix] * (1 - dx) * (1 - dy) + Z[iy, ix + 1] * dx * (1 - dy) +
                Z[iy + 1, ix] * (1 - dx) * dy + Z[iy + 1, ix + 1] * dx * dy)
    return r


def _clip_counter(name):
    parts = name.split("_clip_")
    if len(parts) > 1:
        try: return int(parts[1])
        except ValueError: pass
    return 0


def _build_layer_cover(mesh):
    """把材质层网格在 XY 平面投影为三角单元集合，供精确覆盖判定。

    返回 dict(centroids, radii, polys, tree, r_max)，网格退化时返回 None
    （调用方回退到矩形 bounds 判定）。
    """
    if mesh is None:
        return None
    try:
        surf = mesh.extract_surface().triangulate()
        if surf.n_cells < 1 or surf.n_points < 3:
            return None
        pts = np.asarray(surf.points, dtype=np.float64)[:, :2]
        faces = np.asarray(surf.faces, dtype=np.int64).reshape(-1, 4)
        tris = faces[:, 1:4]
        polys = pts[tris]                                   # (n_cells, 3, 2)
        centroids = polys.mean(axis=1)                      # (n_cells, 2)
        radii = np.max(np.linalg.norm(polys - centroids[:, None, :], axis=2), axis=1)
        r_max = float(radii.max())
        if not np.isfinite(r_max) or r_max <= 0:
            return None
        return {
            "polys": np.ascontiguousarray(polys),
            "centroids": np.ascontiguousarray(centroids),
            "tree": cKDTree(centroids),
            "r_max": r_max,
        }
    except Exception:
        return None


def _points_covered(cover, xp, yp):
    """判定 (xp, yp) 是否落在材质层任一三角单元的 XY 投影内。

    恰好在单元边上的点 ray-casting 判定不稳定（会漏判），
    中心判定失败时以 4 个微扰邻点重试，消除边界退化。
    """
    query = np.column_stack([xp, yp])
    # 候选检索半径 = 最大单元外接半径：点到质心距离 ≤ r_max 才可能被覆盖
    cand_lists = cover["tree"].query_ball_point(query, r=cover["r_max"])
    # 扁平化为 CSR 结构，一次性进入 njit（避免逐点 numba 类型分派开销）
    counts = np.fromiter((len(c) for c in cand_lists), dtype=np.int64, count=len(cand_lists))
    cand_ptr = np.zeros(len(cand_lists) + 1, dtype=np.int64)
    np.cumsum(counts, out=cand_ptr[1:])
    total = int(cand_ptr[-1])
    if total == 0:
        return np.zeros(len(xp), dtype=bool)
    cand_idx = np.fromiter(
        (ci for c in cand_lists for ci in c), dtype=np.int64, count=total)
    eps = max(cover["r_max"] * 1e-3, 1e-6)
    return _covered_batch(
        np.ascontiguousarray(xp, dtype=np.float64),
        np.ascontiguousarray(yp, dtype=np.float64),
        cand_ptr, cand_idx, cover["polys"], eps)


@njit
def _covered_batch(xp, yp, cand_ptr, cand_idx, polys, eps):
    inside = np.zeros(len(xp), dtype=np.bool_)
    for i in range(len(xp)):
        s, e = cand_ptr[i], cand_ptr[i + 1]
        x, y = xp[i], yp[i]
        found = False
        for k in range(s, e):
            if _point_in_poly(x, y, polys[cand_idx[k]]):
                found = True
                break
        if not found:
            for dx, dy in ((eps, 0.0), (-eps, 0.0), (0.0, eps), (0.0, -eps)):
                for k in range(s, e):
                    if _point_in_poly(x + dx, y + dy, polys[cand_idx[k]]):
                        found = True
                        break
                if found:
                    break
        inside[i] = found
    return inside


@njit
def _point_in_poly(x, y, poly):
    n = len(poly); inside = False
    j = n - 1
    for i in range(n):
        if ((poly[i, 1] > y) != (poly[j, 1] > y)) and \
           (x < (poly[j, 0] - poly[i, 0]) * (y - poly[i, 1]) /
            (poly[j, 1] - poly[i, 1] + 1e-30) + poly[i, 0]):
            inside = not inside
        j = i
    return inside
