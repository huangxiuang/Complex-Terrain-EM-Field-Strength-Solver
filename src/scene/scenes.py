"""
场景注册表 — 5 个预置场景，从简单到复杂。
"""

import numpy as np
import pyvista as pv
from src.antenna_types import DEFAULT_ANTENNA_CONFIG
from src.physics.terrain import sample_terrain


# ═══════════════════════════════════════════════════════════════
#  场景注册表
# ═══════════════════════════════════════════════════════════════

SCENE_REGISTRY = {}  # {key: {"name": str, "description": str, "builder": callable}}


def register(key, name, description):
    def deco(fn):
        SCENE_REGISTRY[key] = {"name": name, "description": description, "builder": fn}
        return fn
    return deco


# ═══════════════════════════════════════════════════════════════
#  公共工具
# ═══════════════════════════════════════════════════════════════

ANTENNA_POS = (-5.0, 0.0, 6.0)
ANTENNA_RADIUS = 0.3


def _make_antenna(terrain_z=0.0):
    """天线标记：球体 + 杆子从地面到天线高度。"""
    h = ANTENNA_POS[2]
    pole_base = max(terrain_z, 0.0)
    if h - pole_base < 0.3:
        pole_base = h - 0.3  # 至少 30cm 杆子
    
    sphere = pv.Sphere(radius=ANTENNA_RADIUS, center=ANTENNA_POS)
    pole = pv.Cylinder(
        center=(ANTENNA_POS[0], ANTENNA_POS[1], (pole_base + h) / 2),
        direction=(0, 0, 1), radius=ANTENNA_RADIUS * 0.3, height=h - pole_base,
    )
    return {
        "mesh": sphere.merge([pole]), "type": "mesh", "visible": True,
        "params": {"color": "#e63946", "smooth_shading": True, "opacity": 1.0,
                   "ambient": 0.5, "diffuse": 0.9, "specular": 0.5, "specular_power": 50},
        "extra": {"position": ANTENNA_POS, "is_source": True,
                  "antenna_config": dict(DEFAULT_ANTENNA_CONFIG)},
        "name": "antenna",
    }


def _make_wall(x=0.0, y_half=15.0, z_top=5.5, thickness=1.0, label="金属铝",
               eps_r=1.0, sigma=3.8e7, color="#888888"):
    wall = pv.Box(bounds=(
        x - thickness / 2, x + thickness / 2, -y_half, y_half, 0.0, z_top,
    ))
    return {
        "mesh": wall, "type": "mesh", "visible": True,
        "params": {"color": color, "smooth_shading": True, "opacity": 0.85,
                   "ambient": 0.3, "diffuse": 0.8, "specular": 0.3, "specular_power": 30},
        "extra": {"material": {"label": label, "eps_r": eps_r, "sigma": sigma},
                  "obstacle_type": "wall"},
        "name": "wall",
    }


def _make_terrain(X, Y, Z, is_dem=False):
    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = Z.flatten(order="F")
    return {
        "mesh": grid, "type": "mesh", "visible": True,
        "params": {"color": "#c8b88a", "smooth_shading": True, "opacity": 1.0},
        "extra": {"original_z": Z.copy(), "X": X, "Y": Y, "is_dem": is_dem,
                  "material": {"label": "干燥土壤", "eps_r": 15.0, "sigma": 0.01}},
        "name": "terrain",
    }


# ═══════════════════════════════════════════════════════════════
#  场景 1：金属挡板（最简单）
# ═══════════════════════════════════════════════════════════════

@register("metal_barrier", "金属挡板", "平坦地面 + 单面铝制挡板，经典 knife‑edge 绕射场景")
def build_metal_barrier():
    span, res = 10.0, 50
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    Z += 0.05 * (X + span) / (2 * span)

    actors = {}
    actors["terrain"] = _make_terrain(X, Y, Z)
    actors["wall"] = _make_wall()
    tz = float(Z[np.argmin(np.abs(xs - ANTENNA_POS[0])), np.argmin(np.abs(ys - ANTENNA_POS[1]))])
    actors["antenna"] = _make_antenna(tz)
    return actors


# ═══════════════════════════════════════════════════════════════
#  场景 2：丘陵地带

# ═══════════════════════════════════════════════════════════════
#  场景 2：自然景观
@register("classic", "自然景观", "高斯山丘地形 + 正弦河道 + 沙/草/土分层 + 河岸植被 + 鸟 + 树")
def build_classic():
    res_x, res_y = 60, 60
    xs = np.linspace(-10, 10, res_x)
    ys = np.linspace(-10, 10, res_y)
    X, Y = np.meshgrid(xs, ys)

    # 3 座高斯山丘
    Z = (
        5.0 * np.exp(-((X - 4) ** 2 + (Y - 3) ** 2) / 12)
        + 3.5 * np.exp(-((X + 3) ** 2 + (Y - 4) ** 2) / 9)
        + 2.0 * np.exp(-((X - 1) ** 2 + (Y + 2) ** 2) / 15)
    )

    # 正弦河道
    river_width = 1.2
    river_depth = 0.7
    for i in range(res_x):
        for j in range(res_y):
            dist = abs(X[i, j] - 3.0 * np.sin(Y[i, j] * 0.4))
            if dist < river_width:
                t = dist / river_width
                Z[i, j] = Z[i, j] * (0.05 + 0.15 * t) - river_depth * (1 - t)

    actors = {}

    # ── 地形 + 3 层土壤 ──
    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = Z.flatten(order="F")
    actors["terrain"] = {
        "mesh": grid, "type": "mesh", "visible": True,
        "params": {"color": "#f2e2a8", "smooth_shading": True, "opacity": 1.0},
        "extra": {"original_z": Z.copy(), "X": X, "Y": Y, "is_dem": False,
                  "material": {"label": "干燥土壤", "eps_r": 15.0, "sigma": 0.01}},
        "name": "terrain",
    }

    # ── 河道水面 ──
    n_y, n_w = 100, 12
    river_y = np.linspace(-10, 10, n_y)
    river_w = np.linspace(-0.8, 0.8, n_w)
    Ry, Rw = np.meshgrid(river_y, river_w)
    Rx_center = 3.0 * np.sin(Ry * 0.4)
    Rx = Rx_center + Rw
    river_grid = pv.StructuredGrid(Rx, Ry, np.full_like(Rx, 0.0))
    actors["river"] = {
        "mesh": river_grid, "type": "mesh", "visible": True,
        "params": {"color": "#1488cc", "opacity": 0.88, "smooth_shading": True},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 100.0},
                  "Ry": Ry, "phase": 0.0},
        "name": "river",
    }

    # ── 河岸植被 ──
    bank_n = 100
    bank_y = np.linspace(-10, 10, bank_n)
    bank_x_left = 3.0 * np.sin(bank_y * 0.4) - 0.88
    bank_x_right = 3.0 * np.sin(bank_y * 0.4) + 0.88
    bank_z = np.full_like(bank_y, -0.15)
    left_pts = pv.PolyData(np.column_stack((bank_x_left, bank_y, bank_z)))
    right_pts = pv.PolyData(np.column_stack((bank_x_right, bank_y, bank_z)))
    actors["vegetation"] = {
        "mesh": left_pts.merge(right_pts), "type": "points", "visible": True,
        "params": {"color": "#2d882d", "point_size": 10, "opacity": 0.8},
        "extra": None, "name": "vegetation",
    }

    # ── 鸟 ──
    body = pv.Sphere(radius=0.12, center=(0, 0, 0))
    head = pv.Sphere(radius=0.06, center=(0.15, 0, 0.04))
    lw = pv.Cone(center=(0, -0.15, 0.02), direction=(0, -1, 0.2), height=0.18, radius=0.06)
    rw = pv.Cone(center=(0, 0.15, 0.02), direction=(0, 1, 0.2), height=0.18, radius=0.06)
    tail = pv.Cone(center=(-0.12, 0, 0.02), direction=(-1, 0, 0.1), height=0.08, radius=0.04)
    bird_mesh = body.merge([head, lw, rw, tail]); bird_mesh.translate((5, -1, 7), inplace=True)
    actors["bird"] = {
        "mesh": bird_mesh, "type": "mesh", "visible": True,
        "params": {"color": "#e8c36a", "smooth_shading": True},
        "extra": None, "name": "bird",
    }

    # ── 树（贴合地形） ──
    tree_x, tree_y = 4.5, 2.0
    tz = float(5.0 * np.exp(-((tree_x-4)**2+(tree_y-3)**2)/12) +
               3.5 * np.exp(-((tree_x+3)**2+(tree_y-4)**2)/9) +
               2.0 * np.exp(-((tree_x-1)**2+(tree_y+2)**2)/15))
    trunk = pv.Cylinder(center=(0, 0, -0.25), direction=(0, 0, 1), radius=0.06, height=0.5)
    leaves = pv.Cone(center=(0, 0, 0.15), direction=(0, 0, 1), height=0.5, radius=0.2)
    trunk.points[:, :2] *= 0.5
    tree_mesh = trunk.merge(leaves); tree_mesh.translate((tree_x, tree_y, tz + 0.5), inplace=True)
    actors["tree"] = {
        "mesh": tree_mesh, "type": "mesh", "visible": True,
        "params": {"color": "#5a8f3c", "smooth_shading": True},
        "extra": None, "name": "tree",
    }

    # ── 天线 ──
    ix = np.argmin(np.abs(xs - ANTENNA_POS[0]))
    iy = np.argmin(np.abs(ys - ANTENNA_POS[1]))
    ant_tz = float(Z[iy, ix])
    actors["antenna"] = _make_antenna(ant_tz)
    return actors


# ═══════════════════════════════════════════════════════════════
#  场景 4：城市街区
# ═══════════════════════════════════════════════════════════════

@register("city_block", "城市街区", "平坦地面 + 4 栋混凝土建筑，模拟城区多径与遮挡")
def build_city_block():
    span, res = 10.0, 50
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    actors = {}
    actors["terrain"] = _make_terrain(X, Y, Z)

    # 4 栋建筑：不同位置、高度、大小
    buildings = [
        {"x": 0, "y": -4, "w": 1.5, "d": 2.0, "h": 8.0, "name": "building_a"},
        {"x": 3, "y": 3, "w": 2.0, "d": 1.5, "h": 12.0, "name": "building_b"},
        {"x": -1, "y": 6, "w": 1.0, "d": 2.5, "h": 6.0, "name": "building_c"},
        {"x": 6, "y": -5, "w": 2.5, "d": 2.0, "h": 15.0, "name": "building_d"},
    ]
    for b in buildings:
        box = pv.Box(bounds=(
            b["x"] - b["w"] / 2, b["x"] + b["w"] / 2,
            b["y"] - b["d"] / 2, b["y"] + b["d"] / 2,
            0.0, b["h"],
        ))
        concrete = {"label": "混凝土", "eps_r": 6.0, "sigma": 0.02}
        actors[b["name"]] = {
            "mesh": box, "type": "mesh", "visible": True,
            "params": {"color": "#b0a090", "smooth_shading": True, "opacity": 0.9,
                       "ambient": 0.3, "diffuse": 0.8},
            "extra": {"material": concrete, "obstacle_type": "wall"},
            "name": b["name"],
        }

    actors["antenna"] = _make_antenna(0.0)
    return actors


# ═══════════════════════════════════════════════════════════════
#  场景 4：荒原
# ═══════════════════════════════════════════════════════════════

@register("wilderness", "荒原", "1000×1000m 真实地形 — 山脊/平原/谷地 + 沙地/草地/森林/湖泊, ITU-R P.833")
def build_wilderness():
    span = 500.0           # 半边长
    res = 140              # 分辨率，7.2m 间距
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)

    # ── 多尺度地形生成 ──
    # 大尺度：山脉 + 平原
    Z_large = (
        80.0 * np.exp(-((X - 200) ** 2 + (Y - 150) ** 2) / 80000) +   # 东北山脊
        50.0 * np.exp(-((X - 300) ** 2 + (Y - 300) ** 2) / 60000) +   # 远东山头
        35.0 * np.exp(-((X + 100) ** 2 + (Y - 250) ** 2) / 50000) +   # 北山
        60.0 * np.exp(-((X - 350) ** 2 + Y ** 2) / 70000)             # 东山
    )
    # 中尺度：丘陵
    Z_mid = (
        20.0 * np.exp(-((X + 50) ** 2 + (Y + 50) ** 2) / 15000) +
        25.0 * np.exp(-((X - 100) ** 2 + (Y + 150) ** 2) / 20000) +
        15.0 * np.exp(-((X - 50) ** 2 + (Y - 100) ** 2) / 18000) +
        18.0 * np.exp(-((X + 200) ** 2 + (Y - 50) ** 2) / 25000)
    )
    # 小尺度：地表纹理（多频正弦叠加模拟侵蚀）
    Z_fine = (
        3.0 * np.sin(X * 0.02) * np.cos(Y * 0.03) +
        2.0 * np.cos(X * 0.05 + 1.5) * np.sin(Y * 0.04) +
        1.5 * np.sin(X * 0.08 - Y * 0.06) +
        1.0 * np.cos(X * 0.12) * np.cos(Y * 0.10)
    )
    # 谷地：沿 X 方向的侵蚀沟
    Z_valley = -12.0 * np.exp(-((Y + 50) ** 2) / 3000) * np.abs(np.sin(X * 0.008 + 1.0))
    # 盆地
    Z_basin = -15.0 * np.exp(-((X - 250) ** 2 + (Y + 200) ** 2) / 25000)

    Z = Z_large + Z_mid + Z_fine + Z_valley + Z_basin + 5.0
    Z = np.maximum(Z, 0.0)  # 不低于 0

    actors = {}

    # ── 地形 ──
    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = Z.flatten(order="F")
    actors["terrain"] = {
        "mesh": grid, "type": "mesh", "visible": True,
        "params": {"color": "#c8b88a", "smooth_shading": True, "opacity": 1.0},
        "extra": {"original_z": Z.copy(), "X": X, "Y": Y, "is_dem": False,
                  "material": {"label": "干燥土壤", "eps_r": 15.0, "sigma": 0.01}},
        "name": "terrain",
    }

    surface = grid.extract_surface()

    # ── 沙地区域（西南平坦区，Z<10m 的低处）──
    # ITU-R P.527: dry sand ε_r≈2.5-3.0, σ≈0.0001-0.001
    sand_mask = (X < 50) & (Y < 50) & (Z < 10)
    if sand_mask.any():
        sand_pts = np.column_stack((X[sand_mask], Y[sand_mask], Z[sand_mask] + 0.1))
        sand_poly = pv.PolyData(sand_pts)
        try:
            sand_surf = sand_poly.delaunay_2d()
            actors["sand_zone"] = {
                "mesh": sand_surf, "type": "mesh", "visible": True,
                "params": {"color": "#e8c76a", "opacity": 0.7, "smooth_shading": True},
                "extra": {"material": {"label": "沙地", "eps_r": 3.0, "sigma": 0.001, "thickness_cm": 30.0}},
                "name": "sand_zone",
            }
        except Exception:
            pass

    # ── 草原区域（中部丘陵，Z=10~40m）──
    # ITU-R P.527: medium dry ground ε_r≈15, σ≈0.01
    grass_poly_xy = np.array([
        [-300, -200], [-100, 150], [200, 300], [350, 100], [250, -250], [-50, -350], [-300, -200]
    ])
    grass_mask = np.zeros(X.shape, dtype=bool)
    for i in range(len(xs)):
        for j in range(len(ys)):
            if _point_in_poly_py(X[j, i], Y[j, i], grass_poly_xy) and 8 < Z[j, i] < 45:
                grass_mask[j, i] = True
    if grass_mask.any():
        g_pts = np.column_stack((X[grass_mask], Y[grass_mask], Z[grass_mask] + 0.15))
        g_poly = pv.PolyData(g_pts)
        try:
            g_surf = g_poly.delaunay_2d()
            actors["grassland"] = {
                "mesh": g_surf, "type": "mesh", "visible": True,
                "params": {"color": "#7dcea0", "opacity": 0.6, "smooth_shading": True},
                "extra": {"material": {"label": "草地", "eps_r": 12.0, "sigma": 0.005, "thickness_cm": 150.0}},
                "name": "grassland",
            }
        except Exception:
            pass

    # ── 森林区域（东北山区，Z>35m, 高大树木）──
    # ITU-R P.833-9: in-leaf trees @ 2.8 GHz
    # specific attenuation ≈ 0.3-0.5 dB/m, modeled as ε_r≈1.1, σ≈0.002, height 4m
    forest_poly_xy = np.array([
        [50, 50], [350, 100], [450, 300], [300, 450], [100, 350], [-50, 200], [50, 50]
    ])
    forest_mask = np.zeros(X.shape, dtype=bool)
    for i in range(len(xs)):
        for j in range(len(ys)):
            if _point_in_poly_py(X[j, i], Y[j, i], forest_poly_xy) and Z[j, i] > 30:
                forest_mask[j, i] = True
    if forest_mask.any():
        f_pts = np.column_stack((X[forest_mask], Y[forest_mask], Z[forest_mask] + 0.3))
        f_poly = pv.PolyData(f_pts)
        try:
            f_surf = f_poly.delaunay_2d()
            actors["forest"] = {
                "mesh": f_surf, "type": "mesh", "visible": True,
                "params": {"color": "#1e8449", "opacity": 0.55, "smooth_shading": True},
                "extra": {"material": {"label": "森林冠层", "eps_r": 1.1, "sigma": 0.002, "thickness_cm": 400.0}},
                "name": "forest",
            }
        except Exception:
            pass

    # ── 湖泊（东部盆地）──
    lake_cx, lake_cy = -200.0, -150.0
    lake_r = 60.0
    n_theta_l, n_r_l = 40, 15
    theta_l = np.linspace(0, 2 * np.pi, n_theta_l)
    r_l = np.linspace(0, lake_r, n_r_l)
    T_l, R_l = np.meshgrid(theta_l, r_l)
    lx = lake_cx + R_l * np.cos(T_l)
    ly = lake_cy + R_l * np.sin(T_l)
    lz = np.full_like(lx, 2.5)
    lake_grid = pv.StructuredGrid(lx, ly, lz)
    actors["lake"] = {
        "mesh": lake_grid, "type": "mesh", "visible": True,
        "params": {"color": "#1a5276", "opacity": 0.65, "smooth_shading": True},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 200.0}},
        "name": "lake",
    }

    # ── 点植被标记（散布的灌木丛点）──
    n_bushes = 200
    np.random.seed(42)
    bush_x = np.random.uniform(-400, 400, n_bushes)
    bush_y = np.random.uniform(-400, 400, n_bushes)
    bush_z_vals = np.zeros(n_bushes)
    for k in range(n_bushes):
        ix = np.argmin(np.abs(xs - bush_x[k]))
        iy = np.argmin(np.abs(ys - bush_y[k]))
        bush_z_vals[k] = Z[iy, ix] + 0.2
    bush_pts = pv.PolyData(np.column_stack((bush_x, bush_y, bush_z_vals)))
    actors["bushes"] = {
        "mesh": bush_pts, "type": "points", "visible": True,
        "params": {"color": "#6b8e23", "point_size": 6, "opacity": 0.7},
        "extra": None,
        "name": "bushes",
    }

    # ── 天线 ──
    ANT = (-400.0, 0.0, 55.0)
    ix_a = np.argmin(np.abs(xs - ANT[0]))
    iy_a = np.argmin(np.abs(ys - ANT[1]))
    ant_tz = float(Z[iy_a, ix_a])
    sphere = pv.Sphere(radius=1.5, center=ANT)
    pole_base = max(ant_tz, 0.0)
    h = ANT[2]
    pole = pv.Cylinder(center=(ANT[0], ANT[1], (pole_base + h) / 2),
                       direction=(0, 0, 1), radius=0.5, height=h - pole_base)
    actors["antenna"] = {
        "mesh": sphere.merge([pole]), "type": "mesh", "visible": True,
        "params": {"color": "#e63946", "smooth_shading": True, "opacity": 1.0,
                   "ambient": 0.5, "diffuse": 0.9, "specular": 0.5, "specular_power": 50},
        "extra": {"position": ANT, "is_source": True,
                  "antenna_config": dict(DEFAULT_ANTENNA_CONFIG)},
        "name": "antenna",
    }

    return actors


def _point_in_poly_py(x, y, poly):
    """Python版点包含测试（用于构建阶段）。"""
    n = len(poly); inside = False
    j = n - 1
    for i in range(n):
        if ((poly[i, 1] > y) != (poly[j, 1] > y)) and \
           (x < (poly[j, 0] - poly[i, 0]) * (y - poly[i, 1]) /
            (poly[j, 1] - poly[i, 1] + 1e-30) + poly[i, 0]):
            inside = not inside
        j = i
    return inside
