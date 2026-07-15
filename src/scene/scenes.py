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
                  "material": {"label": "地面", "eps_r": 15.0, "sigma": 0.01}},
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

@register("hills", "丘陵地带", "起伏地形，3 座缓坡山丘，无障碍物，研究地形对传播的影响")
def build_hills():
    span, res = 10.0, 60
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)

    Z = (
        3.0 * np.exp(-((X - 3) ** 2 + (Y - 2) ** 2) / 15) +
        5.0 * np.exp(-((X + 2) ** 2 + (Y + 4) ** 2) / 20) +
        4.0 * np.exp(-((X + 5) ** 2 + (Y - 5) ** 2) / 12) +
        0.02 * X + 0.01 * Y
    )

    actors = {}
    actors["terrain"] = _make_terrain(X, Y, Z)
    tz = float(Z[np.argmin(np.abs(xs - ANTENNA_POS[0])), np.argmin(np.abs(ys - ANTENNA_POS[1]))])
    actors["antenna"] = _make_antenna(tz)
    return actors


# ═══════════════════════════════════════════════════════════════
#  场景 3：经典原始（项目最初默认场景）
# ═══════════════════════════════════════════════════════════════

@register("classic", "经典原始", "高斯山丘 + 正弦河道 + 沙/草/土分层 + 植被 + 双机 + 鸟 + 树，项目最初默认场景")
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
                  "material": {"label": "地面", "eps_r": 15.0, "sigma": 0.01}},
        "name": "terrain",
    }

    # 高程阈值土壤层
    surface = grid.extract_surface()
    z_min, z_max = float(Z.min()), float(Z.max())
    rng = max(z_max - z_min, 0.01)
    s_max = z_min + rng * 0.20
    g_max = z_min + rng * 0.45
    eps = 0.02
    sand = surface.threshold([z_min - 1, s_max + eps], scalars="elevation", preference="point")
    grass = surface.threshold([s_max - eps, g_max + eps], scalars="elevation", preference="point")
    earth = surface.threshold([g_max - eps, z_max + 1], scalars="elevation", preference="point")

    actors["layer_sand"] = {
        "mesh": sand, "type": "mesh", "visible": False,
        "params": {"scalars": "elevation", "cmap": ["#f5e6b8", "#e8c76a", "#d4a843"],
                   "clim": [z_min, s_max], "smooth_shading": True, "opacity": 1.0, "show_scalar_bar": False},
        "extra": {"material": {"label": "沙地", "eps_r": 3.0, "sigma": 0.001, "thickness_cm": 5.0}}, "name": "layer_sand",
    }
    actors["layer_grass"] = {
        "mesh": grass, "type": "mesh", "visible": False,
        "params": {"scalars": "elevation", "cmap": ["#a8d5a2", "#5a9e4c", "#2d6b28"],
                   "clim": [s_max, g_max], "smooth_shading": True, "opacity": 1.0, "show_scalar_bar": False},
        "extra": {"material": {"label": "草地", "eps_r": 12.0, "sigma": 0.005, "thickness_cm": 10.0}}, "name": "layer_grass",
    }
    actors["layer_earth"] = {
        "mesh": earth, "type": "mesh", "visible": False,
        "params": {"scalars": "elevation", "cmap": ["#d4b896", "#8b6f47", "#5c4033"],
                   "clim": [g_max, z_max], "smooth_shading": True, "opacity": 1.0, "show_scalar_bar": False},
        "extra": {"material": {"label": "中等干燥地面", "eps_r": 15.0, "sigma": 0.01, "thickness_cm": 20.0}}, "name": "layer_earth",
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
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 80.0},
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

    # ── 双机 ──
    def _make_jet():
        fuselage = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=0.16, height=1.6)
        nose = pv.Cone(center=(1.05, 0, 0), direction=(1, 0, 0), height=0.5, radius=0.16)
        wl = pv.Box(bounds=(-0.3, 0.3, -0.9, 0, -0.025, 0.025)); wl.translate((0, -0.45, 0), inplace=True)
        wr = pv.Box(bounds=(-0.3, 0.3, 0, 0.9, -0.025, 0.025)); wr.translate((0, 0.45, 0), inplace=True)
        hl = pv.Box(bounds=(-0.125, 0.125, -0.4, 0, -0.015, 0.015)); hl.translate((-0.717, -0.2, 0), inplace=True)
        hr = pv.Box(bounds=(-0.125, 0.125, 0, 0.4, -0.015, 0.015)); hr.translate((-0.717, 0.2, 0), inplace=True)
        vt = pv.Box(bounds=(-0.95, -0.8, -0.015, 0.015, 0, 0.35))
        nozzle = pv.Cylinder(center=(-0.8, 0, 0), direction=(-1, 0, 0), radius=0.14, height=0.03)
        jet = fuselage.merge([nose, wl, wr, hl, hr, vt, nozzle])
        jet.scale(1.5, inplace=True)
        return jet

    jet1 = _make_jet(); jet1.translate((-6, -2, 8), inplace=True)
    actors["aircraft"] = {
        "mesh": jet1, "type": "mesh", "visible": True,
        "params": {"color": "#c4c4c4", "smooth_shading": True, "ambient": 0.35, "diffuse": 0.75, "specular": 0.6, "specular_power": 40},
        "extra": None, "name": "aircraft",
    }

    jet2 = _make_jet(); jet2.translate((-6, 3, 7.5), inplace=True)
    actors["aircraft2"] = {
        "mesh": jet2, "type": "mesh", "visible": True,
        "params": {"color": "#cc3333", "smooth_shading": True, "ambient": 0.35, "diffuse": 0.75, "specular": 0.6, "specular_power": 40},
        "extra": None, "name": "aircraft2",
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
#  场景 5：复合场景
# ═══════════════════════════════════════════════════════════════

@register("complex", "复合场景", "起伏地形 + 金属挡板 + 湖泊 + 草地，综合研究多因素耦合")
def build_complex():
    span, res = 10.0, 60
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)

    # 起伏地形
    Z = (
        4.0 * np.exp(-((X - 3) ** 2 + (Y - 2) ** 2) / 12) +
        3.0 * np.exp(-((X + 4) ** 2 + (Y + 5) ** 2) / 18) +
        0.02 * X + 0.01 * Y
    )

    actors = {}
    actors["terrain"] = _make_terrain(X, Y, Z)

    # 金属挡板
    actors["wall"] = _make_wall(z_top=6.0, label="金属铝", eps_r=1.0, sigma=3.8e7)

    # 湖泊（圆形水面，左侧）
    lake_r = 3.0
    lake_cx, lake_cy = -4.0, -3.0
    lake_z = 0.5
    n_theta, n_r = 40, 20
    theta_l = np.linspace(0, 2 * np.pi, n_theta)
    r_l = np.linspace(0, lake_r, n_r)
    T, R = np.meshgrid(theta_l, r_l)
    lx = lake_cx + R * np.cos(T)
    ly = lake_cy + R * np.sin(T)
    lz = np.full_like(lx, lake_z)
    lake_grid = pv.StructuredGrid(lx, ly, lz)
    actors["lake"] = {
        "mesh": lake_grid, "type": "mesh", "visible": True,
        "params": {"color": "#1a5276", "opacity": 0.6, "smooth_shading": True},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 80.0}},
        "name": "lake",
    }

    # 草地区域（右侧矩形）
    gxs = np.linspace(4, 9, 20)
    gys = np.linspace(1, 8, 20)
    GX, GY = np.meshgrid(gxs, gys)
    GZ = np.full_like(GX, 0.05)
    grass_grid = pv.StructuredGrid(GX, GY, GZ)
    actors["grass_patch"] = {
        "mesh": grass_grid, "type": "mesh", "visible": True,
        "params": {"color": "#27ae60", "opacity": 0.5, "smooth_shading": True},
        "extra": {"material": {"label": "草地", "eps_r": 12.0, "sigma": 0.005, "thickness_cm": 30.0}},
        "name": "grass_patch",
    }

    tz = float(Z[np.argmin(np.abs(xs - ANTENNA_POS[0])), np.argmin(np.abs(ys - ANTENNA_POS[1]))])
    actors["antenna"] = _make_antenna(tz)
    return actors
