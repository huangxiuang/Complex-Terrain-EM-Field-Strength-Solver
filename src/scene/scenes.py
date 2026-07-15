"""
场景注册表 — 5 个预置场景，从简单到复杂。
"""

import numpy as np
import pyvista as pv
from src.antenna_types import DEFAULT_ANTENNA_CONFIG


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


def _make_antenna():
    sphere = pv.Sphere(radius=ANTENNA_RADIUS, center=ANTENNA_POS)
    pole = pv.Cylinder(
        center=(ANTENNA_POS[0], ANTENNA_POS[1], ANTENNA_POS[2] / 2),
        direction=(0, 0, 1), radius=ANTENNA_RADIUS * 0.3, height=ANTENNA_POS[2],
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
    actors["antenna"] = _make_antenna()
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
    actors["antenna"] = _make_antenna()
    return actors


# ═══════════════════════════════════════════════════════════════
#  场景 3：山谷河流
# ═══════════════════════════════════════════════════════════════

@register("valley_river", "山谷河流", "V 形山谷 + 贯穿河流，水体对电磁波的反射/吸收效应")
def build_valley_river():
    span, res = 10.0, 60
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)

    # 山谷：沿 Y 轴的 V 形槽
    valley = 6.0 * np.exp(-X ** 2 / 8) * (1 + 0.3 * np.sin(Y * 0.5))
    # 两侧山脊
    ridges = (3.0 * np.exp(-((X + 6) ** 2) / 12) +
              3.0 * np.exp(-((X - 6) ** 2) / 12))
    Z = ridges - valley + 2.0

    actors = {}
    actors["terrain"] = _make_terrain(X, Y, Z)

    # 河流：山谷底部的水面
    water_z = 1.8
    river_mask = (X > -2.5) & (X < 2.5)
    water_X = X[river_mask].reshape(-1)
    water_Y = Y[river_mask].reshape(-1)
    nx = len(np.unique(water_X))
    ny = len(np.unique(water_Y))
    if nx > 1 and ny > 1:
        wx = water_X.reshape(ny, nx)
        wy = water_Y.reshape(ny, nx)
        wz = np.full_like(wx, water_z)
        water_grid = pv.StructuredGrid(wx, wy, wz)
        actors["river"] = {
            "mesh": water_grid, "type": "mesh", "visible": True,
            "params": {"color": "#2980b9", "opacity": 0.7, "smooth_shading": True},
            "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 0.5}},
            "name": "river",
        }

    actors["antenna"] = _make_antenna()
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
        {"x": -3, "y": 5, "w": 1.0, "d": 2.5, "h": 6.0, "name": "building_c"},
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

    actors["antenna"] = _make_antenna()
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
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 1.0}},
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
        "extra": {"material": {"label": "草地", "eps_r": 12.0, "sigma": 0.005, "thickness_cm": 1.0}},
        "name": "grass_patch",
    }

    actors["antenna"] = _make_antenna()
    return actors
