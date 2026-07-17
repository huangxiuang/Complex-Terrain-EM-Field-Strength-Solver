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


@register("wilderness", "荒原", "1000×1000m 多噪声地形+河道+岩石山体+森林+草地, 底平面防悬空")
def build_wilderness():
    span = 500.0; res = 220
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)
    actors = {}
    np.random.seed(137)

    # ═══════════════════════════════════════════════════════════
    #  1. 多噪声地形
    # ═══════════════════════════════════════════════════════════
    Z1 = 90*np.exp(-((X-180)**2+(Y-120)**2)/90000) + 75*np.exp(-((X-320)**2+(Y-280)**2)/75000)
    Z2 = 28*np.exp(-((X+40)**2+(Y+60)**2)/18000) + 22*np.exp(-((X-120)**2+(Y+180)**2)/22000)
    plateau = np.where((X<-150)&(Y>100)&(X>-400)&(Y<350), 35.0, 0)
    gully = -8*np.exp(-((Y+80)**2)/2500)*np.abs(np.sin(X*0.01+0.8))
    tex = 2.5*np.sin(X*0.015)*np.cos(Y*0.018)+1.8*np.cos(X*0.035+1.2)*np.sin(Y*0.028)
    tex += 1.2*np.sin(X*0.06-Y*0.05)+0.8*np.cos(X*0.09)*np.cos(Y*0.11)
    Z_base = Z1 + Z2 + plateau + gully + tex + 5.0

    # 挖河道
    Z = Z_base.copy()
    for i in range(res):
        for j in range(res):
            dist = abs(X[i,j] - 3.0*np.sin(Y[i,j]*0.3))
            if dist < 2.0:
                t = dist/2.0; Z[i,j] = Z[i,j]*(0.05+0.15*t) - 2.5*(1-t)
    Z = np.maximum(Z, 0.0)

    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = Z.flatten(order="F")
    actors["terrain"] = {
        "mesh": grid, "type": "mesh", "visible": True,
        "params": {"color": "#b8956a", "smooth_shading": True, "opacity": 1.0,
                   "ambient": 0.1, "diffuse": 0.9, "specular": 0.1, "specular_power": 10},
        "extra": {"original_z": Z.copy(), "X": X, "Y": Y, "is_dem": False,
                  "material": {"label": "干燥土壤", "eps_r": 15.0, "sigma": 0.01}},
        "name": "terrain",
    }

    # 底平面（消除悬空感）
    z_min = Z.min()
    bottom = pv.Plane(center=(0,0,z_min-2), direction=(0,0,1), i_size=1000, j_size=1000)
    actors["ground_base"] = {
        "mesh": bottom, "type": "mesh", "visible": True,
        "params": {"color": "#8b6914", "smooth_shading": False, "opacity": 1.0},
        "extra": None, "name": "ground_base",
    }

    # 河道水面
    n_y, n_w = 150, 15
    river_y = np.linspace(-500, 500, n_y)
    river_w = np.linspace(-1.2, 1.2, n_w)
    Ry, Rw = np.meshgrid(river_y, river_w)
    Rx = 3.0*np.sin(Ry*0.3) + Rw
    river_grid = pv.StructuredGrid(Rx, Ry, np.full_like(Rx, -2.3))
    actors["river"] = {
        "mesh": river_grid, "type": "mesh", "visible": True,
        "params": {"color": "#0d4f4f", "opacity": 0.7, "smooth_shading": True,
                   "specular": 0.6, "specular_power": 50, "ambient": 0.2},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 200},
                  "is_material_layer": True},
        "name": "river",
    }

    # ═══════════════════════════════════════════════════════════
    #  2. 岩石山体
    # ═══════════════════════════════════════════════════════════
    Z_mtn = 130*np.exp(-((X-200)**2+(Y-150)**2)/25000) + 90*np.exp(-((X-300)**2+(Y-280)**2)/18000) + 70*np.exp(-((X-350)**2+(Y-20)**2)/15000)
    mg = pv.StructuredGrid(X, Y, Z_mtn)
    mg["Elevation"] = Z_mtn.flatten(order="F")
    try:
        mb = mg.extract_surface().threshold([20,200], scalars="Elevation", preference="point")
        if mb.n_points > 10:
            actors["mountain"] = {
                "mesh": mb, "type": "mesh", "visible": True,
                "params": {"color": "#7a7a7a", "smooth_shading": True, "opacity": 0.95,
                           "ambient": 0.15, "diffuse": 0.7, "specular": 0.2, "specular_power": 10},
                "extra": {"material": {"label": "岩石山体", "eps_r": 7.0, "sigma": 1e6},
                          "obstacle_type": "wall"},
                "name": "mountain",
            }
    except Exception: pass

    # ═══════════════════════════════════════════════════════════
    #  3. 森林（独立树木, 南部）
    # ═══════════════════════════════════════════════════════════
    trees_list = []
    for _ in range(250):
        tx = np.random.uniform(50, 400); ty = np.random.uniform(-300, -50)
        tiy = np.argmin(np.abs(ys-ty)); tix = np.argmin(np.abs(xs-tx))
        if Z_mtn[tiy, tix] < 15 and Z[tiy, tix] >= 0:
            h = np.random.uniform(3, 8)
            trunk = pv.Cylinder(center=(tx,ty,Z[tiy,tix]+h*0.25), direction=(0,0,1), radius=0.15, height=h*0.5, resolution=6)
            canopy = pv.Cone(center=(tx,ty,Z[tiy,tix]+h*0.6), direction=(0,0,1), radius=h*0.25, height=h*0.6, resolution=8)
            trees_list.append(trunk.merge(canopy))
    if trees_list:
        tm = trees_list[0]
        for t in trees_list[1:]: tm = tm.merge(t)
        actors["forest"] = {"mesh": tm, "type": "mesh", "visible": True,
                            "params": {"color": "#2d6b2d", "smooth_shading": False, "opacity": 0.9},
                            "extra": None, "name": "forest"}

    # ═══════════════════════════════════════════════════════════
    #  4. 草地（点精灵, 排除森林/河道/山体）
    # ═══════════════════════════════════════════════════════════
    n_grass = 4000
    gx = np.random.uniform(-450, 450, n_grass)
    gy = np.random.uniform(-450, 450, n_grass)
    gz = np.zeros(n_grass); keep = np.ones(n_grass, dtype=bool)
    for k in range(n_grass):
        gix = np.argmin(np.abs(xs-gx[k])); giy = np.argmin(np.abs(ys-gy[k]))
        gz[k] = Z[giy, gix] + 0.05
        if Z_mtn[giy, gix] > 15: keep[k] = False
        if 50<gx[k]<400 and -300<gy[k]<-50: keep[k] = False  # 森林区
        if abs(gx[k]-3.0*np.sin(gy[k]*0.3)) < 2.5: keep[k] = False  # 河道
    if keep.sum() > 0:
        actors["grass"] = {
            "mesh": pv.PolyData(np.column_stack((gx[keep], gy[keep], gz[keep]))),
            "type": "points", "visible": True,
            "params": {"color": "#4a8c3f", "point_size": 4, "opacity": 0.7},
            "extra": None, "name": "grass",
        }

    # ═══════════════════════════════════════════════════════════
    #  5. 湖泊 + 岩石 + 灌木
    # ═══════════════════════════════════════════════════════════
    lake_cx, lake_cy, lake_r = -300.0, -300.0, 40.0
    tl = np.linspace(0, 2*np.pi, 40); rl = np.linspace(0, lake_r, 15)
    Tl, Rl = np.meshgrid(tl, rl)
    actors["lake"] = {
        "mesh": pv.StructuredGrid(lake_cx+Rl*np.cos(Tl), lake_cy+Rl*np.sin(Tl), np.full_like(Rl, -2.3)),
        "type": "mesh", "visible": True,
        "params": {"color": "#0d4f4f", "opacity": 0.55, "smooth_shading": True,
                   "specular": 0.6, "specular_power": 50, "ambient": 0.2},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 200},
                  "is_material_layer": True},
        "name": "lake",
    }
    rocks_list = []
    for _ in range(40):
        rx = np.random.uniform(100, 400); ry = np.random.uniform(50, 350)
        riy = np.argmin(np.abs(ys-ry)); rix = np.argmin(np.abs(xs-rx))
        rz = Z_mtn[riy, rix]
        if rz > 30:
            s = np.random.uniform(3, 12)
            rock = pv.Icosahedron(radius=s); rock.points += np.array([rx, ry, rz - s*0.3])
            rocks_list.append(rock)
    if rocks_list:
        rm = rocks_list[0]
        for r in rocks_list[1:]: rm = rm.merge(r)
        actors["rocks"] = {"mesh": rm, "type": "mesh", "visible": True,
                           "params": {"color": "#7a7a7a", "smooth_shading": True, "opacity": 0.9},
                           "extra": None, "name": "rocks"}
    n_bush = 400
    bx = np.random.uniform(-450, 450, n_bush)
    by = np.random.uniform(-450, 450, n_bush)
    bz = np.array([Z[np.argmin(np.abs(ys-by[k])), np.argmin(np.abs(xs-bx[k]))]+0.1 for k in range(n_bush)])
    actors["bushes"] = {
        "mesh": pv.PolyData(np.column_stack((bx, by, bz))),
        "type": "points", "visible": True,
        "params": {"color": "#6b8e23", "point_size": 5, "opacity": 0.6},
        "extra": None, "name": "bushes",
    }

    # ═══════════════════════════════════════════════════════════
    #  6. 天线
    # ═══════════════════════════════════════════════════════════
    ANT = (-400.0, 0.0, 100.0)
    sphere = pv.Sphere(radius=2.0, center=ANT)
    pole = pv.Cylinder(center=(ANT[0], ANT[1], 50), direction=(0,0,1), radius=0.5, height=100)
    actors["antenna"] = {
        "mesh": sphere.merge([pole]), "type": "mesh", "visible": True,
        "params": {"color": "#e63946", "smooth_shading": True, "opacity": 1.0,
                   "ambient": 0.5, "diffuse": 0.9, "specular": 0.5, "specular_power": 50},
        "extra": {"position": ANT, "is_source": True,
                  "antenna_config": dict(DEFAULT_ANTENNA_CONFIG, dr_factor=8.0,
                                         fast_nz=512, fast_nphi=64,
                                         sigma_z=40.0, type="gaussian", tilt_angle=15.0)},
        "name": "antenna",
    }
    return actors
        "name": "antenna",
    }
    return actors
