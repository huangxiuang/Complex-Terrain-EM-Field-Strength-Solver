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


@register("wilderness", "荒原", "1000×1000m 平地+岩石山体+森林/湖泊/沙地, ITU-R P.527/P.833")
def build_wilderness():
    span = 500.0; res = 200
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)
    actors = {}
    np.random.seed(137)

    # ── 1. 平坦地面 z=0（干燥土壤）──
    Z_flat = np.zeros_like(X)
    grid = pv.StructuredGrid(X, Y, Z_flat)
    grid["elevation"] = Z_flat.flatten(order="F")
    actors["terrain"] = {
        "mesh": grid, "type": "mesh", "visible": True,
        "params": {"color": "#b8956a", "smooth_shading": True, "opacity": 1.0,
                   "ambient": 0.1, "diffuse": 0.9, "specular": 0.1, "specular_power": 10},
        "extra": {"original_z": Z_flat.copy(), "X": X, "Y": Y, "is_dem": False,
                  "material": {"label": "干燥土壤", "eps_r": 15.0, "sigma": 0.01}},
        "name": "terrain",
    }

    # ── 2. 岩石山体（obstacle, field zeroed inside）──
    Z_mtn = (
        120.0 * np.exp(-((X-200)**2+(Y-150)**2)/30000) +
        80.0  * np.exp(-((X-300)**2+(Y-280)**2)/20000) +
        60.0  * np.exp(-((X-350)**2+(Y+20)**2)/18000)
    )
    mtn_grid = pv.StructuredGrid(X, Y, Z_mtn)
    mtn_grid["Elevation"] = Z_mtn.flatten(order="F")
    mtn_surface = mtn_grid.extract_surface()
    try:
        mtn_body = mtn_surface.threshold([20, 200], scalars="Elevation", preference="point")
        if mtn_body.n_points > 10:
            actors["mountain"] = {
                "mesh": mtn_body, "type": "mesh", "visible": True,
                "params": {"color": "#7a7a7a", "smooth_shading": True, "opacity": 0.95,
                           "ambient": 0.15, "diffuse": 0.7, "specular": 0.2, "specular_power": 10},
                "extra": {"material": {"label": "岩石山体", "eps_r": 7.0, "sigma": 1e6},
                          "obstacle_type": "wall"},
                "name": "mountain",
            }
    except Exception:
        pass

    # ── 3. 沙地区域（西南）— ITU P.527 ──
    sand_mask = (X < 50) & (Y < 50)
    if sand_mask.any():
        s_pts = np.column_stack((X[sand_mask], Y[sand_mask], np.full(sand_mask.sum(), 0.15)))
        sp = pv.PolyData(s_pts)
        try:
            ss = sp.delaunay_2d()
            actors["sand_zone"] = {
                "mesh": ss, "type": "mesh", "visible": True,
                "params": {"color": "#e8c76a", "opacity": 0.7, "smooth_shading": True},
                "extra": {"material": {"label": "沙地", "eps_r": 3.0, "sigma": 0.001, "thickness_cm": 30},
                          "is_material_layer": True},
                "name": "sand_zone",
            }
        except Exception: pass

    # ── 4. 草原（中部）— ITU P.527 ──
    grass_mask = (X > -300) & (X < 250) & (Y > -300) & (Y < 250)
    if grass_mask.any():
        g_pts = np.column_stack((X[grass_mask], Y[grass_mask], np.full(grass_mask.sum(), 0.2)))
        gp = pv.PolyData(g_pts)
        try:
            gs = gp.delaunay_2d()
            actors["grassland"] = {
                "mesh": gs, "type": "mesh", "visible": True,
                "params": {"color": "#7dcea0", "opacity": 0.5, "smooth_shading": True},
                "extra": {"material": {"label": "低矮植被", "eps_r": 1.2, "sigma": 0.00006, "thickness_cm": 150},
                          "is_material_layer": True},
                "name": "grassland",
            }
        except Exception: pass

    # ── 5. 森林冠层（东北）— ITU P.833 ──
    forest_mask = (X > 80) & (X < 400) & (Y > 50) & (Y < 350)
    if forest_mask.any():
        f_pts = np.column_stack((X[forest_mask], Y[forest_mask], np.full(forest_mask.sum(), 0.3)))
        fp = pv.PolyData(f_pts)
        try:
            fs = fp.delaunay_2d()
            actors["forest_canopy"] = {
                "mesh": fs, "type": "mesh", "visible": True,
                "params": {"color": "#1e8449", "opacity": 0.4, "smooth_shading": True},
                "extra": {"material": {"label": "森林冠层", "eps_r": 1.5, "sigma": 0.0002, "thickness_cm": 1000},
                          "is_material_layer": True},
                "name": "forest_canopy",
            }
        except Exception: pass

    # ── 6. 湖泊（西部）──
    lake_cx, lake_cy, lake_r = -250.0, -200.0, 50.0
    tl = np.linspace(0, 2*np.pi, 50); rl = np.linspace(0, lake_r, 20)
    Tl, Rl = np.meshgrid(tl, rl)
    actors["lake"] = {
        "mesh": pv.StructuredGrid(lake_cx+Rl*np.cos(Tl), lake_cy+Rl*np.sin(Tl), np.full_like(Rl, 0.1)),
        "type": "mesh", "visible": True,
        "params": {"color": "#0d4f4f", "opacity": 0.55, "smooth_shading": True,
                   "specular": 0.6, "specular_power": 50, "ambient": 0.2},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 200},
                  "is_material_layer": True},
        "name": "lake",
    }

    # ── 7. 树木（视觉）──
    trees_list = []
    for _ in range(200):
        tx = np.random.uniform(100, 380); ty = np.random.uniform(80, 350)
        if Z_mtn[np.argmin(np.abs(ys-ty)), np.argmin(np.abs(xs-tx))] < 5:
            h = np.random.uniform(3, 7)
            trunk = pv.Cylinder(center=(tx,ty,h*0.25), direction=(0,0,1), radius=0.12, height=h*0.5, resolution=6)
            canopy = pv.Cone(center=(tx,ty,h*0.55), direction=(0,0,1), radius=h*0.2, height=h*0.6, resolution=8)
            trees_list.append(trunk.merge(canopy))
    if trees_list:
        tm = trees_list[0]
        for t in trees_list[1:]: tm = tm.merge(t)
        actors["trees"] = {"mesh": tm, "type": "mesh", "visible": True,
                           "params": {"color": "#2d6b2d", "smooth_shading": False, "opacity": 0.85},
                           "extra": None, "name": "trees"}

    # ── 8. 岩石（视觉）──
    rocks_list = []
    for _ in range(50):
        rx = np.random.uniform(100, 400); ry = np.random.uniform(50, 350)
        riy = np.argmin(np.abs(ys-ry)); rix = np.argmin(np.abs(xs-rx))
        rz = Z_mtn[riy, rix]
        if rz > 20:
            s = np.random.uniform(2, 10)
            rock = pv.Icosahedron(radius=s)
            rock.points += np.array([rx, ry, rz - s*0.3])
            rocks_list.append(rock)
    if rocks_list:
        rm = rocks_list[0]
        for r in rocks_list[1:]: rm = rm.merge(r)
        actors["rocks"] = {"mesh": rm, "type": "mesh", "visible": True,
                           "params": {"color": "#7a7a7a", "smooth_shading": True, "opacity": 0.9},
                           "extra": None, "name": "rocks"}

    # ── 9. 灌木点 ──
    n_bush = 300
    bush_x = np.random.uniform(-450, 450, n_bush)
    bush_y = np.random.uniform(-450, 450, n_bush)
    bush_z = np.random.uniform(0.1, 0.3, n_bush)
    actors["bushes"] = {
        "mesh": pv.PolyData(np.column_stack((bush_x, bush_y, bush_z))),
        "type": "points", "visible": True,
        "params": {"color": "#6b8e23", "point_size": 5, "opacity": 0.6},
        "extra": None, "name": "bushes",
    }

    # ── 10. 天线 — 平地上 100m 塔 ──
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
                                         sigma_z=40.0, type="gaussian",
                                         tilt_angle=15.0)},
        "name": "antenna",
    }
    return actors
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
