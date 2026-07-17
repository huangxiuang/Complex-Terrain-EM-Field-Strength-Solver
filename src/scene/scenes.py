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

# ═══════════════════════════════════════════════════════════════
#  场景 4：荒原 — 精细雕刻
# ═══════════════════════════════════════════════════════════════

@register("wilderness", "荒原", "1000×1000m 精细地形: 山脊·断崖·高原·沟壑 + 森林·岩体·湖泊·沙地, ITU-R P.833/P.527")
def build_wilderness():
    span = 500.0
    res = 250           # 4m 间距，高精度
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)
    actors = {}
    np.random.seed(137)

    # ═══════════════════════════════════════════════════════════
    #  1. 地形 — 8 层叠加
    # ═══════════════════════════════════════════════════════════

    # 大尺度山脉骨架
    Z1 = (
        90.0 * np.exp(-((X-180)**2+(Y-120)**2)/90000) +   # 东北主山脊
        75.0 * np.exp(-((X-320)**2+(Y-280)**2)/75000) +   # 远东山头
        55.0 * np.exp(-((X+120)**2+(Y-280)**2)/60000) +   # 北山
        65.0 * np.exp(-((X-380)**2+(Y+20)**2)/65000) +    # 东南山
        40.0 * np.exp(-((X+300)**2+(Y-100)**2)/55000)     # 西山
    )

    # 中尺度丘陵过渡
    Z2 = (
        28.0 * np.exp(-((X+40)**2+(Y+60)**2)/18000) +
        22.0 * np.exp(-((X-120)**2+(Y+180)**2)/22000) +
        18.0 * np.exp(-((X-60)**2+(Y-120)**2)/20000) +
        25.0 * np.exp(-((X+220)**2+(Y-40)**2)/24000) +
        20.0 * np.exp(-((X+180)**2+(Y+200)**2)/21000)
    )

    # 高原台地（西北方向）
    plateau_mask = (X < -150) & (Y > 100) & (X > -400) & (Y < 350)
    Z_plateau = np.zeros_like(X)
    Z_plateau[plateau_mask] = 35.0
    edge_dist = np.minimum(
        np.abs(X + 150), np.abs(X + 400),
    )
    edge_fade = np.clip((edge_dist - 20) / 40, 0, 1)
    Z_plateau[plateau_mask] *= np.where(edge_fade[plateau_mask] < 0.3, edge_fade[plateau_mask] / 0.3, 1.0)

    # 断崖（Y=200 附近的陡坡）
    cliff = 25.0 / (1.0 + np.exp(-(Y - 200) / 10)) * np.exp(-((X + 250)**2) / 15000)

    # 侵蚀沟壑网络
    gully1 = -8.0 * np.exp(-((Y + 80)**2) / 2500) * np.abs(np.sin(X * 0.01 + 0.8))
    gully2 = -6.0 * np.exp(-((Y - 300)**2) / 2000) * np.abs(np.cos(X * 0.012 - 1.2))
    gully3 = -5.0 * np.exp(-((Y - 50)**2) / 3500) * np.abs(np.sin((X + 200) * 0.009))

    # 多频地表纹理（6 个频率分量）
    Z_texture = (
        2.5 * np.sin(X*0.015) * np.cos(Y*0.018) +
        1.8 * np.cos(X*0.035+1.2) * np.sin(Y*0.028) +
        1.2 * np.sin(X*0.06-Y*0.05) * np.cos(Y*0.04) +
        0.8 * np.cos(X*0.09) * np.cos(Y*0.11+0.7) +
        0.5 * np.sin(X*0.13+Y*0.12) +
        0.3 * np.cos(X*0.18-Y*0.15)
    )

    # 随机起伏（模拟风化碎石地）
    Z_random = 1.5 * np.random.randn(*X.shape)
    Z_random = np.clip(Z_random, -3, 3)

    Z = Z1 + Z2 + Z_plateau + cliff + gully1 + gully2 + gully3 + Z_texture + Z_random + 3.0
    Z = np.maximum(Z, 0.0)

    # ═══════════════════════════════════════════════════════════
    #  2. 地形 Mesh
    # ═══════════════════════════════════════════════════════════
    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = Z.flatten(order="F")
    actors["terrain"] = {
        "mesh": grid, "type": "mesh", "visible": True,
        "params": {"scalars": "elevation", "cmap": "terrain",
                   "show_scalar_bar": False, "smooth_shading": True, "opacity": 1.0,
                   "ambient": 0.1, "diffuse": 0.9, "specular": 0.15, "specular_power": 20},
        "extra": {"original_z": Z.copy(), "X": X, "Y": Y, "is_dem": False,
                  "material": {"label": "干燥土壤", "eps_r": 15.0, "sigma": 0.01}},
        "name": "terrain",
    }

    # ═══════════════════════════════════════════════════════════
    #  3. 沙地区域（西南低处）— ITU-R P.527 dry sand
    # ═══════════════════════════════════════════════════════════
    sand_poly = np.array([[-350,-300],[-50,-350],[100,-150],[50,50],[-200,0],[-350,-100]])
    sand_mask = np.zeros(X.shape, dtype=bool)
    for i in range(res):
        for j in range(res):
            if _point_in_poly_py(X[j,i], Y[j,i], sand_poly) and Z[j,i] < 12:
                sand_mask[j,i] = True
    if sand_mask.any():
        s_pts = np.column_stack((X[sand_mask], Y[sand_mask], Z[sand_mask]+0.2))
        sp = pv.PolyData(s_pts)
        try:
            ss = sp.delaunay_2d()
            actors["sand_zone"] = {
                "mesh": ss, "type": "mesh", "visible": True,
                "params": {"color": "#e8c76a", "opacity": 0.75, "smooth_shading": True},
                "extra": {"material": {"label": "沙地", "eps_r": 3.0, "sigma": 0.001, "thickness_cm": 30}},
                "name": "sand_zone",
            }
        except Exception: pass

    # ═══════════════════════════════════════════════════════════
    #  4. 草原 — ITU-R P.527 medium dry ground
    # ═══════════════════════════════════════════════════════════
    grass_poly = np.array([[-300,-250],[-80,120],[150,280],[320,80],[280,-200],[40,-320],[-250,-300]])
    grass_mask = np.zeros(X.shape, dtype=bool)
    for i in range(res):
        for j in range(res):
            if _point_in_poly_py(X[j,i], Y[j,i], grass_poly) and 5 < Z[j,i] < 50:
                grass_mask[j,i] = True
    if grass_mask.any():
        g_pts = np.column_stack((X[grass_mask], Y[grass_mask], Z[grass_mask]+0.25))
        gp = pv.PolyData(g_pts)
        try:
            gs = gp.delaunay_2d()
            actors["grassland"] = {
                "mesh": gs, "type": "mesh", "visible": True,
                "params": {"color": "#7dcea0", "opacity": 0.55, "smooth_shading": True},
                "extra": {"material": {"label": "草地", "eps_r": 12.0, "sigma": 0.005, "thickness_cm": 200}},
                "name": "grassland",
            }
        except Exception: pass

    # ═══════════════════════════════════════════════════════════
    #  5. 森林冠层（东北山区）— ITU-R P.833-9, 4m canopy
    # ═══════════════════════════════════════════════════════════
    forest_poly = np.array([[30,30],[320,80],[420,280],[280,420],[80,340],[-60,180],[30,30]])
    forest_mask = np.zeros(X.shape, dtype=bool)
    for i in range(res):
        for j in range(res):
            if _point_in_poly_py(X[j,i], Y[j,i], forest_poly) and Z[j,i] > 25:
                forest_mask[j,i] = True
    if forest_mask.any():
        f_pts = np.column_stack((X[forest_mask], Y[forest_mask], Z[forest_mask]+0.5))
        fp = pv.PolyData(f_pts)
        try:
            fs = fp.delaunay_2d()
            actors["forest_canopy"] = {
                "mesh": fs, "type": "mesh", "visible": True,
                "params": {"color": "#1e8449", "opacity": 0.4, "smooth_shading": True},
                "extra": {"material": {"label": "森林冠层", "eps_r": 1.1, "sigma": 0.002, "thickness_cm": 400}},
                "name": "forest_canopy",
            }
        except Exception: pass

    # ═══════════════════════════════════════════════════════════
    #  6. 独立树木（视觉森林 — 200+ 棵）
    # ═══════════════════════════════════════════════════════════
    tree_positions = []
    # 密集林区
    for _ in range(160):
        tx = np.random.uniform(50, 380)
        ty = np.random.uniform(80, 400)
        tix = np.argmin(np.abs(xs - tx))
        tiy = np.argmin(np.abs(ys - ty))
        if _point_in_poly_py(tx, ty, forest_poly) and Z[tiy, tix] > 25:
            tz = float(Z[tiy, tix])
            h = np.random.uniform(3.0, 7.0)
            tree_positions.append((tx, ty, tz, h, "#1e5a1e"))
    # 稀疏林缘
    for _ in range(50):
        tx = np.random.uniform(-50, 420)
        ty = np.random.uniform(20, 430)
        tix = np.argmin(np.abs(xs - tx))
        tiy = np.argmin(np.abs(ys - ty))
        if _point_in_poly_py(tx, ty, forest_poly) and Z[tiy, tix] > 20:
            tz = float(Z[tiy, tix])
            h = np.random.uniform(2.0, 5.0)
            tree_positions.append((tx, ty, tz, h, "#2d6b2d"))
    # 孤立树木（草原上）
    for _ in range(30):
        tx = np.random.uniform(-300, 300)
        ty = np.random.uniform(-300, 300)
        tix = np.argmin(np.abs(xs - tx))
        tiy = np.argmin(np.abs(ys - ty))
        if not _point_in_poly_py(tx, ty, forest_poly) and Z[tiy, tix] > 8:
            tz = float(Z[tiy, tix])
            h = np.random.uniform(1.5, 4.0)
            tree_positions.append((tx, ty, tz, h, "#3a7d3a"))

    if tree_positions:
        all_trees = []
        for tx, ty, tz, h, color in tree_positions:
            trunk_h = h * 0.5
            trunk_r = h * 0.04
            canopy_r = h * 0.25
            trunk = pv.Cylinder(center=(tx, ty, tz + trunk_h/2), direction=(0,0,1),
                                radius=trunk_r, height=trunk_h, resolution=6)
            canopy = pv.Cone(center=(tx, ty, tz + trunk_h + canopy_r*0.6),
                             direction=(0,0,1), radius=canopy_r, height=canopy_r*2.5, resolution=8)
            tree = trunk.merge(canopy)
            all_trees.append(tree)
        all_trees_mesh = all_trees[0]
        for t in all_trees[1:]:
            all_trees_mesh = all_trees_mesh.merge(t)
        actors["trees"] = {
            "mesh": all_trees_mesh, "type": "mesh", "visible": True,
            "params": {"color": "#2d6b2d", "smooth_shading": False, "opacity": 0.9,
                       "ambient": 0.2, "diffuse": 0.8},
            "extra": None, "name": "trees",
        }

    # ═══════════════════════════════════════════════════════════
    #  7. 岩石山体（视觉岩层 — 50+ 块）
    # ═══════════════════════════════════════════════════════════
    rock_colors = ["#6b6b6b", "#787878", "#5a5a5a", "#8a8078", "#706860", "#606060"]
    all_rocks = []
    # 主山脊岩石群
    for _ in range(30):
        rx = np.random.uniform(120, 350)
        ry = np.random.uniform(80, 300)
        rix = np.argmin(np.abs(xs - rx))
        riy = np.argmin(np.abs(ys - ry))
        rz = float(Z[riy, rix])
        if rz > 30:
            s = np.random.uniform(3, 12)
            rock = pv.Icosahedron(radius=s)
            rock = rock.subdivide(1, subfilter="linear")
            rock.points += np.array([rx, ry, rz - s*0.3])
            rock.points += np.random.randn(*rock.points.shape) * s * 0.1
            all_rocks.append(rock)
    # 高原边缘碎石
    for _ in range(15):
        rx = np.random.uniform(-380, -160)
        ry = np.random.uniform(130, 330)
        rix = np.argmin(np.abs(xs - rx))
        riy = np.argmin(np.abs(ys - ry))
        rz = float(Z[riy, rix])
        if 30 < rz < 45:
            s = np.random.uniform(2, 8)
            rock = pv.Icosahedron(radius=s)
            rock.points += np.array([rx, ry, rz - s*0.2])
            all_rocks.append(rock)
    # 散落岩石（全图）
    for _ in range(20):
        rx = np.random.uniform(-450, 450)
        ry = np.random.uniform(-450, 450)
        rix = np.argmin(np.abs(xs - rx))
        riy = np.argmin(np.abs(ys - ry))
        rz = float(Z[riy, rix])
        s = np.random.uniform(1, 5)
        rock = pv.Icosahedron(radius=s)
        rock.points += np.array([rx, ry, rz])
        all_rocks.append(rock)
    if all_rocks:
        rocks_mesh = all_rocks[0]
        for r in all_rocks[1:]:
            rocks_mesh = rocks_mesh.merge(r)
        actors["rocks"] = {
            "mesh": rocks_mesh, "type": "mesh", "visible": True,
            "params": {"color": "#7a7a7a", "smooth_shading": True, "opacity": 0.95,
                       "ambient": 0.15, "diffuse": 0.7, "specular": 0.2, "specular_power": 10},
            "extra": None, "name": "rocks",
        }

    # ═══════════════════════════════════════════════════════════
    #  8. 水体 — 主湖 + 小水塘（采样地形 Z）
    # ═══════════════════════════════════════════════════════════
    def _terrain_z_at(x, y):
        ix = np.argmin(np.abs(xs - x)); iy = np.argmin(np.abs(ys - y))
        return float(Z[iy, ix])

    lake_cx, lake_cy, lake_r = -300.0, -350.0, 45.0
    ntl, nrl = 50, 20
    tl = np.linspace(0, 2*np.pi, ntl)
    rl = np.linspace(0, lake_r, nrl)
    Tl, Rl = np.meshgrid(tl, rl)
    lx = lake_cx + Rl*np.cos(Tl)
    ly = lake_cy + Rl*np.sin(Tl)
    lz = np.maximum(np.full_like(lx, _terrain_z_at(lake_cx, lake_cy) + 0.3), 0.5)
    actors["lake"] = {
        "mesh": pv.StructuredGrid(lx, ly, np.full_like(lx, lz[0,0])), "type": "mesh", "visible": True,
        "params": {"color": "#0d4f4f", "opacity": 0.6, "smooth_shading": True,
                   "specular": 0.6, "specular_power": 50, "ambient": 0.2},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 250}},
        "name": "lake",
    }
    for px, py, pr in [(-200, -400, 12), (0, -300, 10), (300, -200, 8)]:
        ntp, nrp = 25, 8
        tp = np.linspace(0, 2*np.pi, ntp)
        rp = np.linspace(0, pr, nrp)
        Tp, Rp = np.meshgrid(tp, rp)
        pz = max(_terrain_z_at(px, py) + 0.2, 0.3)
        pond = pv.StructuredGrid(px+Rp*np.cos(Tp), py+Rp*np.sin(Tp), np.full_like(Rp, pz))
        actors[f"pond_{px:.0f}"] = {
            "mesh": pond, "type": "mesh", "visible": True,
            "params": {"color": "#0d4f4f", "opacity": 0.55, "smooth_shading": True,
                       "specular": 0.7, "specular_power": 60, "ambient": 0.2},
            "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 100}},
            "name": f"pond_{px:.0f}",
        }

    # ═══════════════════════════════════════════════════════════
    #  9. 灌木丛点
    # ═══════════════════════════════════════════════════════════
    n_bush = 350
    bush_x = np.random.uniform(-450, 450, n_bush)
    bush_y = np.random.uniform(-450, 450, n_bush)
    bush_z = np.array([float(Z[np.argmin(np.abs(ys-bush_y[k])), np.argmin(np.abs(xs-bush_x[k]))])+0.15
                       for k in range(n_bush)])
    actors["bushes"] = {
        "mesh": pv.PolyData(np.column_stack((bush_x, bush_y, bush_z))),
        "type": "points", "visible": True,
        "params": {"color": "#6b8e23", "point_size": 5, "opacity": 0.6},
        "extra": None, "name": "bushes",
    }

    # ═══════════════════════════════════════════════════════════
    #  10. 枯木/倒木
    # ═══════════════════════════════════════════════════════════
    dead_trees = []
    for _ in range(15):
        dx = np.random.uniform(-400, 400)
        dy = np.random.uniform(-400, 400)
        dix = np.argmin(np.abs(xs-dx)); diy = np.argmin(np.abs(ys-dy))
        dz = float(Z[diy, dix]) + 0.1
        h = np.random.uniform(1, 3)
        angle = np.random.uniform(0, np.pi)
        trunk = pv.Cylinder(center=(dx, dy, dz+h/2), direction=(np.cos(angle), np.sin(angle), 0.3),
                            radius=0.15, height=h, resolution=5)
        dead_trees.append(trunk)
    if dead_trees:
        dt_mesh = dead_trees[0]
        for dt in dead_trees[1:]: dt_mesh = dt_mesh.merge(dt)
        actors["deadwood"] = {
            "mesh": dt_mesh, "type": "mesh", "visible": True,
            "params": {"color": "#8b7355", "smooth_shading": False, "opacity": 0.8},
            "extra": None, "name": "deadwood",
        }

    # ═══════════════════════════════════════════════════════════
    #  11. 天线
    # ═══════════════════════════════════════════════════════════
    ANT = (-400.0, 0.0, 55.0)
    ant_ix = np.argmin(np.abs(xs-ANT[0])); ant_iy = np.argmin(np.abs(ys-ANT[1]))
    ant_tz = float(Z[ant_iy, ant_ix])
    sphere = pv.Sphere(radius=1.5, center=ANT)
    pb = max(ant_tz, 0.0)
    pole = pv.Cylinder(center=(ANT[0], ANT[1], (pb+ANT[2])/2), direction=(0,0,1),
                       radius=0.5, height=ANT[2]-pb)
    actors["antenna"] = {
        "mesh": sphere.merge([pole]), "type": "mesh", "visible": True,
        "params": {"color": "#e63946", "smooth_shading": True, "opacity": 1.0,
                   "ambient": 0.5, "diffuse": 0.9, "specular": 0.5, "specular_power": 50},
        "extra": {"position": ANT, "is_source": True,
                  "antenna_config": dict(DEFAULT_ANTENNA_CONFIG, dr_factor=8.0, fast_nz=768, fast_nphi=48)},
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
