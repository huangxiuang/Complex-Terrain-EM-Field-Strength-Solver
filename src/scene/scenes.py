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


@register("wilderness", "荒原", "1000×1000m 山脉融于地形+河道+湖沼+草地/疏林/密林+道路桥梁电线杆")
def build_wilderness():
    span = 500.0; res = 200
    xs = np.linspace(-span, span, res); ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)
    actors = {}; np.random.seed(42)

    # ═══════════════════════════════════════════════════════════
    #  1. 地形（山脉融于 Z）+ 挖河
    # ═══════════════════════════════════════════════════════════
    Z = (
        120*np.exp(-((X-200)**2+(Y-150)**2)/25000) + 100*np.exp(-((X-300)**2+(Y-280)**2)/20000) +
        80*np.exp(-((X-350)**2+(Y+20)**2)/18000) + 60*np.exp(-((X-100)**2+(Y-250)**2)/30000) +
        30*np.exp(-((X+40)**2+(Y+60)**2)/40000) + 20*np.exp(-((X-120)**2+(Y+180)**2)/35000) +
        5*np.sin(X*0.01)*np.cos(Y*0.015) + 3*np.cos(X*0.02)*np.sin(Y*0.025) +
        2*np.sin(X*0.04-Y*0.03) + 5.0
    )
    # 挖河道
    for i in range(res):
        for j in range(res):
            d = abs(X[i,j] - 3.0*np.sin(Y[i,j]*0.3))
            if d < 2.0: t = d/2.0; Z[i,j] = Z[i,j]*(0.03+0.15*t) - 2.5*(1-t)
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
    # 底平面
    actors["ground_base"] = {
        "mesh": pv.Plane(center=(0,0,-2), direction=(0,0,1), i_size=1000, j_size=1000),
        "type": "mesh", "visible": True,
        "params": {"color": "#6b4c1e", "smooth_shading": False, "opacity": 1.0},
        "extra": None, "name": "ground_base",
    }

    # ═══════════════════════════════════════════════════════════
    #  2. 岩石崖面（山体 Z>40m 区, 视觉覆盖）
    # ═══════════════════════════════════════════════════════════
    try:
        rk = grid.extract_surface().threshold([40, 200], scalars="elevation", preference="point")
        if rk.n_points > 10:
            actors["rock_face"] = {
                "mesh": rk, "type": "mesh", "visible": True,
                "params": {"color": "#7a7a7a", "smooth_shading": True, "opacity": 0.85,
                           "ambient": 0.15, "diffuse": 0.7, "specular": 0.2, "specular_power": 10},
                "extra": {"material": {"label": "岩石崖面", "eps_r": 7.0, "sigma": 1e6, "thickness_cm": 5},
                          "is_material_layer": True},
                "name": "rock_face",
            }
    except Exception: pass

    # ═══════════════════════════════════════════════════════════
    #  3. 水域: 河 + 湖 + 沼泽
    # ═══════════════════════════════════════════════════════════
    # 河
    ny, nw = 150, 15
    ry = np.linspace(-500, 500, ny); rw = np.linspace(-1.2, 1.2, nw)
    Ry, Rw = np.meshgrid(ry, rw)
    Rx = 3.0*np.sin(Ry*0.3) + Rw
    actors["river"] = {
        "mesh": pv.StructuredGrid(Rx, Ry, np.full_like(Rx, -2.3)),
        "type": "mesh", "visible": True,
        "params": {"color": "#0d4f4f", "opacity": 0.75, "smooth_shading": True,
                   "specular": 0.6, "specular_power": 50, "ambient": 0.2},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 150},
                  "is_material_layer": True},
        "name": "river",
    }
    # 湖
    clx, cly, cr = -300.0, -300.0, 40.0
    tl = np.linspace(0, 2*np.pi, 40); rl = np.linspace(0, cr, 15)
    Tl, Rl = np.meshgrid(tl, rl)
    actors["lake"] = {
        "mesh": pv.StructuredGrid(clx+Rl*np.cos(Tl), cly+Rl*np.sin(Tl), np.full_like(Rl, -2.0)),
        "type": "mesh", "visible": True,
        "params": {"color": "#0d4f4f", "opacity": 0.6, "smooth_shading": True,
                   "specular": 0.6, "specular_power": 50, "ambient": 0.2},
        "extra": {"material": {"label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 200},
                  "is_material_layer": True},
        "name": "lake",
    }
    # 沼泽（低洼湿地, 0<Z<3m 区域）
    marsh = (Z > 0) & (Z < 3) & (X < 0) & (Y < -100)
    if marsh.any():
        mp = np.column_stack((X[marsh], Y[marsh], np.full(marsh.sum(), 0.15)))
        actors["swamp"] = {
            "mesh": pv.PolyData(mp).delaunay_2d() if marsh.sum() > 3 else pv.PolyData(mp),
            "type": "mesh", "visible": True,
            "params": {"color": "#2d5016", "opacity": 0.5, "smooth_shading": True},
            "extra": {"material": {"label": "沼泽湿地", "eps_r": 30.0, "sigma": 0.05, "thickness_cm": 50},
                      "is_material_layer": True},
            "name": "swamp",
        }

    # ═══════════════════════════════════════════════════════════
    #  4. 植被分区
    # ═══════════════════════════════════════════════════════════
    # 草地 (西部, 20<Z<60)
    n_grass = 5000; gx=[]; gy=[]; gz=[]
    for _ in range(n_grass):
        tx=np.random.uniform(-400,0); ty=np.random.uniform(-300,300)
        gix=np.argmin(np.abs(xs-tx)); giy=np.argmin(np.abs(ys-ty))
        if 20<Z[giy,gix]<60:
            gx.append(tx); gy.append(ty); gz.append(Z[giy,gix]+0.05)
    if gx: actors["grassland"] = {
        "mesh": pv.PolyData(np.column_stack((gx,gy,gz))), "type": "points", "visible": True,
        "params": {"color": "#4a8c3f", "point_size": 4, "opacity": 0.7},
        "extra": None, "name": "grassland",
    }
    # 稀疏林 (南部, Z<40)
    st=[]; 
    for _ in range(120):
        tx=np.random.uniform(50,300); ty=np.random.uniform(-200,-50)
        tix=np.argmin(np.abs(xs-tx)); tiy=np.argmin(np.abs(ys-ty))
        if Z[tiy,tix]<40:
            h=np.random.uniform(3,6)
            tr=pv.Cylinder(center=(tx,ty,Z[tiy,tix]+h*0.25),direction=(0,0,1),radius=0.12,height=h*0.5,resolution=6)
            cn=pv.Cone(center=(tx,ty,Z[tiy,tix]+h*0.6),direction=(0,0,1),radius=h*0.2,height=h*0.6,resolution=8)
            st.append(tr.merge(cn))
    if st:
        sm=st[0]; 
        for t in st[1:]: sm=sm.merge(t)
        actors["sparse_forest"] = {"mesh":sm,"type":"mesh","visible":True,
            "params":{"color":"#3a7d3a","smooth_shading":False,"opacity":0.85},"extra":None,"name":"sparse_forest"}
    # 密林 (东南, Z<30)
    dt=[];
    for _ in range(200):
        tx=np.random.uniform(250,450); ty=np.random.uniform(-250,-80)
        tix=np.argmin(np.abs(xs-tx)); tiy=np.argmin(np.abs(ys-ty))
        if Z[tiy,tix]<30:
            h=np.random.uniform(4,10)
            tr=pv.Cylinder(center=(tx,ty,Z[tiy,tix]+h*0.25),direction=(0,0,1),radius=0.18,height=h*0.5,resolution=5)
            cn=pv.Cone(center=(tx,ty,Z[tiy,tix]+h*0.6),direction=(0,0,1),radius=h*0.3,height=h*0.6,resolution=7)
            dt.append(tr.merge(cn))
    if dt:
        dm=dt[0]; 
        for t in dt[1:]: dm=dm.merge(t)
        actors["dense_forest"] = {"mesh":dm,"type":"mesh","visible":True,
            "params":{"color":"#1e5a1e","smooth_shading":False,"opacity":0.9},"extra":None,"name":"dense_forest"}

    # ═══════════════════════════════════════════════════════════
    #  5. 人造: 道路 + 桥梁 + 电线杆
    # ═══════════════════════════════════════════════════════════
    road_x = np.linspace(-400, 400, 200)
    road_y = road_x * 0.2 + 50
    road_z = np.array([Z[np.argmin(np.abs(ys-road_y[k])), np.argmin(np.abs(xs-road_x[k]))]+0.05 for k in range(200)])
    road_pts = np.column_stack((road_x, road_y, road_z))
    if len(road_pts) > 4:
        try:
            actors["road"] = {
                "mesh": pv.PolyData(road_pts).delaunay_2d(),
                "type": "mesh", "visible": True,
                "params": {"color": "#555555", "opacity": 0.7, "smooth_shading": False},
                "extra": {"material": {"label": "沥青路面", "eps_r": 4.0, "sigma": 0.001, "thickness_cm": 10},
                          "is_material_layer": True},
                "name": "road",
            }
        except Exception: pass
    # 电线杆
    poles=[]; 
    for k in range(0,200,15):
        px,py,pz=road_x[k],road_y[k],road_z[k]
        pole=pv.Cylinder(center=(px,py,pz+8),direction=(0,0,1),radius=0.3,height=16,resolution=6)
        poles.append(pole)
    if poles:
        pm=poles[0]; 
        for p in poles[1:]: pm=pm.merge(p)
        actors["power_poles"] = {"mesh":pm,"type":"mesh","visible":True,
            "params":{"color":"#8b4513","smooth_shading":False,"opacity":0.8},"extra":None,"name":"power_poles"}
    # 桥（跨河）
    bridge_cy = 0; bridge_cx = 3.0*np.sin(bridge_cy*0.3)
    bridge = pv.Box(bounds=(bridge_cx-4,bridge_cx+4,-3,3,-2.5,-1.5))
    actors["bridge"] = {
        "mesh": bridge, "type": "mesh", "visible": True,
        "params": {"color": "#888888", "smooth_shading": True, "opacity": 0.85,
                   "ambient": 0.2, "diffuse": 0.8},
        "extra": {"material": {"label": "混凝土", "eps_r": 6.0, "sigma": 0.02, "thickness_cm": 50},
                  "is_material_layer": True},
        "name": "bridge",
    }

    # ═══════════════════════════════════════════════════════════
    #  6. 岩石 + 雪顶 + 裸土
    # ═══════════════════════════════════════════════════════════
    # 雪顶 (山体 Z>100m)
    try:
        sn = grid.extract_surface().threshold([100,200], scalars="elevation", preference="point")
        if sn.n_points > 5:
            actors["snow_cap"] = {"mesh":sn,"type":"mesh","visible":True,
                "params":{"color":"#f0f0f0","smooth_shading":True,"opacity":0.7,
                          "ambient":0.3,"diffuse":0.9,"specular":0.3,"specular_power":30},
                "extra":{"material":{"label":"雪/冰层","eps_r":2.0,"sigma":0.0001,"thickness_cm":30},
                         "is_material_layer":True},
                "name":"snow_cap"}
    except Exception: pass
    # 裸土 (低处, Z<5m, 无植被)
    bare = (Z<5) & (X>200) & (Y>200)
    if bare.any():
        bp=np.column_stack((X[bare],Y[bare],np.full(bare.sum(),0.1)))
        actors["bare_soil"] = {"mesh":pv.PolyData(bp).delaunay_2d() if bare.sum()>3 else pv.PolyData(bp),
            "type":"mesh","visible":True,
            "params":{"color":"#a08060","opacity":0.6,"smooth_shading":True},
            "extra":None,"name":"bare_soil"}
    # 散落岩石
    rl=[]; 
    for _ in range(30):
        rx=np.random.uniform(100,400); ry=np.random.uniform(50,350)
        riy=np.argmin(np.abs(ys-ry)); rix=np.argmin(np.abs(xs-rx))
        if Z[riy,rix]>40:
            s=np.random.uniform(3,12); rk=pv.Icosahedron(radius=s)
            rk.points+=np.array([rx,ry,Z[riy,rix]-s*0.3]); rl.append(rk)
    if rl:
        rm=rl[0]; 
        for r in rl[1:]: rm=rm.merge(r)
        actors["rocks"] = {"mesh":rm,"type":"mesh","visible":True,
            "params":{"color":"#7a7a7a","smooth_shading":True,"opacity":0.9},"extra":None,"name":"rocks"}
    # 灌木
    nb=300; bx=np.random.uniform(-450,450,nb); by=np.random.uniform(-450,450,nb)
    bz=np.array([Z[np.argmin(np.abs(ys-by[k])),np.argmin(np.abs(xs-bx[k]))]+0.1 for k in range(nb)])
    actors["bushes"] = {"mesh":pv.PolyData(np.column_stack((bx,by,bz))),"type":"points","visible":True,
        "params":{"color":"#6b8e23","point_size":5,"opacity":0.6},"extra":None,"name":"bushes"}

    # ═══════════════════════════════════════════════════════════
    #  7. 天线
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
