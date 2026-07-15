"""
Simple test scene factory — flat ground, wall obstacle, antenna source.

Replaces ``scene_builder.build_default_scene()`` during CPE development.
Compatible with the existing scene_objects dict protocol.

Objects:
    terrain       — flat ground plane (z=0, XY: -10..10)
    wall          — rectangular barrier at x=0, extending along y-axis, z=0.5
    antenna       — source marker sphere at (-5, 0, 6)

Usage::

    from src.simple_scene_builder import build_simple_scene
    scene_objects = build_simple_scene()
"""

import numpy as np
import pyvista as pv


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

GROUND_HALF_SPAN = 10.0         # XY half‑span of ground plane
GROUND_RES       = 50           # grid resolution per axis

WALL_X            = 0.0         # wall centre X
WALL_Y_HALF       = 15.0        # wall half‑length along y (宽于地形，防 phi 泄漏)
WALL_Z_BASE       = 0.0         # wall bottom z
WALL_Z_TOP        = 5.5         # wall top z (略低于天线 6m，经典 knife‑edge)
WALL_THICKNESS    = 1.0         # wall thickness in x (≈10 PE 步长，足以形成阴影)

from src.antenna_types import DEFAULT_ANTENNA_CONFIG

ANTENNA_POS       = (-5.0, 0.0, 6.0)   # (x, y, z) world coords
ANTENNA_RADIUS    = 0.3                # visual marker size


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def build_simple_scene():
    """Build a minimal test scene with flat ground, wall, and antenna.

    Returns
    -------
    dict  {name: scene_object, ...}
        Same protocol as ``scene_builder.build_default_scene()``.
    """
    actors = {}

    # ── 1. Flat ground plane ────────────────────────────
    xs = np.linspace(-GROUND_HALF_SPAN, GROUND_HALF_SPAN, GROUND_RES)
    ys = np.linspace(-GROUND_HALF_SPAN, GROUND_HALF_SPAN, GROUND_RES)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)   # flat at z=0, with tiny slope for layer thresholding
    Z += 0.05 * (X + GROUND_HALF_SPAN) / (2 * GROUND_HALF_SPAN)  # 0→0.05m slope

    grid = pv.StructuredGrid(X, Y, Z)
    grid["elevation"] = Z.flatten(order="F")

    actors["terrain"] = {
        "mesh": grid,
        "type": "mesh",
        "visible": True,
        "params": {
            "color": "#c8b88a",
            "smooth_shading": True,
            "opacity": 1.0,
        },
        "extra": {
            "original_z": Z.copy(),
            "X": X,
            "Y": Y,
            "is_dem": False,
            "material": {
                "label": "地面",
                "eps_r": 15.0,
                "sigma": 0.01,
            },
        },
        "name": "terrain",
    }

    # ── Overlay layers: sand/grass/earth ──
    for name, obj in _build_layer_meshes(grid, Z).items():
        actors[name] = obj

    # ── 1b. Water surface (flat plane at z=0.02) ──
    water = pv.StructuredGrid(X, Y, np.full_like(X, 0.02))
    actors["layer_water"] = {
        "mesh": water, "type": "mesh", "visible": False,
        "params": {"color": "#2980b9", "opacity": 0.7, "smooth_shading": True},
        "extra": {"material": {
            "label": "水面（淡水）", "eps_r": 80.0, "sigma": 0.01, "thickness_cm": 1.0,
        }},
        "name": "layer_water",
    }

    # ── 2. Wall obstacle ────────────────────────────────
    # Thin box:  thickness along x,  wide along y,  tall along z
    wall = pv.Box(bounds=(
        WALL_X - WALL_THICKNESS / 2,
        WALL_X + WALL_THICKNESS / 2,
        -WALL_Y_HALF,
        +WALL_Y_HALF,
        WALL_Z_BASE,
        WALL_Z_TOP,
    ))

    actors["wall"] = {
        "mesh": wall,
        "type": "mesh",
        "visible": True,
        "params": {
            "color": "#888888",
            "smooth_shading": True,
            "opacity": 0.85,
            "ambient": 0.3,
            "diffuse": 0.8,
            "specular": 0.3,
            "specular_power": 30,
        },
        "extra": {
            "material": {
                "label": "金属铝",
                "eps_r": 1.0,
                "sigma": 3.8e7,
            },
            "obstacle_type": "wall",
        },
        "name": "wall",
    }

    # ── 3. Antenna source (visual marker) ───────────────
    antenna = pv.Sphere(radius=ANTENNA_RADIUS, center=ANTENNA_POS)
    # Add a small cylinder "pole" for visual clarity
    pole = pv.Cylinder(
        center=(ANTENNA_POS[0], ANTENNA_POS[1], ANTENNA_POS[2] / 2),
        direction=(0, 0, 1),
        radius=ANTENNA_RADIUS * 0.3,
        height=ANTENNA_POS[2],
    )
    antenna_mesh = antenna.merge([pole])

    actors["antenna"] = {
        "mesh": antenna_mesh,
        "type": "mesh",
        "visible": True,
        "params": {
            "color": "#e63946",
            "smooth_shading": True,
            "opacity": 1.0,
            "ambient": 0.5,
            "diffuse": 0.9,
            "specular": 0.5,
            "specular_power": 50,
        },
        "extra": {
            "position": ANTENNA_POS,
            "is_source": True,
            "antenna_config": dict(DEFAULT_ANTENNA_CONFIG),
        },
        "name": "antenna",
    }

    return actors


def _build_layer_meshes(grid, Z):
    """Create 3 elevation-thresholded terrain layers (sand/grass/earth)."""
    surface = grid.extract_surface()
    z_min, z_max = float(Z.min()), float(Z.max())
    elev_range = max(z_max - z_min, 0.01)

    sand_max = z_min + elev_range * 0.20
    grass_max = z_min + elev_range * 0.45
    eps = 0.005

    sand_mesh = surface.threshold(
        [z_min - 0.1, sand_max + eps], scalars="elevation", preference="point")
    grass_mesh = surface.threshold(
        [sand_max - eps, grass_max + eps], scalars="elevation", preference="point")
    earth_mesh = surface.threshold(
        [grass_max - eps, z_max + 0.1], scalars="elevation", preference="point")

    return {
        "layer_sand": {
            "mesh": sand_mesh, "type": "mesh", "visible": False,
            "params": {"scalars": "elevation",
                       "cmap": ["#f5e6b8", "#e8c76a", "#d4a843"],
                       "clim": [z_min, sand_max],
                       "smooth_shading": True, "opacity": 1.0,
                       "show_scalar_bar": False},
            "extra": None, "name": "layer_sand",
        },
        "layer_grass": {
            "mesh": grass_mesh, "type": "mesh", "visible": False,
            "params": {"scalars": "elevation",
                       "cmap": ["#a8d5a2", "#5a9e4c", "#2d6b28"],
                       "clim": [sand_max, grass_max],
                       "smooth_shading": True, "opacity": 1.0,
                       "show_scalar_bar": False},
            "extra": None, "name": "layer_grass",
        },
        "layer_earth": {
            "mesh": earth_mesh, "type": "mesh", "visible": False,
            "params": {"scalars": "elevation",
                       "cmap": ["#d4b896", "#8b6f47", "#5c4033"],
                       "clim": [grass_max, z_max],
                       "smooth_shading": True, "opacity": 1.0,
                       "show_scalar_bar": False},
            "extra": None, "name": "layer_earth",
        },
    }
