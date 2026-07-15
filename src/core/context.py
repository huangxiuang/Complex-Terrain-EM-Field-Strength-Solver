"""
Context — 管道对象，在 problem/prep/solver/post 模块间传递。
所有字段在对应模块的 run() 中填充。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from src.core.config import EMConfig


@dataclass
class Context:
    # ── 输入 ──
    config: EMConfig = field(default_factory=EMConfig)
    rx_points: list = field(default_factory=list)     # [(x,y,z), ...]

    # ── Problem 模块填充 ──
    scene: dict = field(default_factory=dict)          # {name: scene_object}

    # ── Prep 模块填充 ──
    r_vals: np.ndarray = field(default_factory=lambda: np.array([]))
    z_vals: np.ndarray = field(default_factory=lambda: np.array([]))
    dr: float = 0.0
    dz: float = 0.0
    z_grd: np.ndarray = field(default_factory=lambda: np.array([]))
    z_top: np.ndarray = field(default_factory=lambda: np.array([]))
    n_grd: np.ndarray = field(default_factory=lambda: np.array([]))
    c_grd: np.ndarray = field(default_factory=lambda: np.array([]))
    n_lay: np.ndarray = field(default_factory=lambda: np.array([]))
    c_lay: np.ndarray = field(default_factory=lambda: np.array([]))
    dmft: Optional[dict] = None
    u_init: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Solver 模块填充 ──
    u_total: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_vals: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Post 模块填充 ──
    results: list = field(default_factory=list)
