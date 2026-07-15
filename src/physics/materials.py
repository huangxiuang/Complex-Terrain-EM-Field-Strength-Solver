"""
材料模型 — 频率相关复折射率。
"""

import numpy as np
from typing import Optional, Tuple

C0 = 2.99792458e8
EPS0 = 8.854187817e-12
SIGMA_CONDUCTOR = 1e5


def material_n(freq: float, eps_r: float, sigma: float) -> complex:
    """频率相关的复折射率 n = sqrt(eps_r - j*sigma/(omega*eps0))"""
    omega = 2.0 * np.pi * freq
    return np.sqrt(eps_r - 1j * sigma / (omega * EPS0 + 1e-30))


def is_conductor(sigma: float) -> bool:
    return sigma > SIGMA_CONDUCTOR


def resolve_material(freq: float, mat: dict) -> tuple:
    """从材料字典解析复折射率和导体标志。"""
    nc = material_n(freq, mat["eps_r"], mat["sigma"])
    return nc, is_conductor(mat["sigma"])


def get_material(obj: dict) -> Optional[dict]:
    """从场景对象提取材料字典。"""
    extra = obj.get("extra")
    return extra.get("material") if extra else None
