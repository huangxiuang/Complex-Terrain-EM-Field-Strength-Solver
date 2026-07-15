"""
DMFT (离散混合傅里叶变换) — 阻抗地面边界条件。
"""

import numpy as np
from typing import Optional
from src.physics.materials import material_n, is_conductor, get_material


def build_dmft(freq: float, k0: float, n_z: int, z_max: float,
               scene: dict) -> Optional[dict]:
    """为有损地面构建 DMFT 算子。导体地面返回 None（用 FFT+PEC）。"""
    terrain = scene.get("terrain")
    if terrain is None:
        return None
    mat = get_material(terrain)
    if mat is None or is_conductor(mat["sigma"]):
        return None

    n_c = material_n(freq, mat["eps_r"], mat["sigma"])
    alpha = 1j * k0 * np.sqrt(n_c ** 2 - 1.0) / (n_c ** 2)
    return _make_dmft(alpha, n_z, z_max)


def _make_dmft(alpha: complex, n_z: int, z_max: float) -> dict:
    dz = z_max / (n_z - 1)
    z_vals = np.linspace(0, z_max, n_z)
    k_n = (np.arange(n_z) + 0.5) * np.pi / (z_max + dz)
    kzg, zg = k_n[:, None], z_vals[None, :]
    phi_nz = np.cos(kzg * zg) + (alpha / (kzg + 1e-30)) * np.sin(kzg * zg)
    N_n = (z_max / 2.0) * (1.0 + np.abs(alpha / (k_n + 1e-30)) ** 2)
    F = phi_nz * (dz / N_n[:, None])
    G = phi_nz.T
    return {"F": F, "G": G, "kz": k_n}
