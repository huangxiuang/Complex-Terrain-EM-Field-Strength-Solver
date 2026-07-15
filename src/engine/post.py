"""
Post-processing 模块 — 场提取、路径损耗、结果打包。
"""

import numpy as np
from src.core.context import Context


def run(ctx: Context) -> Context:
    cfg = ctx.config
    results = []

    for rx in ctx.rx_points:
        r, phi = _cart_to_cyl(rx, cfg.antenna_pos)
        # 找到每个接收点对应的径向索引
        ri = np.argmin(np.abs(ctx.r_vals - r))
        pi = np.argmin(np.abs(ctx.phi_vals - phi))
        zi = np.argmin(np.abs(ctx.z_vals - rx[2]))

        E_rx = float(np.abs(ctx.u_total[ri, pi, zi]))
        E_fs = float(ctx.r_vals[0] / max(r, 1e-6))

        L_fs = (20 * np.log10(cfg.frequency / 1e6) +
                20 * np.log10(r / 1000) + 32.45)
        L_pe = L_fs - 20 * np.log10(max(E_rx / E_fs, 1e-12))

        results.append({
            "rx": rx,
            "dist": float(r),
            "E_rx": E_rx,
            "E_fs": E_fs,
            "L_fs_dB": float(L_fs),
            "path_loss_dB": float(L_pe),
            "r_grid": ctx.r_vals,
            "z_grid": ctx.z_vals,
            "phi_grid": ctx.phi_vals,
            "z_terrain_2d": ctx.z_top,
        })

    ctx.results = results
    return ctx


def _cart_to_cyl(rx, antenna):
    dx, dy = rx[0] - antenna[0], rx[1] - antenna[1]
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    if phi < 0:
        phi += 2 * np.pi
    return r, phi
