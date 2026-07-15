"""
PE Solver 模块 — 前向抛物方程步进。
"""

import numpy as np
from scipy.special import hankel1 as _hankel1
from src.core.context import Context

N_ATM = 1.0003


def run(ctx: Context) -> Context:
    cfg = ctx.config

    # 只取前向传播
    u_fwd = _pe_march(
        ctx.u_init.copy(),
        ctx.r_vals, ctx.dr,
        ctx.z_vals, ctx.dz,
        ctx.z_grd, ctx.z_top,
        ctx.n_grd, ctx.c_grd,
        ctx.n_lay, ctx.c_lay,
        ctx.dmft,
        cfg.k0, cfg.n_atm, cfg.n_phi, cfg.n_z,
        forward=True,
    )
    ctx.u_total = u_fwd
    return ctx


def _pe_march(u, r_vals, dr, z_vals, dz,
              z_grd, z_top, n_grd, c_grd, n_lay, c_lay,
              dmft, k0, n_atm, n_phi, n_z, forward):
    nr = len(r_vals)
    if dmft is not None:
        kz_vals = dmft["kz"]
    else:
        kz_vals = 2.0 * np.pi * np.fft.fftfreq(n_z, dz)

    k_r = np.sqrt(np.maximum((k0 * n_atm) ** 2 - kz_vals ** 2, 0.0))
    k_r_ok = k_r > 1e-12

    refract = np.exp(1j * k0 * (n_atm - 2.0) * abs(dr))

    n_taper = n_z // 8
    taper = np.ones(n_z)
    for k in range(n_taper):
        taper[n_z - n_taper + k] = 0.5 + 0.5 * np.cos(8.0 * np.pi * k / n_z)

    g_idx = np.clip((z_grd / dz).astype(int), 0, n_z)
    t_idx = np.clip((z_top / dz).astype(int), 0, n_z)

    m_vals = np.fft.fftfreq(n_phi, 1.0 / n_phi).astype(int)

    u_full = np.zeros((nr, n_phi, n_z), dtype=np.complex64)
    u_full[0] = u

    for i in range(1, nr):
        rp, rc = r_vals[i - 1], r_vals[i]

        if dmft is not None:
            Uz = dmft["F"] @ u_full[i - 1].T
        else:
            Uz = np.fft.fft(u_full[i - 1], axis=1).T

        for ki in range(n_z):
            if not k_r_ok[ki]:
                Uz[ki, :] = 0.0
                continue
            Um = np.fft.fft(Uz[ki, :])
            xp_val, xc_val = k_r[ki] * rp, k_r[ki] * rc
            hp = _hankel1(m_vals, xp_val)
            hc = _hankel1(m_vals, xc_val)
            Um *= np.where(np.abs(hp) > 1e-15, hc / hp, 0.0)
            Uz[ki, :] = np.fft.ifft(Um)

        if dmft is not None:
            u_step = (dmft["G"] @ Uz).T
        else:
            u_step = np.fft.ifft(Uz.T, axis=1)

        u_step *= refract
        if min(rp, rc) > 0:
            u_step *= np.sqrt(rp / rc)

        # 地面边界
        for pi in range(n_phi):
            gi = g_idx[pi, i]
            ti = t_idx[pi, i]
            if gi <= 0:
                continue
            if c_grd[pi, i]:
                u_step[pi, :gi] = 0.0 + 0.0j
            else:
                u_step[pi, :gi] *= np.exp(1j * k0 * (n_grd[pi, i] - n_atm) * abs(dr))
            if ti > gi:
                if c_lay[pi, i]:
                    u_step[pi, gi:ti] = 0.0 + 0.0j
                else:
                    u_step[pi, gi:ti] *= np.exp(1j * k0 * (n_lay[pi, i] - n_atm) * abs(dr))

        u_step[:, :] *= taper[np.newaxis, :]
        u_full[i] = u_step

    return u_full
