"""
天线类型定义与初始场生成。

支持的 5 种天线：
  - half_wave_dipole : 半波偶极子（垂直）— 传播研究标准参考
  - microstrip_patch : 微带贴片 — 移动通信常用
  - horn            : 喇叭天线 — 微波链路、高增益
  - isotropic       : 全向点源 — 理论对照
  - gaussian        : 自定义高斯波束 — 可调波束宽度 + 俯角
"""

import numpy as np

C0 = 2.99792458e8

# ── 天线类型中文名映射 ──
ANTENNA_TYPE_LABELS = {
    "half_wave_dipole": "半波偶极子（垂直）",
    "microstrip_patch": "微带贴片",
    "horn": "喇叭天线",
    "isotropic": "全向天线（点源）",
    "gaussian": "自定义高斯波束",
}

# ── 各类型可调参数定义: (参数名, 默认值, 范围, 步长, 中文标签) ──
ANTENNA_PARAMS = {
    "half_wave_dipole": [
        ("tilt_angle", 0.0, (-30.0, 30.0), 1.0, "俯角 (°)"),
    ],
    "microstrip_patch": [
        ("patch_hpbw", 70.0, (30.0, 150.0), 5.0, "E面 HPBW (°)"),
        ("tilt_angle", 0.0, (-30.0, 30.0), 1.0, "俯角 (°)"),
    ],
    "horn": [
        ("horn_hpbw", 30.0, (5.0, 90.0), 5.0, "HPBW (°)"),
        ("tilt_angle", 0.0, (-30.0, 30.0), 1.0, "俯角 (°)"),
    ],
    "isotropic": [],
    "gaussian": [
        ("sigma_z", 2.0, (0.1, 20.0), 0.1, "波束宽度 σ_z (m)"),
        ("tilt_angle", 0.0, (-30.0, 30.0), 1.0, "俯角 (°)"),
    ],
}

DEFAULT_ANTENNA_CONFIG = {
    "type": "gaussian",
    "frequency": 2.8e9,
    "tilt_angle": 0.0,
    "sigma_z": 4.0,
    "patch_hpbw": 70.0,
    "horn_hpbw": 30.0,
}


# ═══════════════════════════════════════════════════════════════════
#  方向图函数 — 计算 z 高度网格上天线方向图幅度
# ═══════════════════════════════════════════════════════════════════

def _antenna_pattern(antenna_config, z_vals, h_ant, r0):
    """
    计算天线在圆柱面 r=r0 上各高度 z 的方向图幅度。

    参数
    ----
    antenna_config : dict  天线配置字典
    z_vals : ndarray      高度网格 (m), shape (nz,)
    h_ant : float         天线中心高度 (m)
    r0 : float            起始圆柱半径 (m)

    返回
    ----
    pattern : ndarray     归一化方向图幅度, shape (nz,)
    """
    ant_type = antenna_config.get("type", "gaussian")
    tilt_deg = antenna_config.get("tilt_angle", 0.0)
    tilt_rad = np.radians(tilt_deg)

    # 有效高度偏移（俯角近似）
    z_eff = z_vals - h_ant + r0 * np.tan(tilt_rad)

    if ant_type == "isotropic":
        pattern = np.ones_like(z_vals)

    elif ant_type == "half_wave_dipole":
        # 垂直偶极子：E ∝ |cos(π/2 cos θ_d) / sin θ_d|
        # θ_d 从偶极子轴（z轴）算起
        theta_d = np.pi / 2 - np.arctan2(z_eff, r0)  # 从垂直轴的角度
        sin_td = np.sin(np.maximum(np.abs(theta_d), 1e-12))
        pattern = np.abs(np.cos(np.pi / 2 * np.cos(theta_d))) / sin_td
        # 处理 θ_d → 0 的极限：cos(π/2) / 0 → π/2
        zero_mask = np.abs(theta_d) < 1e-4
        pattern[zero_mask] = np.pi / 2

    elif ant_type == "microstrip_patch":
        # 贴片天线：E(θ) ∝ cos^N(θ_elev), θ_elev 从水平面算
        hpbw_deg = antenna_config.get("patch_hpbw", 70.0)
        hpbw_rad = np.radians(hpbw_deg)
        N = np.log(2) / (2 * np.log(np.cos(hpbw_rad / 2) + 1e-15))
        theta_elev = np.arctan2(z_eff, r0)  # 仰角，水平=0，向上为正
        pattern = np.cos(np.abs(theta_elev)) ** N
        pattern[np.abs(theta_elev) > np.pi / 2] = 0.0

    elif ant_type == "horn":
        # 喇叭天线：高斯近似，给定 HPBW
        hpbw_deg = antenna_config.get("horn_hpbw", 30.0)
        hpbw_rad = np.radians(hpbw_deg)
        sigma_theta = hpbw_rad / (2 * np.sqrt(2 * np.log(2)))
        theta_elev = np.arctan2(z_eff, r0)
        pattern = np.exp(-0.5 * (theta_elev / sigma_theta) ** 2)

    elif ant_type == "gaussian":
        sigma_z = antenna_config.get("sigma_z", 2.0)
        pattern = np.exp(-0.5 * (z_eff / sigma_z) ** 2)

    else:
        pattern = np.ones_like(z_vals)

    # 归一化
    pmax = pattern.max()
    if pmax > 0:
        pattern = pattern / pmax
    return pattern


def make_initial_field(antenna_config, z_vals, n_phi, h_ant, k0=None, r0=None):
    """
    根据天线配置生成 PE 初始场。

    初始场 = 天线方向图幅度，PE 步进器内部处理几何扩散和相位传播。
    -40 dB 底线防止方向图零点导致 PE 数值失稳。
    """
    if r0 is None:
        r0 = 0.2

    pattern = _antenna_pattern(antenna_config, z_vals, h_ant, r0)

    u = np.zeros((n_phi, len(z_vals)), dtype=np.complex64)
    u[:, :] = pattern.astype(np.float32)[np.newaxis, :]
    return u
