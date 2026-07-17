"""
所有求解参数集中定义。
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EMConfig:
    # ── 物理参数 ──
    frequency: float = 2.8e9          # Hz
    n_atm: float = 1.0003             # 大气折射率

    # ── 天线 ──
    antenna_pos: Tuple[float, float, float] = (-5.0, 0.0, 6.0)
    antenna_type: str = "gaussian"
    antenna_sigma_z: float = 4.0
    antenna_tilt: float = 0.0         # 俯角 (deg)
    antenna_patch_hpbw: float = 70.0
    antenna_horn_hpbw: float = 30.0

    # ── PE 网格 ──
    n_z: int = 2048
    n_phi: int = 128
    dr_factor: float = 1.0            # 步长 = dr_factor * lambda
    r0_factor: float = 2.0            # 起始半径 = r0_factor * lambda
    z_pad_above: float = 500.0         # 高度网格上方留白，推高 taper 区

    # ── TWPE (暂不使用) ──
    twpe_max_iter: int = 0
    twpe_epsilon: float = 1e-3

    @property
    def wavelength(self) -> float:
        return 2.99792458e8 / self.frequency

    @property
    def k0(self) -> float:
        return 2.0 * 3.141592653589793 * self.frequency / 2.99792458e8

    def antenna_config(self) -> dict:
        return {
            "type": self.antenna_type,
            "frequency": self.frequency,
            "tilt_angle": self.antenna_tilt,
            "sigma_z": self.antenna_sigma_z,
            "patch_hpbw": self.antenna_patch_hpbw,
            "horn_hpbw": self.antenna_horn_hpbw,
        }
