"""
画图对话框 — 传输损耗分布图、电场伪彩图、剖面图。
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotDialog(QtWidgets.QDialog):
    """多标签页绘图对话框，每页可独立导出。"""

    def __init__(self, parent, field_data):
        super().__init__(parent)
        self.setWindowTitle("传输损耗分布图")
        self.setMinimumSize(900, 650)
        self._data = field_data
        self._figures = []

        layout = QtWidgets.QVBoxLayout(self)

        # 标签页
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, 1)

        # 按钮
        btn = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("导出当前图…")
        btn_export.clicked.connect(self._export_current)
        btn.addWidget(btn_export)
        btn_export_all = QtWidgets.QPushButton("导出全部…")
        btn_export_all.clicked.connect(self._export_all)
        btn.addWidget(btn_export_all)
        btn.addStretch()
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn.addWidget(btn_close)
        layout.addLayout(btn)

        self._build_plots()

    def _build_plots(self):
        d = self._data
        u = d["u_total"]  # (nr, nphi, nz)
        r = d["r_vals"]
        z = d["z_vals"]
        phi = d["phi_vals"]
        cfg = d["config"]

        # phi=0 场
        pi0 = 0
        u_r0 = u[:, pi0, :]  # (nr, nz)
        E_r0 = np.abs(u_r0)

        # 传输损耗 TL = -20*log10(|u|/|u_ref|)
        u_ref = np.abs(u[0, pi0, :]).max() or 1e-30
        TL = -20 * np.log10(np.maximum(E_r0 / u_ref, 1e-15))
        # 补偿柱面扩散: TL_comp = TL + 10*log10(r/r0)
        r0 = r[0]
        for i in range(len(r)):
            TL[i, :] += 10 * np.log10(max(r[i] / r0, 1.0))

        R, Z = np.meshgrid(r, z, indexing="ij")

        # ── 图1: 距离-深度 TL 分布 (r-z contour) ──
        self._add_plot("距离-深度 (r-z) TL 分布", self._fig_r_z_tl(R, Z, TL, cfg))

        # ── 图2: 距离-深度 电场伪彩 (r-z |E|) ──
        self._add_plot("距离-深度 (r-z) 电场分布", self._fig_r_z_field(R, Z, E_r0, cfg))

        # ── 图3: 方位角-深度 TL 分布 (φ-z) ──
        ri = len(r) - 1  # 最远距离处
        self._add_plot("方位角-深度 (φ-z) TL 分布", self._fig_phi_z(phi, z, u[ri, :, :], r[ri], r0, cfg))

        # ── 图4: 路径损耗 vs 距离 (fix z=antenna height) ──
        zi_ant = np.argmin(np.abs(z - cfg.antenna_pos[2]))
        self._add_plot("路径损耗 vs 距离", self._fig_tl_vs_range(r, TL[:, zi_ant], cfg))

        # ── 图5: 场强 vs 高度 (fix r at max range) ──
        self._add_plot("场强 vs 高度", self._fig_E_vs_height(z, E_r0[-1, :], TL[-1, :], cfg))

        # ── 图6: 3D 表面图 ──
        self._add_plot("TL 3D 表面", self._fig_3d(R, Z, TL, cfg))

    def _add_plot(self, title, fig):
        self._figures.append((title, fig))
        w = QtWidgets.QWidget()
        lo = QtWidgets.QVBoxLayout(w)
        canvas = FigureCanvasQTAgg(fig)
        lo.addWidget(canvas)
        self._tabs.addTab(w, title)

    # ═══════════════════════════════════════
    #  各图绘制函数
    # ═══════════════════════════════════════

    def _fig_r_z_tl(self, R, Z, TL, cfg):
        fig, ax = plt.subplots(figsize=(8, 5))
        lev = np.linspace(TL.min(), min(TL.max(), TL.min() + 80), 30)
        cf = ax.contourf(R, Z, TL, levels=lev, cmap="jet", extend="both")
        cbar = fig.colorbar(cf, ax=ax, label="传输损耗 TL (dB)")
        ax.set_xlabel("距离 r (m)")
        ax.set_ylabel("高度 z (m)")
        ax.set_title(f"距离-深度 传输损耗分布 @ {cfg.frequency/1e9:.1f} GHz")
        # 天线标记
        ax.plot(0, cfg.antenna_pos[2], "w*", markersize=10, markeredgecolor="k")
        fig.tight_layout()
        return fig

    def _fig_r_z_field(self, R, Z, E, cfg):
        fig, ax = plt.subplots(figsize=(8, 5))
        E_db = 20 * np.log10(np.maximum(E, 1e-15))
        lev = np.linspace(E_db.max() - 60, E_db.max(), 30)
        cf = ax.contourf(R, Z, E_db, levels=lev, cmap="hot", extend="both")
        cbar = fig.colorbar(cf, ax=ax, label="|E| (dB)")
        ax.set_xlabel("距离 r (m)")
        ax.set_ylabel("高度 z (m)")
        ax.set_title(f"距离-深度 电场分布 @ {cfg.frequency/1e9:.1f} GHz")
        ax.plot(0, cfg.antenna_pos[2], "c*", markersize=10, markeredgecolor="k")
        fig.tight_layout()
        return fig

    def _fig_phi_z(self, phi, z, u_phi, r_max, r0, cfg):
        fig, ax = plt.subplots(figsize=(8, 5))
        E_phi = np.abs(u_phi)  # (nphi, nz)
        u_ref = E_phi.max() or 1e-30
        TL_phi = -20 * np.log10(np.maximum(E_phi / u_ref, 1e-15))
        TL_phi += 10 * np.log10(max(r_max / r0, 1.0))
        phi_deg = np.degrees(phi)
        P, Zp = np.meshgrid(phi_deg, z, indexing="ij")
        lev = np.linspace(TL_phi.min(), min(TL_phi.max(), TL_phi.min() + 60), 25)
        cf = ax.contourf(P, Zp, TL_phi, levels=lev, cmap="jet", extend="both")
        cbar = fig.colorbar(cf, ax=ax, label="传输损耗 TL (dB)")
        ax.set_xlabel("方位角 φ (°)")
        ax.set_ylabel("高度 z (m)")
        ax.set_title(f"方位角-深度 TL 分布 @ r={r_max:.1f}m, {cfg.frequency/1e9:.1f} GHz")
        fig.tight_layout()
        return fig

    def _fig_tl_vs_range(self, r, TL_line, cfg):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(r, TL_line, "b-", linewidth=1.5)
        ax.set_xlabel("距离 r (m)")
        ax.set_ylabel("传输损耗 TL (dB)")
        ax.set_title(f"路径损耗 vs 距离 @ z={cfg.antenna_pos[2]:.1f}m (天线高度)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _fig_E_vs_height(self, z, E_line, TL_line, cfg):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        E_db = 20 * np.log10(np.maximum(E_line, 1e-15))
        ax1.plot(E_db, z, "r-", linewidth=1.5)
        ax1.set_xlabel("|E| (dB)")
        ax1.set_ylabel("高度 z (m)")
        ax1.set_title("电场 vs 高度")
        ax1.grid(True, alpha=0.3)
        ax2.plot(TL_line, z, "b-", linewidth=1.5)
        ax2.set_xlabel("TL (dB)")
        ax2.set_title("传输损耗 vs 高度")
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f"垂直剖面 @ 最远距离 r={self._data['r_vals'][-1]:.1f}m")
        fig.tight_layout()
        return fig

    def _fig_3d(self, R, Z, TL, cfg):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        step_r = max(1, R.shape[0] // 80)
        step_z = max(1, R.shape[1] // 80)
        Rs, Zs, Ts = R[::step_r, ::step_z], Z[::step_r, ::step_z], TL[::step_r, ::step_z]
        surf = ax.plot_surface(Rs, Zs, Ts, cmap="jet", alpha=0.85, linewidth=0)
        fig.colorbar(surf, ax=ax, label="TL (dB)", shrink=0.6)
        ax.set_xlabel("r (m)")
        ax.set_ylabel("z (m)")
        ax.set_zlabel("TL (dB)")
        ax.set_title(f"TL 3D 表面 @ {cfg.frequency/1e9:.1f} GHz")
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════
    #  导出
    # ═══════════════════════════════════════

    def _export_current(self):
        idx = self._tabs.currentIndex()
        if idx < 0 or idx >= len(self._figures):
            return
        title, fig = self._figures[idx]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"导出 — {title}", f"{title}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")

    def _export_all(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not folder:
            return
        import os
        for i, (title, fig) in enumerate(self._figures):
            name = f"{i+1:02d}_{title}.png"
            fig.savefig(os.path.join(folder, name), dpi=150, bbox_inches="tight")
