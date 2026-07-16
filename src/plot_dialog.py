"""
画图对话框 — 勾选图类型→设参数→生成。
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

# 图定义: (标签, key, 参数列表)
# 参数: (name, label, min, max, default_getter)
PLOT_DEFS = [
    ("r-z 传输损耗分布", "rz_tl", [
        ("phi_deg", "方位角 φ (°)", -90, 90, lambda d: 0),
    ]),
    ("r-z 电场分布", "rz_field", [
        ("phi_deg", "方位角 φ (°)", -90, 90, lambda d: 0),
    ]),
    ("φ-z 传输损耗分布", "phiz_tl", [
        ("r_fix", "固定距离 r (m)", 0, 999, lambda d: float(d["r_vals"][-1])),
    ]),
    ("路径损耗 vs 距离", "tl_vs_r", [
        ("z_fix", "固定高度 z (m)", 0, 200, lambda d: float(d["config"].antenna_pos[2])),
    ]),
    ("场强 vs 高度", "e_vs_z", [
        ("r_fix", "固定距离 r (m)", 0, 999, lambda d: float(d["r_vals"][-1])),
    ]),
    ("TL 3D 表面", "tl_3d", []),
]


class PlotDialog(QtWidgets.QDialog):
    def __init__(self, parent, field_data):
        super().__init__(parent)
        self.setWindowTitle("画图")
        self.setMinimumSize(700, 500)
        self._data = field_data
        self._figures = []
        self._param_widgets = {}

        layout = QtWidgets.QVBoxLayout(self)

        # 上方: 图类型列表 + 参数
        top = QtWidgets.QHBoxLayout()
        # 左侧: 勾选列表
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("选择图类型:"))
        self._list = QtWidgets.QListWidget()
        for label, key, params in PLOT_DEFS:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, key)
            item.setCheckState(QtCore.Qt.Unchecked)
            self._list.addItem(item)
        self._list.itemChanged.connect(self._on_check_changed)
        left.addWidget(self._list, 1)
        top.addLayout(left)

        # 右侧: 参数面板
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("参数设置:"))
        self._param_stack = QtWidgets.QStackedWidget()
        self._param_pages = {}
        for label, key, params in PLOT_DEFS:
            page = QtWidgets.QWidget()
            flo = QtWidgets.QFormLayout(page)
            pw = {}
            for pname, plabel, pmin, pmax, pdef in params:
                val = pdef(self._data)
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(pmin, pmax)
                spin.setDecimals(2)
                spin.setValue(val)
                flo.addRow(plabel + ":", spin)
                pw[pname] = spin
            self._param_pages[key] = pw
            self._param_stack.addWidget(page)
        self._param_stack.setCurrentIndex(0)
        right.addWidget(self._param_stack, 1)
        top.addLayout(right)
        layout.addLayout(top)

        # 按钮
        btn = QtWidgets.QHBoxLayout()
        self._btn_gen = QtWidgets.QPushButton("▶ 生成")
        self._btn_gen.clicked.connect(self._generate)
        self._btn_gen.setStyleSheet("QPushButton { font-weight: bold; background: #2196F3; color: white; padding: 6px 20px; }")
        btn.addWidget(self._btn_gen)
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

        # 结果区: 标签页
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, 1)

    def _on_check_changed(self, item):
        key = item.data(QtCore.Qt.UserRole)
        for i, (_, pk, _) in enumerate(PLOT_DEFS):
            if pk == key:
                if item.checkState() == QtCore.Qt.Checked:
                    self._param_stack.setCurrentIndex(i)
                break

    def _generate(self):
        self._tabs.clear()
        self._figures.clear()
        d = self._data
        u = d["u_total"]  # (nr, nphi, nz)
        r = d["r_vals"]
        z = d["z_vals"]
        phi = d["phi_vals"]
        cfg = d["config"]
        r0 = r[0]

        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() != QtCore.Qt.Checked:
                continue
            key = item.data(QtCore.Qt.UserRole)
            pw = self._param_pages.get(key, {})

            if key == "rz_tl" or key == "rz_field":
                phi_deg = pw.get("phi_deg", None)
                if phi_deg is None: continue
                phi_target = np.radians(phi_deg.value())
                pi = np.argmin(np.abs(phi - phi_target))
                u_slice = np.abs(u[:, pi, :])  # (nr, nz)
                R, Z = np.meshgrid(r, z, indexing="ij")

                if key == "rz_tl":
                    u_ref = u_slice[0, :].max() or 1e-30
                    TL = -20 * np.log10(np.maximum(u_slice / u_ref, 1e-15))
                    for ri in range(len(r)):
                        TL[ri, :] += 10 * np.log10(max(r[ri] / r0, 1.0))
                    title = f"r-z TL分布 φ={phi_deg.value():.0f}°"
                    fig = self._contour(R, Z, TL, "TL (dB)", title, "jet", cfg)
                else:
                    E_db = 20 * np.log10(np.maximum(u_slice, 1e-15))
                    title = f"r-z 电场分布 φ={phi_deg.value():.0f}°"
                    fig = self._contour(R, Z, E_db, "|E| (dB)", title, "hot", cfg)
                self._add_tab(title, fig)

            elif key == "phiz_tl":
                r_fix = pw.get("r_fix", None)
                if r_fix is None: continue
                ri = np.argmin(np.abs(r - r_fix.value()))
                u_slice = np.abs(u[ri, :, :])  # (nphi, nz)
                u_ref = u_slice.max() or 1e-30
                TL = -20 * np.log10(np.maximum(u_slice / u_ref, 1e-15))
                TL += 10 * np.log10(max(r[ri] / r0, 1.0))
                phi_deg = np.degrees(phi)
                P, Zp = np.meshgrid(phi_deg, z, indexing="ij")
                title = f"φ-z TL分布 r={r[ri]:.1f}m"
                fig = self._contour(P, Zp, TL, "TL (dB)", title, "jet", cfg)
                self._add_tab(title, fig)

            elif key == "tl_vs_r":
                z_fix = pw.get("z_fix", None)
                if z_fix is None: continue
                zi = np.argmin(np.abs(z - z_fix.value()))
                u_line = np.abs(u[:, 0, zi])
                u_ref = u_line[0] if u_line[0] > 0 else 1e-30
                TL = -20 * np.log10(np.maximum(u_line / u_ref, 1e-15))
                for ri in range(len(r)):
                    TL[ri] += 10 * np.log10(max(r[ri] / r0, 1.0))
                title = f"路径损耗 vs 距离 z={z[zi]:.1f}m"
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(r, TL, "b-", linewidth=1.5)
                ax.set_xlabel("距离 r (m)"); ax.set_ylabel("TL (dB)")
                ax.set_title(title); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                self._add_tab(title, fig)

            elif key == "e_vs_z":
                r_fix = pw.get("r_fix", None)
                if r_fix is None: continue
                ri = np.argmin(np.abs(r - r_fix.value()))
                E_line = np.abs(u[ri, 0, :])
                E_db = 20 * np.log10(np.maximum(E_line, 1e-15))
                title = f"场强 vs 高度 r={r[ri]:.1f}m"
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(E_db, z, "r-", linewidth=1.5)
                ax.set_xlabel("|E| (dB)"); ax.set_ylabel("高度 z (m)")
                ax.set_title(title); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                self._add_tab(title, fig)

            elif key == "tl_3d":
                pi0 = 0
                u_slice = np.abs(u[:, pi0, :])
                u_ref = u_slice[0, :].max() or 1e-30
                TL = -20 * np.log10(np.maximum(u_slice / u_ref, 1e-15))
                for ri in range(len(r)):
                    TL[ri, :] += 10 * np.log10(max(r[ri] / r0, 1.0))
                R, Z = np.meshgrid(r, z, indexing="ij")
                title = "TL 3D 表面"
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(111, projection="3d")
                sr = max(1, len(r)//80); sz = max(1, len(z)//80)
                surf = ax.plot_surface(R[::sr,::sz], Z[::sr,::sz], TL[::sr,::sz],
                                       cmap="jet", alpha=0.85, linewidth=0)
                fig.colorbar(surf, ax=ax, label="TL (dB)", shrink=0.6)
                ax.set_xlabel("r (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("TL (dB)")
                ax.set_title(title); fig.tight_layout()
                self._add_tab(title, fig)

    def _contour(self, X, Y, data, cbar_label, title, cmap, cfg):
        fig, ax = plt.subplots(figsize=(8, 5))
        lev = np.linspace(data.min(), min(data.max(), data.min() + 80), 30)
        cf = ax.contourf(X, Y, data, levels=lev, cmap=cmap, extend="both")
        fig.colorbar(cf, ax=ax, label=cbar_label)
        ax.set_xlabel("r (m)" if X.max() > 10 else "φ (°)")
        ax.set_ylabel("z (m)")
        ax.set_title(title)
        if cfg.antenna_pos[2] <= Y.max():
            ax.plot(0 if X.max() > 10 else 0, cfg.antenna_pos[2], "w*", markersize=8)
        fig.tight_layout()
        return fig

    def _add_tab(self, title, fig):
        self._figures.append((title, fig))
        w = QtWidgets.QWidget()
        lo = QtWidgets.QVBoxLayout(w)
        canvas = FigureCanvasQTAgg(fig)
        lo.addWidget(canvas)
        self._tabs.addTab(w, title)

    def _export_current(self):
        idx = self._tabs.currentIndex()
        if idx < 0 or idx >= len(self._figures): return
        title, fig = self._figures[idx]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"导出 — {title}", f"{title}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")

    def _export_all(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not folder: return
        import os
        for i, (title, fig) in enumerate(self._figures):
            fig.savefig(os.path.join(folder, f"{i+1:02d}_{title}.png"), dpi=150, bbox_inches="tight")
