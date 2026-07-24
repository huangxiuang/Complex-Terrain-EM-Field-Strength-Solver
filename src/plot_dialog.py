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
        ("r_fix", "固定距离 r (m)", 0, 9999, lambda d: float(d["r_vals"][-1])),
    ]),
    ("路径损耗 vs 距离", "tl_vs_r", [
        ("z_fix", "固定高度 z (m)", 0, 9999, lambda d: float(d["config"].antenna_pos[2])),
        ("phi_deg", "方位角 φ (°)", -90, 90, lambda d: 0),
    ]),
    ("场强 vs 高度", "e_vs_z", [
        ("r_fix", "固定距离 r (m)", 0, 9999, lambda d: float(d["r_vals"][-1])),
        ("phi_deg", "方位角 φ (°)", -90, 90, lambda d: 0),
    ]),
    ("TL 3D 表面", "tl_3d", []),
]


class PlotDialog(QtWidgets.QDialog):
    def __init__(self, parent, field_data):
        super().__init__(parent)
        self.setWindowTitle("画图")
        self.setMinimumSize(700, 500)
        self._data = field_data
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
        btn.addStretch()
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn.addWidget(btn_close)
        layout.addLayout(btn)

        # 结果区: 占位（图在新窗口显示）
        layout.addStretch(0)

    def _on_check_changed(self, item):
        key = item.data(QtCore.Qt.UserRole)
        for i, (_, pk, _) in enumerate(PLOT_DEFS):
            if pk == key:
                if item.checkState() == QtCore.Qt.Checked:
                    self._param_stack.setCurrentIndex(i)
                break

    def _generate(self):
        figs = []
        d = self._data
        u = d["u_total"]; r = d["r_vals"]; z = d["z_vals"]; phi = d["phi_vals"]
        cfg = d["config"]; r0 = r[0]
        zi_ant = np.argmin(np.abs(z - cfg.antenna_pos[2]))

        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() != QtCore.Qt.Checked: continue
            key = item.data(QtCore.Qt.UserRole)
            pw = self._param_pages.get(key, {})

            if key in ("rz_tl", "rz_field"):
                phi_deg = pw.get("phi_deg"); 
                if phi_deg is None: continue
                pi = np.argmin(np.abs(phi - np.radians(phi_deg.value())))
                u_slice = np.abs(u[:, pi, :]); R, Z = np.meshgrid(r, z, indexing="ij")
                if key == "rz_tl":
                    zi_ant = np.argmin(np.abs(z - cfg.antenna_pos[2]))
                    E_ref = np.abs(u[0, pi, zi_ant]) or 1e-30
                    TL = -20*np.log10(np.maximum(u_slice/E_ref, 1e-15))
                    for ri2 in range(len(r)): TL[ri2,:] += 10*np.log10(max(r[ri2]/r0, 1.0))
                    TL = np.nan_to_num(TL, nan=200, posinf=200, neginf=0)
                    # 地形剖面 — 优先用求解器的 z_top（含障碍物+clip图层）
                    terrain_z_profile = self._get_terrain_profile(r, phi[pi], cfg, d)
                    fig = self._contour(R,Z,TL,"TL (dB)",f"r-z TL分布 φ={phi_deg.value():.0f}°","jet",cfg)
                    # 叠加地形线
                    ax = fig.axes[0]
                    if terrain_z_profile is not None:
                        ax.plot(r, terrain_z_profile, 'w-', linewidth=2, zorder=10)
                        ax.fill_between(r, 0, terrain_z_profile, color='black', alpha=0.5, zorder=1)
                    figs.append((f"r-z TL分布 φ={phi_deg.value():.0f}°", fig))
                else:
                    E_db = 20*np.log10(np.maximum(u_slice,1e-15))
                    E_db = np.nan_to_num(E_db, nan=-200, neginf=-200)
                    figs.append((f"r-z 电场分布 φ={phi_deg.value():.0f}°", self._contour(R,Z,E_db,"|E| (dB)",f"r-z 电场分布 φ={phi_deg.value():.0f}°","hot",cfg)))

            elif key == "phiz_tl":
                r_fix = pw.get("r_fix")
                if r_fix is None: continue
                ri = np.argmin(np.abs(r - r_fix.value()))
                u_slice = np.abs(u[ri,:,:])  # (nphi, nz)
                # 参考：r0 处天线高度 φ=0
                zi_ant = np.argmin(np.abs(z - cfg.antenna_pos[2]))
                E_ref = np.abs(u[0, 0, zi_ant]) or 1e-30
                TL = -20*np.log10(np.maximum(u_slice/E_ref, 1e-15))
                TL += 10*np.log10(max(r[ri]/r0, 1.0))
                TL = np.nan_to_num(TL, nan=200, posinf=200, neginf=0)
                # phi: 0-360 → -90..90 convention
                phi_deg = np.degrees(phi)
                phi_deg = np.where(phi_deg > 180, phi_deg - 360, phi_deg)
                mask = (phi_deg >= -90) & (phi_deg <= 90)
                phi_deg = phi_deg[mask]; TL = TL[mask, :]
                # 重排序：phi必须单调递增，pcolormesh才能正确渲染
                sort_idx = np.argsort(phi_deg)
                phi_deg = phi_deg[sort_idx]; TL = TL[sort_idx, :]
                P, Zp = np.meshgrid(phi_deg, z, indexing="ij")
                figs.append((f"φ-z TL分布 r={r[ri]:.1f}m", self._contour(P,Zp,TL,"TL (dB)",f"φ-z TL分布 r={r[ri]:.1f}m","jet",cfg)))

            elif key == "tl_vs_r":
                z_fix = pw.get("z_fix"); 
                if z_fix is None: continue
                zi = np.argmin(np.abs(z - z_fix.value()))
                u_line = np.abs(u[:,0,zi]); u_ref = u_line[0] if u_line[0]>0 else 1e-30
                TL = -20*np.log10(np.maximum(u_line/u_ref,1e-15))
                for ri in range(len(r)): TL[ri] += 10*np.log10(max(r[ri]/r0,1.0))
                title = f"路径损耗 vs 距离 z={z[zi]:.1f}m"
                fig, ax = plt.subplots(figsize=(9,5))
                ax.plot(r, TL, "b-", linewidth=1.5); ax.set_xlabel("距离 r (m)"); ax.set_ylabel("TL (dB)")
                ax.set_title(title); ax.grid(True, alpha=0.3); fig.tight_layout()
                figs.append((title, fig))

            elif key == "e_vs_z":
                r_fix = pw.get("r_fix"); 
                if r_fix is None: continue
                ri = np.argmin(np.abs(r - r_fix.value()))
                E_db = 20*np.log10(np.maximum(np.abs(u[ri,0,:]),1e-15))
                E_db = np.nan_to_num(E_db, nan=-200, neginf=-200)
                title = f"场强 vs 高度 r={r[ri]:.1f}m"
                fig, ax = plt.subplots(figsize=(6,6))
                ax.plot(E_db, z, "r-", linewidth=1.5); ax.set_xlabel("|E| (dB)"); ax.set_ylabel("高度 z (m)")
                ax.set_title(title); ax.grid(True, alpha=0.3); fig.tight_layout()
                figs.append((title, fig))

            elif key == "tl_3d":
                pi0=0; u_slice=np.abs(u[:,pi0,:])
                E_ref = np.abs(u[0, 0, zi_ant]) or 1e-30
                TL=-20*np.log10(np.maximum(u_slice/E_ref,1e-15))
                for ri in range(len(r)): TL[ri,:]+=10*np.log10(max(r[ri]/r0,1.0))
                R,Z=np.meshgrid(r,z,indexing="ij");title="TL 3D 表面"
                from mpl_toolkits.mplot3d import Axes3D
                fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection="3d")
                sr=max(1,len(r)//80);sz=max(1,len(z)//80)
                surf=ax.plot_surface(R[::sr,::sz],Z[::sr,::sz],TL[::sr,::sz],cmap="jet",alpha=0.85,linewidth=0)
                fig.colorbar(surf,ax=ax,label="TL (dB)",shrink=0.6)
                ax.set_xlabel("r (m)");ax.set_ylabel("z (m)");ax.set_zlabel("TL (dB)");ax.set_title(title);fig.tight_layout()
                figs.append((title,fig))

        if not figs:
            QtWidgets.QMessageBox.information(self, "提示", "请至少勾选一种图类型")
            return

        # 在新独立窗口显示
        self._viewer = FigureViewer(figs, self, self._data)
        self._viewer.show()

    def _get_terrain_profile(self, r_vals, phi_rad, cfg, data=None):
        # 优先用求解器的 z_top（含障碍物+clip图层高度）
        if data is not None:
            z_top = data.get("z_top")
            if z_top is not None:
                phi_idx = np.argmin(np.abs(data["phi_vals"] - phi_rad))
                if z_top.ndim == 2:
                    return z_top[phi_idx, :]
                return z_top
        # 回退到 terrain mesh 采样
        scene = self._data.get("scene")
        if scene is None: return None
        terrain = scene.get("terrain")
        if terrain is None: return None
        extra = terrain.get("extra") or {}
        Z = extra.get("original_z"); X = extra.get("X"); Y = extra.get("Y")
        if Z is None or X is None: return None
        xs = X[0, :]; ys = Y[:, 0]
        ant = cfg.antenna_pos
        c, s = np.cos(phi_rad), np.sin(phi_rad)
        profile = np.empty(len(r_vals))
        for i, rv in enumerate(r_vals):
            xp = ant[0] + rv * c; yp = ant[1] + rv * s
            xp = max(xs[0], min(xp, xs[-1]))
            yp = max(ys[0], min(yp, ys[-1]))
            ix = max(0, min(np.searchsorted(xs, xp) - 1, len(xs) - 2))
            iy = max(0, min(np.searchsorted(ys, yp) - 1, len(ys) - 2))
            x1, x2 = xs[ix], xs[ix + 1]; y1, y2 = ys[iy], ys[iy + 1]
            dx = (xp - x1) / (x2 - x1) if x2 != x1 else 0.5
            dy = (yp - y1) / (y2 - y1) if y2 != y1 else 0.5
            profile[i] = (Z[iy, ix]*(1-dx)*(1-dy) + Z[iy, ix+1]*dx*(1-dy) +
                          Z[iy+1, ix]*(1-dx)*dy + Z[iy+1, ix+1]*dx*dy)
        return profile

    def _contour(self, X, Y, data, cbar_label, title, cmap, cfg):
        fig, ax = plt.subplots(figsize=(10, 6))
        data = np.nan_to_num(data, nan=0.0, posinf=300, neginf=0)
        # 画图裁剪 z≤250
        z_mask = Y[0, :] <= 250
        X, Y, data = X[:, z_mask], Y[:, z_mask], data[:, z_mask]
        # 固定色阶范围
        finite = data[np.isfinite(data) & (data < 300)]
        if len(finite) < 10:
            fig.tight_layout(); return fig

        if "TL" in cbar_label or "损耗" in title:
            # TL: 0~200 dB, 0=自由空间, 200=深度阴影
            vmin = max(float(finite.min()), -10)
            vmax = min(float(finite.max()), 220)
            if vmax - vmin < 10:
                vmax = vmin + 40
        else:
            # 电场: 以最大值为参考，向下60dB
            vmax = float(finite.max())
            vmin = max(vmax - 80, -150)
            if vmax - vmin < 10:
                vmin = vmax - 40

        # colormap: TL蓝→红(200dB深红), 电场inferno
        if "TL" in cbar_label or "损耗" in title:
            _cmap = plt.cm.jet
        else:
            _cmap = plt.cm.inferno

        # pcolormesh + Gouraud 平滑 → 无锯齿、无漏白
        cf = ax.pcolormesh(X, Y, data, shading="gouraud", cmap=_cmap,
                           vmin=vmin, vmax=vmax, zorder=5, rasterized=True)
        cbar = fig.colorbar(cf, ax=ax, label=cbar_label, extend="both")
        # 判断轴类型：有负值→φ轴，全是正→r轴
        is_phi = (X.min() < 0)
        ax.set_xlabel("φ (°)" if is_phi else "r (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(title)
        if cfg.antenna_pos[2] <= Y.max():
            ax.plot(0, cfg.antenna_pos[2], "w*", markersize=10, markeredgecolor="k", markeredgewidth=0.5)
        fig.tight_layout()
        return fig


class FigureViewer(QtWidgets.QDialog):
    """独立大窗口，标签页显示生成的图。"""

    def __init__(self, figures, parent=None, field_data=None):
        super().__init__(parent)
        self.setWindowTitle("绘图结果")
        self.resize(1000, 700)
        self._figures = figures

        layout = QtWidgets.QVBoxLayout(self)

        # 范围说明
        if field_data:
            d = field_data
            r0, r1 = d["r_vals"][0], d["r_vals"][-1]
            info = QtWidgets.QLabel(
                f"绘制范围：径向 r ∈ [{r0:.1f}, {r1:.1f}] m  |  "
                f"高度 z ∈ [0, {d['z_vals'][-1]:.1f}] m  |  "
                f"频率 {d['config'].frequency/1e9:.1f} GHz\n"
                f"注：最远测量点决定径向范围；地面以下及天顶吸收层无有效数据"
            )
            info.setStyleSheet("color: #666; padding: 2px 6px; font-size: 11px;")
            layout.addWidget(info)
        tabs = QtWidgets.QTabWidget()
        for title, fig in figures:
            w = QtWidgets.QWidget()
            lo = QtWidgets.QVBoxLayout(w)
            canvas = FigureCanvasQTAgg(fig)
            lo.addWidget(canvas)
            tabs.addTab(w, title)
        layout.addWidget(tabs, 1)

        btn = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("导出当前图…")
        btn_export.clicked.connect(lambda: self._export(tabs.currentIndex()))
        btn.addWidget(btn_export)
        btn_export_all = QtWidgets.QPushButton("导出全部…")
        btn_export_all.clicked.connect(self._export_all)
        btn.addWidget(btn_export_all)
        btn.addStretch()
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn.addWidget(btn_close)
        layout.addLayout(btn)
        self._tabs = tabs

    def _export(self, idx):
        if idx < 0 or idx >= len(self._figures): return
        title, fig = self._figures[idx]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"导出 — {title}", f"{title}.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path: fig.savefig(path, dpi=150, bbox_inches="tight")

    def _export_all(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not folder: return
        import os
        for i, (title, fig) in enumerate(self._figures):
            fig.savefig(os.path.join(folder, f"{i+1:02d}_{title}.png"), dpi=150, bbox_inches="tight")
