"""
Field solver dialogs — XY plane point picker + results table.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class FieldPointDialog(QtWidgets.QDialog):
    """XY-plane dialog for selecting receiver points."""

    def __init__(self, parent, scene, antenna_pos, frequency):
        super().__init__(parent)
        self.setWindowTitle("求解电场 — 选择接收点坐标")
        self.setMinimumSize(750, 600)

        self.scene = scene
        self.tx = np.asarray(antenna_pos, dtype=float)
        self.freq = frequency
        self.points = []          # list of (x, y, z) world coords
        self._z_default = 3.0     # default receiver height

        self._build_ui()
        self._draw_scene()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # ── Instructions ──
        instr = QtWidgets.QLabel(
            f"天线位置: ({self.tx[0]:.0f}, {self.tx[1]:.0f}, {self.tx[2]:.0f})  |  "
            f"频率: {self.freq/1e9:.1f} GHz\n"
            "左键点击 XY 平面添加接收点  |  右键删除最近一点  |  滚轮缩放"
        )
        instr.setStyleSheet("font-size: 12px; padding: 6px;")
        layout.addWidget(instr)

        # ── Canvas ──
        self._fig = Figure(figsize=(6, 5))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        layout.addWidget(self._canvas, 1)

        # ── Z-height spinbox ──
        z_row = QtWidgets.QHBoxLayout()
        z_row.addWidget(QtWidgets.QLabel("接收点高度 Z:"))
        self._z_spin = QtWidgets.QDoubleSpinBox()
        self._z_spin.setRange(-10, 200)
        self._z_spin.setValue(self._z_default)
        self._z_spin.setDecimals(3)
        self._z_spin.valueChanged.connect(self._on_z_changed)
        z_row.addWidget(self._z_spin)
        z_row.addStretch()
        self._point_count = QtWidgets.QLabel("已选: 0 个点")
        z_row.addWidget(self._point_count)
        layout.addLayout(z_row)

        # ── Buttons ──
        btn_row = QtWidgets.QHBoxLayout()
        btn_undo = QtWidgets.QPushButton("撤销上一点")
        btn_undo.clicked.connect(self._undo)
        btn_row.addWidget(btn_undo)
        btn_clear = QtWidgets.QPushButton("清除全部")
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        self._btn_ok = QtWidgets.QPushButton("确认计算 ✓")
        self._btn_ok.clicked.connect(self.accept)
        self._btn_ok.setEnabled(False)
        self._btn_ok.setStyleSheet(
            "QPushButton { font-weight: bold; background: #4CAF50; color: white; "
            "padding: 6px 16px; } QPushButton:disabled { background: #ccc; }"
        )
        btn_row.addWidget(self._btn_ok)
        layout.addLayout(btn_row)

    # ── Drawing ────────────────────────────────────────

    def _draw_scene(self):
        self._ax.clear()

        # Terrain bounds from scene
        t = self.scene.get("terrain")
        if t is not None:
            b = t["mesh"].bounds
            xlim = max(abs(b[0]), abs(b[1])) * 1.1
            ylim = max(abs(b[2]), abs(b[3])) * 1.1
        else:
            xlim = ylim = 12

        self._ax.set_xlim(-xlim, xlim)
        self._ax.set_ylim(-ylim, ylim)
        self._ax.set_aspect("equal")
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.grid(True, linestyle="--", alpha=0.4)
        self._ax.axhline(0, color="gray", linewidth=0.5)
        self._ax.axvline(0, color="gray", linewidth=0.5)

        # Draw terrain elevation contours
        if t is not None:
            try:
                mesh = t["mesh"]
                pts = np.asarray(mesh.points)
                if len(pts) >= 50:
                    if len(pts) > 8000:
                        step = max(1, len(pts) // 8000)
                        pts = pts[::step]
                    triang = tri.Triangulation(pts[:, 0], pts[:, 1])
                    z = pts[:, 2]
                    z_min, z_max = z.min(), z.max()
                    n_levels = 8 if len(pts) > 2000 else 6
                    if z_max - z_min < 0.001:
                        # 平坦地形：画单条等高线
                        self._ax.tricontour(triang, z, levels=[z_min],
                                            colors="gray", linewidths=0.4, alpha=0.35, zorder=0)
                    else:
                        levels = np.linspace(z_min, z_max, n_levels)
                        self._ax.tricontour(triang, z, levels=levels,
                                            colors="gray", linewidths=0.4, alpha=0.35, zorder=0)
                        try:
                            self._ax.tricontourf(triang, z, levels=levels,
                                                 cmap="terrain", alpha=0.15, zorder=0)
                        except Exception:
                            pass
            except Exception:
                pass

        # Antenna marker
        self._ax.plot(self.tx[0], self.tx[1], "o", color="cyan",
                      markersize=12, markeredgecolor="black",
                      markeredgewidth=1.5, zorder=5)
        self._ax.annotate("Tx", (self.tx[0], self.tx[1]),
                          textcoords="offset points", xytext=(8, 8),
                          fontsize=11, fontweight="bold", color="cyan")

        # Draw selected points
        for i, (px, py, pz) in enumerate(self.points):
            self._ax.plot(px, py, "o", color="#FF5722", markersize=8, zorder=4)
            self._ax.annotate(str(i+1), (px, py),
                              textcoords="offset points", xytext=(6, 6),
                              fontsize=9, fontweight="bold", color="#FF5722")

        self._canvas.draw_idle()

    # ── Interaction ────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return

        if event.button == 3:  # right-click → undo
            self._undo()
            return

        if event.button != 1:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        z = self._z_spin.value()
        self.points.append((float(x), float(y), float(z)))
        self._update_ui()
        self._draw_scene()

    def _on_z_changed(self, val):
        self._z_default = val
        # Update Z of all existing points
        if self.points:
            self.points = [(px, py, val) for (px, py, _) in self.points]

    def _undo(self):
        if self.points:
            self.points.pop()
        self._update_ui()
        self._draw_scene()

    def _clear(self):
        self.points.clear()
        self._update_ui()
        self._draw_scene()

    def _update_ui(self):
        n = len(self.points)
        self._point_count.setText(f"已选: {n} 个点")
        self._btn_ok.setEnabled(n > 0)

    def get_points(self):
        return list(self.points)


# ═══════════════════════════════════════════════════════════════
#  Results dialog
# ═══════════════════════════════════════════════════════════════

class FieldResultDialog(QtWidgets.QDialog):
    """Display field strength results in a table."""

    def __init__(self, parent, results, antenna_pos, frequency):
        super().__init__(parent)
        self.setWindowTitle("电场求解结果")
        self.setMinimumSize(700, 400)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            f"天线: ({antenna_pos[0]:.0f}, {antenna_pos[1]:.0f}, {antenna_pos[2]:.0f}) m  |  "
            f"频率: {frequency/1e9:.1f} GHz  |  "
            f"共 {len(results)} 个接收点"
        )
        info.setStyleSheet("font-size: 12px; padding: 6px;")
        layout.addWidget(info)

        # Table
        table = QtWidgets.QTableWidget()
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels([
            "#", "X (m)", "Y (m)", "Z (m)", "距离 (m)",
            "E (dBμV/m)", "L_fs (dB)", "L_pe (dB)"
        ])
        table.setRowCount(len(results))

        for i, r in enumerate(results):
            E_val = r.get("E_rx_dbuv", 0.0)
            items = [
                str(i+1),
                f"{r['rx'][0]:.3f}", f"{r['rx'][1]:.3f}", f"{r['rx'][2]:.3f}",
                f"{r['dist']:.3f}",
                f"{E_val:.3f}",
                f"{r['L_fs_dB']:.3f}", f"{r['path_loss_dB']:.3f}",
            ]
            for j, text in enumerate(items):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if j == 5:      # E field — highlight
                    item.setBackground(QtGui.QColor("#e3f2fd"))
                    item.setForeground(QtGui.QColor("#1565C0"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                elif j == 6:
                    item.setBackground(QtGui.QColor("#e8f5e9"))
                elif j == 7:
                    item.setBackground(QtGui.QColor("#fff3e0"))
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        # Close button
        btn = QtWidgets.QPushButton("关闭")
        btn.clicked.connect(self.accept)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)


# ═══════════════════════════════════════════════════════════════
#  Precise measurement point dialog (two-step: XY → Z)
# ═══════════════════════════════════════════════════════════════

class PreciseRxDialog(QtWidgets.QDialog):
    """Two-step wizard: pick XY points → adjust Z heights."""

    def __init__(self, parent, scene, antenna_pos, frequency):
        super().__init__(parent)
        self.setWindowTitle("精准添加测量点")
        self.setMinimumSize(750, 560)
        self.scene = scene; self.tx = np.asarray(antenna_pos, dtype=float)
        self.freq = frequency
        self._xy_points = []    # list of (x, y)
        self._z_values = []     # list of z (same length)
        self._selected_idx = -1

        layout = QtWidgets.QVBoxLayout(self)
        self._stack = QtWidgets.QStackedWidget()
        layout.addWidget(self._stack)
        self._stack.addWidget(self._build_page1())
        self._stack.addWidget(self._build_page2())
        self._stack.setCurrentIndex(0)
        self._draw_xy()   # ensure proper axis range on init

    def _build_page1(self):
        w = QtWidgets.QWidget(); lo = QtWidgets.QVBoxLayout(w)
        lo.addWidget(QtWidgets.QLabel(
            "步骤 1/2：在 XY 平面上点击添加测量点\n"
            "左键添加  |  右键撤销  |  至少添加 1 个点"))
        self._fig1 = Figure(figsize=(5, 4.5))
        self._ax1 = self._fig1.add_subplot(111)
        self._canvas1 = FigureCanvasQTAgg(self._fig1)
        self._canvas1.mpl_connect("button_press_event", self._on_xy_click)
        lo.addWidget(self._canvas1, 1)

        # Point table
        self._pt_table = QtWidgets.QTableWidget()
        self._pt_table.setColumnCount(3)
        self._pt_table.setHorizontalHeaderLabels(["#", "X", "Y"])
        self._pt_table.setMaximumHeight(120)
        self._pt_table.itemChanged.connect(self._on_table_edit)
        lo.addWidget(self._pt_table)

        # Manual input row
        manual_row = QtWidgets.QHBoxLayout()
        manual_row.addWidget(QtWidgets.QLabel("X:"))
        self._spin_x = QtWidgets.QDoubleSpinBox(); self._spin_x.setRange(-100,100); self._spin_x.setDecimals(3)
        manual_row.addWidget(self._spin_x)
        manual_row.addWidget(QtWidgets.QLabel("Y:"))
        self._spin_y = QtWidgets.QDoubleSpinBox(); self._spin_y.setRange(-100,100); self._spin_y.setDecimals(3)
        manual_row.addWidget(self._spin_y)
        btn_add_manual = QtWidgets.QPushButton("添加坐标"); btn_add_manual.clicked.connect(self._add_manual_xy)
        manual_row.addWidget(btn_add_manual); manual_row.addStretch()
        lo.addLayout(manual_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_undo = QtWidgets.QPushButton("撤销"); btn_undo.clicked.connect(self._undo_xy)
        btn_row.addWidget(btn_undo)
        btn_clear = QtWidgets.QPushButton("清除"); btn_clear.clicked.connect(self._clear_xy)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        btn_cancel = QtWidgets.QPushButton("取消"); btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        self._btn_next = QtWidgets.QPushButton("下一步 →")
        self._btn_next.clicked.connect(self._go_step2); self._btn_next.setEnabled(False)
        btn_row.addWidget(self._btn_next)
        lo.addLayout(btn_row)
        return w

    def _build_page2(self):
        w = QtWidgets.QWidget(); lo = QtWidgets.QVBoxLayout(w)
        lo.addWidget(QtWidgets.QLabel("步骤 2/2：调整每个点的高度 Z"))
        self._fig2 = Figure(figsize=(5, 3))
        self._ax2 = self._fig2.add_subplot(111)
        self._canvas2 = FigureCanvasQTAgg(self._fig2)
        self._canvas2.mpl_connect("button_press_event", self._on_z_click)
        lo.addWidget(self._canvas2, 1)

        z_ctrl = QtWidgets.QHBoxLayout()
        z_ctrl.addWidget(QtWidgets.QLabel("选中点 Z:"))
        self._z_spin = QtWidgets.QDoubleSpinBox()
        self._z_spin.setRange(-10, 50); self._z_spin.setDecimals(3)
        self._z_spin.valueChanged.connect(self._on_z_spin)
        z_ctrl.addWidget(self._z_spin)
        self._z_info = QtWidgets.QLabel("点击图表选择点")
        z_ctrl.addWidget(self._z_info); z_ctrl.addStretch()
        lo.addLayout(z_ctrl)

        btn_row = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("← 上一步"); btn_prev.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        btn_row.addWidget(btn_prev); btn_row.addStretch()
        btn_ok = QtWidgets.QPushButton("确认 ✓")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet("QPushButton { font-weight: bold; background: #4CAF50; color: white; padding: 4px 16px; }")
        btn_row.addWidget(btn_ok)
        lo.addLayout(btn_row)
        return w

    def _draw_xy(self):
        self._ax1.clear()
        t = self.scene.get("terrain")
        if t is not None:
            b = t["mesh"].bounds; lim = max(abs(b[0]),abs(b[1]),abs(b[2]),abs(b[3]))*1.1
        else: lim = 12
        self._ax1.set_xlim(-lim, lim); self._ax1.set_ylim(-lim, lim)
        self._ax1.set_aspect("equal"); self._ax1.grid(True, alpha=0.3)
        self._ax1.plot(self.tx[0], self.tx[1], "o", color="cyan", markersize=10)

        # 地形等高线
        if t is not None:
            try:
                mesh = t["mesh"]
                pts = np.asarray(mesh.points)
                if len(pts) >= 50:
                    step = max(1, len(pts) // 8000)
                    sp = pts[::step]
                    triang = tri.Triangulation(sp[:, 0], sp[:, 1])
                    z_vals = sp[:, 2]
                    zmin, zmax = z_vals.min(), z_vals.max()
                    nlv = 8 if len(sp) > 2000 else 6
                    if zmax - zmin < 0.001:
                        self._ax1.tricontour(triang, z_vals, levels=[zmin],
                                             colors="gray", linewidths=0.4, alpha=0.35, zorder=0)
                    else:
                        levels = np.linspace(zmin, zmax, nlv)
                        self._ax1.tricontour(triang, z_vals, levels=levels,
                                             colors="gray", linewidths=0.4, alpha=0.35, zorder=0)
                        try:
                            self._ax1.tricontourf(triang, z_vals, levels=levels,
                                                  cmap="terrain", alpha=0.12, zorder=0)
                        except Exception:
                            pass
            except Exception:
                pass

        # 测量点
        xys = np.array(self._xy_points) if self._xy_points else np.empty((0,2))
        if len(xys)>0:
            self._ax1.scatter(xys[:,0], xys[:,1], c="orange", s=30, zorder=5)
            for i,(x,y) in enumerate(self._xy_points):
                self._ax1.annotate(str(i+1),(x,y),textcoords="offset points",xytext=(5,5),fontsize=9)
        self._canvas1.draw_idle()

    def _draw_z(self):
        self._ax2.clear()
        if not self._xy_points: self._canvas2.draw_idle(); return
        pts = np.array(self._xy_points)
        dists = [0.0]
        for i in range(1, len(pts)):
            dists.append(dists[-1]+np.hypot(pts[i,0]-pts[i-1,0],pts[i,1]-pts[i-1,1]))
        zs = self._z_values
        self._ax2.plot(dists, zs, "o-", color="#2196F3", markersize=8)
        for i,(d,z) in enumerate(zip(dists,zs)):
            self._ax2.annotate(str(i+1),(d,z),textcoords="offset points",xytext=(0,8),fontsize=10)
        if 0<=self._selected_idx<len(dists):
            self._ax2.plot(dists[self._selected_idx],zs[self._selected_idx],"o",color="red",markersize=12)
        self._ax2.set_xlabel("累积距离 (m)"); self._ax2.set_ylabel("Z (m)")
        self._ax2.set_ylim(-5, 50)
        self._ax2.grid(True, alpha=0.3)
        self._canvas2.draw_idle()

    def _add_manual_xy(self):
        self._xy_points.append((self._spin_x.value(), self._spin_y.value()))
        self._z_values.append(3.0)
        self._update_xy_table(); self._btn_next.setEnabled(True); self._draw_xy()

    def _on_table_edit(self, item):
        row, col = item.row(), item.column()
        try:
            val = float(item.text())
        except ValueError:
            return
        if row < len(self._xy_points):
            x, y = self._xy_points[row]
            if col == 1: self._xy_points[row] = (val, y)
            elif col == 2: self._xy_points[row] = (x, val)
        self._draw_xy()

    def _on_xy_click(self, event):
        if event.inaxes!=self._ax1: return
        if event.button==3: self._undo_xy(); return
        if event.button!=1: return
        x,y=event.xdata,event.ydata
        if x is None: return
        self._xy_points.append((float(x),float(y)))
        self._z_values.append(3.0)
        self._update_xy_table(); self._btn_next.setEnabled(True); self._draw_xy()

    def _undo_xy(self):
        if self._xy_points: self._xy_points.pop(); self._z_values.pop()
        self._update_xy_table(); self._btn_next.setEnabled(len(self._xy_points)>0); self._draw_xy()

    def _clear_xy(self):
        self._xy_points.clear(); self._z_values.clear()
        self._update_xy_table(); self._btn_next.setEnabled(False); self._draw_xy()

    def _update_xy_table(self):
        self._pt_table.setRowCount(len(self._xy_points))
        for i,(x,y) in enumerate(self._xy_points):
            for j,v in enumerate([str(i+1),f"{x:.3f}",f"{y:.3f}"]):
                self._pt_table.setItem(i,j,QtWidgets.QTableWidgetItem(v))

    def _go_step2(self):
        if not self._xy_points: return
        self._selected_idx=0; self._z_spin.setValue(self._z_values[0])
        self._stack.setCurrentIndex(1); self._draw_z()

    def _on_z_click(self, event):
        if event.inaxes!=self._ax2 or not self._xy_points: return
        pts=np.array(self._xy_points)
        dists=[0.0]
        for i in range(1,len(pts)): dists.append(dists[-1]+np.hypot(pts[i,0]-pts[i-1,0],pts[i,1]-pts[i-1,1]))
        xd=event.xdata; idx=np.argmin(np.abs(np.array(dists)-xd)) if xd is not None else 0
        self._selected_idx=idx
        # Set Z from click Y position
        if event.ydata is not None:
            z_val = max(-10, min(50, event.ydata))
            self._z_values[idx] = z_val
            self._z_spin.blockSignals(True)
            self._z_spin.setValue(z_val)
            self._z_spin.blockSignals(False)
        else:
            self._z_spin.setValue(self._z_values[idx])
        self._z_info.setText(f"点 {idx+1}: 距离 {dists[idx]:.1f}m"); self._draw_z()

    def _on_z_spin(self, val):
        if 0<=self._selected_idx<len(self._z_values):
            self._z_values[self._selected_idx]=val; self._draw_z()

    def get_points(self):
        return [(x, y, z) for (x,y),z in zip(self._xy_points, self._z_values)]
