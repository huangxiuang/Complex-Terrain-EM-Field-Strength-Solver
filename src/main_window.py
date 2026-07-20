"""
MainWindow — simplified 3D scene viewer with CPE field solver menu.
"""

import sys
import os
import copy
# Make 'src' package importable regardless of how this file is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyvista as pv
from PyQt5 import QtWidgets, QtCore
import pyvistaqt as pvqt
from vtkmodules.vtkRenderingCore import vtkActor

from src.simple_scene_builder import build_simple_scene
from src.core.config import EMConfig
from src.core.scheduler import Scheduler
from src.field_dialog import FieldPointDialog, FieldResultDialog, PreciseRxDialog
from src.layer_dialog import LayerManagementDialog, ClipManagerDialog
from src.material_params_dialog import MaterialParamsDialog, LAYER_MATERIAL_MAP
from src.antenna_dialog import AntennaDialog
from src.sweep_dialog import SweepDialog
from src.scene.scenes import SCENE_REGISTRY, build_metal_barrier
from src.scene.scene_dialog import SceneSelectDialog, ScenePropsDialog
from src.plot_dialog import PlotDialog
from src.template_dialog import TemplateDialog
from src import io_utils

import matplotlib.path as mpath


class SolveWorker(QtCore.QObject):
    """后台求解 worker：在 QThread 中执行 prep → solver → post。"""

    progress = QtCore.pyqtSignal(str, int, int)   # stage, done, total
    finished = QtCore.pyqtSignal(object)          # Context
    failed = QtCore.pyqtSignal(str)               # 错误信息 / "__cancelled__"

    CANCELLED = "__cancelled__"

    def __init__(self, config, rx_points, scene):
        super().__init__()
        self._config = config
        self._rx_points = rx_points
        self._scene = scene
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @QtCore.pyqtSlot()
    def run(self):
        from src.core.context import Context
        from src.engine import prep, solver as pe_solver, post
        try:
            ctx = Context(config=self._config, rx_points=self._rx_points,
                          scene=self._scene)
            ctx.progress_cb = lambda s, d, t: self.progress.emit(s, d, t)
            ctx.cancel_cb = lambda: self._cancelled
            ctx = prep.run(ctx)
            ctx = pe_solver.run(ctx)
            if self._cancelled:
                self.failed.emit(self.CANCELLED)
                return
            ctx = post.run(ctx)
            self.finished.emit(ctx)
        except pe_solver.SolveCancelled:
            self.failed.emit(self.CANCELLED)
        except MemoryError:
            self.failed.emit("内存不足：网格过大。请减小 N_Z/N_PHI 或缩短计算距离。")
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D EM Solver — CPE 场强计算")
        self.resize(1100, 720)

        # Scene objects
        self.scene_objects = {}
        self.plotter_actors = {}
        self._pending_rx_points = []
        self._current_scene_key = "metal_barrier"
        self._last_field_data = None
        self._last_results = None      # 最近一次求解结果
        self._tree_results_root = None  # Results 树节点引用（避免按文本查找）
        self._solve_thread = None      # 求解后台线程（非 None 表示求解中）
        self._solve_worker = None
        self._solve_progress = None

        # ── 左侧树形面板 ──
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(200)
        self._tree.setMaximumWidth(320)
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._tree.itemDoubleClicked.connect(self._on_tree_double_click)

        self._tree_rx_root = QtWidgets.QTreeWidgetItem(["📁 测量点"])
        self._tree_func_root = QtWidgets.QTreeWidgetItem(["⚙ 功能"])
        self._tree.addTopLevelItem(self._tree_rx_root)
        self._tree.addTopLevelItem(self._tree_func_root)
        self._tree_rx_root.setExpanded(True)
        self._tree_func_root.setExpanded(True)

        for label, role in [("➕ 添加", "add_rx"), ("⚡ 求解", "solve"), ("📊 画图", "plot")]:
            item = QtWidgets.QTreeWidgetItem([label])
            item.setData(0, QtCore.Qt.UserRole, role)
            self._tree_func_root.addChild(item)

        dock = QtWidgets.QDockWidget("导航", self)
        dock.setWidget(self._tree)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        # ── Central 3D viewport ──
        self.plotter = pvqt.QtInteractor(self)
        self.setCentralWidget(self.plotter)

        # ── Build scene ──
        self._init_scene()

        # ── Menus ──
        self._setup_menus()

        # ── Status bar ──
        self._status_summary_label = QtWidgets.QLabel("")
        self._status_summary_label.setStyleSheet("color: #555; padding-right: 8px;")
        self.statusBar().addPermanentWidget(self._status_summary_label)
        self.statusBar().showMessage(
            "就绪  |  左键旋转 · 滚轮缩放 · 中键平移  |  工具 → 求解电场功能"
        )
        self._update_rx_tree()

        self.show()

    # ═══════════════════════════════════════════════════════════
    #  Scene
    # ═══════════════════════════════════════════════════════════

    def _init_scene(self):
        p = self.plotter
        p.background_color = (0.82, 0.90, 1.0)

        p.add_light(pv.Light(position=(10, -10, 15), intensity=0.8))
        p.add_light(pv.Light(position=(-5, 5, 8), intensity=0.4))

        p.show_axes()
        p.show_grid()

        self.scene_objects = SCENE_REGISTRY[self._current_scene_key]["builder"]()
        # Ensure all materials have defaults (safe for missing user input)
        self._load_material_defaults()
        for name, obj in self.scene_objects.items():
            if obj.get("visible", True):
                self._add_actor(name, obj)

        self._reset_camera()
        p.render()

    def _add_actor(self, name, obj):
        mesh = obj["mesh"]
        params = obj["params"]
        if obj["type"] == "points":
            actor = self.plotter.add_points(mesh, **params)
        else:
            actor = self.plotter.add_mesh(mesh, **params)
        self.plotter_actors[name] = actor

    # ── 场景尺度自适应工具 ──

    def _scene_extent(self):
        """返回 (center, diag)：terrain bounds 中心与对角线长度。"""
        t = self.scene_objects.get("terrain")
        if t is None:
            return (0.0, 0.0, 0.0), 30.0
        b = t["mesh"].bounds
        center = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
        diag = float(np.hypot(np.hypot(b[1] - b[0], b[3] - b[2]),
                              max(b[5] - b[4], 1.0)))
        return center, max(diag, 10.0)

    def _reset_camera(self):
        center, diag = self._scene_extent()
        self.plotter.camera_position = [
            (center[0] + 0.8 * diag, center[1] - 0.6 * diag, center[2] + 0.45 * diag),
            center, (0, 0, 1),
        ]
        self.plotter.camera.clipping_range = (diag / 500.0, diag * 5.0)

    def _marker_scale(self):
        """测量点标记尺寸（随场景尺度缩放，保证大场景下可见）。"""
        _, diag = self._scene_extent()
        return float(np.clip(diag * 0.005, 0.15, 10.0))

    # ═══════════════════════════════════════════════════════════
    #  Menus
    # ═══════════════════════════════════════════════════════════

    def _setup_menus(self):
        mb = self.menuBar()

        # ── Scene menu ──
        menu_scene = mb.addMenu("场景 (&S)")
        menu_scene.addAction("场景选择…", self._on_select_scene)
        menu_scene.addAction("场景属性…", self._on_scene_props)

        # ── Template menu ──
        menu_preset = mb.addMenu("预设 (&T)")
        menu_preset.addAction("模版测量点…", self._on_template)

        # ── View menu ──
        menu_view = mb.addMenu("视图 (&V)")
        menu_view.addAction("俯视", lambda: self._set_view("top"))
        menu_view.addAction("正视", lambda: self._set_view("front"))
        menu_view.addAction("侧视", lambda: self._set_view("side"))
        menu_view.addAction("复位", lambda: self._set_view("reset"))
        menu_view.addSeparator()
        menu_view.addAction("截图保存…", self._screenshot)

        # ── Layer menu ──
        menu_layer = mb.addMenu("图层 (&L)")
        menu_layer.addAction("增加图层…", self._open_layer_dialog)
        menu_layer.addAction("管理裁剪图层…", self._open_clip_manager)

        # ── Import/Export menu ──
        menu_io = mb.addMenu("导入/导出 (&I)")
        menu_io.addAction("导入 DEM / GeoTIFF…", self._on_import_dem)
        menu_io.addAction("导入 ASC 高程栅格…", self._on_import_asc)
        menu_io.addSeparator()
        menu_io.addAction("导出地形为 ASC…", self._on_export_asc)

        # ── Parameters menu ──
        menu_params = mb.addMenu("参数 (&P)")
        menu_params.addAction("材料参数设置…", self._open_material_params)
        menu_params.addAction("天线设置…", self._open_antenna_dialog)
        menu_tools = mb.addMenu("工具 (&T)")
        action_sweep = menu_tools.addAction("扫频模式… (&S)")
        action_sweep.setShortcut("Ctrl+S")
        action_sweep.triggered.connect(self._on_sweep)
        menu_tools.addSeparator()
        action_quick_add = menu_tools.addAction("快速添加测量点（同高）…")
        action_quick_add.triggered.connect(self._on_add_rx_points)
        action_mgr = menu_tools.addAction("管理测量点…")
        action_mgr.triggered.connect(self._on_manage_rx)

        # ── Toolbar with solve button ──
        tb = self.addToolBar("求解")
        tb.setMovable(False)
        self._btn_solve = QtWidgets.QPushButton("⚡ 求解")
        self._btn_solve.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; "
            "background: #e65100; color: white; padding: 6px 20px; "
            "border-radius: 4px; } "
            "QPushButton:disabled { background: #ccc; }"
        )
        self._btn_solve.clicked.connect(self._on_solve)
        self._btn_solve.setEnabled(False)
        self._btn_solve.setToolTip("对已添加的测量点进行 CPE 计算")
        tb.addWidget(self._btn_solve)

        self._btn_plot = QtWidgets.QPushButton("📊 画图")
        self._btn_plot.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; "
            "background: #00796B; color: white; padding: 6px 20px; "
            "border-radius: 4px; } "
            "QPushButton:disabled { background: #ccc; }"
        )
        self._btn_plot.clicked.connect(self._on_plot)
        self._btn_plot.setEnabled(False)
        self._btn_plot.setToolTip("绘制传输损耗分布图")
        tb.addWidget(self._btn_plot)
        # Pending rx point count label
        self._rx_count_label = QtWidgets.QLabel("未添加测量点")
        self._rx_count_label.setStyleSheet("padding: 4px 8px; color: #888;")
        tb.addWidget(self._rx_count_label)

    def _set_view(self, which):
        p = self.plotter
        c, diag = self._scene_extent()
        if which == "top":
            p.camera_position = [(c[0], c[1], c[2] + 1.5 * diag), c, (0, 1, 0)]
        elif which == "front":
            p.camera_position = [(c[0] + 1.2 * diag, c[1], c[2] + 0.15 * diag), c, (0, 0, 1)]
        elif which == "side":
            p.camera_position = [(c[0], c[1] + 1.2 * diag, c[2] + 0.15 * diag), c, (0, 0, 1)]
        elif which == "reset":
            self._reset_camera()
            p.render()
            return
        p.camera.clipping_range = (diag / 500.0, diag * 5.0)
        p.render()

    def _screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "截图保存", "screenshot.png", "PNG (*.png);;JPG (*.jpg)"
        )
        if path:
            self.plotter.screenshot(path)
            self.statusBar().showMessage(f"已保存: {path}")

    # ═══════════════════════════════════════════════════════════
    #  Field solver — two-step: add points → solve
    # ═══════════════════════════════════════════════════════════

    def _antenna_frequency(self):
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None:
            return 2.8e9
        ac = (ant_obj.get("extra") or {}).get("antenna_config") or {}
        return float(ac.get("frequency", 2.8e9))

    def _on_add_rx_points(self):
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None:
            QtWidgets.QMessageBox.warning(self, "错误", "场景中未找到天线！")
            return
        tx = tuple(ant_obj["extra"]["position"])
        dlg = FieldPointDialog(self, self.scene_objects, tx, self._antenna_frequency())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        pts = dlg.get_points()
        if pts:
            self._pending_rx_points.extend(pts)
            self._invalidate_results("测量点已变更")
            self._draw_rx_markers()
            self._update_rx_tree()

    def _on_precise_rx(self):
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None: return
        tx = tuple(ant_obj["extra"]["position"])
        dlg = PreciseRxDialog(self, self.scene_objects, tx, self._antenna_frequency())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        pts = dlg.get_points()
        if pts:
            self._pending_rx_points.extend(pts)
            self._invalidate_results("测量点已变更")
            self._draw_rx_markers()
            self._update_rx_tree()

    def _on_manage_rx(self):
        ant_obj = self.scene_objects.get("antenna")
        tx = tuple(ant_obj["extra"]["position"]) if ant_obj else (0, 0, 0)
        dlg = RxManageDialog(self, self._pending_rx_points, tx)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_points = dlg.get_points()
            if new_points != self._pending_rx_points:
                self._pending_rx_points = new_points
                self._invalidate_results("测量点已变更")
            self._draw_rx_markers()
            self._update_rx_tree()

    def _draw_rx_markers(self):
        self._clear_rx_markers()
        pts = np.array(self._pending_rx_points, dtype=np.float32) if self._pending_rx_points else np.empty((0, 3))
        radius = self._marker_scale()
        font_size = int(np.clip(radius * 8, 10, 24))
        for i, (x, y, z) in enumerate(self._pending_rx_points):
            sphere = pv.Sphere(radius=radius, center=(x, y, z))
            name = f"__rx_{i}"
            actor = self.plotter.add_mesh(sphere, color="#FF5722", ambient=0.5)
            self.plotter_actors[name] = actor
        if len(pts) > 0:
            labels = [str(i + 1) for i in range(len(pts))]
            self.plotter.add_point_labels(
                pts, labels, font_size=font_size, text_color="white",
                shape_color="#FF5722", point_size=int(radius * 60),
                name="__rx_labels", always_visible=True, shadow=True,
            )
        self.plotter.render()

    def _clear_rx_markers(self):
        for name in list(self.plotter_actors.keys()):
            if name.startswith("__rx_"):
                self.plotter.remove_actor(self.plotter_actors.pop(name))
        # 清除标签
        try:
            self.plotter.remove_actor("__rx_labels")
        except Exception:
            pass
        self.plotter.render()

    def _update_rx_tree(self):
        self._tree_rx_root.takeChildren()
        if not self._pending_rx_points:
            item = QtWidgets.QTreeWidgetItem(["(无测量点)"])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsSelectable)
            self._tree_rx_root.addChild(item)
        else:
            for i, (x, y, z) in enumerate(self._pending_rx_points):
                item = QtWidgets.QTreeWidgetItem([f"点{i+1}: ({x:.2f}, {y:.2f}, {z:.2f})"])
                item.setData(0, QtCore.Qt.UserRole, f"rx:{i}")
                self._tree_rx_root.addChild(item)

    def _update_results_tree(self, results):
        # 按引用移除旧 Results 根节点（比文本匹配可靠）
        if self._tree_results_root is not None:
            idx = self._tree.indexOfTopLevelItem(self._tree_results_root)
            if idx >= 0:
                self._tree.takeTopLevelItem(idx)
            self._tree_results_root = None
        if not results:
            return
        root = QtWidgets.QTreeWidgetItem([f"📊 Results ({len(results)}点)"])
        root.setData(0, QtCore.Qt.UserRole, "results_root")
        for i, r in enumerate(results):
            d = r["path_loss_dB"] - r["L_fs_dB"]
            rx_item = QtWidgets.QTreeWidgetItem([f"点{i+1}: Δ={d:+.1f}dB"])
            rx_item.setData(0, QtCore.Qt.UserRole, f"result:{i}")
            for c in [
                f"L_fs: {r['L_fs_dB']:.2f} dB",
                f"L_pe: {r['path_loss_dB']:.2f} dB",
                f"E_rx: {r['E_rx_vm']:.4e} V/m",
                f"E: {r['E_rx_dbuv']:.1f} dBμV/m",
                f"距离: {r['dist']:.2f} m",
            ]:
                rx_item.addChild(QtWidgets.QTreeWidgetItem([c]))
            root.addChild(rx_item)
        table_item = QtWidgets.QTreeWidgetItem(["📋 查看完整结果表"])
        table_item.setData(0, QtCore.Qt.UserRole, "result_table")
        root.addChild(table_item)
        self._tree.addTopLevelItem(root)
        root.setExpanded(True)
        self._tree_results_root = root

    # ── 统一失效模型 ──
    # 两级缓存：results 依赖点集；field_data 依赖场景+天线+频率+材料。
    # 点集变化 → 只失效 results（场数据与点无关，画图仍可用）；
    # 场景/天线/材料/图层变化 → 全部失效。

    def _invalidate_results(self, reason=""):
        self._last_results = None
        self._update_results_tree(None)
        if reason:
            self.statusBar().showMessage(f"{reason}，结果已失效 — 请重新求解")
        self._refresh_action_states()

    def _invalidate_field(self, reason=""):
        self._last_field_data = None
        self._last_results = None
        self._update_results_tree(None)
        if reason:
            self.statusBar().showMessage(f"{reason}，请重新求解")
        self._refresh_action_states()

    def _refresh_action_states(self):
        solving = self._solve_thread is not None
        n = len(self._pending_rx_points)
        self._btn_solve.setEnabled(n > 0 and not solving)
        self._btn_plot.setEnabled(self._last_field_data is not None and not solving)
        self._rx_count_label.setText(
            f"待求解: {n} 个点" if n else "未添加测量点")
        # 状态栏摘要：场景 | 天线 | 频率 | 点数
        info = SCENE_REGISTRY.get(self._current_scene_key, {})
        ant = self.scene_objects.get("antenna", {})
        ac = (ant.get("extra") or {}).get("antenna_config", {})
        pos = (ant.get("extra") or {}).get("position", (0, 0, 0))
        self._status_summary_label.setText(
            f"{info.get('name', '?')} | 天线 ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) "
            f"@ {ac.get('frequency', 2.8e9)/1e9:.2f} GHz | 点 {n}"
        )

    def _on_tree_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item: return
        role = item.data(0, QtCore.Qt.UserRole)
        if role and role.startswith("rx:"):
            idx = int(role.split(":")[1])
            menu = QtWidgets.QMenu()
            act_del = menu.addAction("🗑 删除")
            act_del.triggered.connect(lambda: self._delete_rx_point(idx))
            menu.exec_(self._tree.viewport().mapToGlobal(pos))
        elif item is self._tree_rx_root:
            if not self._pending_rx_points:
                return
            menu = QtWidgets.QMenu()
            act_clear = menu.addAction("🗑 清空全部测量点")
            act_clear.triggered.connect(self._clear_all_rx_points)
            menu.exec_(self._tree.viewport().mapToGlobal(pos))
        elif role == "results_root":
            menu = QtWidgets.QMenu()
            act_clear = menu.addAction("🗑 清空结果")
            act_clear.triggered.connect(
                lambda: self._invalidate_results("结果已清空"))
            menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def _clear_all_rx_points(self):
        reply = QtWidgets.QMessageBox.question(
            self, "清空测量点",
            f"确定删除全部 {len(self._pending_rx_points)} 个测量点吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self._pending_rx_points.clear()
        self._invalidate_results("测量点已清空")
        self._draw_rx_markers()
        self._update_rx_tree()

    def _delete_rx_point(self, idx):
        if 0 <= idx < len(self._pending_rx_points):
            del self._pending_rx_points[idx]
            self._invalidate_results(f"已删除点{idx+1}")
            self._draw_rx_markers()
            self._update_rx_tree()

    def _on_tree_double_click(self, item, col):
        role = item.data(0, QtCore.Qt.UserRole)
        if role == "solve": self._on_solve()
        elif role == "plot": self._on_plot()
        elif role == "add_rx": self._on_precise_rx()
        elif role and role.startswith("rx:"):
            idx = int(role.split(":")[1])
            if idx < len(self._pending_rx_points):
                x, y, z = self._pending_rx_points[idx]
                # 编辑范围随场景尺度自适应（大场景 ±100 会钳坏坐标）
                t = self.scene_objects.get("terrain")
                if t is not None:
                    b = t["mesh"].bounds
                    xy_lim = max(abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3])) * 1.3 + 1.0
                    z_lim = max(abs(b[4]), abs(b[5])) * 2.0 + 50.0
                else:
                    xy_lim, z_lim = 1000.0, 500.0
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle(f"编辑 点{idx+1}")
                lo = QtWidgets.QFormLayout(dlg)
                sx = QtWidgets.QDoubleSpinBox(); sx.setRange(-xy_lim, xy_lim); sx.setDecimals(3); sx.setValue(x)
                sy = QtWidgets.QDoubleSpinBox(); sy.setRange(-xy_lim, xy_lim); sy.setDecimals(3); sy.setValue(y)
                sz = QtWidgets.QDoubleSpinBox(); sz.setRange(-10, z_lim); sz.setDecimals(3); sz.setValue(z)
                lo.addRow("X (m):", sx); lo.addRow("Y (m):", sy); lo.addRow("Z (m):", sz)
                btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                btn.accepted.connect(dlg.accept); btn.rejected.connect(dlg.reject); lo.addRow(btn)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    self._pending_rx_points[idx] = (sx.value(), sy.value(), sz.value())
                    self._invalidate_results(f"点{idx+1} 已更新")
                    self._draw_rx_markers()
                    self._update_rx_tree()
        elif role and role.startswith("result:"):
            idx = int(role.split(":")[1])
            if self._last_results and idx < len(self._last_results):
                r = self._last_results[idx]
                QtWidgets.QMessageBox.information(self, f"点{idx+1} 求解结果",
                    f"L_fs = {r['L_fs_dB']:.2f} dB\n"
                    f"L_pe = {r['path_loss_dB']:.2f} dB\n"
                    f"Δ = {r['path_loss_dB']-r['L_fs_dB']:+.2f} dB\n"
                    f"E_rx = {r['E_rx_vm']:.4e} V/m\n"
                    f"E = {r['E_rx_dbuv']:.1f} dBμV/m\n"
                    f"距离 = {r['dist']:.2f} m")
        elif role == "result_table" and self._last_results:
            from src.field_dialog import FieldResultDialog
            ant_obj = self.scene_objects.get("antenna")
            tx = tuple(ant_obj["extra"]["position"]) if ant_obj else (-5, 0, 6)
            FieldResultDialog(self, self._last_results, tx,
                              self._antenna_frequency()).exec_()

    def _on_select_scene(self):
        dlg = SceneSelectDialog(self, self._current_scene_key)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_key = dlg.selected_key()
        if new_key == self._current_scene_key:
            return
        self._current_scene_key = new_key
        self._load_scene()

    def _on_scene_props(self):
        dlg = ScenePropsDialog(self, self.scene_objects)
        dlg.exec_()

    def _on_template(self):
        dlg = TemplateDialog(self, self._current_scene_key)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        t = dlg.selected_template()
        if t is None:
            return
        pts = t.get("points", [])
        self._pending_rx_points.extend(pts)
        # 点集变化只失效结果；场数据与点集无关，画图功能保留
        self._invalidate_results(f"已加载模版：{t['name']}（+{len(pts)} 点）")
        self._draw_rx_markers()
        self._update_rx_tree()

    def _load_scene(self):
        # 清除旧场景（含测量点标签 actor —— 它不在 plotter_actors 字典里）
        for name in list(self.plotter_actors.keys()):
            self.plotter.remove_actor(self.plotter_actors.pop(name))
        self._clear_rx_markers()
        self.scene_objects.clear()
        self._pending_rx_points.clear()
        self._invalidate_field()
        self._update_rx_tree()

        # 加载新场景
        info = SCENE_REGISTRY[self._current_scene_key]
        self.scene_objects = info["builder"]()
        self._load_material_defaults()
        for name, obj in self.scene_objects.items():
            if obj.get("visible", True):
                self._add_actor(name, obj)

        self._reset_camera()
        self.plotter.render()
        self.statusBar().showMessage(f"已加载场景：{info['name']}")
        self._refresh_action_states()

    def _on_sweep(self):
        if not self._pending_rx_points:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加测量点")
            return
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None:
            return
        tx = tuple(ant_obj["extra"]["position"])
        ant_cfg = (ant_obj.get("extra") or {}).get("antenna_config") or {}
        dlg = SweepDialog(self, self.scene_objects, tx,
                          self._pending_rx_points, antenna_config=ant_cfg)
        dlg.exec_()

    def _on_plot(self):
        if self._last_field_data is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先完成求解")
            return
        dlg = PlotDialog(self, self._last_field_data)
        dlg.exec_()

    # ── 后台求解 ──

    def _on_solve(self):
        if self._solve_thread is not None:
            return  # 求解中，防重入
        if not self._pending_rx_points:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加测量点")
            return
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None:
            QtWidgets.QMessageBox.warning(self, "错误", "场景中未找到天线！")
            return
        tx = tuple(ant_obj["extra"]["position"])

        # 从天线配置中读取场景特定参数（含频率 —— 之前硬编码 2.8 GHz 被忽略）
        ant_cfg = (ant_obj.get("extra") or {}).get("antenna_config") or {}
        freq = float(ant_cfg.get("frequency", 2.8e9))
        config = EMConfig(
            frequency=freq, antenna_pos=tx,
            dr_factor=ant_cfg.get("dr_factor", 1.0),
            n_z=ant_cfg.get("fast_nz", 2048),
            n_phi=ant_cfg.get("fast_nphi", 128),
            antenna_type=ant_cfg.get("type", "gaussian"),
            antenna_sigma_z=ant_cfg.get("sigma_z", 4.0),
            antenna_tilt=ant_cfg.get("tilt_angle", 0.0),
            antenna_patch_hpbw=ant_cfg.get("patch_hpbw", 70.0),
            antenna_horn_hpbw=ant_cfg.get("horn_hpbw", 30.0),
            z_pad_above=ant_cfg.get("z_pad", 20.0),
        )

        # 过近点（r < r0）无物理意义，剔除并提示
        r0 = config.r0_factor * config.wavelength
        rx_points, too_close = [], []
        for p in self._pending_rx_points:
            r = np.hypot(p[0] - tx[0], p[1] - tx[1])
            (too_close if r < r0 else rx_points).append(p)
        if too_close:
            QtWidgets.QMessageBox.warning(
                self, "测量点过近",
                f"以下 {len(too_close)} 个点距天线不足 {r0:.2f} m（起始半径 r0），已剔除：\n"
                + "\n".join(f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in too_close))
        if not rx_points:
            return

        # 场景快照（extra 深拷贝）：求解期间用户可继续修改场景，互不影响
        scene_snap = {}
        for k, o in self.scene_objects.items():
            o2 = dict(o)
            if o.get("extra") is not None:
                o2["extra"] = copy.deepcopy(o["extra"])
            scene_snap[k] = o2
        scene_fp = self._scene_fingerprint()

        progress = QtWidgets.QProgressDialog(
            "CPE 计算中…", "取消", 0, 0, self)
        progress.setWindowTitle("求解电场")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(360)

        self._solve_worker = SolveWorker(config, list(rx_points), scene_snap)
        self._solve_thread = QtCore.QThread(self)
        self._solve_worker.moveToThread(self._solve_thread)
        self._solve_thread.started.connect(self._solve_worker.run)
        self._solve_worker.progress.connect(self._on_solve_progress)
        self._solve_worker.finished.connect(self._on_solve_finished)
        self._solve_worker.failed.connect(self._on_solve_failed)
        progress.canceled.connect(self._solve_worker.cancel)
        self._solve_progress = progress
        self._solve_fp = scene_fp
        self._solve_freq = freq
        self._solve_tx = tx

        self._refresh_action_states()
        self.statusBar().showMessage("CPE 计算中…（后台进行，可继续操作视图）")
        self._solve_thread.start()
        progress.show()

    def _on_solve_progress(self, stage, done, total):
        if self._solve_progress is None:
            return
        if stage == "求解" and total > 0:
            if self._solve_progress.maximum() != total:
                self._solve_progress.setMaximum(total)
            self._solve_progress.setValue(done)
            self._solve_progress.setLabelText(f"CPE 步进求解中… {done}/{total}")
        else:
            self._solve_progress.setLabelText(f"{stage}中…")

    def _on_solve_finished(self, ctx):
        self._teardown_solve_thread()
        freq = self._solve_freq
        tx = self._solve_tx

        # 保存完整场数据供绘图（场景用求解时快照，保证一致性）
        self._last_field_data = {
            "u_total": ctx.u_total,
            "r_vals": ctx.r_vals,
            "z_vals": ctx.z_vals,
            "phi_vals": ctx.phi_vals,
            "z_top": ctx.z_top,
            "config": ctx.config,
            "scene": ctx.scene,
        }
        results = []
        for res in ctx.results:
            rx = res["rx"]
            E_rx_dbuv = 20.0 * np.log10(max(res["E_rx"], 1e-15) / 1e-6)
            results.append({
                "rx": rx,
                "dist": res["dist"],
                "path_loss_dB": res["path_loss_dB"],
                "L_fs_dB": res["L_fs_dB"],
                "E_rx_vm": res["E_rx"],
                "E_rx_dbuv": E_rx_dbuv,
            })
        self._last_results = results
        self._update_results_tree(results)
        self._refresh_action_states()

        msg = f"计算完成 — {len(results)} 个点 @ {freq/1e9:.2f} GHz"
        self.statusBar().showMessage(msg)

        # 求解期间场景被修改 → 提醒结果基于旧快照
        if self._scene_fingerprint() != self._solve_fp:
            QtWidgets.QMessageBox.information(
                self, "场景已变更",
                "求解期间场景/材料/天线参数发生变化。\n"
                "当前结果基于求解开始时的场景快照，如需反映最新修改请重新求解。")
        # 网格精度等警告
        if ctx.warnings:
            QtWidgets.QMessageBox.warning(
                self, "求解精度提示", "\n\n".join(ctx.warnings))

        rdlg = FieldResultDialog(self, results, tx, freq)
        rdlg.exec_()

    def _on_solve_failed(self, msg):
        self._teardown_solve_thread()
        self._refresh_action_states()
        if msg == SolveWorker.CANCELLED:
            self.statusBar().showMessage("求解已取消")
        else:
            self.statusBar().showMessage("求解失败")
            QtWidgets.QMessageBox.critical(
                self, "求解失败", f"计算过程中发生错误：\n{msg}")

    def _teardown_solve_thread(self):
        if self._solve_progress is not None:
            self._solve_progress.reset()
            self._solve_progress.deleteLater()
            self._solve_progress = None
        if self._solve_thread is not None:
            self._solve_thread.quit()
            self._solve_thread.wait(5000)
            self._solve_thread = None
            self._solve_worker = None

    def _scene_fingerprint(self):
        """场景 EM 相关状态指纹：求解完成后检测场景是否被改动。"""
        parts = [self._current_scene_key]
        for name, obj in sorted(self.scene_objects.items()):
            extra = obj.get("extra") or {}
            mat = extra.get("material") or {}
            parts.append(
                f"{name}:{mat.get('eps_r')},{mat.get('sigma')},"
                f"{mat.get('thickness_cm')},{extra.get('obstacle_type')},"
                f"{extra.get('is_material_layer')}")
        ant = self.scene_objects.get("antenna", {})
        ac = (ant.get("extra") or {}).get("antenna_config", {})
        parts.append(str(sorted(ac.items())))
        return hash(tuple(parts))

    # ═══════════════════════════════════════════════════════════
    #  Layer management
    # ═══════════════════════════════════════════════════════════

    def _compute_terrain_extent(self):
        t = self.scene_objects.get("terrain")
        if t is None:
            return (10.0, 20.0)
        b = t["mesh"].bounds
        # Use actual terrain bounds, not half-span
        xy_half = max(abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3])) * 1.1
        zspan = max(b[5] - b[4], 0.5)
        return (xy_half, zspan)

    def _is_dem_scene(self):
        t = self.scene_objects.get("terrain")
        if t is None or t.get("extra") is None:
            return False
        return t["extra"].get("is_dem", False)

    # ── 导入/导出 ──

    def _on_import_dem(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "导入 DEM 高程模型",
            "", "DEM 文件 (*.tif *.tiff *.img *.dem);;所有文件 (*)")
        if not path:
            return
        try:
            # 降采样因子自动估算：目标网格 150×150
            import rasterio
            with rasterio.open(path) as src:
                w, h = src.width, src.height
            ds = max(1, max(w, h) // 150)
            dem_info = io_utils.read_dem(path, downsample=ds)
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "缺少依赖",
                "导入 DEM 需要 rasterio 库。\n请在终端执行: pip install rasterio")
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导入失败", str(e))
            return
        self._replace_terrain(io_utils.dem_to_terrain_obj(dem_info, center=True),
                              f"已导入 DEM: {os.path.basename(path)}")

    def _on_import_asc(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "导入 ASC 高程栅格",
            "", "ASC 文件 (*.asc *.txt);;所有文件 (*)")
        if not path:
            return
        try:
            asc_info = io_utils.read_asc(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导入失败", str(e))
            return
        self._replace_terrain(io_utils.dem_to_terrain_obj(asc_info, center=True),
                              f"已导入 ASC: {os.path.basename(path)}")

    def _on_export_asc(self):
        terrain = self.scene_objects.get("terrain")
        if terrain is None:
            QtWidgets.QMessageBox.warning(self, "提示", "场景中没有地形数据")
            return
        info = io_utils.terrain_to_dem_info(terrain)
        if info is None:
            QtWidgets.QMessageBox.warning(self, "导出失败", "无法从当前地形提取高程数据")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出地形为 ASC", "terrain_export.asc", "ASC (*.asc)")
        if not path:
            return
        try:
            io_utils.write_asc(path, info["data"],
                               xll=info["xll"], yll=info["yll"],
                               cellsize=info["cellsize"])
            QtWidgets.QMessageBox.information(
                self, "导出完成",
                f"已导出 {info['ncols']}×{info['nrows']} 高程栅格\n→ {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(e))

    def _replace_terrain(self, new_terrain: dict, status_msg: str):
        old_actor = self.plotter_actors.pop("terrain", None)
        if old_actor is not None:
            self.plotter.remove_actor(old_actor)

        self.scene_objects["terrain"] = new_terrain
        self._add_actor("terrain", new_terrain)

        self._invalidate_field(status_msg)
        self._reset_camera()
        self.plotter.render()
        self.statusBar().showMessage(status_msg)

    def _open_layer_dialog(self):
        layer_names = {
            "layer_sand": "沙地", "layer_grass": "草地",
            "layer_earth": "土地", "layer_water": "水面",
        }
        extent = self._compute_terrain_extent()
        is_dem = self._is_dem_scene()
        terrain_mesh = self.scene_objects.get("terrain", {}).get("mesh")

        dlg = LayerManagementDialog(
            self, layer_names, extent, is_dem, terrain_mesh=terrain_mesh)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        result = dlg.get_results()
        shapes = result.get("layer_shapes", {})
        if shapes:
            self._apply_layer_shapes(shapes)

    def _apply_layer_shapes(self, shapes_by_layer):
        terrain_obj = self.scene_objects.get("terrain")
        if terrain_obj is None:
            return
        # Use the full terrain surface as the source mesh for clipping
        terrain_mesh = terrain_obj["mesh"]
        if hasattr(terrain_mesh, 'extract_surface'):
            source_mesh = terrain_mesh.extract_surface()
        else:
            source_mesh = terrain_mesh
        all_pts = np.asarray(source_mesh.points)

        for layer_key, shapes in shapes_by_layer.items():
            if not shapes:
                continue
            combined = np.zeros(all_pts.shape[0], dtype=bool)
            all_polys = []
            for shape in shapes:
                poly = np.array(shape["xy"])
                all_polys.append(poly)
                path = mpath.Path(poly)
                inside = path.contains_points(all_pts[:, :2])
                combined |= inside
            if not combined.any():
                continue
            sub = source_mesh.extract_points(combined)
            clip_name = f"{layer_key}_clip"
            count = 1
            while clip_name in self.scene_objects:
                count += 1
                clip_name = f"{layer_key}_clip_{count}"
            # Get material from base layer or use default
            base_obj = self.scene_objects.get(layer_key, {})
            mat = base_obj.get("extra", {}).get("material", {})
            self.scene_objects[clip_name] = {
                "mesh": sub, "type": "mesh", "visible": True,
                "params": {"color": "#FF5722" if "sand" in layer_key
                           else "#4CAF50" if "grass" in layer_key
                           else "#2980b9" if "water" in layer_key
                           else "#795548",
                           "opacity": 0.7, "smooth_shading": True},
                "extra": {"material": mat, "polygons": all_polys, "layer_key": layer_key},
                "name": clip_name,
            }
            self._add_actor(clip_name, self.scene_objects[clip_name])
        self.plotter.render()
        if shapes_by_layer:
            self._invalidate_field("图层已更新")

    def _toggle_clip_visibility(self, name, visible):
        actor = self.plotter_actors.get(name)
        if actor is not None:
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass
        obj = self.scene_objects.get(name)
        if obj is not None:
            obj["visible"] = visible
        self.plotter.render()

    def _set_clip_opacity(self, name, opacity):
        actor = self.plotter_actors.get(name)
        if actor is not None:
            try:
                actor.GetProperty().SetOpacity(opacity)
            except Exception:
                pass
        self.plotter.render()

    def _remove_clip(self, name):
        actor = self.plotter_actors.pop(name, None)
        if actor is not None:
            self.plotter.remove_actor(actor)
        self.scene_objects.pop(name, None)
        self.plotter.render()
        self._invalidate_field("裁剪图层已删除")

    def _open_clip_manager(self):
        dlg = ClipManagerDialog(
            self,
            self.scene_objects,
            self.plotter_actors,
            self._toggle_clip_visibility,
            self._set_clip_opacity,
            self._remove_clip,
            self._open_clip_manager,
        )
        dlg.exec_()

    def _load_material_defaults(self):
        from src.material_params_dialog import LAYER_MATERIAL_MAP, DEFAULT_MATERIALS
        # 纯视觉装饰对象 — 不参与 EM 计算，不分配材料
        _visual_only = {"bird", "tree", "vegetation", "aircraft", "aircraft2"}
        for name, obj in self.scene_objects.items():
            if name == "antenna" or name in _visual_only:
                continue
            extra = obj.get("extra")
            if extra is None:
                obj["extra"] = {}
                extra = obj["extra"]
            if extra.get("material") is None:
                label = LAYER_MATERIAL_MAP.get(name, "干燥土壤")
                dflt = DEFAULT_MATERIALS.get(label, DEFAULT_MATERIALS["干燥土壤"])
                extra["material"] = {"label": label, **dflt}

    def _open_material_params(self):
        dlg = MaterialParamsDialog(self, self.scene_objects, self._antenna_frequency())
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._invalidate_field("材料参数已更新")

    def _open_antenna_dialog(self):
        ant = self.scene_objects.get("antenna")
        if ant is None:
            QtWidgets.QMessageBox.warning(self, "错误", "场景中未找到天线！")
            return
        config = dict(ant.get("extra", {}).get("antenna_config", {}))
        cur_pos = ant.get("extra", {}).get("position", (-5, 0, 6))
        config["position"] = cur_pos
        # 位置编辑范围随场景尺度自适应（大场景 ±100 会把天线坐标钳坏）
        _, diag = self._scene_extent()
        dlg = AntennaDialog(self, config, pos_limit=max(diag * 0.65, 120.0))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_config = dlg.get_config()
            if "extra" not in ant:
                ant["extra"] = {}
            # 保留场景特定的性能参数
            for k in ("dr_factor", "fast_nz", "fast_nphi", "z_pad"):
                old_val = (ant.get("extra") or {}).get("antenna_config", {}).get(k)
                if old_val is not None:
                    new_config[k] = old_val
            ant["extra"]["antenna_config"] = new_config
            # 同步位置到场景
            new_pos = new_config.get("position", cur_pos)
            ant["extra"]["position"] = new_pos
            # 检查天线是否低于地形
            terrain = self.scene_objects.get("terrain")
            if terrain:
                from src.physics.terrain import sample_terrain
                tz = sample_terrain(terrain["mesh"], np.array([new_pos[0]]), np.array([new_pos[1]]))[0]
                if new_pos[2] <= tz + 0.5:
                    QtWidgets.QMessageBox.warning(
                        self, "天线位置异常",
                        f"天线高度 {new_pos[2]:.1f}m 低于此处地形 {tz:.1f}m！\n"
                        f"电磁波将被地面吸收，建议天线高度至少设为 {tz+5:.0f}m 以上。"
                    )
            # 更新可视化
            self._rebuild_antenna_marker(new_pos)
            self._invalidate_field(
                f"天线已更新：{new_config['type']} @ {new_config['frequency']/1e9:.2f} GHz")

    def _rebuild_antenna_marker(self, new_pos):
        import pyvista as pv
        from src.physics.terrain import sample_terrain
        radius = max(self._marker_scale() * 0.6, 0.3)
        sphere = pv.Sphere(radius=radius, center=new_pos)
        # 杆子从地面到天线
        terrain = self.scene_objects.get("terrain")
        if terrain:
            tz = float(sample_terrain(terrain["mesh"],
                      np.array([new_pos[0]]), np.array([new_pos[1]]))[0])
        else:
            tz = 0.0
        tz = max(tz, 0.0)
        h = new_pos[2]
        if h - tz > 0.3:
            pole = pv.Cylinder(
                center=(new_pos[0], new_pos[1], (tz + h) / 2),
                direction=(0, 0, 1), radius=radius * 0.3, height=h - tz,
            )
            ant_mesh = sphere.merge([pole])
        else:
            ant_mesh = sphere
        self.scene_objects["antenna"]["mesh"] = ant_mesh
        # 替换 3D 场景中的 actor
        old_actor = self.plotter_actors.pop("antenna", None)
        if old_actor is not None:
            self.plotter.remove_actor(old_actor)
        params = self.scene_objects["antenna"]["params"]
        actor = self.plotter.add_mesh(ant_mesh, **params)
        self.plotter_actors["antenna"] = actor
        self.plotter.render()


class RxManageDialog(QtWidgets.QDialog):
    """Manage measurement points: view, edit, delete, import/export CSV."""

    def __init__(self, parent, points, antenna_pos=(0, 0, 0)):
        super().__init__(parent)
        self.setWindowTitle("管理测量点")
        self.setMinimumSize(680, 380)
        self._points = [list(p) for p in points]
        self._tx = antenna_pos
        self._invalid_warned = False

        layout = QtWidgets.QVBoxLayout(self)
        self._count_label = QtWidgets.QLabel(f"共 {len(self._points)} 个测量点")
        layout.addWidget(self._count_label)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["#", "X (m)", "Y (m)", "Z (m)", "距天线 (m)"])
        self._reload_table()
        layout.addWidget(self._table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_del = QtWidgets.QPushButton("删除选中")
        btn_del.clicked.connect(self._delete)
        btn_row.addWidget(btn_del)
        btn_clear = QtWidgets.QPushButton("清空全部")
        btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(btn_clear)
        btn_row.addSpacing(16)
        btn_import = QtWidgets.QPushButton("导入 CSV…")
        btn_import.clicked.connect(self._import_csv)
        btn_row.addWidget(btn_import)
        btn_export = QtWidgets.QPushButton("导出 CSV…")
        btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        btn_ok = QtWidgets.QPushButton("确定")
        btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def _reload_table(self):
        self._table.setRowCount(len(self._points))
        for i, (x, y, z) in enumerate(self._points):
            dist = float(np.hypot(x - self._tx[0], y - self._tx[1]))
            values = [str(i + 1), f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{dist:.2f}"]
            for j, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(v)
                if j in (0, 4):
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self._table.setItem(i, j, item)
        self._count_label.setText(f"共 {len(self._points)} 个测量点")

    def _delete(self):
        row = self._table.currentRow()
        if row >= 0:
            del self._points[row]
            self._reload_table()

    def _clear_all(self):
        if not self._points:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "清空全部", f"确定删除全部 {len(self._points)} 个测量点吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self._points.clear()
            self._reload_table()

    def _import_csv(self):
        import csv
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "导入测量点", "", "CSV (*.csv);;文本 (*.txt)")
        if not path:
            return
        added, skipped = 0, 0
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.reader(f):
                    if not row or row[0].strip().startswith("#"):
                        continue
                    try:
                        vals = [float(v) for v in row[:3]]
                        if len(vals) != 3:
                            raise ValueError
                        self._points.append(vals)
                        added += 1
                    except ValueError:
                        skipped += 1
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导入失败", str(e))
            return
        self._reload_table()
        QtWidgets.QMessageBox.information(
            self, "导入完成", f"成功导入 {added} 个点" +
            (f"，跳过 {skipped} 行无效数据" if skipped else ""))

    def _export_csv(self):
        import csv
        if not self._points:
            QtWidgets.QMessageBox.information(self, "提示", "没有可导出的测量点")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出测量点", "rx_points.csv", "CSV (*.csv)")
        if not path:
            return
        pts = self._read_table_points()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["x_m", "y_m", "z_m"])
            for x, y, z in pts:
                w.writerow([x, y, z])
        QtWidgets.QMessageBox.information(
            self, "导出完成", f"已导出 {len(pts)} 个点到\n{path}")

    def _read_table_points(self):
        pts = []
        for r in range(self._table.rowCount()):
            try:
                x = float(self._table.item(r, 1).text())
                y = float(self._table.item(r, 2).text())
                z = float(self._table.item(r, 3).text())
            except (ValueError, AttributeError):
                continue
            pts.append((x, y, z))
        return pts

    def _on_ok(self):
        n_table = self._table.rowCount()
        pts = self._read_table_points()
        if len(pts) < n_table:
            QtWidgets.QMessageBox.warning(
                self, "无效输入",
                f"有 {n_table - len(pts)} 行坐标不是有效数字，将被忽略。\n"
                "请检查输入后重试。")
        self._points = [list(p) for p in pts]
        self.accept()

    def get_points(self):
        return [tuple(p) for p in self._points]


if __name__ == "__main__":
    import sys
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
