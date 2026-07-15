"""
MainWindow — simplified 3D scene viewer with CPE field solver menu.
"""

import sys
import os
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

import matplotlib.path as mpath


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

        # ── Central 3D viewport ──
        self.plotter = pvqt.QtInteractor(self)
        self.setCentralWidget(self.plotter)

        # ── Build scene ──
        self._init_scene()

        # ── Menus ──
        self._setup_menus()

        # ── Status bar ──
        self.statusBar().showMessage(
            "就绪  |  左键旋转 · 滚轮缩放 · 中键平移  |  工具 → 求解电场功能"
        )

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

        # Camera: position to see antenna, wall, and beyond
        p.camera_position = [(20, -14, 10), (0, 0, 3), (0, 0, 1)]
        p.camera.clipping_range = (0.1, 100.0)
        p.render()

    def _add_actor(self, name, obj):
        mesh = obj["mesh"]
        params = obj["params"]
        if obj["type"] == "points":
            actor = self.plotter.add_points(mesh, **params)
        else:
            actor = self.plotter.add_mesh(mesh, **params)
        self.plotter_actors[name] = actor

    # ═══════════════════════════════════════════════════════════
    #  Menus
    # ═══════════════════════════════════════════════════════════

    def _setup_menus(self):
        mb = self.menuBar()

        # ── Scene menu ──
        menu_scene = mb.addMenu("场景 (&S)")
        menu_scene.addAction("场景选择…", self._on_select_scene)
        menu_scene.addAction("场景属性…", self._on_scene_props)

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

        # ── Parameters menu ──
        menu_params = mb.addMenu("参数 (&P)")
        menu_params.addAction("材料参数设置…", self._open_material_params)
        menu_params.addAction("天线设置…", self._open_antenna_dialog)
        menu_tools = mb.addMenu("工具 (&T)")
        action_rx = menu_tools.addAction("添加测量点… (&A)")
        action_rx.setShortcut("Ctrl+A")
        action_rx.triggered.connect(self._on_add_rx_points)
        action_precise = menu_tools.addAction("精准添加测量点…")
        action_precise.triggered.connect(self._on_precise_rx)
        menu_tools.addSeparator()
        action_sweep = menu_tools.addAction("扫频模式… (&S)")
        action_sweep.setShortcut("Ctrl+S")
        action_sweep.triggered.connect(self._on_sweep)
        menu_tools.addSeparator()
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
        # Pending rx point count label
        self._rx_count_label = QtWidgets.QLabel("未添加测量点")
        self._rx_count_label.setStyleSheet("padding: 4px 8px; color: #888;")
        tb.addWidget(self._rx_count_label)

    def _set_view(self, which):
        p = self.plotter
        if which == "top":
            p.camera_position = [(0, 0, 25), (0, 0, 3), (0, 1, 0)]
        elif which == "front":
            p.camera_position = [(20, 0, 3), (0, 0, 3), (0, 0, 1)]
        elif which == "side":
            p.camera_position = [(0, 20, 3), (0, 0, 3), (0, 0, 1)]
        elif which == "reset":
            p.camera_position = [(20, -14, 10), (0, 0, 3), (0, 0, 1)]
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

    def _on_add_rx_points(self):
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None:
            QtWidgets.QMessageBox.warning(self, "错误", "场景中未找到天线！")
            return
        tx = tuple(ant_obj["extra"]["position"])
        dlg = FieldPointDialog(self, self.scene_objects, tx, 2.8e9)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        pts = dlg.get_points()
        if pts:
            self._pending_rx_points = pts
            self._draw_rx_markers()
            self._update_solve_ui()

    def _on_precise_rx(self):
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None: return
        tx = tuple(ant_obj["extra"]["position"])
        dlg = PreciseRxDialog(self, self.scene_objects, tx, 2.8e9)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        pts = dlg.get_points()
        if pts:
            self._pending_rx_points = pts
            self._draw_rx_markers()
            self._update_solve_ui()

    def _on_manage_rx(self):
        dlg = RxManageDialog(self, self._pending_rx_points)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._pending_rx_points = dlg.get_points()
            self._draw_rx_markers()
            self._update_solve_ui()
            if not self._pending_rx_points:
                self._clear_rx_markers()

    def _draw_rx_markers(self):
        self._clear_rx_markers()
        for i, (x, y, z) in enumerate(self._pending_rx_points):
            sphere = pv.Sphere(radius=0.15, center=(x, y, z))
            name = f"__rx_{i}"
            actor = self.plotter.add_mesh(sphere, color="#FF5722", ambient=0.5)
            self.plotter_actors[name] = actor
        self.plotter.render()

    def _clear_rx_markers(self):
        for name in list(self.plotter_actors.keys()):
            if name.startswith("__rx_"):
                self.plotter.remove_actor(self.plotter_actors.pop(name))
        self.plotter.render()

    def _update_solve_ui(self):
        n = len(self._pending_rx_points)
        self._btn_solve.setEnabled(n > 0)
        self._rx_count_label.setText(f"待求解: {n} 个点" if n else "未添加测量点")

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

    def _load_scene(self):
        # 清除旧场景
        for name in list(self.plotter_actors.keys()):
            self.plotter.remove_actor(self.plotter_actors.pop(name))
        self.scene_objects.clear()
        self._pending_rx_points.clear()
        self._update_solve_ui()

        # 加载新场景
        info = SCENE_REGISTRY[self._current_scene_key]
        self.scene_objects = info["builder"]()
        self._load_material_defaults()
        for name, obj in self.scene_objects.items():
            if obj.get("visible", True):
                self._add_actor(name, obj)

        self.plotter.camera_position = [(20, -14, 10), (0, 0, 3), (0, 0, 1)]
        self.plotter.render()
        self.statusBar().showMessage(f"已加载场景：{info['name']}")

    def _on_sweep(self):
        if not self._pending_rx_points:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加测量点")
            return
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None:
            return
        tx = tuple(ant_obj["extra"]["position"])
        dlg = SweepDialog(self, self.scene_objects, tx, self._pending_rx_points)
        dlg.exec_()

    def _on_solve(self):
        if not self._pending_rx_points:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加测量点")
            return
        ant_obj = self.scene_objects.get("antenna")
        if ant_obj is None:
            return
        tx = tuple(ant_obj["extra"]["position"])
        freq = 2.8e9
        rx_points = self._pending_rx_points

        self.statusBar().showMessage("CPE 计算中…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        progress = QtWidgets.QProgressDialog(
            f"CPE 计算中… (0/{len(rx_points)})", None, 0, len(rx_points), self)
        progress.setWindowTitle("求解电场")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        config = EMConfig(
            frequency=freq,
            antenna_pos=tx,
        )
        sched = Scheduler()
        results_raw = sched.run(config=config, rx_points=rx_points, scene=self.scene_objects)
        results = []
        for i, res in enumerate(results_raw):
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
            progress.setValue(i + 1)
            progress.setLabelText(
                f"CPE 计算中… ({i+1}/{len(rx_points)})\n"
                f"({rx[0]:.1f}, {rx[1]:.1f}, {rx[2]:.1f})")
            QtWidgets.QApplication.processEvents()

        progress.setValue(len(rx_points))
        progress.close()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"计算完成 — {len(results)} 个点")

        rdlg = FieldResultDialog(self, results, tx, freq)
        rdlg.exec_()

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

    def _toggle_clip_visibility(self, name, visible):
        actor = self.plotter_actors.get(name)
        if actor is not None:
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass
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
        dlg = MaterialParamsDialog(self, self.scene_objects)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.statusBar().showMessage("材料参数已更新")

    def _open_antenna_dialog(self):
        ant = self.scene_objects.get("antenna")
        if ant is None:
            QtWidgets.QMessageBox.warning(self, "错误", "场景中未找到天线！")
            return
        config = ant.get("extra", {}).get("antenna_config", {})
        cur_pos = ant.get("extra", {}).get("position", (-5, 0, 6))
        config["position"] = cur_pos
        dlg = AntennaDialog(self, config)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_config = dlg.get_config()
            if "extra" not in ant:
                ant["extra"] = {}
            ant["extra"]["antenna_config"] = new_config
            # 同步位置到场景
            new_pos = new_config.get("position", cur_pos)
            ant["extra"]["position"] = new_pos
            # 更新可视化：移除旧天线标记，重建新的
            self._rebuild_antenna_marker(new_pos)
            self.statusBar().showMessage(
                f"天线已更新：{new_config['type']} @ {new_config['frequency']/1e9:.1f} GHz"
            )

    def _rebuild_antenna_marker(self, new_pos):
        import pyvista as pv
        from src.physics.terrain import sample_terrain
        radius = 0.3
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
    """Manage measurement points: view, edit, delete."""

    def __init__(self, parent, points):
        super().__init__(parent)
        self.setWindowTitle("管理测量点")
        self.setMinimumSize(550, 350)
        self._points = [list(p) for p in points]

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"共 {len(self._points)} 个测量点"))

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["#", "X (m)", "Y (m)", "Z (m)"])
        self._table.setRowCount(len(self._points))
        for i, (x, y, z) in enumerate(self._points):
            for j, v in enumerate([str(i+1), f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"]):
                self._table.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        layout.addWidget(self._table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_del = QtWidgets.QPushButton("删除选中")
        btn_del.clicked.connect(self._delete)
        btn_row.addWidget(btn_del)
        btn_clear = QtWidgets.QPushButton("清空全部")
        btn_clear.clicked.connect(lambda: (self._points.clear(), self._table.setRowCount(0)))
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        btn_ok = QtWidgets.QPushButton("确定")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def _delete(self):
        row = self._table.currentRow()
        if row >= 0:
            del self._points[row]
            self._table.removeRow(row)

    def get_points(self):
        # Re-read from table (user may have edited cells)
        pts = []
        for r in range(self._table.rowCount()):
            try:
                x = float(self._table.item(r, 1).text()) if self._table.item(r, 1) else self._points[r][0]
                y = float(self._table.item(r, 2).text()) if self._table.item(r, 2) else self._points[r][1]
                z = float(self._table.item(r, 3).text()) if self._table.item(r, 3) else self._points[r][2]
            except (ValueError, IndexError):
                continue
            pts.append((x, y, z))
        return pts


if __name__ == "__main__":
    import sys
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
