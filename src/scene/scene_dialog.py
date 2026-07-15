"""
场景对话框 — 场景选择 + 场景属性详情。
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from src.scene.scenes import SCENE_REGISTRY


class SceneSelectDialog(QtWidgets.QDialog):
    """场景选择对话框。"""

    def __init__(self, parent, current_key):
        super().__init__(parent)
        self.setWindowTitle("场景选择")
        self.setMinimumSize(550, 380)
        self._selected = current_key

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("选择一个场景："))
        layout.addWidget(QtWidgets.QLabel("（切换场景将替换当前所有场景对象）"))

        self._list = QtWidgets.QListWidget()
        for key, info in SCENE_REGISTRY.items():
            item = QtWidgets.QListWidgetItem(f"{info['name']}")
            item.setData(QtCore.Qt.UserRole, key)
            item.setToolTip(info["description"])
            self._list.addItem(item)
            if key == current_key:
                self._list.setCurrentItem(item)
        layout.addWidget(self._list, 1)

        # 描述
        self._desc = QtWidgets.QLabel("")
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color: #555; padding: 6px;")
        layout.addWidget(self._desc)
        self._list.currentItemChanged.connect(self._on_select)

        # 按钮
        btn = QtWidgets.QHBoxLayout()
        btn.addStretch()
        cancel = QtWidgets.QPushButton("取消"); cancel.clicked.connect(self.reject)
        btn.addWidget(cancel)
        ok = QtWidgets.QPushButton("加载场景 ✓")
        ok.clicked.connect(self.accept)
        ok.setStyleSheet("QPushButton { font-weight: bold; background: #2196F3; color: white; padding: 4px 16px; }")
        btn.addWidget(ok)
        layout.addLayout(btn)

        if self._list.currentItem():
            self._on_select(self._list.currentItem(), None)

    def _on_select(self, current, _prev):
        if current:
            key = current.data(QtCore.Qt.UserRole)
            self._selected = key
            self._desc.setText(SCENE_REGISTRY[key]["description"])

    def selected_key(self):
        return self._selected


class ScenePropsDialog(QtWidgets.QDialog):
    """场景属性详情对话框。"""

    def __init__(self, parent, scene_objects):
        super().__init__(parent)
        self.setWindowTitle("场景属性")
        self.setMinimumSize(600, 500)

        layout = QtWidgets.QVBoxLayout(self)

        # 场景概览
        _vis = {"bird", "tree", "vegetation", "aircraft", "aircraft2"}
        n_em = sum(1 for k in scene_objects if k not in _vis and k != "antenna")
        obstacles = [k for k, v in scene_objects.items()
                     if (v.get("extra") or {}).get("obstacle_type") == "wall"]
        layers = [k for k in scene_objects if k.startswith("layer_") or
                  k in ("river", "lake", "grass_patch")]
        terrain = scene_objects.get("terrain")

        summary = (
            f"EM 对象：{n_em}  |  障碍物：{len(obstacles)} 个  |  "
            f"图层：{len(layers)} 个  |  "
            f"天线：{'有' if 'antenna' in scene_objects else '无'}  |  "
            f"装饰：{sum(1 for k in _vis if k in scene_objects)} 个"
        )
        layout.addWidget(QtWidgets.QLabel(summary))

        # 详情表格
        table = QtWidgets.QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "名称", "类型", "位置/尺寸", "材料", "ε_r", "σ (S/m)"
        ])

        # 第一行：空气（背景介质）
        rows = [["（背景）", "空气", "全空间", "空气", "1.0006", "0.0"]]

        for name, obj in scene_objects.items():
            if name in _vis:
                continue
            mesh = obj.get("mesh")
            if mesh is None:
                continue
            b = mesh.bounds
            obj_type = obj.get("type", "?")
            extra = obj.get("extra") or {}
            mat = extra.get("material") or {}

            if extra.get("obstacle_type") == "wall":
                obj_type = "障碍物"
            elif name == "terrain":
                obj_type = "地面"
            elif name == "antenna":
                obj_type = "天线源"
            elif "layer" in name or name in ("river", "lake", "grass_patch"):
                obj_type = "介质层"

            pos_str = (f"x:{b[0]:.1f}~{b[1]:.1f}  "
                       f"y:{b[2]:.1f}~{b[3]:.1f}  "
                       f"z:{b[4]:.1f}~{b[5]:.1f}m")
            mat_label = mat.get("label", "—")
            eps = f"{mat.get('eps_r', '—')}"
            sigma = f"{mat.get('sigma', '—')}"

            rows.append([name, obj_type, pos_str, mat_label, eps, sigma])

        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                table.setItem(i, j, QtWidgets.QTableWidgetItem(str(val)))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, 1)

        # 地形详情
        if terrain:
            t = terrain
            t_extra = t.get("extra") or {}
            t_mat = t_extra.get("material") or {}
            b = t["mesh"].bounds
            detail = (
                f"地面材料：{t_mat.get('label', '—')}  "
                f"ε_r={t_mat.get('eps_r', '—')}  σ={t_mat.get('sigma', '—')} S/m\n"
                f"范围：X {b[0]:.1f}~{b[1]:.1f}m  Y {b[2]:.1f}~{b[3]:.1f}m  "
                f"高程 {b[4]:.1f}~{b[5]:.1f}m  "
                f"类型：{'DEM 导入' if t_extra.get('is_dem') else '程序生成'}"
            )
        else:
            detail = "无地面"

        dlabel = QtWidgets.QLabel(detail)
        dlabel.setWordWrap(True)
        layout.addWidget(dlabel)

        btn = QtWidgets.QPushButton("关闭"); btn.clicked.connect(self.accept)
        layout.addWidget(btn)
