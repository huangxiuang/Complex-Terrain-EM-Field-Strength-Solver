"""
MaterialParamsDialog — edit ε_r, σ, thickness for each non-conductor layer.

Stores a material registry dict shared with main_window.
Supports JSON export/import for persistence.
"""

import json
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui


# ── Default material database ─────────────────────────
DEFAULT_MATERIALS = {
    "地面": {"eps_r": 15.0, "sigma": 0.01,   "thickness_cm": 0.0},
    "湿润地面":     {"eps_r": 25.0, "sigma": 0.1,    "thickness_cm": 0.0},
    "沙地":         {"eps_r": 3.0,  "sigma": 0.001,  "thickness_cm": 0.5},
    "草地":         {"eps_r": 12.0, "sigma": 0.005,  "thickness_cm": 1.0},
    "土地":         {"eps_r": 5.0,  "sigma": 0.005,  "thickness_cm": 2.0},
    "水面（淡水）":  {"eps_r": 80.0, "sigma": 0.01,   "thickness_cm": 0.5},
    "水面（海水）":  {"eps_r": 80.0, "sigma": 4.0,    "thickness_cm": 0.5},
    "混凝土":       {"eps_r": 6.0,  "sigma": 0.02,   "thickness_cm": 20.0},
    "玻璃":         {"eps_r": 7.0,  "sigma": 1e-6,   "thickness_cm": 1.0},
    "木材":         {"eps_r": 2.0,  "sigma": 1e-4,   "thickness_cm": 5.0},
    "金属铝":       {"eps_r": 1.0,  "sigma": 3.8e7,  "thickness_cm": 0.0},
}

# ── Layer → default material mapping ──────────────────
LAYER_MATERIAL_MAP = {
    "layer_sand":  "沙地",
    "layer_grass": "草地",
    "layer_earth": "土地",
    "layer_water": "水面（淡水）",
    "wall":        "金属铝",
    "terrain":     "地面",
}


class MaterialParamsDialog(QtWidgets.QDialog):
    """Non-modal dialog to manage material parameters for all non-conductor layers."""

    def __init__(self, parent, scene_objects, frequency=2.8e9):
        super().__init__(parent)
        self.setWindowTitle("材料参数设置")
        self.setMinimumSize(950, 480)
        self._scene = scene_objects
        self._freq = frequency

        layout = QtWidgets.QVBoxLayout(self)

        # Frequency row
        freq_row = QtWidgets.QHBoxLayout()
        freq_row.addWidget(QtWidgets.QLabel("频率 f:"))
        self._freq_spin = QtWidgets.QDoubleSpinBox()
        self._freq_spin.setRange(0.1, 100.0)
        self._freq_spin.setDecimals(1)
        self._freq_spin.setSuffix(" GHz")
        self._freq_spin.setValue(frequency / 1e9)
        self._freq_spin.valueChanged.connect(self._on_freq_changed)
        freq_row.addWidget(self._freq_spin)
        freq_row.addStretch()
        freq_row.addWidget(QtWidgets.QLabel(
            "n = √(ε_r − jσ/ωε₀)  |  σ: 电导率 S/m（值越大损耗越大）"))
        layout.addLayout(freq_row)

        # Table
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "材料", "材料标签", "ε_r", "σ (S/m)", "厚度 (cm)",
            "Re(n)", "Im(n)"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("导出参数…")
        btn_export.clicked.connect(self._export)
        btn_row.addWidget(btn_export)
        btn_import = QtWidgets.QPushButton("导入参数…")
        btn_import.clicked.connect(self._import)
        btn_row.addWidget(btn_import)
        btn_row.addStretch()
        btn_defaults = QtWidgets.QPushButton("恢复默认")
        btn_defaults.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_defaults)
        btn_apply = QtWidgets.QPushButton("应用")
        btn_apply.clicked.connect(self._apply)
        btn_apply.setStyleSheet("QPushButton { font-weight: bold; }")
        btn_row.addWidget(btn_apply)
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self._populate()
        self._load_defaults()

    def _populate(self):
        """Fill table with all non-conductor scene layers."""
        self._table.setRowCount(0)
        self._rows = []
        omega = 2.0 * np.pi * self._freq
        eps0 = 8.854187817e-12

        for name, obj in self._scene.items():
            if name in ("antenna",):
                continue
            # Skip base visual layers (only show physical materials)
            if name.startswith("layer_") and "_clip" not in name:
                continue
            mat = None
            extra = obj.get("extra")
            if extra is not None:
                mat = extra.get("material")
            if mat is None:
                # Assign default
                label = LAYER_MATERIAL_MAP.get(name, "地面")
                mat = dict(DEFAULT_MATERIALS.get(label, DEFAULT_MATERIALS["地面"]))
                mat["label"] = label
                if extra is None:
                    obj["extra"] = {}
                obj["extra"]["material"] = mat
            # Skip conductors (PEC, no penetration params needed)
            if mat.get("sigma", 0) > 1e5:
                continue

            row = self._table.rowCount()
            self._table.insertRow(row)

            # Layer name (read-only)
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self._table.setItem(row, 0, item)

            # Material label (editable)
            item = QtWidgets.QTableWidgetItem(mat.get("label", ""))
            self._table.setItem(row, 1, item)

            # ε_r
            spin_eps = QtWidgets.QDoubleSpinBox()
            spin_eps.setRange(1.0, 100.0)
            spin_eps.setDecimals(2)
            spin_eps.setValue(mat.get("eps_r", 15.0))
            spin_eps.valueChanged.connect(
                lambda v, r=row: self._update_n(r))
            self._table.setCellWidget(row, 2, spin_eps)

            # σ
            spin_sig = QtWidgets.QDoubleSpinBox()
            spin_sig.setRange(1e-6, 100.0)
            spin_sig.setDecimals(6)
            spin_sig.setValue(mat.get("sigma", 0.01))
            spin_sig.valueChanged.connect(
                lambda v, r=row: self._update_n(r))
            self._table.setCellWidget(row, 3, spin_sig)

            # Thickness
            spin_thick = QtWidgets.QDoubleSpinBox()
            spin_thick.setRange(0.0, 1000.0)
            spin_thick.setDecimals(1)
            spin_thick.setValue(mat.get("thickness_cm", 0.0))
            spin_thick.setSuffix(" cm")
            self._table.setCellWidget(row, 4, spin_thick)

            # n = sqrt(ε_r - jσ/(ωε₀))  — computed, read-only
            eps_r = mat.get("eps_r", 15.0)
            sigma = mat.get("sigma", 0.01)
            eps_c = eps_r - 1j * sigma / (omega * eps0 + 1e-30)
            n_c = np.sqrt(eps_c)

            item_re = QtWidgets.QTableWidgetItem(f"{n_c.real:.4f}")
            item_re.setFlags(item_re.flags() & ~QtCore.Qt.ItemIsEditable)
            self._table.setItem(row, 5, item_re)

            item_im = QtWidgets.QTableWidgetItem(f"{n_c.imag:.4e}")
            item_im.setFlags(item_im.flags() & ~QtCore.Qt.ItemIsEditable)
            self._table.setItem(row, 6, item_im)

            self._rows.append((name, obj, spin_eps, spin_sig, spin_thick, item_re, item_im))

        self._table.resizeColumnsToContents()

    def _update_n(self, row):
        """Recompute n for row when ε_r or σ changes."""
        if row < 0 or row >= len(self._rows):
            return
        _, _, spin_eps, spin_sig, _, item_re, item_im = self._rows[row]
        eps_r = spin_eps.value()
        sigma = spin_sig.value()
        omega = 2.0 * np.pi * self._freq
        eps0 = 8.854187817e-12
        n_c = np.sqrt(eps_r - 1j * sigma / (omega * eps0 + 1e-30))
        item_re.setText(f"{n_c.real:.4f}")
        item_im.setText(f"{n_c.imag:.4e}")

    def _on_freq_changed(self, val):
        self._freq = val * 1e9
        self._table.setRowCount(0)
        self._rows = []
        self._populate()

    def _apply(self):
        """Write table values back to scene objects."""
        for name, obj, s_eps, s_sig, s_thick, _, _ in self._rows:
            extra = obj.get("extra")
            if extra is None:
                obj["extra"] = {}
                extra = obj["extra"]
            mat = extra.get("material", {})
            mat["eps_r"] = s_eps.value()
            mat["sigma"] = s_sig.value()
            mat["thickness_cm"] = s_thick.value()
            mat["label"] = self._table.item(
                self._table.indexAt(QtCore.QPoint()).row() if False else 0, 1
            ).text() if False else mat.get("label", "")
            extra["material"] = mat
        self.accept()

    def _load_defaults(self):
        """Fill missing materials with defaults (safe for missing user input)."""
        for name, obj in self._scene.items():
            if name == "antenna" or (name.startswith("layer_") and "_clip" not in name):
                continue
            extra = obj.get("extra")
            if extra is None:
                obj["extra"] = {}
                extra = obj["extra"]
            mat = extra.get("material")
            if mat is None:
                label = LAYER_MATERIAL_MAP.get(name, "地面")
                mat = dict(DEFAULT_MATERIALS.get(label, DEFAULT_MATERIALS["地面"]))
                mat["label"] = label
                extra["material"] = mat
            elif "thickness_cm" not in mat:
                mat["thickness_cm"] = 0.0

    def _reset_defaults(self):
        reply = QtWidgets.QMessageBox.question(
            self, "恢复默认", "将所有材料参数恢复为默认值？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        for name, obj in self._scene.items():
            if name == "antenna" or (name.startswith("layer_") and "_clip" not in name):
                continue
            label = LAYER_MATERIAL_MAP.get(name, "地面")
            dflt = DEFAULT_MATERIALS.get(label, DEFAULT_MATERIALS["地面"])
            extra = obj.get("extra", {})
            extra["material"] = {"label": label, **dflt}
            obj["extra"] = extra
        self._table.setRowCount(0)
        self._rows = []
        self._populate()

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出材料参数", "material_params.json", "JSON (*.json)")
        if not path:
            return
        data = {}
        for name, obj in self._scene.items():
            extra = obj.get("extra")
            if extra is None:
                continue
            mat = extra.get("material")
            if mat is None:
                continue
            data[name] = {
                "label": mat.get("label", ""),
                "eps_r": mat.get("eps_r", 15.0),
                "sigma": mat.get("sigma", 0.01),
                "thickness_cm": mat.get("thickness_cm", 0.0),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        QtWidgets.QMessageBox.information(self, "导出完成",
                                           f"已导出 {len(data)} 个材料参数到\n{path}")

    def _import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "导入材料参数", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "导入失败", str(e))
            return
        for name, params in data.items():
            obj = self._scene.get(name)
            if obj is None:
                continue
            extra = obj.get("extra", {})
            extra["material"] = {
                "label": params.get("label", ""),
                "eps_r": params.get("eps_r", 15.0),
                "sigma": params.get("sigma", 0.01),
                "thickness_cm": params.get("thickness_cm", 0.0),
            }
            obj["extra"] = extra
        self._table.setRowCount(0)
        self._rows = []
        self._populate()
        QtWidgets.QMessageBox.information(self, "导入完成",
                                           f"已导入 {len(data)} 个材料参数。")
