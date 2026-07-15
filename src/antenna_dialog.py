"""
天线设置对话框 — 选择天线类型、调整参数、预览垂直面方向图。
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.antenna_types import (
    ANTENNA_TYPE_LABELS,
    ANTENNA_PARAMS,
    DEFAULT_ANTENNA_CONFIG,
    _antenna_pattern,
)


class AntennaDialog(QtWidgets.QDialog):
    """天线参数设置对话框。"""

    def __init__(self, parent, antenna_config):
        super().__init__(parent)
        self.setWindowTitle("天线设置")
        self.setMinimumSize(650, 520)

        # 深拷贝配置
        self._config = dict(antenna_config) if antenna_config else dict(DEFAULT_ANTENNA_CONFIG)
        self._param_spinboxes = {}

        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # ── 左侧：参数面板 ──
        left = QtWidgets.QVBoxLayout()

        # 天线类型下拉框
        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel("天线类型:"))
        self._type_combo = QtWidgets.QComboBox()
        for key, label in ANTENNA_TYPE_LABELS.items():
            self._type_combo.addItem(label, key)
        # 选中当前类型
        cur_type = self._config.get("type", "half_wave_dipole")
        idx = self._type_combo.findData(cur_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo, 1)
        left.addLayout(type_row)

        # 分隔线
        left.addWidget(self._h_line())

        # 天线位置
        pos_group = QtWidgets.QGroupBox("天线位置")
        pos_layout = QtWidgets.QFormLayout()
        self._spin_pos_x = QtWidgets.QDoubleSpinBox()
        self._spin_pos_x.setRange(-100, 100); self._spin_pos_x.setDecimals(2)
        self._spin_pos_x.setSuffix(" m")
        self._spin_pos_x.valueChanged.connect(self._on_param_changed)
        pos_layout.addRow("X:", self._spin_pos_x)

        self._spin_pos_y = QtWidgets.QDoubleSpinBox()
        self._spin_pos_y.setRange(-100, 100); self._spin_pos_y.setDecimals(2)
        self._spin_pos_y.setSuffix(" m")
        self._spin_pos_y.valueChanged.connect(self._on_param_changed)
        pos_layout.addRow("Y:", self._spin_pos_y)

        self._spin_pos_z = QtWidgets.QDoubleSpinBox()
        self._spin_pos_z.setRange(-10, 200); self._spin_pos_z.setDecimals(2)
        self._spin_pos_z.setSuffix(" m")
        self._spin_pos_z.valueChanged.connect(self._on_param_changed)
        pos_layout.addRow("Z:", self._spin_pos_z)
        pos_group.setLayout(pos_layout)

        pos = self._config.get("position", (-5, 0, 6))
        self._spin_pos_x.setValue(pos[0])
        self._spin_pos_y.setValue(pos[1])
        self._spin_pos_z.setValue(pos[2])
        left.addWidget(pos_group)

        left.addWidget(self._h_line())

        # 频率
        freq_row = QtWidgets.QHBoxLayout()
        freq_row.addWidget(QtWidgets.QLabel("频率:"))
        self._freq_spin = QtWidgets.QDoubleSpinBox()
        self._freq_spin.setRange(0.1, 100.0)
        self._freq_spin.setDecimals(2)
        self._freq_spin.setSuffix(" GHz")
        self._freq_spin.setValue(self._config.get("frequency", 2.8e9) / 1e9)
        self._freq_spin.valueChanged.connect(self._on_param_changed)
        freq_row.addWidget(self._freq_spin)
        freq_row.addStretch()
        left.addLayout(freq_row)

        left.addWidget(self._h_line())

        # 动态参数区域
        self._params_group = QtWidgets.QGroupBox("天线参数")
        self._params_layout = QtWidgets.QFormLayout()
        self._params_group.setLayout(self._params_layout)
        left.addWidget(self._params_group)

        left.addStretch()

        # 按钮
        btn_row = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("恢复默认")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_ok = QtWidgets.QPushButton("确定 ✓")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet(
            "QPushButton { font-weight: bold; background: #4CAF50; color: white; padding: 4px 16px; }"
        )
        btn_row.addWidget(btn_ok)
        left.addLayout(btn_row)

        layout.addLayout(left, 1)

        # ── 右侧：方向图预览 ──
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("垂直面方向图"))
        self._fig = Figure(figsize=(4, 4))
        self._ax = self._fig.add_subplot(111, projection="polar")
        self._canvas = FigureCanvasQTAgg(self._fig)
        right.addWidget(self._canvas, 1)
        layout.addLayout(right, 2)

        # 初始化动态参数
        self._rebuild_params()

    def _h_line(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # ── 动态参数 ──

    def _on_type_changed(self):
        self._rebuild_params()
        self._update_preview()

    def _rebuild_params(self):
        # 清空现有参数控件
        while self._params_layout.rowCount() > 0:
            self._params_layout.removeRow(0)
        self._param_spinboxes.clear()

        ant_type = self._type_combo.currentData()
        params = ANTENNA_PARAMS.get(ant_type, [])

        for key, default, rng, step, label in params:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(rng[0], rng[1])
            spin.setSingleStep(step)
            spin.setDecimals(1)
            spin.setSuffix(f" {label.split('(')[-1].replace(')', '')}" if '(' in label else "")
            spin.setValue(self._config.get(key, default))
            spin.valueChanged.connect(self._on_param_changed)
            self._params_layout.addRow(f"{label}:", spin)
            self._param_spinboxes[key] = spin

        if not params:
            self._params_layout.addRow(
                QtWidgets.QLabel("此天线类型无可调参数"),
                QtWidgets.QWidget()
            )

    def _on_param_changed(self):
        self._config["frequency"] = self._freq_spin.value() * 1e9
        self._config["position"] = (
            self._spin_pos_x.value(),
            self._spin_pos_y.value(),
            self._spin_pos_z.value(),
        )
        for key, spin in self._param_spinboxes.items():
            self._config[key] = spin.value()
        self._config["type"] = self._type_combo.currentData()
        self._update_preview()

    def _reset_defaults(self):
        self._config = dict(DEFAULT_ANTENNA_CONFIG)
        self._config["position"] = (-5.0, 0.0, 6.0)
        self._config["frequency"] = 2.8e9
        self._spin_pos_x.setValue(-5.0)
        self._spin_pos_y.setValue(0.0)
        self._spin_pos_z.setValue(6.0)
        for key, spin in self._param_spinboxes.items():
            params = ANTENNA_PARAMS.get(self._type_combo.currentData(), [])
            for pk, default, _, _, _ in params:
                if pk == key:
                    spin.setValue(default)
                    break
        self._update_preview()

    # ── 方向图预览 ──

    def _update_preview(self):
        self._ax.clear()
        ant_type = self._type_combo.currentData()
        self._ax.set_title(
            f"{ANTENNA_TYPE_LABELS.get(ant_type, ant_type)}\n"
            f"{self._config.get('frequency', 2.8e9)/1e9:.1f} GHz",
            va="bottom", fontsize=10
        )

        h_ant = 0.0
        r0 = 1.0

        # 用固定 z 范围映射到仰角（避免 tan 在 ±90° 处发散）
        z_max = 20.0
        z_vals = np.linspace(-z_max, z_max, 721)
        valid = z_vals != h_ant
        theta_elev = np.arctan2(z_vals[valid], r0)

        pattern_raw = _antenna_pattern(self._config, z_vals[valid], h_ant, r0)
        # 极坐标 θ=0° 为天顶(N)，只绘制上半球 (θ_elev ∈ [-90°, 90°])
        polar_theta = np.pi / 2 - theta_elev  # θ_elev=90°→0, θ_elev=-90°→π

        # 按 polar_theta 升序排序（fill_between 要求 x 单调递增）
        sort_idx = np.argsort(polar_theta)
        pt_sorted = polar_theta[sort_idx]
        p_sorted = pattern_raw[sort_idx]

        self._ax.fill_between(pt_sorted, 0, p_sorted, alpha=0.3, color="tab:blue")
        self._ax.plot(pt_sorted, p_sorted, "b-", linewidth=1.5)
        self._ax.axhline(y=1 / np.sqrt(2), color="red", linestyle="--", linewidth=0.8, alpha=0.6)

        self._ax.set_theta_zero_location("N")
        self._ax.set_theta_direction(-1)
        self._ax.set_thetamin(0)
        self._ax.set_thetamax(180)
        self._ax.set_ylim(0, 1.1)
        self._ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        self._ax.set_yticklabels([])

        self._canvas.draw_idle()

    # ── 公共接口 ──

    def get_config(self):
        return {
            "type": self._type_combo.currentData(),
            "frequency": self._freq_spin.value() * 1e9,
            "position": (
                self._spin_pos_x.value(),
                self._spin_pos_y.value(),
                self._spin_pos_z.value(),
            ),
            "tilt_angle": self._config.get("tilt_angle", 0.0),
            "sigma_z": self._config.get("sigma_z", 4.0),
            "patch_hpbw": self._config.get("patch_hpbw", 70.0),
            "horn_hpbw": self._config.get("horn_hpbw", 30.0),
        }
