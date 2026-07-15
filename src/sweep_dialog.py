"""
扫频模式对话框 — 多频率批量求解 + 结果表格 + 曲线图 + CSV 导出。
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SweepDialog(QtWidgets.QDialog):
    """扫频求解对话框。"""

    def __init__(self, parent, scene_objects, antenna_pos, rx_points):
        super().__init__(parent)
        self.setWindowTitle("扫频模式")
        self.setMinimumSize(820, 620)

        self._scene = scene_objects
        self._antenna_pos = antenna_pos
        self._rx_points = list(rx_points)
        self._results = []          # list of dict: freq, rx_results
        self._running = False

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # ── 顶部：参数设置 ──
        top = QtWidgets.QHBoxLayout()

        # 频率范围
        top.addWidget(QtWidgets.QLabel("频率:"))
        self._freq_start = QtWidgets.QDoubleSpinBox()
        self._freq_start.setRange(0.1, 100); self._freq_start.setDecimals(1)
        self._freq_start.setSuffix(" GHz"); self._freq_start.setValue(2.0)
        top.addWidget(self._freq_start)

        top.addWidget(QtWidgets.QLabel("–"))
        self._freq_stop = QtWidgets.QDoubleSpinBox()
        self._freq_stop.setRange(0.1, 100); self._freq_stop.setDecimals(1)
        self._freq_stop.setSuffix(" GHz"); self._freq_stop.setValue(4.0)
        top.addWidget(self._freq_stop)

        top.addWidget(QtWidgets.QLabel("  步数:"))
        self._n_steps = QtWidgets.QSpinBox()
        self._n_steps.setRange(3, 200); self._n_steps.setValue(11)
        top.addWidget(self._n_steps)

        # 线性/对数
        self._log_scale = QtWidgets.QCheckBox("对数")
        top.addWidget(self._log_scale)

        top.addSpacing(20)

        # 降分辨率加速
        self._fast_mode = QtWidgets.QCheckBox("快速模式 (N_Z=1024,N_PHI=64)")
        self._fast_mode.setChecked(True)
        self._fast_mode.setToolTip("分辨率减半，速度约 4×，精度略降")
        top.addWidget(self._fast_mode)

        top.addStretch()
        layout.addLayout(top)

        # ── 接收点信息 ──
        info = QtWidgets.QLabel(
            f"接收点: {len(self._rx_points)} 个  |  "
            f"天线: ({self._antenna_pos[0]:.1f}, {self._antenna_pos[1]:.1f}, {self._antenna_pos[2]:.1f})"
        )
        info.setStyleSheet("color: #666; padding: 2px;")
        layout.addWidget(info)

        # ── 按钮行 ──
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_run = QtWidgets.QPushButton("▶ 开始扫频")
        self._btn_run.clicked.connect(self._run_sweep)
        self._btn_run.setStyleSheet(
            "QPushButton { font-weight: bold; background: #2196F3; color: white; "
            "padding: 6px 20px; border-radius: 4px; }"
            "QPushButton:disabled { background: #ccc; }"
        )
        btn_row.addWidget(self._btn_run)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        btn_row.addWidget(self._progress, 1)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color: #888;")
        btn_row.addWidget(self._status_label)

        btn_row.addStretch()

        self._btn_csv = QtWidgets.QPushButton("导出 CSV")
        self._btn_csv.clicked.connect(self._export_csv)
        self._btn_csv.setEnabled(False)
        btn_row.addWidget(self._btn_csv)

        self._btn_plot = QtWidgets.QPushButton("绘制曲线")
        self._btn_plot.clicked.connect(self._show_plot)
        self._btn_plot.setEnabled(False)
        btn_row.addWidget(self._btn_plot)

        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        # ── 结果表格 ──
        self._table = QtWidgets.QTableWidget()
        n_rx = len(self._rx_points)
        self._table.setColumnCount(2 + n_rx * 3)
        headers = ["频率 (GHz)", "步"]
        for k in range(n_rx):
            rx = self._rx_points[k]
            headers += [
                f"Rx{k+1} L_pe(dB)",
                f"Rx{k+1} E(dBμV/m)",
                f"Rx{k+1} E_rx",
            ]
        self._table.setHorizontalHeaderLabels(headers)
        layout.addWidget(self._table, 1)

    # ── 扫频主体 ──

    def _run_sweep(self):
        if not self._rx_points:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加测量点")
            return

        self._running = True
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._results.clear()
        self._table.setRowCount(0)

        f_start = self._freq_start.value() * 1e9
        f_stop = self._freq_stop.value() * 1e9
        n = self._n_steps.value()
        use_log = self._log_scale.isChecked()
        fast = self._fast_mode.isChecked()

        if use_log:
            freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n)
        else:
            freqs = np.linspace(f_start, f_stop, n)

        self._progress.setMaximum(n)
        self._progress.setValue(0)

        nz = 1024 if fast else 2048
        nphi = 64 if fast else 128

        for i, freq in enumerate(freqs):
            if not self._running:
                break
            self._status_label.setText(f"计算 {freq/1e9:.1f} GHz ({i+1}/{n})…")
            self._progress.setValue(i)
            QtWidgets.QApplication.processEvents()

            try:
                from src.cpe_solver import CPESolver2D
                solver = CPESolver2D(
                    frequency=freq,
                    antenna_pos=self._antenna_pos,
                    scene_objects=self._scene,
                    n_z=nz, n_phi=nphi,
                )
                rx_results = []
                for rx in self._rx_points:
                    res = solver.compute(rx)
                    E_dbuv = 20.0 * np.log10(max(res["E_rx"], 1e-15) / 1e-6)
                    rx_results.append({
                        "L_pe": res["path_loss_dB"],
                        "E_dbuv": E_dbuv,
                        "E_rx": res["E_rx"],
                        "L_fs": res["L_fs_dB"],
                    })
                self._results.append({
                    "freq": freq,
                    "rx_results": rx_results,
                })
            except Exception as e:
                self._status_label.setText(f"错误 @ {freq/1e9:.1f} GHz: {e}")
                break

        self._progress.setValue(n)
        self._running = False
        self._btn_run.setEnabled(True)
        self._status_label.setText(f"完成 — {len(self._results)} 个频点")
        self._btn_csv.setEnabled(len(self._results) > 0)
        self._btn_plot.setEnabled(len(self._results) > 0)

        self._fill_table()

    def _fill_table(self):
        self._table.setRowCount(len(self._results))
        for i, r in enumerate(self._results):
            freq_ghz = r["freq"] / 1e9
            self._table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{freq_ghz:.3f}"))
            self._table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(i + 1)))
            for k, rxr in enumerate(r["rx_results"]):
                col = 2 + k * 3
                self._table.setItem(i, col,
                    QtWidgets.QTableWidgetItem(f"{rxr['L_pe']:.3f}"))
                self._table.setItem(i, col + 1,
                    QtWidgets.QTableWidgetItem(f"{rxr['E_dbuv']:.3f}"))
                self._table.setItem(i, col + 2,
                    QtWidgets.QTableWidgetItem(f"{rxr['E_rx']:.6e}"))
        self._table.resizeColumnsToContents()

    # ── 导出 ──

    def _export_csv(self):
        import csv
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出 CSV", "sweep_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            n_rx = len(self._rx_points)
            header = ["Freq_GHz"]
            for k in range(n_rx):
                header += [f"Rx{k+1}_L_pe_dB", f"Rx{k+1}_E_dBuVpm", f"Rx{k+1}_E_rx_Vpm"]
            w.writerow(header)
            for r in self._results:
                row = [r["freq"] / 1e9]
                for rxr in r["rx_results"]:
                    row += [rxr["L_pe"], rxr["E_dbuv"], rxr["E_rx"]]
                w.writerow(row)
        self._status_label.setText(f"已导出: {path}")

    def _show_plot(self):
        if not self._results:
            return

        freqs_ghz = np.array([r["freq"] for r in self._results]) / 1e9
        n_rx = len(self._rx_points)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("扫频曲线")
        dlg.setMinimumSize(700, 450)
        lo = QtWidgets.QVBoxLayout(dlg)

        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasQTAgg(fig)
        lo.addWidget(canvas)

        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
        for k in range(n_rx):
            lpe = [r["rx_results"][k]["L_pe"] for r in self._results]
            rx = self._rx_points[k]
            label = f"Rx{k+1} ({rx[0]:.1f},{rx[1]:.1f},{rx[2]:.1f})"
            ax.plot(freqs_ghz, lpe, "o-", color=colors[k % len(colors)],
                    markersize=4, linewidth=1.5, label=label)

        ax.set_xlabel("频率 (GHz)")
        ax.set_ylabel("路径损耗 L_pe (dB)")
        ax.set_title("扫频结果 — 路径损耗 vs 频率")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        if self._log_scale.isChecked():
            ax.set_xscale("log")

        canvas.draw_idle()

        btn = QtWidgets.QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        lo.addWidget(btn)
        dlg.exec_()

    def closeEvent(self, event):
        self._running = False
        super().closeEvent(event)
