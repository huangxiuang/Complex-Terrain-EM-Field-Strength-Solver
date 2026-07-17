"""
模版测量点选择对话框。
"""

from PyQt5 import QtWidgets, QtCore
from src.template_points import TEMPLATES


class TemplateDialog(QtWidgets.QDialog):
    def __init__(self, parent, scene_key):
        super().__init__(parent)
        self.setWindowTitle("预设模版")
        self.setMinimumSize(500, 380)
        self._scene_key = scene_key
        self._selected = None

        scene_info = TEMPLATES.get(scene_key, {})
        scene_name = scene_info.get("name", scene_key)
        templates = scene_info.get("templates", [])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"场景：{scene_name}"))
        layout.addWidget(QtWidgets.QLabel("选择一个测量点模版（将追加到现有测量点）："))

        self._list = QtWidgets.QListWidget()
        for t in templates:
            item = QtWidgets.QListWidgetItem(f"{t['name']}")
            item.setData(QtCore.Qt.UserRole, t)
            item.setToolTip(t["desc"])
            self._list.addItem(item)
        layout.addWidget(self._list, 1)

        self._desc = QtWidgets.QLabel("")
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color: #555; padding: 6px;")
        layout.addWidget(self._desc)
        self._list.currentItemChanged.connect(self._on_select)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

        btn = QtWidgets.QHBoxLayout()
        btn.addStretch()
        btn_cancel = QtWidgets.QPushButton("取消"); btn_cancel.clicked.connect(self.reject)
        btn.addWidget(btn_cancel)
        btn_ok = QtWidgets.QPushButton("加载模版 ✓")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet("QPushButton { font-weight: bold; background: #2196F3; color: white; padding: 4px 16px; }")
        btn.addWidget(btn_ok)
        layout.addLayout(btn)

    def _on_select(self, current, _prev):
        if current:
            t = current.data(QtCore.Qt.UserRole)
            pts = t.get("points", [])
            self._desc.setText(f"{t['desc']}\n共 {len(pts)} 个测量点")
            self._selected = t

    def selected_template(self):
        return self._selected
