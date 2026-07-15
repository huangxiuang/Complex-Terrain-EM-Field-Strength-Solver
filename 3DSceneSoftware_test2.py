#!/usr/bin/env python3
"""
3D EM Solver — CPE 场强计算 + 3D 场景可视化

Usage:  python 3DSceneSoftware_test2.py
"""

import sys
import os
os.environ["QT_MAC_WANTS_LAYER"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from PyQt5 import QtWidgets, QtCore

from src.main_window import MainWindow


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("3D EM Solver")

    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
