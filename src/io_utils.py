"""
IO 工具 — DEM（GeoTIFF/IMG）和 ASC（ESRI ASCII Raster）导入导出。
"""

import os
import numpy as np
import pyvista as pv


# ═══════════════════════════════════════════════════════════════
#  ASC — ESRI ASCII Raster 格式
# ═══════════════════════════════════════════════════════════════

def read_asc(filepath: str) -> dict:
    """读取 ESRI ASCII Raster 文件。

    返回
    ----
    dict:
        data     — 2D ndarray (nrows, ncols) 高程值
        xll      — 左下角 X 坐标（或 xllcenter）
        yll      — 左下角 Y 坐标（或 yllcenter）
        cellsize — 网格分辨率
        nodata   — 无数据值（或 None）
        ncols, nrows
    """
    header = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        # 读 6 行头
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            key, _, val = line.strip().partition(" ")
            key = key.strip().lower()
            if key in ("ncols", "nrows"):
                header[key] = int(val)
            elif key in ("xllcorner", "yllcorner", "xllcenter", "yllcenter",
                         "cellsize", "dx", "dy"):
                header[key] = float(val)
            elif key == "nodata_value":
                header[key] = float(val)
    # 统一 cellsize
    if "cellsize" not in header:
        header["cellsize"] = header.get("dx", header.get("dy", 30.0))
    # 统一左下角
    if "xllcorner" not in header:
        header["xllcorner"] = header.get("xllcenter", 0.0)
    if "yllcorner" not in header:
        header["yllcorner"] = header.get("yllcenter", 0.0)

    data = np.loadtxt(filepath, skiprows=6, dtype=np.float32)
    if data.shape != (header["nrows"], header["ncols"]):
        data = data.reshape(header["nrows"], header["ncols"])
    header["data"] = data
    return header


def write_asc(filepath: str, Z: np.ndarray,
              xll: float = 0.0, yll: float = 0.0,
              cellsize: float = 1.0, nodata: float = -9999.0):
    """将高程数组写入 ESRI ASCII Raster 文件。"""
    nrows, ncols = Z.shape
    Z_out = np.where(np.isfinite(Z), Z, nodata)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xll}\n")
        f.write(f"yllcorner     {yll}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  {nodata}\n")
        np.savetxt(f, Z_out, fmt="%.3f")


# ═══════════════════════════════════════════════════════════════
#  DEM — GeoTIFF / IMG（via rasterio）
# ═══════════════════════════════════════════════════════════════

def read_dem(filepath: str, downsample: int = 1) -> dict:
    """从 GeoTIFF / IMG 读取 DEM，可选降采样。

    参数
    ----
    filepath   : str  DEM 文件路径（.tif / .img）
    downsample : int  降采样因子（1=原始分辨率）

    返回
    ----
    dict  同 read_asc：data, xll, yll, cellsize, nodata, ncols, nrows
    """
    import rasterio

    with rasterio.open(filepath) as src:
        band = src.read(1)
        # 降采样
        if downsample > 1:
            band = band[::downsample, ::downsample]
        nrows, ncols = band.shape
        transform = src.transform
        cellsize = abs(transform[0]) * downsample
        xll = transform[2]  # 左上角 X
        yll = transform[5] - nrows * cellsize  # 左下角 Y（左上角 Y - nrows*cellsize）
        nodata = src.nodata

    data = band.astype(np.float32)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    return {
        "data": data,
        "xll": float(xll),
        "yll": float(yll),
        "cellsize": float(cellsize),
        "nodata": nodata,
        "ncols": ncols,
        "nrows": nrows,
    }


# ═══════════════════════════════════════════════════════════════
#  高程数据 → PyVista StructuredGrid（替换场景地形）
# ═══════════════════════════════════════════════════════════════

def dem_to_terrain_obj(dem_info: dict,
                       center: bool = True,
                       z_scale: float = 1.0) -> dict:
    """将 DEM/ASC 高程数据转换为场景 terrain 对象。

    参数
    ----
    dem_info : dict   read_dem / read_asc 的输出
    center   : bool   是否将 XY 坐标中心化
    z_scale  : float  垂直缩放因子

    返回
    ----
    dict  场景对象（可直接替换 scene_objects["terrain"]）
    """
    data = dem_info["data"]
    nrows, ncols = data.shape
    cellsize = dem_info["cellsize"]

    xs = np.linspace(0, (ncols - 1) * cellsize, ncols)
    ys = np.linspace(0, (nrows - 1) * cellsize, nrows)
    X, Y = np.meshgrid(xs, ys)

    if center:
        X -= X.mean()
        Y -= Y.mean()

    Z = np.nan_to_num(data, nan=0.0).astype(np.float32) * z_scale

    grid = pv.StructuredGrid(X.astype(np.float32),
                              Y.astype(np.float32),
                              Z)
    grid["elevation"] = Z.flatten(order="F")

    return {
        "mesh": grid,
        "type": "mesh",
        "visible": True,
        "params": {
            "color": "#b8956a",
            "smooth_shading": True,
            "opacity": 1.0,
            "ambient": 0.1,
            "diffuse": 0.9,
            "specular": 0.1,
            "specular_power": 10,
        },
        "extra": {
            "original_z": Z.copy(),
            "X": X.astype(np.float32),
            "Y": Y.astype(np.float32),
            "is_dem": True,
            "material": {
                "label": "干燥土壤",
                "eps_r": 15.0,
                "sigma": 0.01,
            },
        },
        "name": "terrain",
    }


def terrain_to_dem_info(terrain_obj: dict) -> dict:
    """从当前场景 terrain 对象提取高程信息（供导出 ASC 用）。"""
    extra = terrain_obj.get("extra") or {}
    Z = extra.get("original_z")
    X = extra.get("X")
    Y = extra.get("Y")

    if Z is None or X is None or Y is None:
        # 从 mesh 提取
        mesh = terrain_obj["mesh"]
        pts = np.asarray(mesh.points, dtype=np.float64)
        try:
            dims = mesh.dimensions
            nx, ny = dims[0], dims[1]
            Z_2d = pts[:, 2].reshape((nx, ny)).T
            xs = pts[::ny, 0]
            ys = pts[:ny, 1]
        except Exception:
            return None
    else:
        Z_2d = Z
        xs = X[0, :] if X.ndim == 2 else X
        ys = Y[:, 0] if Y.ndim == 2 else Y

    nrows, ncols = Z_2d.shape
    cellsize = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    xll = float(xs[0])
    yll = float(ys[0])

    return {
        "data": Z_2d.astype(np.float32),
        "xll": xll,
        "yll": yll,
        "cellsize": cellsize,
        "nodata": -9999.0,
        "ncols": ncols,
        "nrows": nrows,
    }
