"""
Visualisation tools for CPE solver output.

Saves publication-quality PNG figures to the data/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib import ticker


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DPI = 150
CMAP = "inferno"


def save_all_plots(result, antenna_pos, rx_positions, scene, frequency):
    """Generate and save all visualisations to data/.

    Parameters
    ----------
    result : dict
        Output of CPESolver.compute() for the LAST receiver.
    antenna_pos : tuple
    rx_positions : list of tuples
    scene : dict
    frequency : float
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    r_vals = result["r_grid"]
    z_vals = result["z_grid"]
    field_rz = result["field_rz"]          # 2D: (N_r, N_z)  |u|
    z_terrain = result["z_terrain"]
    freq_ghz = frequency / 1e9

    _plot_field_2d(r_vals, z_vals, field_rz, z_terrain,
                   antenna_pos, freq_ghz)
    _plot_field_vs_height(r_vals, z_vals, field_rz, z_terrain,
                          antenna_pos, freq_ghz)
    _plot_path_loss_vs_range(r_vals, z_vals, field_rz, z_terrain,
                             antenna_pos, freq_ghz)
    _plot_terrain_profile(r_vals, z_terrain, antenna_pos, freq_ghz)
    _plot_free_space_comparison(r_vals, z_vals, field_rz,
                                antenna_pos, freq_ghz)
    _plot_interference_detail(r_vals, z_vals, field_rz, z_terrain,
                              antenna_pos, freq_ghz)

    print(f"Saved {6} plots → {OUT_DIR}/")


# ═══════════════════════════════════════════════════════════════

def _plot_field_2d(r, z, field, terrain, ant, fg):
    """Pseudocolor: |u(r, z)| in dB with terrain overlay."""
    field_db = 20.0 * np.log10(np.maximum(field, 1e-12))
    vmin = field_db.max() - 60
    vmax = field_db.max()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(r, z, field_db.T, shading="auto",
                        cmap=CMAP, vmin=vmin, vmax=vmax)

    # Terrain boundary
    ax.fill_between(r, terrain, terrain.min() - 5,
                     color="black", alpha=0.5, linewidth=0)

    # Antenna marker
    ax.plot(r[0], ant[2], "o", color="cyan", markersize=10,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.annotate("Tx", (r[0], ant[2]),
                textcoords="offset points", xytext=(10, 8),
                color="white", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, label="Field strength (dB)")
    ax.set_xlabel("Range r (m)")
    ax.set_ylabel("Height z (m)")
    ax.set_title(f"CPE Field Distribution  |  f = {fg:.1f} GHz")
    ax.set_ylim(terrain.min() - 1, z.max())
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "field_2d.png"), dpi=DPI)
    plt.close(fig)


def _plot_field_vs_height(r, z, field, terrain, ant, fg):
    """Field strength vs height at 3 selected ranges."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Choose ranges: before wall, just after wall, far
    n = len(r)
    idx_slices = [max(0, n // 3), max(0, n // 2), n - 1]

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    labels = [f"r = {r[i]:.1f} m" for i in idx_slices]

    for i, c, lb in zip(idx_slices, colors, labels):
        field_db = 20.0 * np.log10(np.maximum(field[i, :], 1e-12))
        ax.plot(field_db, z, color=c, linewidth=1.5, label=lb)
        # Mark terrain at this range
        if i < len(terrain):
            ax.axhline(terrain[i], color=c, linestyle="--", alpha=0.4)

    # Antenna reference line
    ax.axhline(ant[2], color="gray", linestyle=":", alpha=0.5)
    ax.annotate("Tx height", xy=(0.02, ant[2]),
                xycoords=("axes fraction", "data"),
                fontsize=9, color="gray")

    ax.set_xlabel("Field strength (dB)")
    ax.set_ylabel("Height z (m)")
    ax.set_title(f"Field vs Height at Different Ranges  |  f = {fg:.1f} GHz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "field_vs_height.png"), dpi=DPI)
    plt.close(fig)


def _plot_path_loss_vs_range(r, z, field, terrain, ant, fg):
    """Path loss vs range at several fixed heights."""
    fig, ax = plt.subplots(figsize=(8, 5))

    heights = [2.0, 4.0, 6.0, 8.0, 10.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(heights)))

    for h, c in zip(heights, colors):
        z_idx = np.argmin(np.abs(z - h))
        field_at_h = field[:, z_idx]
        field_db = 20.0 * np.log10(np.maximum(field_at_h, 1e-12))

        # Normalise: reference = max value at this height
        ref = field_db.max()
        loss = ref - field_db
        ax.plot(r, loss, color=c, linewidth=1.2, label=f"z = {h:.0f} m")

    # Mark wall position with terrain
    wall_idx = np.argmax(terrain)
    if terrain[wall_idx] > 1:
        ax.axvline(r[wall_idx], color="red", linestyle="--", alpha=0.5)
        ax.annotate("wall", xy=(r[wall_idx], 0), xytext=(r[wall_idx]+0.3, 3),
                    fontsize=9, color="red")

    ax.set_xlabel("Range r (m)")
    ax.set_ylabel("Relative loss (dB)")
    ax.set_title(f"Path Loss vs Range at Fixed Heights  |  f = {fg:.1f} GHz")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "path_loss_vs_range.png"), dpi=DPI)
    plt.close(fig)


def _plot_terrain_profile(r, terrain, ant, fg):
    """Terrain elevation profile along the propagation path."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: terrain profile
    ax1.fill_between(r, terrain, terrain.min() - 2,
                      color="saddlebrown", alpha=0.4)
    ax1.plot(r, terrain, color="saddlebrown", linewidth=2)
    ax1.plot(r[0], ant[2], "o", color="cyan", markersize=10, zorder=5)
    ax1.annotate(f"Tx ({ant[0]:.0f},{ant[1]:.0f},{ant[2]:.0f})",
                 (r[0], ant[2]), textcoords="offset points",
                 xytext=(10, 8), fontsize=10, fontweight="bold")

    # Antenna LOS lines at ±30° (typical PE valid range)
    los_up = ant[2] + r * np.tan(np.deg2rad(30))
    los_dn = np.maximum(ant[2] - r * np.tan(np.deg2rad(30)), terrain)
    ax1.fill_between(r, los_dn, los_up, color="cyan", alpha=0.06)
    ax1.annotate("±30° beam", (r[-1]*0.6, los_up[int(len(r)*0.6)]+1),
                 fontsize=9, color="cyan", alpha=0.7)

    ax1.set_ylabel("Height z (m)")
    ax1.set_title(f"Terrain Profile & Antenna Geometry  |  f = {fg:.1f} GHz")
    ax1.grid(True, alpha=0.3)

    # Bottom: terrain derivative (slope)
    slope = np.gradient(terrain, r[1] - r[0])
    ax2.plot(r, slope, color="darkred", linewidth=1)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax2.fill_between(r, 0, slope, where=(slope>0),
                      color="darkred", alpha=0.3)
    ax2.fill_between(r, 0, slope, where=(slope<0),
                      color="steelblue", alpha=0.3)
    ax2.set_xlabel("Range r (m)")
    ax2.set_ylabel("Slope (dz/dr)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "terrain_profile.png"), dpi=DPI)
    plt.close(fig)


def _plot_free_space_comparison(r, z, field, ant, fg):
    """Compare PE field at antenna height with free-space 1/r decay."""
    fig, ax = plt.subplots(figsize=(8, 5))

    z_idx = np.argmin(np.abs(z - ant[2]))
    field_pe = np.abs(field[:, z_idx])

    # Free-space: 1/r decay from r0
    r0 = r[0]
    field_fs = r0 / r

    ax.plot(r, 20*np.log10(field_pe), color="#2196F3", linewidth=1.8,
            label="PE (SSFT)")
    ax.plot(r, 20*np.log10(field_fs), color="gray", linewidth=1.2,
            linestyle="--", label="Free space (1/r)")

    # Difference
    diff = 20*np.log10(np.maximum(field_pe / field_fs, 1e-12))
    ax2 = ax.twinx()
    ax2.plot(r, diff, color="#FF5722", linewidth=1, alpha=0.7,
             label="Δ (PE − FS)")
    ax2.set_ylabel("Δ (dB)", color="#FF5722")
    ax2.axhline(0, color="#FF5722", linestyle=":", alpha=0.3)

    ax.set_xlabel("Range r (m)")
    ax.set_ylabel("Field strength (dB)")
    ax.set_title(f"PE vs Free Space at z = {ant[2]:.0f} m  |  f = {fg:.1f} GHz")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "free_space_comparison.png"), dpi=DPI)
    plt.close(fig)


def _plot_interference_detail(r, z, field, terrain, ant, fg):
    """Zoomed view of field near the obstacle for interference analysis."""
    # Find the region around the obstacle
    obs_ranges = np.where(terrain > 0.5)[0]
    if len(obs_ranges) == 0:
        return  # No obstacle, skip

    r_centre = r[obs_ranges[len(obs_ranges)//2]]
    r_min = max(r_centre - 5, r[0])
    r_max = min(r_centre + 8, r[-1])
    z_max = min(terrain.max() + 10, z.max())

    r_mask = (r >= r_min) & (r <= r_max)
    z_mask = z <= z_max

    r_sub = r[r_mask]
    z_sub = z[z_mask]
    field_sub = field[np.ix_(r_mask, z_mask)]
    field_db = 20.0 * np.log10(np.maximum(field_sub, 1e-12))
    vmax = field_db.max()
    vmin = vmax - 40

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(r_sub, z_sub, field_db.T, shading="auto",
                        cmap=CMAP, vmin=vmin, vmax=vmax)

    # Terrain
    t_sub = terrain[r_mask]
    ax.fill_between(r_sub, t_sub, t_sub.min() - 2,
                     color="black", alpha=0.6, linewidth=0)

    cbar = fig.colorbar(im, ax=ax, label="Field (dB)")
    ax.set_xlabel("Range r (m)")
    ax.set_ylabel("Height z (m)")
    ax.set_title(f"Interference Pattern Near Obstacle  |  f = {fg:.1f} GHz")

    # Annotate diffraction zones
    y_mid = (z.max() + terrain[r_mask].max()) / 2
    x_mid = r_centre
    ax.annotate("Shadow", (x_mid + 1, 1.5), fontsize=13,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
    ax.annotate("LOS", (x_mid + 1, terrain[r_mask].max() + 1), fontsize=13,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="navy", alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "interference_detail.png"), dpi=DPI)
    plt.close(fig)
