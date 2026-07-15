#!/usr/bin/env python3
"""
CPE field strength calculator — entry point.

Usage
-----
    python run.py                        # compute test points + table
    python run.py --plot                 # compute + generate plots in data/
    python run.py --rx 5,0,3             # single receiver
    python run.py --freq 1e9 --rx 8,0,5  # custom frequency
    python run.py --help                 # all options
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.core.config import EMConfig
from src.core.scheduler import Scheduler
from src.simple_scene_builder import build_simple_scene


def main():
    p = argparse.ArgumentParser(
        description="CPE field strength calculator (SSFT-PE)"
    )
    p.add_argument("--freq", type=float, default=2.8e9,
                   help="Frequency in Hz (default: 2.8e9 = 2.8 GHz)")
    p.add_argument("--tx", type=str, default="-5,0,6",
                   help="Antenna position x,y,z (default: -5,0,6)")
    p.add_argument("--rx", type=str, default=None,
                   help="Receiver position x,y,z. Omit to run default test set.")
    p.add_argument("--nz", type=int, default=256,
                   help="Height FFT points, power of 2 (default: 256)")
    p.add_argument("--dr", type=float, default=1.0,
                   help="Range step in wavelengths (default: 1.0)")
    p.add_argument("--plot", action="store_true",
                   help="Generate and save visualisation plots to data/")
    args = p.parse_args()

    # Parse positions
    tx = tuple(float(v) for v in args.tx.split(","))
    if len(tx) != 3:
        p.error("--tx must be x,y,z")

    # Build scene
    print("Building test scene …")
    scene = build_simple_scene()
    print(f"  Objects: {sorted(scene.keys())}")

    for name, obj in scene.items():
        b = obj["mesh"].bounds
        print(f"  {name}: bounds [{b[0]:.3f},{b[1]:.3f}, "
              f"{b[2]:.3f},{b[3]:.3f}, {b[4]:.3f},{b[5]:.3f}]")

    # Config
    config = EMConfig(
        frequency=args.freq,
        antenna_pos=tx,
        n_z=args.nz,
        dr_factor=args.dr,
    )

    # Receiver list
    if args.rx:
        rx_positions = [tuple(float(v) for v in args.rx.split(","))]
        if len(rx_positions[0]) != 3:
            p.error("--rx must be x,y,z")
    else:
        ftx = tx[2]
        rx_positions = [
            (-2, 0, ftx),
            (3, 0, ftx),
            (3, 0, 2),
            (3, 0, 4),
            (3, 0, 7),
            (8, 0, 2),
            (8, 0, ftx),
        ]

    # ── Pipeline ──────────────────────────────────────
    print(f"\nInitialising pipeline @ {args.freq/1e9:.1f} GHz …")
    print(f"  λ = {config.wavelength:.3f} m,  k₀ = {config.k0:.1f} rad/m")

    sched = Scheduler()
    results = sched.run(config=config, rx_points=rx_positions, scene=scene)

    # ── Output ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Antenna: ({tx[0]:.0f}, {tx[1]:.0f}, {tx[2]:.0f}) m")
    print(f"{'='*70}")
    print(f"  {'Rx (x,y,z)':<26s} {'Dist(m)':>10s}  {'L_fs(dB)':>10s}  "
          f"{'L_pe(dB)':>10s}  {'Δ(dB)':>9s}  Notes")
    print(f"  {'-'*80}")

    last_result = None
    for result in results:
        rx = result["rx"]
        r = result["dist"]
        last_result = result

        delta = result["path_loss_dB"] - result["L_fs_dB"]
        if delta > 15:
            note = "deep shadow"
        elif delta > 5:
            note = "diffraction loss"
        elif delta > 0.5:
            note = "slight obstruction"
        else:
            note = "≈ free space"

        print(f"  ({rx[0]:6.1f},{rx[1]:6.1f},{rx[2]:6.1f})  "
              f"{r:10.3f}  {result['L_fs_dB']:10.3f}  "
              f"{result['path_loss_dB']:10.3f}  {delta:+9.3f}  {note}")

    print(f"{'='*70}\n")

    if args.plot and last_result is not None:
        from src.visualizer import save_all_plots
        save_all_plots(last_result, tx, rx_positions, scene, args.freq)

    print("Done.  Run with --help for options.")


if __name__ == "__main__":
    main()
