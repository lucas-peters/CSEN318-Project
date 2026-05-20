#!/usr/bin/env python3
"""
Chiplet-Based Multi-Systolic Array Simulator

Usage:
    python run_chiplet_sim.py                     # full sweep + plots
    python run_chiplet_sim.py --quick             # small sweep for testing
    python run_chiplet_sim.py --plot-only         # regenerate plots from CSV
    python run_chiplet_sim.py --validate          # compare analytical model vs SCALE-Sim
    python run_chiplet_sim.py --single            # single-point evaluation for debugging
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chiplet_sim.compute import analytical_compute, scalesim_compute
from chiplet_sim.interconnect import Interconnect, Topology
from chiplet_sim.partition import evaluate_all_strategies
from chiplet_sim.workloads import (
    ALL_WORKLOADS, RESNET50_REPRESENTATIVE, BERT_LAYERS,
    bert_attention_workloads, GEMMWorkload
)
from chiplet_sim.sweep import run_sweep, run_bandwidth_crossover_sweep
from chiplet_sim.plot import plot_all


def cmd_single():
    """Run a single-point evaluation to verify the model works."""
    print("=" * 70)
    print("SINGLE-POINT EVALUATION")
    print("=" * 70)

    # ResNet-50 conv4_1_2: 14×14, 3×3, 256→256
    # GEMM equivalent: M=196, N=256, K=2304
    wl = RESNET50_REPRESENTATIVE[3]
    print(f"\nWorkload: {wl}")

    array_h, array_w = 128, 128
    dataflow = "os"

    # Single chiplet baseline
    baseline = analytical_compute(wl.m, wl.n, wl.k, array_h, array_w, dataflow)
    print(f"\nSingle chiplet ({array_h}×{array_w}, {dataflow}):")
    print(f"  Cycles:     {baseline.total_cycles:>12,}")
    print(f"  Map. Eff:   {baseline.mapping_efficiency:>11.1%}")
    print(f"  Comp. Util: {baseline.compute_utilization:>11.1%}")

    # 4-chiplet evaluation across topologies
    for topo in [Topology.RING, Topology.ALL_TO_ALL]:
        print(f"\n4 chiplets, {topo.value} topology, 32 B/cyc link BW:")
        ic = Interconnect(4, topo, link_bw=32, link_latency=10)

        results = evaluate_all_strategies(
            wl.m, wl.n, wl.k, array_h, array_w, dataflow, 4, ic
        )
        for r in results:
            print(f"  {r.strategy:>10s}: "
                  f"speedup={r.speedup_no_overlap:.2f}x "
                  f"(eff={r.efficiency_no_overlap:.1%}), "
                  f"compute={r.compute.total_cycles:,}, "
                  f"dist={r.distribute.cycles:,}, "
                  f"reduce={r.reduce.cycles:,}")

    # BERT attention for contrast
    wl2 = BERT_LAYERS[0]
    print(f"\n{'=' * 70}")
    print(f"Workload: {wl2}")

    baseline2 = analytical_compute(wl2.m, wl2.n, wl2.k, array_h, array_w, dataflow)
    print(f"\nSingle chiplet: {baseline2.total_cycles:,} cycles")

    ic = Interconnect(4, Topology.RING, link_bw=32, link_latency=10)
    results2 = evaluate_all_strategies(
        wl2.m, wl2.n, wl2.k, array_h, array_w, dataflow, 4, ic
    )
    for r in results2:
        print(f"  {r.strategy:>10s}: "
              f"speedup={r.speedup_no_overlap:.2f}x "
              f"(compute={r.compute.total_cycles:,}, "
              f"dist={r.distribute.cycles:,}, "
              f"reduce={r.reduce.cycles:,})")


def cmd_validate():
    """Compare analytical model against SCALE-Sim for a set of configs."""
    print("=" * 70)
    print("VALIDATION: Analytical vs SCALE-Sim")
    print("=" * 70)

    test_cases = [
        (1024, 256, 256),
        (512, 512, 64),
        (196, 256, 2304),
        (49, 512, 4608),
    ]

    for m, n, k in test_cases:
        for df in ["os", "ws", "is"]:
            ana = analytical_compute(m, n, k, 128, 128, df)
            sim = scalesim_compute(m, n, k, 128, 128, df)

            ratio = sim.total_cycles / ana.total_cycles if ana.total_cycles > 0 else 0
            print(f"GEMM({m:>4},{n:>4},{k:>4}) {df}: "
                  f"analytical={ana.total_cycles:>10,}  "
                  f"scalesim={sim.total_cycles:>10,}  "
                  f"ratio={ratio:.2f}  "
                  f"stalls={sim.stall_cycles:,}")


def cmd_sweep(quick=False):
    """Run the full parameter sweep."""
    if quick:
        workloads = [RESNET50_REPRESENTATIVE[3], BERT_LAYERS[0]]
        bandwidths = [4, 16, 64, 256]
        chiplet_counts = [1, 2, 4]
        topologies = [Topology.RING]
    else:
        workloads = ALL_WORKLOADS
        bandwidths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        chiplet_counts = [1, 2, 4, 8]
        topologies = [Topology.RING, Topology.MESH, Topology.ALL_TO_ALL]

    print("Running main sweep...")
    run_sweep(
        array_h=128, array_w=128,
        chiplet_counts=chiplet_counts,
        bandwidths=bandwidths,
        topologies=topologies,
        dataflows=["os", "ws"],
        workloads=workloads,
    )

    print("\nRunning crossover sweep...")
    run_bandwidth_crossover_sweep(
        array_h=128, array_w=128,
        workloads=workloads,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Chiplet-based multi-systolic array simulator"
    )
    parser.add_argument("--single", action="store_true",
                        help="Single-point evaluation for debugging")
    parser.add_argument("--validate", action="store_true",
                        help="Validate analytical model against SCALE-Sim")
    parser.add_argument("--quick", action="store_true",
                        help="Small sweep for testing")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate plots from existing CSV")
    args = parser.parse_args()

    if args.single:
        cmd_single()
    elif args.validate:
        cmd_validate()
    elif args.plot_only:
        plot_all()
    else:
        cmd_sweep(quick=args.quick)
        plot_all()


if __name__ == "__main__":
    main()
