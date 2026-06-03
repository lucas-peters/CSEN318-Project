#!/usr/bin/env python3
"""
Chiplet-Based Multi-Systolic Array Simulator

Usage:
    python run_chiplet_sim.py                     # full sweep + all plots
    python run_chiplet_sim.py --quick             # small sweep for testing
    python run_chiplet_sim.py --plot-only         # regenerate plots from CSV
    python run_chiplet_sim.py --single            # single-point debug evaluation
    python run_chiplet_sim.py --validate          # quick analytical vs SCALE-Sim check

Per-experiment runners (each does its own targeted sweep then plots):
    python run_chiplet_sim.py --exp1              # crossover B* + correlation
    python run_chiplet_sim.py --exp2              # topology comparison at P=8
    python run_chiplet_sim.py --exp3              # strategy phase diagram
    python run_chiplet_sim.py --exp4              # scaling efficiency
    python run_chiplet_sim.py --exp5              # compute model validation
    python run_chiplet_sim.py --all-experiments   # run Exp 1-5 in sequence
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chiplet_sim.compute import analytical_compute, scalesim_compute
from chiplet_sim.interconnect import Interconnect, Topology
from chiplet_sim.partition import evaluate_all_strategies
from chiplet_sim.schedule import schedule_layers
from chiplet_sim.workloads import (
    ALL_WORKLOADS, RESNET50_REPRESENTATIVE, BERT_LAYERS,
    REPORT_WORKLOADS, VALIDATION_WORKLOADS,
    bert_attention_workloads, GEMMWorkload,
)
from chiplet_sim.sweep import (
    run_sweep, run_bandwidth_crossover_sweep,
    run_phase_diagram_sweep, run_scaling_sweep, run_validation_sweep,
)
from chiplet_sim.plot import (
    plot_all, plot_roofline, plot_crossover, plot_strategy_map,
    plot_crossover_bstar, plot_topology_comparison,
    plot_phase_diagram, plot_scaling_efficiency, plot_validation_mape,
)


# ── Experiment 1: Crossover bandwidth and B* correlation ──────────────────────

def cmd_exp1():
    """
    Experiment 1: For each workload and P in {2, 4, 8}, sweep B from 1 to
    2048 bytes/cycle on a ring topology.  Extract B* and correlate with the
    GEMM's compute-to-communication ratio.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Crossover Bandwidth and B* Correlation")
    print("=" * 70)
    print("Running crossover sweep (ring topology, P in {2,4,8})...")
    run_bandwidth_crossover_sweep(
        array_h=128, array_w=128,
        workloads=REPORT_WORKLOADS,
        output_csv="crossover_results.csv",
    )
    print("Generating plots...")
    plot_crossover(output_dir="plots")
    plot_crossover_bstar(output_dir="plots")
    print("Exp 1 complete.")


# ── Experiment 2: Topology comparison ────────────────────────────────────────

def cmd_exp2():
    """
    Experiment 2: Fix P=8, sweep B in {16, 64, 256}.  Compare ring, 2D mesh
    (2×4), all-to-all, and hierarchical topologies on all workloads.

    Data is drawn from the main sweep CSV.  If it doesn't exist, a targeted
    sweep is run first.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Topology Comparison  (P=8, ring/mesh/all-to-all/hier)")
    print("=" * 70)

    print("Running topology comparison sweep (P=8, all topologies)...")
    run_sweep(
        array_h=128, array_w=128,
        chiplet_counts=[8],
        bandwidths=[16, 64, 256],
        topologies=[Topology.RING, Topology.MESH,
                    Topology.ALL_TO_ALL, Topology.HIERARCHICAL],
        dataflows=["os"],
        workloads=REPORT_WORKLOADS,
        output_csv="exp2_sweep_results.csv",
    )
    print("Generating topology comparison plots...")
    plot_topology_comparison(sweep_csv="exp2_sweep_results.csv", output_dir="plots")
    print("Exp 2 complete.")


# ── Experiment 3: Partition strategy phase diagram ────────────────────────────

def cmd_exp3():
    """
    Experiment 3: 2D heatmap per workload with B on x-axis (1–256 B/cyc) and
    P on y-axis (2–8, every integer).  Each cell shows which partition strategy
    minimises execution time.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Partition Strategy Phase Diagram  (P=2–8, ring)")
    print("=" * 70)
    print("Running phase diagram sweep...")
    run_phase_diagram_sweep(
        array_h=128, array_w=128,
        workloads=REPORT_WORKLOADS,
        output_csv="phase_diagram_results.csv",
    )
    print("Generating phase diagram plots...")
    plot_phase_diagram(output_dir="plots")
    print("Exp 3 complete.")


# ── Experiment 4: Scaling efficiency ──────────────────────────────────────────

def cmd_exp4():
    """
    Experiment 4: Speedup and efficiency vs. P (1–8) at B in {16, 64, 256}
    bytes/cycle using the best strategy at each point.  Decomposes T_multi
    into T_compute and T_comm.
    """
    print("=" * 70)
    print("EXPERIMENT 4: Scaling Efficiency  (P=1–8, B in {16,64,256})")
    print("=" * 70)
    print("Running scaling sweep...")
    run_scaling_sweep(
        array_h=128, array_w=128,
        workloads=REPORT_WORKLOADS,
        output_csv="scaling_results.csv",
    )
    print("Generating scaling efficiency plots...")
    plot_scaling_efficiency(output_dir="plots")
    print("Exp 4 complete.")


# ── Experiment 5: Compute model validation ────────────────────────────────────

def cmd_exp5():
    """
    Experiment 5: Compare analytical cycle counts (with SRAM stall model)
    against SCALE-Sim for all 8 workloads at array sizes {16×16, 32×32, 64×64}
    under WS and OS dataflows.  Reports MAPE per configuration.
    """
    print("=" * 70)
    print("EXPERIMENT 5: Compute Model Validation  (analytical vs. SCALE-Sim)")
    print("=" * 70)
    print("Array sizes: 16×16, 32×32, 64×64  |  Dataflows: ws, os")
    print("Workloads:", [w.name for w in VALIDATION_WORKLOADS])
    print()
    run_validation_sweep(
        workloads=VALIDATION_WORKLOADS,
        output_csv="validation_results.csv",
    )
    print("\nGenerating validation plots...")
    plot_validation_mape(output_dir="plots")
    print("Exp 5 complete.")


# ── Full sweep (all experiments combined) ─────────────────────────────────────

def cmd_all_experiments():
    """Run all five experiments in sequence."""
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS (1 – 5)")
    print("=" * 70)
    cmd_exp5()   # validation first (independent of sweep)
    cmd_exp1()   # crossover (uses its own sweep)
    # Full main sweep needed for Exp 2 topology comparison
    print("\nRunning full main sweep for Exp 2 (all topologies)...")
    run_sweep(
        array_h=128, array_w=128,
        chiplet_counts=[1, 2, 4, 8],
        bandwidths=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        topologies=[Topology.RING, Topology.MESH,
                    Topology.ALL_TO_ALL, Topology.HIERARCHICAL],
        dataflows=["os"],
        workloads=REPORT_WORKLOADS,
    )
    cmd_exp2()
    cmd_exp3()
    cmd_exp4()
    print("\nAll experiments complete.  Plots saved to plots/")


# ── Legacy / debug commands ───────────────────────────────────────────────────

def cmd_single():
    """Run a single-point evaluation to verify the model works."""
    print("=" * 70)
    print("SINGLE-POINT EVALUATION")
    print("=" * 70)

    wl = RESNET50_REPRESENTATIVE[3]   # conv4_1_2
    print(f"\nWorkload: {wl}")

    array_h, array_w = 128, 128
    dataflow = "os"

    baseline = analytical_compute(wl.m, wl.n, wl.k, array_h, array_w, dataflow)
    print(f"\nSingle chiplet ({array_h}×{array_w}, {dataflow}):")
    print(f"  Cycles:     {baseline.total_cycles:>12,}")
    print(f"  Map. Eff:   {baseline.mapping_efficiency:>11.1%}")
    print(f"  Comp. Util: {baseline.compute_utilization:>11.1%}")

    # Show SRAM stall effect at a small array size
    ana_sram = analytical_compute(wl.m, wl.n, wl.k, 16, 16, dataflow,
                                   sram_ifmap_kb=32, sram_filter_kb=32,
                                   sram_ofmap_kb=16, dram_bandwidth=10.0)
    print(f"\nSingle chiplet (16×16, SRAM 32/32/16 KiB, DRAM 10 B/cyc):")
    print(f"  Total:      {ana_sram.total_cycles:>12,}")
    print(f"  Stalls:     {ana_sram.stall_cycles:>12,}  (SRAM-limited: {ana_sram.sram_limited})")

    # Multi-chiplet comparison
    for topo in [Topology.RING, Topology.ALL_TO_ALL, Topology.HIERARCHICAL]:
        print(f"\n4 chiplets, {topo.value} topology, 32 B/cyc:")
        ic = Interconnect(4, topo, link_bw=32, link_latency=10)
        results = evaluate_all_strategies(
            wl.m, wl.n, wl.k, array_h, array_w, dataflow, 4, ic
        )
        for r in results:
            print(f"  {r.strategy:>10s}: speedup={r.speedup_no_overlap:.2f}x  "
                  f"compute={r.compute.total_cycles:,}  "
                  f"dist={r.distribute.cycles:,}  "
                  f"reduce={r.reduce.cycles:,}")
        info = ic.connection_info()
        print(f"  Topology stats: links={info['num_links']}  "
              f"bisect_bw={info['bisection_bandwidth']} B/cyc  "
              f"diameter={info['diameter']}")

    # Multi-layer scheduling demo
    print(f"\n{'=' * 70}")
    print("MULTI-LAYER SCHEDULING (ResNet-50 representative stages)")
    sched = schedule_layers(
        workloads=RESNET50_REPRESENTATIVE,
        array_h=128, array_w=128,
        dataflow="os",
        sram_ifmap_kb=6144, sram_filter_kb=6144, sram_ofmap_kb=2048,
        dram_bandwidth=10.0,
    )
    print(sched.summary())


def cmd_validate():
    """Quick analytical vs. SCALE-Sim check on a handful of configs."""
    print("=" * 70)
    print("QUICK VALIDATION: Analytical vs SCALE-Sim")
    print("=" * 70)

    test_cases = [
        (1024, 256, 256),
        (512, 512, 64),
        (196, 256, 2304),
        (49, 512, 4608),
    ]

    sram_cfg = dict(sram_ifmap_kb=6144, sram_filter_kb=6144, sram_ofmap_kb=2048)
    print(f"{'GEMM':<25} {'DF':>3}  {'Analytical':>12}  {'SCALE-Sim':>12}  "
          f"{'Ratio':>6}  {'Stalls(sim)':>11}")
    print("-" * 80)
    for m, n, k in test_cases:
        for df in ["os", "ws"]:
            ana = analytical_compute(m, n, k, 128, 128, df,
                                     dram_bandwidth=10.0, **sram_cfg)
            sim = scalesim_compute(m, n, k, 128, 128, df,
                                   bandwidth=10, **sram_cfg)
            ratio = sim.total_cycles / ana.total_cycles if ana.total_cycles else 0
            print(f"GEMM({m:>4},{n:>4},{k:>4}) {df}:  "
                  f"{ana.total_cycles:>12,}  {sim.total_cycles:>12,}  "
                  f"{ratio:>6.2f}  {sim.stall_cycles:>11,}")


def cmd_sweep(quick=False):
    """Run the full parameter sweep."""
    if quick:
        workloads      = REPORT_WORKLOADS[:2]
        bandwidths     = [4, 16, 64, 256]
        chiplet_counts = [1, 2, 4]
        topologies     = [Topology.RING, Topology.HIERARCHICAL]
    else:
        workloads      = REPORT_WORKLOADS
        bandwidths     = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        chiplet_counts = [1, 2, 4, 8]
        topologies     = [Topology.RING, Topology.MESH,
                          Topology.ALL_TO_ALL, Topology.HIERARCHICAL]

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Chiplet-based multi-systolic array simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── per-experiment flags ──────────────────────────────────────────
    parser.add_argument("--exp1", action="store_true",
                        help="Exp 1: crossover bandwidth B* and CTC correlation")
    parser.add_argument("--exp2", action="store_true",
                        help="Exp 2: topology comparison at P=8")
    parser.add_argument("--exp3", action="store_true",
                        help="Exp 3: partition strategy phase diagram (P=2–8)")
    parser.add_argument("--exp4", action="store_true",
                        help="Exp 4: scaling efficiency vs. P")
    parser.add_argument("--exp5", action="store_true",
                        help="Exp 5: compute model validation vs. SCALE-Sim")
    parser.add_argument("--all-experiments", action="store_true",
                        help="Run all five experiments in sequence")

    # ── legacy / utility flags ────────────────────────────────────────
    parser.add_argument("--single", action="store_true",
                        help="Single-point evaluation for debugging")
    parser.add_argument("--validate", action="store_true",
                        help="Quick analytical vs. SCALE-Sim check")
    parser.add_argument("--quick", action="store_true",
                        help="Small sweep for testing")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate all plots from existing CSVs")

    args = parser.parse_args()

    if args.exp1:
        cmd_exp1()
    elif args.exp2:
        cmd_exp2()
    elif args.exp3:
        cmd_exp3()
    elif args.exp4:
        cmd_exp4()
    elif args.exp5:
        cmd_exp5()
    elif args.all_experiments:
        cmd_all_experiments()
    elif args.single:
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
