"""
Parameter sweep engine.

Sweeps across:
  - Chiplet counts: 1, 2, 4, 8
  - Interconnect bandwidths: range of bytes/cycle
  - Topologies: ring, mesh, all_to_all
  - Partition strategies: split_m, split_n, split_k, split_mn
  - Dataflows: os, ws, is

Produces CSV results for plotting.
"""

import csv
import os
import time
import logging
from dataclasses import asdict

from .compute import analytical_compute
from .interconnect import Interconnect, Topology
from .partition import evaluate_all_strategies, PartitionResult
from .compute import analytical_compute, scalesim_compute
from .workloads import (GEMMWorkload, ALL_WORKLOADS, RESNET50_REPRESENTATIVE,
                        BERT_LAYERS, REPORT_WORKLOADS, VALIDATION_WORKLOADS)


SWEEP_CSV = "chiplet_sweep_results.csv"

CSV_FIELDS = [
    "workload", "m", "n", "k", "total_macs",
    "dataflow", "topology", "num_chiplets",
    "link_bw", "link_latency",
    "strategy",
    "sub_m", "sub_n", "sub_k",
    "compute_cycles", "distribute_cycles", "reduce_cycles",
    "total_cycles_no_overlap", "total_cycles_overlap",
    "speedup_no_overlap", "speedup_overlap",
    "efficiency_no_overlap", "efficiency_overlap",
    "baseline_cycles",
]


def _result_to_row(wl: GEMMWorkload, dataflow: str, topo: Topology,
                   link_bw: float, link_latency: int,
                   pr: PartitionResult, baseline_cycles: int) -> dict:
    return {
        "workload": wl.name,
        "m": wl.m,
        "n": wl.n,
        "k": wl.k,
        "total_macs": wl.macs,
        "dataflow": dataflow,
        "topology": topo.value,
        "num_chiplets": pr.num_chiplets,
        "link_bw": link_bw,
        "link_latency": link_latency,
        "strategy": pr.strategy,
        "sub_m": pr.sub_m,
        "sub_n": pr.sub_n,
        "sub_k": pr.sub_k,
        "compute_cycles": pr.compute.total_cycles,
        "distribute_cycles": pr.distribute.cycles,
        "reduce_cycles": pr.reduce.cycles,
        "total_cycles_no_overlap": pr.total_cycles_no_overlap,
        "total_cycles_overlap": pr.total_cycles_overlap,
        "speedup_no_overlap": pr.speedup_no_overlap,
        "speedup_overlap": pr.speedup_overlap,
        "efficiency_no_overlap": pr.efficiency_no_overlap,
        "efficiency_overlap": pr.efficiency_overlap,
        "baseline_cycles": baseline_cycles,
    }


def run_sweep(array_h: int = 128,
              array_w: int = 128,
              chiplet_counts: list[int] = None,
              bandwidths: list[float] = None,
              topologies: list[Topology] = None,
              dataflows: list[str] = None,
              workloads: list[GEMMWorkload] = None,
              link_latency: int = 10,
              output_csv: str = SWEEP_CSV):
    """
    Run the full parameter sweep and write results to CSV.

    Defaults model a realistic design space:
      - 128×128 array per chiplet (16K PEs, TPU-like)
      - 1-8 chiplets
      - Link bandwidths from 1 to 256 bytes/cycle
        (1 B/cyc ≈ 1 GB/s at 1GHz, 256 B/cyc ≈ 256 GB/s)
      - All three topologies
      - OS dataflow (most common for inference)
    """
    if chiplet_counts is None:
        chiplet_counts = [1, 2, 4, 8]
    if bandwidths is None:
        bandwidths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if topologies is None:
        topologies = [Topology.RING, Topology.MESH, Topology.ALL_TO_ALL]
    if dataflows is None:
        dataflows = ["os"]
    if workloads is None:
        workloads = ALL_WORKLOADS

    log = logging.getLogger(__name__)

    # Precompute single-chiplet baselines
    baselines = {}
    for wl in workloads:
        for df in dataflows:
            result = analytical_compute(wl.m, wl.n, wl.k, array_h, array_w, df)
            baselines[(wl.name, df)] = result.total_cycles

    total_runs = (len(workloads) * len(dataflows) * len(topologies) *
                  len(chiplet_counts) * len(bandwidths) * 4)  # ~4 strategies each
    log.info(f"Estimated sweep size: ~{total_runs} evaluations")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

    count = 0
    t_start = time.time()

    for wl in workloads:
        for df in dataflows:
            baseline_cyc = baselines[(wl.name, df)]

            for topo in topologies:
                for P in chiplet_counts:
                    if P == 1:
                        # Single chiplet: no interconnect, just log baseline
                        bw_val = 0
                        ic = Interconnect(1, topo, 1, link_latency)
                        results = evaluate_all_strategies(
                            wl.m, wl.n, wl.k,
                            array_h, array_w, df, 1, ic
                        )
                        for pr in results:
                            row = _result_to_row(wl, df, topo, bw_val,
                                                 link_latency, pr, baseline_cyc)
                            with open(output_csv, "a", newline="") as f:
                                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
                            count += 1
                        continue

                    # Mesh requires specific chiplet counts
                    if topo == Topology.MESH and P == 2:
                        continue  # 2 chiplets can't form a useful mesh

                    for bw in bandwidths:
                        ic = Interconnect(P, topo, bw, link_latency)
                        results = evaluate_all_strategies(
                            wl.m, wl.n, wl.k,
                            array_h, array_w, df, P, ic
                        )
                        for pr in results:
                            row = _result_to_row(wl, df, topo, bw,
                                                 link_latency, pr, baseline_cyc)
                            with open(output_csv, "a", newline="") as f:
                                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
                            count += 1

    elapsed = time.time() - t_start
    log.info(f"Sweep complete: {count} rows in {elapsed:.1f}s -> {output_csv}")


def run_bandwidth_crossover_sweep(
        array_h: int = 128,
        array_w: int = 128,
        workloads: list[GEMMWorkload] = None,
        output_csv: str = "crossover_results.csv"):
    """
    Targeted sweep to find the compute-vs-interconnect crossover bandwidth.

    For each workload and chiplet count, finds the minimum link bandwidth
    where adding more chiplets actually improves throughput.
    """
    if workloads is None:
        workloads = RESNET50_REPRESENTATIVE + BERT_LAYERS

    chiplet_counts = [2, 4, 8]
    # Fine-grained bandwidth sweep
    bandwidths = [2**i for i in range(0, 12)]  # 1 to 2048 bytes/cycle
    dataflow = "os"
    topo = Topology.RING  # most realistic for chiplet interconnect

    log = logging.getLogger(__name__)

    fields = ["workload", "num_chiplets", "link_bw",
              "best_strategy", "speedup", "efficiency",
              "compute_fraction", "interconnect_fraction"]

    with open(output_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for wl in workloads:
        baseline = analytical_compute(wl.m, wl.n, wl.k, array_h, array_w, dataflow)

        for P in chiplet_counts:
            for bw in bandwidths:
                ic = Interconnect(P, topo, bw, 10)
                results = evaluate_all_strategies(
                    wl.m, wl.n, wl.k,
                    array_h, array_w, dataflow, P, ic
                )
                best = results[0]  # sorted by total_cycles_no_overlap

                total = best.total_cycles_no_overlap
                comp_frac = best.compute.total_cycles / total if total > 0 else 0
                ic_frac = 1 - comp_frac

                row = {
                    "workload": wl.name,
                    "num_chiplets": P,
                    "link_bw": bw,
                    "best_strategy": best.strategy,
                    "speedup": best.speedup_no_overlap,
                    "efficiency": best.efficiency_no_overlap,
                    "compute_fraction": comp_frac,
                    "interconnect_fraction": ic_frac,
                }
                with open(output_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)

    log.info(f"Crossover sweep -> {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Partition strategy phase diagram
# ─────────────────────────────────────────────────────────────────────────────

PHASE_CSV = "phase_diagram_results.csv"

PHASE_FIELDS = [
    "workload", "m", "n", "k",
    "num_chiplets", "link_bw",
    "best_strategy", "speedup",
    "compute_cycles", "distribute_cycles", "reduce_cycles",
]


def run_phase_diagram_sweep(
        array_h: int = 128,
        array_w: int = 128,
        workloads: list = None,
        chiplet_counts: list = None,
        bandwidths: list = None,
        link_latency: int = 10,
        dataflow: str = "os",
        output_csv: str = PHASE_CSV):
    """
    Experiment 3 sweep: for each (workload, P, B) cell on a ring topology
    record the winning partition strategy and its speedup.

    P sweeps every integer from 2 to 8 (not just powers of two) so that the
    phase diagram has continuous coverage on the y-axis.
    B sweeps 1–256 bytes/cycle on a log2 scale.
    """
    if workloads is None:
        workloads = REPORT_WORKLOADS
    if chiplet_counts is None:
        chiplet_counts = list(range(2, 9))          # 2,3,4,5,6,7,8
    if bandwidths is None:
        bandwidths = [2**i for i in range(0, 9)]    # 1,2,4,...,256

    log = logging.getLogger(__name__)
    topo = Topology.RING

    with open(output_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=PHASE_FIELDS).writeheader()

    for wl in workloads:
        for P in chiplet_counts:
            for bw in bandwidths:
                ic = Interconnect(P, topo, bw, link_latency)
                results = evaluate_all_strategies(
                    wl.m, wl.n, wl.k, array_h, array_w, dataflow, P, ic
                )
                best = results[0]   # sorted by total_cycles_no_overlap
                row = {
                    "workload":           wl.name,
                    "m": wl.m, "n": wl.n, "k": wl.k,
                    "num_chiplets":       P,
                    "link_bw":            bw,
                    "best_strategy":      best.strategy,
                    "speedup":            best.speedup_no_overlap,
                    "compute_cycles":     best.compute.total_cycles,
                    "distribute_cycles":  best.distribute.cycles,
                    "reduce_cycles":      best.reduce.cycles,
                }
                with open(output_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=PHASE_FIELDS).writerow(row)

    log.info(f"Phase diagram sweep -> {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Scaling efficiency
# ─────────────────────────────────────────────────────────────────────────────

SCALING_CSV = "scaling_results.csv"

SCALING_FIELDS = [
    "workload", "m", "n", "k",
    "num_chiplets", "link_bw",
    "best_strategy", "speedup", "efficiency",
    "compute_cycles", "distribute_cycles", "reduce_cycles",
    "total_cycles", "baseline_cycles",
    "compute_fraction", "comm_fraction",
]


def run_scaling_sweep(
        array_h: int = 128,
        array_w: int = 128,
        workloads: list = None,
        chiplet_counts: list = None,
        bandwidths: list = None,
        link_latency: int = 10,
        dataflow: str = "os",
        output_csv: str = SCALING_CSV):
    """
    Experiment 4 sweep: best-strategy speedup and efficiency vs. chiplet count
    at three fixed bandwidths (16, 64, 256 bytes/cycle).

    P sweeps every integer 1–8 so the speedup curve is smooth.
    Results include a compute/comm decomposition for each point.
    """
    if workloads is None:
        workloads = REPORT_WORKLOADS
    if chiplet_counts is None:
        chiplet_counts = list(range(1, 9))      # 1,2,3,...,8
    if bandwidths is None:
        bandwidths = [16, 64, 256]

    log = logging.getLogger(__name__)
    topo = Topology.RING

    with open(output_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=SCALING_FIELDS).writeheader()

    for wl in workloads:
        baseline = analytical_compute(wl.m, wl.n, wl.k, array_h, array_w, dataflow)

        for bw in bandwidths:
            for P in chiplet_counts:
                if P == 1:
                    row = {
                        "workload": wl.name, "m": wl.m, "n": wl.n, "k": wl.k,
                        "num_chiplets": 1, "link_bw": bw,
                        "best_strategy": "single", "speedup": 1.0, "efficiency": 1.0,
                        "compute_cycles":    baseline.total_cycles,
                        "distribute_cycles": 0, "reduce_cycles": 0,
                        "total_cycles":      baseline.total_cycles,
                        "baseline_cycles":   baseline.total_cycles,
                        "compute_fraction":  1.0, "comm_fraction": 0.0,
                    }
                    with open(output_csv, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=SCALING_FIELDS).writerow(row)
                    continue

                ic = Interconnect(P, topo, bw, link_latency)
                results = evaluate_all_strategies(
                    wl.m, wl.n, wl.k, array_h, array_w, dataflow, P, ic
                )
                best = results[0]
                total = best.total_cycles_no_overlap
                comm  = best.distribute.cycles + best.reduce.cycles
                row = {
                    "workload": wl.name, "m": wl.m, "n": wl.n, "k": wl.k,
                    "num_chiplets":       P,
                    "link_bw":            bw,
                    "best_strategy":      best.strategy,
                    "speedup":            best.speedup_no_overlap,
                    "efficiency":         best.efficiency_no_overlap,
                    "compute_cycles":     best.compute.total_cycles,
                    "distribute_cycles":  best.distribute.cycles,
                    "reduce_cycles":      best.reduce.cycles,
                    "total_cycles":       total,
                    "baseline_cycles":    baseline.total_cycles,
                    "compute_fraction":   best.compute.total_cycles / total if total else 0,
                    "comm_fraction":      comm / total if total else 0,
                }
                with open(output_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=SCALING_FIELDS).writerow(row)

    log.info(f"Scaling sweep -> {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Compute model validation (analytical vs. SCALE-Sim)
# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_CSV = "validation_results.csv"

VALIDATION_FIELDS = [
    "workload", "m", "n", "k",
    "array_h", "array_w", "dataflow",
    "sram_ifmap_kb", "sram_filter_kb", "sram_ofmap_kb", "dram_bandwidth",
    "analytical_cycles", "scalesim_cycles",
    "analytical_stalls", "scalesim_stalls",
    "abs_error", "rel_error_pct",
    "mapping_efficiency",
]


def run_validation_sweep(
        array_sizes: list = None,
        dataflows: list = None,
        workloads: list = None,
        sram_ifmap_kb: int = 6144,
        sram_filter_kb: int = 6144,
        sram_ofmap_kb: int = 2048,
        dram_bandwidth: int = 10,
        output_csv: str = VALIDATION_CSV):
    """
    Experiment 5: compare analytical cycle counts against SCALE-Sim for all
    eight workloads across array sizes {16×16, 32×32, 64×64} and dataflows
    {ws, os}.  Computes per-row absolute and relative error; MAPE is computed
    in the plot function.

    Falls back to the analytical model with SRAM stall modeling if SCALE-Sim
    is not installed, so the CSV is always produced.
    """
    if array_sizes is None:
        array_sizes = [(16, 16), (32, 32), (64, 64)]
    if dataflows is None:
        dataflows = ["ws", "os"]
    if workloads is None:
        workloads = VALIDATION_WORKLOADS

    log = logging.getLogger(__name__)

    with open(output_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=VALIDATION_FIELDS).writeheader()

    for wl in workloads:
        for (ah, aw) in array_sizes:
            for df in dataflows:
                ana = analytical_compute(
                    wl.m, wl.n, wl.k, ah, aw, df,
                    sram_ifmap_kb=sram_ifmap_kb,
                    sram_filter_kb=sram_filter_kb,
                    sram_ofmap_kb=sram_ofmap_kb,
                    dram_bandwidth=float(dram_bandwidth),
                )
                sim = scalesim_compute(
                    wl.m, wl.n, wl.k, ah, aw, df,
                    sram_ifmap_kb=sram_ifmap_kb,
                    sram_filter_kb=sram_filter_kb,
                    sram_ofmap_kb=sram_ofmap_kb,
                    bandwidth=dram_bandwidth,
                )
                abs_err = abs(ana.total_cycles - sim.total_cycles)
                rel_err = 100.0 * abs_err / sim.total_cycles if sim.total_cycles else 0.0

                row = {
                    "workload":           wl.name,
                    "m": wl.m, "n": wl.n, "k": wl.k,
                    "array_h":            ah,
                    "array_w":            aw,
                    "dataflow":           df,
                    "sram_ifmap_kb":      sram_ifmap_kb,
                    "sram_filter_kb":     sram_filter_kb,
                    "sram_ofmap_kb":      sram_ofmap_kb,
                    "dram_bandwidth":     dram_bandwidth,
                    "analytical_cycles":  ana.total_cycles,
                    "scalesim_cycles":    sim.total_cycles,
                    "analytical_stalls":  ana.stall_cycles,
                    "scalesim_stalls":    sim.stall_cycles,
                    "abs_error":          abs_err,
                    "rel_error_pct":      round(rel_err, 3),
                    "mapping_efficiency": round(ana.mapping_efficiency, 4),
                }
                with open(output_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=VALIDATION_FIELDS).writerow(row)
                log.info(f"{wl.name} {ah}×{aw} {df}: "
                         f"ana={ana.total_cycles:,}  sim={sim.total_cycles:,}  "
                         f"err={rel_err:.1f}%")

    log.info(f"Validation sweep -> {output_csv}")
