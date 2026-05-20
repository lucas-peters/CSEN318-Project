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
from .workloads import GEMMWorkload, ALL_WORKLOADS, RESNET50_REPRESENTATIVE, BERT_LAYERS


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
