"""
Workload partitioning for multi-chiplet systolic arrays.

Given a GEMM M×N×K and P chiplets, evaluates partition strategies:

  Strategy     Sub-problem per chiplet    Distribution cost        Reduction cost
  ────────────────────────────────────────────────────────────────────────────────
  split_m      (M/P) × N × K             broadcast weights K×N    none
  split_n      M × (N/P) × K             broadcast inputs M×K     none
  split_k      M × N × (K/P)             scatter inputs+weights   allreduce M×N
  split_mn     (M/√P) × (N/√P) × K      scatter both             none (if exact)

For each strategy, total latency is:

  T_total = T_distribute + max(T_compute_per_chiplet) + T_reduce

The distribute phase moves input data to chiplets. The compute phase runs
on all chiplets in parallel. The reduce phase combines partial results.

Whether distribution overlaps with compute depends on the system's ability
to double-buffer. We model both cases:
  - no_overlap: T_total = T_distribute + T_compute + T_reduce
  - full_overlap: T_total = max(T_distribute, T_compute) + T_reduce
"""

import math
from dataclasses import dataclass, field

from .compute import analytical_compute, ComputeResult
from .interconnect import Interconnect, TransferResult


@dataclass
class PartitionResult:
    strategy: str
    num_chiplets: int
    sub_m: int
    sub_n: int
    sub_k: int
    compute: ComputeResult
    distribute: TransferResult
    reduce: TransferResult
    total_cycles_no_overlap: int
    total_cycles_overlap: int
    speedup_no_overlap: float      # vs single-chiplet baseline
    speedup_overlap: float
    efficiency_no_overlap: float   # speedup / num_chiplets
    efficiency_overlap: float


def _ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)


def evaluate_partition(m: int, n: int, k: int,
                       array_h: int, array_w: int,
                       dataflow: str,
                       num_chiplets: int,
                       interconnect: Interconnect,
                       strategy: str) -> PartitionResult:
    """
    Evaluate a single partition strategy.

    Args:
        m, n, k: GEMM dimensions
        array_h, array_w: per-chiplet systolic array size
        dataflow: os, ws, or is
        num_chiplets: number of chiplets
        interconnect: Interconnect object with topology and bandwidth
        strategy: split_m, split_n, split_k, or split_mn
    """
    P = num_chiplets

    # Compute single-chiplet baseline
    baseline = analytical_compute(m, n, k, array_h, array_w, dataflow)

    # Determine sub-problem dimensions and communication patterns
    if strategy == "split_m":
        sub_m = _ceil_div(m, P)
        sub_n = n
        sub_k = k
        # Every chiplet needs full weight matrix
        distribute = interconnect.broadcast(k * n)
        # No reduction needed: outputs are independent
        reduce = TransferResult(0, 0, 0, "none")

    elif strategy == "split_n":
        sub_m = m
        sub_n = _ceil_div(n, P)
        sub_k = k
        # Every chiplet needs full input matrix
        distribute = interconnect.broadcast(m * k)
        reduce = TransferResult(0, 0, 0, "none")

    elif strategy == "split_k":
        sub_m = m
        sub_n = n
        sub_k = _ceil_div(k, P)
        # Inputs and weights are partitioned along K, scatter both
        # Each chiplet gets M × K/P inputs and K/P × N weights
        dist_input = interconnect.scatter(m * k)
        dist_weight = interconnect.scatter(k * n)
        distribute = TransferResult(
            cycles=dist_input.cycles + dist_weight.cycles,
            data_bytes=dist_input.data_bytes + dist_weight.data_bytes,
            effective_bw=(dist_input.data_bytes + dist_weight.data_bytes) /
                         max(dist_input.cycles + dist_weight.cycles, 1),
            description=f"scatter inputs ({m}×{sub_k}) + weights ({sub_k}×{n})"
        )
        # Allreduce the M×N partial output sums
        reduce = interconnect.allreduce(m * n)

    elif strategy == "split_mn":
        # 2D split: sqrt(P) along M, sqrt(P) along N
        sqrt_p = math.isqrt(P)
        if sqrt_p * sqrt_p != P:
            # Not a perfect square, fall back to rectangular
            split_m_count = sqrt_p
            split_n_count = _ceil_div(P, sqrt_p)
        else:
            split_m_count = sqrt_p
            split_n_count = sqrt_p

        sub_m = _ceil_div(m, split_m_count)
        sub_n = _ceil_div(n, split_n_count)
        sub_k = k
        # Each chiplet needs full K dimension but only its M and N slices
        # Multicast inputs to chiplets sharing same M slice: split_n_count groups
        # Multicast weights to chiplets sharing same N slice: split_m_count groups
        input_multicast = interconnect.broadcast(sub_m * k)
        weight_multicast = interconnect.broadcast(sub_n * k)
        distribute = TransferResult(
            cycles=max(input_multicast.cycles, weight_multicast.cycles),
            data_bytes=input_multicast.data_bytes + weight_multicast.data_bytes,
            effective_bw=(input_multicast.data_bytes + weight_multicast.data_bytes) /
                         max(input_multicast.cycles, weight_multicast.cycles, 1),
            description=f"multicast inputs and weights for 2D split"
        )
        reduce = TransferResult(0, 0, 0, "none")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Compute per-chiplet execution time
    chiplet_compute = analytical_compute(
        sub_m, sub_n, sub_k, array_h, array_w, dataflow
    )

    # Total time under two overlap assumptions
    t_no_overlap = distribute.cycles + chiplet_compute.total_cycles + reduce.cycles
    t_overlap = max(distribute.cycles, chiplet_compute.total_cycles) + reduce.cycles

    # Speedup relative to single-chiplet baseline
    sp_no = baseline.total_cycles / t_no_overlap if t_no_overlap > 0 else 0
    sp_ov = baseline.total_cycles / t_overlap if t_overlap > 0 else 0

    return PartitionResult(
        strategy=strategy,
        num_chiplets=P,
        sub_m=sub_m,
        sub_n=sub_n,
        sub_k=sub_k,
        compute=chiplet_compute,
        distribute=distribute,
        reduce=reduce,
        total_cycles_no_overlap=t_no_overlap,
        total_cycles_overlap=t_overlap,
        speedup_no_overlap=sp_no,
        speedup_overlap=sp_ov,
        efficiency_no_overlap=sp_no / P,
        efficiency_overlap=sp_ov / P,
    )


def evaluate_all_strategies(m: int, n: int, k: int,
                            array_h: int, array_w: int,
                            dataflow: str,
                            num_chiplets: int,
                            interconnect: Interconnect) -> list[PartitionResult]:
    """Evaluate all partition strategies and return sorted by best no-overlap time."""
    strategies = ["split_m", "split_n", "split_k"]
    if num_chiplets >= 4:
        strategies.append("split_mn")

    results = []
    for strat in strategies:
        r = evaluate_partition(
            m, n, k, array_h, array_w, dataflow,
            num_chiplets, interconnect, strat
        )
        results.append(r)

    results.sort(key=lambda r: r.total_cycles_no_overlap)
    return results
