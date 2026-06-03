"""
chiplet_sim – Closed-form analytical simulator for multi-chiplet systolic arrays.

Public API
----------
Compute:
    analytical_compute(m, n, k, array_h, array_w, dataflow, ...)
    scalesim_compute(m, n, k, array_h, array_w, dataflow, ...)
    ComputeResult

Interconnect:
    Interconnect(num_chiplets, topology, link_bw, ...)
    Topology            – RING | MESH | ALL_TO_ALL | HIERARCHICAL
    TransferResult

Partition:
    evaluate_partition(...)
    evaluate_all_strategies(...)
    PartitionResult

Schedule:
    schedule_layers(workloads, array_h, array_w, dataflow, ...)
    ScheduleResult, LayerResult

Workloads:
    RESNET50_LAYERS
    RESNET50_REPRESENTATIVE
    BERT_LAYERS              – seq_len = 512
    BERT_LAYERS_2048         – seq_len = 2048
    REPORT_WORKLOADS         – the five workloads in the mid-project report table
    VALIDATION_WORKLOADS     – all eight workloads for Experiment 5
    ALL_WORKLOADS
    GEMMWorkload

Sweep:
    run_sweep(...)
    run_bandwidth_crossover_sweep(...)
    run_phase_diagram_sweep(...)   – Experiment 3
    run_scaling_sweep(...)         – Experiment 4
    run_validation_sweep(...)      – Experiment 5

Plot:
    plot_all(...)
    plot_roofline(...)
    plot_crossover(...)
    plot_crossover_bstar(...)      – Experiment 1
    plot_topology_comparison(...)  – Experiment 2
    plot_phase_diagram(...)        – Experiment 3
    plot_scaling_efficiency(...)   – Experiment 4
    plot_validation_mape(...)      – Experiment 5
"""

from .compute     import analytical_compute, scalesim_compute, ComputeResult
from .interconnect import Interconnect, Topology, TransferResult
from .partition   import evaluate_partition, evaluate_all_strategies, PartitionResult
from .schedule    import schedule_layers, ScheduleResult, LayerResult
from .workloads   import (
    RESNET50_LAYERS, RESNET50_REPRESENTATIVE,
    BERT_LAYERS, BERT_LAYERS_2048,
    REPORT_WORKLOADS, VALIDATION_WORKLOADS,
    ALL_WORKLOADS, GEMMWorkload,
    bert_attention_workloads,
)
from .sweep       import (
    run_sweep, run_bandwidth_crossover_sweep,
    run_phase_diagram_sweep, run_scaling_sweep, run_validation_sweep,
)
from .plot        import (
    plot_all, plot_roofline, plot_crossover, plot_strategy_map,
    plot_crossover_bstar, plot_topology_comparison,
    plot_phase_diagram, plot_scaling_efficiency, plot_validation_mape,
)

__all__ = [
    # compute
    "analytical_compute", "scalesim_compute", "ComputeResult",
    # interconnect
    "Interconnect", "Topology", "TransferResult",
    # partition
    "evaluate_partition", "evaluate_all_strategies", "PartitionResult",
    # schedule
    "schedule_layers", "ScheduleResult", "LayerResult",
    # workloads
    "RESNET50_LAYERS", "RESNET50_REPRESENTATIVE",
    "BERT_LAYERS", "BERT_LAYERS_2048",
    "REPORT_WORKLOADS", "VALIDATION_WORKLOADS",
    "ALL_WORKLOADS", "GEMMWorkload", "bert_attention_workloads",
    # sweep
    "run_sweep", "run_bandwidth_crossover_sweep",
    "run_phase_diagram_sweep", "run_scaling_sweep", "run_validation_sweep",
    # plot
    "plot_all", "plot_roofline", "plot_crossover", "plot_strategy_map",
    "plot_crossover_bstar", "plot_topology_comparison",
    "plot_phase_diagram", "plot_scaling_efficiency", "plot_validation_mape",
]
