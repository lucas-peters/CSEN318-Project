"""
Multi-layer scheduling for sequences of GEMM workloads.

When consecutive layers are executed on the same chiplet, the output tensor
of layer i is the input tensor of layer i+1.  If that tensor fits inside the
chiplet's ofmap SRAM bank it can be passed on-chip (zero DRAM round-trip).
Otherwise it must be written to DRAM and read back, adding a spill penalty:

    spill_cycles = ceil(output_bytes / dram_bandwidth)   [write]
                 + ceil(input_bytes_next / dram_bandwidth) [read, if different size]

The scheduler also reports the total compute cycles, SRAM stall cycles, and
spill cycles separately so that the dominant bottleneck can be identified.

Usage::

    from chiplet_sim.workloads import RESNET50_REPRESENTATIVE
    from chiplet_sim.schedule import schedule_layers

    result = schedule_layers(
        workloads    = RESNET50_REPRESENTATIVE,
        array_h=128, array_w=128,
        dataflow     = "os",
        sram_ifmap_kb=6144, sram_filter_kb=6144, sram_ofmap_kb=2048,
        dram_bandwidth=10.0,
    )
    print(result.summary())
"""

import math
from dataclasses import dataclass, field

from .compute import analytical_compute, ComputeResult
from .workloads import GEMMWorkload


@dataclass
class LayerResult:
    workload: GEMMWorkload
    compute_cycles: int          # pure compute + pipeline fill/drain
    stall_cycles: int            # SRAM-capacity stalls (DRAM refill)
    spill_cycles: int            # DRAM write/read at layer boundary
    total_cycles: int            # compute + stall + spill


@dataclass
class ScheduleResult:
    layers: list                  # list[LayerResult]
    total_cycles: int
    total_compute_cycles: int
    total_stall_cycles: int
    total_spill_cycles: int
    on_chip_reuse_count: int      # layer boundaries with zero spill cost
    array_h: int
    array_w: int
    dataflow: str
    sram_ifmap_kb: int
    sram_filter_kb: int
    sram_ofmap_kb: int
    dram_bandwidth: float

    def summary(self) -> str:
        lines = [
            f"Multi-layer schedule  ({self.array_h}×{self.array_w} array, "
            f"{self.dataflow.upper()}, DRAM bw={self.dram_bandwidth} B/cyc)",
            f"  SRAM: ifmap={self.sram_ifmap_kb} KiB  "
            f"filter={self.sram_filter_kb} KiB  "
            f"ofmap={self.sram_ofmap_kb} KiB",
            "",
            f"{'Layer':<28} {'Compute':>10} {'Stalls':>10} "
            f"{'Spill':>10} {'Total':>10}  SRAM-ltd",
        ]
        for lr in self.layers:
            limited = "YES" if lr.stall_cycles > 0 else "no"
            lines.append(
                f"  {lr.workload.name:<26} {lr.compute_cycles:>10,} "
                f"{lr.stall_cycles:>10,} {lr.spill_cycles:>10,} "
                f"{lr.total_cycles:>10,}  {limited}"
            )
        lines += [
            "",
            f"  {'TOTAL':<26} {self.total_compute_cycles:>10,} "
            f"{self.total_stall_cycles:>10,} {self.total_spill_cycles:>10,} "
            f"{self.total_cycles:>10,}",
            f"  On-chip reuse at {self.on_chip_reuse_count} of "
            f"{max(len(self.layers)-1, 0)} layer boundaries",
        ]
        return "\n".join(lines)


def schedule_layers(workloads: list,
                    array_h: int,
                    array_w: int,
                    dataflow: str,
                    sram_ifmap_kb: int = 6144,
                    sram_filter_kb: int = 6144,
                    sram_ofmap_kb: int = 2048,
                    dram_bandwidth: float = 10.0,
                    element_bytes: int = 2) -> ScheduleResult:
    """
    Schedule a sequence of GEMM workloads on a single chiplet and compute
    total cycle count including SRAM stalls and inter-layer DRAM spill costs.

    Args:
        workloads:      list of GEMMWorkload objects in execution order
        array_h/w:      systolic array dimensions
        dataflow:       "os" | "ws" | "is"
        sram_ifmap_kb:  ifmap SRAM bank size in KiB
        sram_filter_kb: filter SRAM bank size in KiB
        sram_ofmap_kb:  ofmap SRAM bank size in KiB
        dram_bandwidth: off-chip bandwidth in bytes/cycle
        element_bytes:  bytes per element (2 = fp16, 4 = fp32)

    Returns:
        ScheduleResult with per-layer breakdown and aggregate totals.
    """
    sram_ofmap_bytes = sram_ofmap_kb * 1024
    sram_ifmap_bytes = sram_ifmap_kb * 1024
    layer_results = []
    on_chip_reuse = 0

    for i, wl in enumerate(workloads):
        result: ComputeResult = analytical_compute(
            wl.m, wl.n, wl.k,
            array_h, array_w, dataflow,
            sram_ifmap_kb=sram_ifmap_kb,
            sram_filter_kb=sram_filter_kb,
            sram_ofmap_kb=sram_ofmap_kb,
            dram_bandwidth=dram_bandwidth,
            element_bytes=element_bytes,
        )

        # ── determine inter-layer spill cost ──────────────────────────────
        spill = 0
        if i < len(workloads) - 1:
            next_wl = workloads[i + 1]
            output_bytes    = wl.m * wl.n * element_bytes
            next_ifmap_bytes = next_wl.m * next_wl.k * element_bytes

            # On-chip reuse is possible when:
            #   (a) the output tensor fits in the ofmap SRAM, and
            #   (b) that same tensor fits in the next layer's ifmap SRAM
            if output_bytes <= sram_ofmap_bytes and output_bytes <= sram_ifmap_bytes:
                on_chip_reuse += 1
            else:
                # Write output to DRAM
                spill = int(math.ceil(output_bytes / dram_bandwidth))
                # If sizes differ the next layer reads a different amount;
                # the read cost is already captured by its own stall_cycles,
                # so we only add the write penalty here.

        layer_results.append(LayerResult(
            workload      = wl,
            compute_cycles= result.compute_cycles,
            stall_cycles  = result.stall_cycles,
            spill_cycles  = spill,
            total_cycles  = result.total_cycles + spill,
        ))

    return ScheduleResult(
        layers               = layer_results,
        total_cycles         = sum(lr.total_cycles for lr in layer_results),
        total_compute_cycles = sum(lr.compute_cycles for lr in layer_results),
        total_stall_cycles   = sum(lr.stall_cycles for lr in layer_results),
        total_spill_cycles   = sum(lr.spill_cycles for lr in layer_results),
        on_chip_reuse_count  = on_chip_reuse,
        array_h              = array_h,
        array_w              = array_w,
        dataflow             = dataflow,
        sram_ifmap_kb        = sram_ifmap_kb,
        sram_filter_kb       = sram_filter_kb,
        sram_ofmap_kb        = sram_ofmap_kb,
        dram_bandwidth       = dram_bandwidth,
    )
