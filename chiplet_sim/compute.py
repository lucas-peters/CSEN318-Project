"""
Single-chiplet systolic array compute model.

Two modes:
  - Analytical: fast closed-form cycle estimate for sweeps
  - SCALE-Sim: accurate simulation for validation

The analytical model follows the SCALE-Sim paper's spatio-temporal mapping.
For a GEMM M×N×K on an array H×W:

  Dataflow    S_r    S_c    T
  ─────────────────────────────
  OS          M      N      K
  WS          K      N      M
  IS          M      K      N

  Folds_r = ceil(S_r / H)
  Folds_c = ceil(S_c / W)
  Compute_cycles = Folds_r × Folds_c × T

  Mapping_efficiency = (S_r × S_c) / (H × Folds_r × W × Folds_c)
  This captures PE waste from array dimensions not dividing evenly.

SRAM capacity modeling (optional):
  Each chiplet has three on-chip SRAM banks: ifmap, filter, and ofmap.
  For each fold, the working-set sizes depend on the dataflow:

  Dataflow   Ifmap bank              Filter bank           Ofmap bank
  ──────────────────────────────────────────────────────────────────────
  OS         array_h × K             K × array_w           array_h × array_w
  WS         M × array_h             array_h × array_w     M × array_w
  IS         array_h × array_w       array_w × N           array_h × N

  When a working-set exceeds its bank's capacity the overflow must be fetched
  from DRAM before the fold can proceed:

    stall_cycles = ceil(overflow_bytes / dram_bandwidth) × Folds_r × Folds_c

  Pass sram_ifmap_kb / sram_filter_kb / sram_ofmap_kb to enable this model.
  When those arguments are None the model returns stall_cycles = 0 (ideal
  on-chip storage, backward-compatible default).
"""

import math
import os
import csv
import shutil
import tempfile
from dataclasses import dataclass, field


@dataclass
class ComputeResult:
    total_cycles: int
    compute_cycles: int
    stall_cycles: int
    mapping_efficiency: float
    compute_utilization: float
    total_macs: int
    num_pes: int
    sram_limited: bool = False   # True when stall_cycles > 0


def _sram_stalls(m: int, n: int, k: int,
                 array_h: int, array_w: int,
                 dataflow: str,
                 sram_ifmap_bytes: int,
                 sram_filter_bytes: int,
                 sram_ofmap_bytes: int,
                 dram_bandwidth: float,
                 element_bytes: int = 2) -> int:
    """
    Closed-form SRAM capacity stall estimate.

    For each fold the working-set footprint is computed for the three SRAM
    banks.  Any bytes that exceed the bank's capacity must be streamed from
    DRAM before the fold can proceed, adding stall cycles:

        stall_per_fold = overflow_bytes / dram_bandwidth
        total_stalls   = stall_per_fold × Folds_r × Folds_c

    The model is conservative: it assumes no overlap between DRAM fetch and
    array computation.  SCALE-Sim's stall counts will be similar but may
    differ slightly due to its cycle-accurate pipeline tracking.
    """
    if dataflow == "os":
        # S_r = M, S_c = N, T = K
        folds_r = math.ceil(m / array_h)
        folds_c = math.ceil(n / array_w)
        ifmap_per_fold   = array_h * k * element_bytes
        filter_per_fold  = k * array_w * element_bytes
        ofmap_per_fold   = array_h * array_w * element_bytes

    elif dataflow == "ws":
        # S_r = K, S_c = N, T = M  (weights stationary along rows)
        folds_r = math.ceil(k / array_h)
        folds_c = math.ceil(n / array_w)
        ifmap_per_fold   = m * array_h * element_bytes
        filter_per_fold  = array_h * array_w * element_bytes
        ofmap_per_fold   = m * array_w * element_bytes

    elif dataflow == "is":
        # S_r = M, S_c = K, T = N  (inputs stationary along rows)
        folds_r = math.ceil(m / array_h)
        folds_c = math.ceil(k / array_w)
        ifmap_per_fold   = array_h * array_w * element_bytes
        filter_per_fold  = array_w * n * element_bytes
        ofmap_per_fold   = array_h * n * element_bytes

    else:
        return 0

    overflow_per_fold = (
        max(0, ifmap_per_fold  - sram_ifmap_bytes)
        + max(0, filter_per_fold - sram_filter_bytes)
        + max(0, ofmap_per_fold  - sram_ofmap_bytes)
    )
    if overflow_per_fold == 0:
        return 0

    stall_per_fold = overflow_per_fold / dram_bandwidth
    return int(math.ceil(stall_per_fold * folds_r * folds_c))


def analytical_compute(m: int, n: int, k: int,
                       array_h: int, array_w: int,
                       dataflow: str,
                       sram_ifmap_kb: int = None,
                       sram_filter_kb: int = None,
                       sram_ofmap_kb: int = None,
                       dram_bandwidth: float = 10.0,
                       element_bytes: int = 2) -> ComputeResult:
    """
    Analytical cycle estimate for GEMM M×N×K on systolic array H×W.

    Args:
        m, n, k:          GEMM dimensions
        array_h, array_w: systolic array height and width (PEs)
        dataflow:         "os" | "ws" | "is"
        sram_ifmap_kb:    ifmap SRAM bank size in KiB  (None → no stall model)
        sram_filter_kb:   filter SRAM bank size in KiB (None → no stall model)
        sram_ofmap_kb:    ofmap SRAM bank size in KiB  (None → no stall model)
        dram_bandwidth:   off-chip bandwidth in bytes/cycle (used only when
                          SRAM sizes are provided)
        element_bytes:    bytes per element (2 = fp16, 4 = fp32)

    Returns ComputeResult with stall_cycles = 0 when SRAM params are omitted
    (backward-compatible behaviour identical to the previous implementation).
    """
    if dataflow == "os":
        s_r, s_c, t = m, n, k
    elif dataflow == "ws":
        s_r, s_c, t = k, n, m
    elif dataflow == "is":
        s_r, s_c, t = m, k, n
    else:
        raise ValueError(f"Unknown dataflow: {dataflow}")

    folds_r = math.ceil(s_r / array_h)
    folds_c = math.ceil(s_c / array_w)

    # Pure compute cycles (no memory effects)
    compute_cycles = folds_r * folds_c * t

    # Pipeline fill/drain overhead: one-time cost per fold
    pipeline_overhead = folds_r * folds_c * (array_h + array_w - 2)

    # SRAM capacity stall cycles (zero when SRAM params are not supplied)
    stall_cycles = 0
    if (sram_ifmap_kb is not None
            and sram_filter_kb is not None
            and sram_ofmap_kb is not None):
        stall_cycles = _sram_stalls(
            m, n, k, array_h, array_w, dataflow,
            sram_ifmap_bytes=sram_ifmap_kb * 1024,
            sram_filter_bytes=sram_filter_kb * 1024,
            sram_ofmap_bytes=sram_ofmap_kb * 1024,
            dram_bandwidth=dram_bandwidth,
            element_bytes=element_bytes,
        )

    total_cycles = compute_cycles + pipeline_overhead + stall_cycles

    # Mapping efficiency: fraction of PEs doing useful work per cycle
    used_rows = s_r / (folds_r * array_h)
    used_cols = s_c / (folds_c * array_w)
    mapping_eff = used_rows * used_cols

    total_macs = m * n * k
    num_pes = array_h * array_w
    compute_util = total_macs / (num_pes * total_cycles) if total_cycles > 0 else 0

    return ComputeResult(
        total_cycles=total_cycles,
        compute_cycles=compute_cycles + pipeline_overhead,
        stall_cycles=stall_cycles,
        mapping_efficiency=mapping_eff,
        compute_utilization=compute_util,
        total_macs=total_macs,
        num_pes=num_pes,
        sram_limited=(stall_cycles > 0),
    )


def scalesim_compute(m: int, n: int, k: int,
                     array_h: int, array_w: int,
                     dataflow: str,
                     sram_ifmap_kb: int = 6144,
                     sram_filter_kb: int = 6144,
                     sram_ofmap_kb: int = 2048,
                     bandwidth: int = 10,
                     is_gemm: bool = True) -> ComputeResult:
    """
    Run SCALE-Sim for a single GEMM and return structured results.
    Falls back to analytical model if SCALE-Sim is not installed or fails.
    The analytical fallback uses the same SRAM sizes and bandwidth so that
    the two models are directly comparable in Experiment 5.
    """
    try:
        from scalesim.scale_sim import scalesim
    except ImportError:
        print("SCALE-Sim not available, using analytical model")
        return analytical_compute(m, n, k, array_h, array_w, dataflow,
                                  sram_ifmap_kb=sram_ifmap_kb,
                                  sram_filter_kb=sram_filter_kb,
                                  sram_ofmap_kb=sram_ofmap_kb,
                                  dram_bandwidth=float(bandwidth))

    tmpdir = tempfile.mkdtemp()
    try:
        run_name = f"h{array_h}_w{array_w}_{dataflow}"

        # Write config
        cfg_path = os.path.join(tmpdir, "config.cfg")
        with open(cfg_path, "w") as f:
            f.write(f"[general]\nrun_name = {run_name}\n\n")
            f.write("[architecture_presets]\n")
            f.write(f"ArrayHeight:    {array_h}\n")
            f.write(f"ArrayWidth:     {array_w}\n")
            f.write(f"IfmapSramSzkB:    {sram_ifmap_kb}\n")
            f.write(f"FilterSramSzkB:   {sram_filter_kb}\n")
            f.write(f"OfmapSramSzkB:    {sram_ofmap_kb}\n")
            f.write(f"IfmapOffset:    0\nFilterOffset:   10000000\n")
            f.write(f"OfmapOffset:    20000000\n")
            f.write(f"Dataflow : {dataflow}\n")
            f.write(f"Bandwidth : {bandwidth}\n")
            f.write("ReadRequestBuffer: 512\nWriteRequestBuffer: 512\n\n")
            f.write("[layout]\nIfmapCustomLayout: False\n")
            f.write("IfmapSRAMBankBandwidth: 10\nIfmapSRAMBankNum: 10\n")
            f.write("IfmapSRAMBankPort: 2\nFilterCustomLayout: False\n")
            f.write("FilterSRAMBankBandwidth: 10\nFilterSRAMBankNum: 10\n")
            f.write("FilterSRAMBankPort: 2\n\n")
            f.write("[sparsity]\nSparsitySupport : false\n")
            f.write("SparseRep : ellpack_block\nOptimizedMapping : false\n")
            f.write("BlockSize : 8\nRandomNumberGeneratorSeed : 40\n\n")
            f.write("[run_presets]\nInterfaceBandwidth: USER\n")
            f.write("UseRamulatorTrace: False\n")

        # Write topology
        topo_path = os.path.join(tmpdir, "topo.csv")
        with open(topo_path, "w") as f:
            if is_gemm:
                f.write("Layer name, M, N, K,\n")
                f.write(f"GEMM,{m},{n},{k},\n")
            else:
                raise ValueError("Conv mode requires direct topology file")

        # Write dummy layout
        layout_path = os.path.join(tmpdir, "layout.csv")
        with open(layout_path, "w") as f:
            f.write(
                "Layer name, IFMAP Height Intraline Factor, "
                "IFMAP Width Intraline Factor, "
                "Filter Height Intraline Factor, "
                "Filter Width Intraline Factor, "
                "Channel Intraline Factor, "
                "Num Filter Intraline Factor, "
                "IFMAP Height Intraline Order, "
                "IFMAP Width Intraline Order, "
                "Channel Intraline Order, "
                "IFMAP Height Interline Order, "
                "IFMAP Width Interline Order, "
                "Channel Interline Order, "
                "Num Filter Intraline Order, "
                "Channel Intraline Order, "
                "Filter Height Intraline Order, "
                "Filter Width Intraline Order, "
                "Num Filter Interline Order, "
                "Channel Interline Order, "
                "Filter Height Interline Order, "
                "Filter Width Interline Order,\n"
            )
            f.write("GEMM,1,1,1,1,1,1,0,1,2,3,4,5,3,2,1,0,7,4,5,6,\n")

        out_path = os.path.join(tmpdir, "output")
        s = scalesim(
            save_disk_space=True,
            verbose=False,
            config=cfg_path,
            topology=topo_path,
            layout=layout_path,
            input_type_gemm=is_gemm,
        )
        s.run_scale(top_path=out_path)

        report = os.path.join(out_path, run_name, "COMPUTE_REPORT.csv")
        with open(report) as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)

        total_cycles = int(row[2].strip())
        stall_cycles = int(row[3].strip())
        overall_util = float(row[4].strip()) / 100.0
        mapping_eff = float(row[5].strip()) / 100.0
        compute_util = float(row[6].strip()) / 100.0
        num_pes = array_h * array_w

        return ComputeResult(
            total_cycles=total_cycles,
            compute_cycles=total_cycles - stall_cycles,
            stall_cycles=stall_cycles,
            mapping_efficiency=mapping_eff,
            compute_utilization=compute_util,
            total_macs=m * n * k,
            num_pes=num_pes,
            sram_limited=(stall_cycles > 0),
        )

    except Exception as e:
        print(f"SCALE-Sim failed ({e}), falling back to analytical model")
        return analytical_compute(m, n, k, array_h, array_w, dataflow,
                                  sram_ifmap_kb=sram_ifmap_kb,
                                  sram_filter_kb=sram_filter_kb,
                                  sram_ofmap_kb=sram_ofmap_kb,
                                  dram_bandwidth=float(bandwidth))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
