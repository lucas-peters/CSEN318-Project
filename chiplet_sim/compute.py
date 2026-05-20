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

Memory stalls are not modeled analytically. Use SCALE-Sim for that.
"""

import math
import os
import csv
import shutil
import tempfile
from dataclasses import dataclass


@dataclass
class ComputeResult:
    total_cycles: int
    compute_cycles: int
    stall_cycles: int
    mapping_efficiency: float
    compute_utilization: float
    total_macs: int
    num_pes: int


def analytical_compute(m: int, n: int, k: int,
                       array_h: int, array_w: int,
                       dataflow: str) -> ComputeResult:
    """
    Analytical cycle estimate for GEMM M×N×K on systolic array H×W.
    Returns compute-only cycles (no memory stalls).
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

    # Pure compute cycles
    compute_cycles = folds_r * folds_c * t

    # Pipeline fill/drain overhead: one-time cost per fold sequence
    pipeline_overhead = folds_r * folds_c * (array_h + array_w - 2)

    total_cycles = compute_cycles + pipeline_overhead

    # Mapping efficiency: fraction of PEs doing useful work
    used_rows = s_r / (folds_r * array_h)
    used_cols = s_c / (folds_c * array_w)
    mapping_eff = used_rows * used_cols

    total_macs = m * n * k
    num_pes = array_h * array_w
    compute_util = total_macs / (num_pes * total_cycles) if total_cycles > 0 else 0

    return ComputeResult(
        total_cycles=total_cycles,
        compute_cycles=compute_cycles,
        stall_cycles=0,
        mapping_efficiency=mapping_eff,
        compute_utilization=compute_util,
        total_macs=total_macs,
        num_pes=num_pes,
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
    """
    try:
        from scalesim.scale_sim import scalesim
    except ImportError:
        print("SCALE-Sim not available, using analytical model")
        return analytical_compute(m, n, k, array_h, array_w, dataflow)

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
                # Caller must provide conv params via m=H, n=W, etc.
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
        )

    except Exception as e:
        print(f"SCALE-Sim failed ({e}), falling back to analytical model")
        return analytical_compute(m, n, k, array_h, array_w, dataflow)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
