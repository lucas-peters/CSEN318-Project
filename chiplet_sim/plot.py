"""
Visualization for chiplet sweep results.

Generates:
  1. Roofline-style plots: speedup vs. link bandwidth for each chiplet count
  2. Strategy comparison: best partition strategy as a function of bandwidth
  3. Efficiency heatmaps: chiplet count vs. bandwidth, colored by parallel efficiency
  4. Compute vs. interconnect fraction breakdown
"""

import csv
import os
import numpy as np


def _load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in row:
                if k not in ("workload", "dataflow", "topology", "strategy",
                             "best_strategy"):
                    try:
                        row[k] = float(row[k])
                    except (ValueError, KeyError):
                        pass
            rows.append(row)
    return rows


def plot_roofline(sweep_csv: str = "chiplet_sweep_results.csv",
                  output_dir: str = "plots"):
    """
    Roofline-style: X = link bandwidth, Y = speedup.
    One line per chiplet count. One plot per workload.
    Uses best strategy at each point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(sweep_csv)

    workloads = sorted(set(r["workload"] for r in data))
    topologies = sorted(set(r["topology"] for r in data))

    for wl in workloads:
        for topo in topologies:
            fig, ax = plt.subplots(figsize=(10, 6))

            wl_data = [r for r in data
                       if r["workload"] == wl and r["topology"] == topo
                       and r["num_chiplets"] > 1]

            chiplet_counts = sorted(set(int(r["num_chiplets"]) for r in wl_data))

            for P in chiplet_counts:
                p_data = [r for r in wl_data if int(r["num_chiplets"]) == P]
                # Group by bandwidth, take best strategy
                bw_best = {}
                for r in p_data:
                    bw = r["link_bw"]
                    if bw not in bw_best or r["speedup_no_overlap"] > bw_best[bw]["speedup_no_overlap"]:
                        bw_best[bw] = r

                bws = sorted(bw_best.keys())
                speedups = [bw_best[b]["speedup_no_overlap"] for b in bws]
                ax.plot(bws, speedups, "o-", label=f"P={P}", markersize=4)

                # Draw ideal speedup line
                ax.axhline(y=P, color="gray", linestyle=":", alpha=0.3)

            ax.set_xscale("log", base=2)
            ax.set_xlabel("Link Bandwidth (bytes/cycle)")
            ax.set_ylabel("Speedup vs. Single Chiplet")
            ax.set_title(f"{wl} — {topo} topology")
            ax.legend()
            ax.grid(True, alpha=0.3)

            fname = f"{output_dir}/roofline_{wl}_{topo}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname}")


def plot_crossover(crossover_csv: str = "crossover_results.csv",
                   output_dir: str = "plots"):
    """
    Compute vs. interconnect fraction as stacked area.
    X = link bandwidth, Y = fraction. One plot per workload.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(crossover_csv)

    workloads = sorted(set(r["workload"] for r in data))

    for wl in workloads:
        wl_data = [r for r in data if r["workload"] == wl]
        chiplet_counts = sorted(set(int(r["num_chiplets"]) for r in wl_data))

        fig, axes = plt.subplots(1, len(chiplet_counts),
                                 figsize=(5 * len(chiplet_counts), 5),
                                 sharey=True)
        if len(chiplet_counts) == 1:
            axes = [axes]

        for ax, P in zip(axes, chiplet_counts):
            p_data = sorted(
                [r for r in wl_data if int(r["num_chiplets"]) == P],
                key=lambda r: r["link_bw"]
            )
            bws = [r["link_bw"] for r in p_data]
            comp = [r["compute_fraction"] for r in p_data]
            ic = [r["interconnect_fraction"] for r in p_data]

            ax.stackplot(bws, comp, ic,
                         labels=["Compute", "Interconnect"],
                         colors=["#4C72B0", "#DD8452"], alpha=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Link Bandwidth (B/cyc)")
            ax.set_title(f"P={P}")
            ax.set_ylim(0, 1)

            # Mark crossover point where speedup > 1
            speedups = [r["speedup"] for r in p_data]
            for i, sp in enumerate(speedups):
                if sp >= 1.0 and i > 0 and speedups[i-1] < 1.0:
                    ax.axvline(x=bws[i], color="red", linestyle="--",
                               alpha=0.7, label=f"Crossover @ {bws[i]} B/cyc")
                    ax.legend(fontsize=8)
                    break

        axes[0].set_ylabel("Time Fraction")
        axes[0].legend(loc="lower left", fontsize=8)
        fig.suptitle(f"{wl} — Compute vs. Interconnect Bound", fontsize=13)
        plt.tight_layout()

        fname = f"{output_dir}/crossover_{wl}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def plot_strategy_map(sweep_csv: str = "chiplet_sweep_results.csv",
                      output_dir: str = "plots"):
    """
    Which partition strategy wins at each bandwidth, for each chiplet count.
    Color-coded bar chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(sweep_csv)

    strategy_colors = {
        "split_m": "#4C72B0",
        "split_n": "#55A868",
        "split_k": "#C44E52",
        "split_mn": "#8172B2",
    }

    workloads = sorted(set(r["workload"] for r in data))
    topo = "ring"  # most practical topology

    for wl in workloads:
        wl_data = [r for r in data
                   if r["workload"] == wl and r["topology"] == topo
                   and r["num_chiplets"] > 1]
        if not wl_data:
            continue

        chiplet_counts = sorted(set(int(r["num_chiplets"]) for r in wl_data))

        fig, axes = plt.subplots(1, len(chiplet_counts),
                                 figsize=(5 * len(chiplet_counts), 4),
                                 sharey=True)
        if len(chiplet_counts) == 1:
            axes = [axes]

        for ax, P in zip(axes, chiplet_counts):
            p_data = [r for r in wl_data if int(r["num_chiplets"]) == P]
            bw_best = {}
            for r in p_data:
                bw = r["link_bw"]
                if bw not in bw_best or r["speedup_no_overlap"] > bw_best[bw]["speedup_no_overlap"]:
                    bw_best[bw] = r

            bws = sorted(bw_best.keys())
            strategies = [bw_best[b]["strategy"] for b in bws]
            colors = [strategy_colors.get(s, "gray") for s in strategies]
            speedups = [bw_best[b]["speedup_no_overlap"] for b in bws]

            ax.bar(range(len(bws)), speedups, color=colors, width=0.8)
            ax.set_xticks(range(len(bws)))
            ax.set_xticklabels([str(int(b)) for b in bws], rotation=45, fontsize=8)
            ax.set_xlabel("Link BW (B/cyc)")
            ax.set_title(f"P={P}")
            ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3)

        axes[0].set_ylabel("Speedup")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=s)
                           for s, c in strategy_colors.items()]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
        fig.suptitle(f"{wl} — Best Strategy by Bandwidth (ring)", fontsize=13)
        plt.tight_layout()

        fname = f"{output_dir}/strategy_{wl}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Crossover B* and compute-to-communication correlation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_bstar(data: list) -> dict:
    """
    From crossover CSV rows, find B* for each (workload, P): the lowest
    bandwidth at which speedup ≥ 1.0.  Returns {(workload, P): bstar_or_None}.
    """
    keys = set((r["workload"], int(float(r["num_chiplets"]))) for r in data)
    result = {}
    for (wl, P) in keys:
        subset = sorted(
            [r for r in data if r["workload"] == wl
             and int(float(r["num_chiplets"])) == P],
            key=lambda r: float(r["link_bw"])
        )
        bstar = None
        for r in subset:
            if float(r["speedup"]) >= 1.0:
                bstar = float(r["link_bw"])
                break
        result[(wl, P)] = bstar
    return result


def _workload_dims():
    """Return {name: (m, n, k)} for all known workloads."""
    from .workloads import ALL_WORKLOADS, REPORT_WORKLOADS, VALIDATION_WORKLOADS
    lookup = {}
    for wl in ALL_WORKLOADS + REPORT_WORKLOADS + VALIDATION_WORKLOADS:
        lookup[wl.name] = (wl.m, wl.n, wl.k)
    return lookup


def plot_crossover_bstar(crossover_csv: str = "crossover_results.csv",
                         output_dir: str = "plots"):
    """
    Experiment 1: scatter-plot B* (minimum bandwidth for net-positive scaling)
    vs. the GEMM's intrinsic compute-to-communication (CTC) ratio.

    CTC ratio for split-M: (M/P * N * K MACs) / (K*N elements communicated)
                         = M / P   (in elements; elem cancels)

    A strong linear relationship between CTC and B* means a simple closed-form
    predictor of B* exists from GEMM shape alone.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(crossover_csv):
        print(f"Skipping B* plot: {crossover_csv} not found")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(crossover_csv)
    bstar_map = _extract_bstar(data)

    # Gather workload metadata from the workload registry
    wl_meta = _workload_dims()

    markers = {2: "o", 4: "s", 8: "^"}
    colors  = {2: "#4C72B0", 4: "#DD8452", 8: "#55A868"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for P in [2, 4, 8]:
        xs, ys, labels = [], [], []
        for (wl, p), bstar in bstar_map.items():
            if p != P or bstar is None:
                continue
            m, n, k = wl_meta[wl]
            # CTC = compute operations / elements communicated (split-M)
            ctc = (m / P) * n * k / (k * n)   # simplifies to m/P
            xs.append(ctc)
            ys.append(bstar)
            labels.append(wl)
        if not xs:
            continue

        ax.scatter(xs, ys, marker=markers[P], color=colors[P],
                   s=80, label=f"P={P}", zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab.replace("resnet50_", "").replace("_seq2048", ""),
                        (x, y), textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=colors[P])

        # Linear regression
        if len(xs) >= 2:
            coeffs = np.polyfit(np.log2(xs), np.log2(ys), 1)
            x_fit = np.linspace(min(xs), max(xs), 100)
            y_fit = 2 ** np.polyval(coeffs, np.log2(x_fit))
            ax.plot(x_fit, y_fit, "--", color=colors[P], alpha=0.5, linewidth=1)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Compute-to-Communication Ratio  (M/P)", fontsize=11)
    ax.set_ylabel("Crossover Bandwidth B*  (bytes/cycle)", fontsize=11)
    ax.set_title("Exp 1: B* vs. Compute-to-Communication Ratio", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fname = f"{output_dir}/exp1_bstar_correlation.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Topology comparison at P=8
# ─────────────────────────────────────────────────────────────────────────────

def plot_topology_comparison(sweep_csv: str = "chiplet_sweep_results.csv",
                              output_dir: str = "plots"):
    """
    Experiment 2: grouped bar chart showing speedup for ring, 2D mesh, and
    all-to-all at P=8 and B in {16, 64, 256} bytes/cycle for each workload.

    Also annotates at which bandwidth ring becomes competitive with all-to-all
    (defined as ring speedup within 10% of all-to-all speedup).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(sweep_csv):
        print(f"Skipping topology plot: {sweep_csv} not found")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(sweep_csv)

    target_P  = 8
    target_bw = [16.0, 64.0, 256.0]
    topos     = ["ring", "mesh", "all_to_all", "hierarchical"]
    topo_colors = {
        "ring":         "#4C72B0",
        "mesh":         "#55A868",
        "all_to_all":   "#C44E52",
        "hierarchical": "#8172B2",
    }
    topo_labels = {
        "ring": "Ring", "mesh": "2×4 Mesh",
        "all_to_all": "All-to-All", "hierarchical": "Hierarchical",
    }

    workloads = sorted(set(r["workload"] for r in data))

    for wl in workloads:
        fig, axes = plt.subplots(1, len(target_bw),
                                 figsize=(5 * len(target_bw), 5),
                                 sharey=True)

        for ax, bw in zip(axes, target_bw):
            subset = [r for r in data
                      if r["workload"] == wl
                      and int(float(r["num_chiplets"])) == target_P
                      and abs(float(r["link_bw"]) - bw) < 0.01]

            topo_speedup = {}
            for r in subset:
                t = r["topology"]
                sp = float(r["speedup_no_overlap"])
                if t not in topo_speedup or sp > topo_speedup[t]:
                    topo_speedup[t] = sp

            present_topos = [t for t in topos if t in topo_speedup]
            bar_x = np.arange(len(present_topos))
            bar_h = [topo_speedup[t] for t in present_topos]
            bar_c = [topo_colors[t] for t in present_topos]

            ax.bar(bar_x, bar_h, color=bar_c, width=0.6, zorder=2)
            ax.set_xticks(bar_x)
            ax.set_xticklabels([topo_labels.get(t, t) for t in present_topos],
                               rotation=20, ha="right", fontsize=8)
            ax.axhline(y=target_P, color="gray", linestyle=":",
                       alpha=0.5, label=f"ideal {target_P}x")
            ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3)
            ax.set_title(f"B = {int(bw)} B/cyc")
            ax.set_xlabel("Topology")
            ax.grid(True, axis="y", alpha=0.3, zorder=0)

            # Annotate ring-vs-all-to-all gap
            if "ring" in topo_speedup and "all_to_all" in topo_speedup:
                gap = topo_speedup["all_to_all"] - topo_speedup["ring"]
                ax.annotate(f"Δ={gap:.2f}x",
                            xy=(0, topo_speedup["ring"]),
                            xytext=(0.05, topo_speedup["ring"] + 0.15),
                            fontsize=8, color="#4C72B0")

        axes[0].set_ylabel(f"Speedup vs. Single Chiplet  (P={target_P})")
        fig.suptitle(f"{wl} — Topology Comparison", fontsize=13)
        plt.tight_layout()
        fname = f"{output_dir}/exp2_topology_{wl}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Partition strategy phase diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_diagram(phase_csv: str = "phase_diagram_results.csv",
                       output_dir: str = "plots"):
    """
    Experiment 3: 2-D heatmap per workload.
      X-axis: link bandwidth (log2 scale, 1–256 B/cyc)
      Y-axis: chiplet count P (2–8)
      Cell colour: winning partition strategy

    A secondary speedup contour overlaid in grey shows the iso-speedup lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if not os.path.exists(phase_csv):
        print(f"Skipping phase diagram: {phase_csv} not found")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(phase_csv)

    strategy_list = ["split_m", "split_n", "split_k", "split_mn"]
    strategy_idx  = {s: i for i, s in enumerate(strategy_list)}
    strategy_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    cmap = ListedColormap(strategy_colors)

    workloads  = sorted(set(r["workload"] for r in data))
    all_bws    = sorted(set(float(r["link_bw"]) for r in data))
    all_Ps     = sorted(set(int(float(r["num_chiplets"])) for r in data))

    for wl in workloads:
        wl_data = [r for r in data if r["workload"] == wl]

        grid_strategy = np.full((len(all_Ps), len(all_bws)), np.nan)
        grid_speedup  = np.full((len(all_Ps), len(all_bws)), np.nan)

        for r in wl_data:
            pi = all_Ps.index(int(float(r["num_chiplets"])))
            bi = all_bws.index(float(r["link_bw"]))
            grid_strategy[pi, bi] = strategy_idx.get(r["best_strategy"], 0)
            grid_speedup[pi, bi]  = float(r["speedup"])

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(grid_strategy, aspect="auto", origin="lower",
                       cmap=cmap, vmin=-0.5, vmax=len(strategy_list) - 0.5,
                       interpolation="nearest")

        # Speedup contours
        bw_vals = np.arange(len(all_bws))
        p_vals  = np.arange(len(all_Ps))
        cs = ax.contour(bw_vals, p_vals, grid_speedup,
                        levels=[1.0, 2.0, 4.0, 6.0],
                        colors="white", linewidths=0.8, alpha=0.7)
        ax.clabel(cs, fmt="%.0fx", fontsize=7)

        ax.set_xticks(range(len(all_bws)))
        ax.set_xticklabels([str(int(b)) for b in all_bws], fontsize=8)
        ax.set_yticks(range(len(all_Ps)))
        ax.set_yticklabels([str(p) for p in all_Ps])
        ax.set_xlabel("Link Bandwidth (bytes/cycle)")
        ax.set_ylabel("Chiplet Count P")
        ax.set_title(f"{wl} — Partition Strategy Phase Diagram")

        legend_patches = [Patch(facecolor=c, label=s)
                          for s, c in zip(strategy_list, strategy_colors)]
        ax.legend(handles=legend_patches, loc="upper left",
                  fontsize=8, framealpha=0.8)

        fname = f"{output_dir}/exp3_phase_{wl}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Scaling efficiency
# ─────────────────────────────────────────────────────────────────────────────

def plot_scaling_efficiency(scaling_csv: str = "scaling_results.csv",
                             output_dir: str = "plots"):
    """
    Experiment 4: two-panel figure per workload.

    Top panel: speedup vs. P at B ∈ {16, 64, 256} bytes/cycle with the ideal
    linear speedup shown as a dashed reference.

    Bottom panel: stacked bar showing the fraction of total cycles spent in
    compute vs. communication at each (P, B) point, making the communication
    bottleneck visible directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(scaling_csv):
        print(f"Skipping scaling plot: {scaling_csv} not found")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(scaling_csv)

    workloads  = sorted(set(r["workload"] for r in data))
    bandwidths = sorted(set(float(r["link_bw"]) for r in data))
    bw_colors  = {16.0: "#4C72B0", 64.0: "#DD8452", 256.0: "#55A868"}

    for wl in workloads:
        wl_data = [r for r in data if r["workload"] == wl]
        fig, (ax_sp, ax_eff, ax_decomp) = plt.subplots(
            3, 1, figsize=(9, 11), sharex=False)

        all_Ps = sorted(set(int(float(r["num_chiplets"])) for r in wl_data))

        # ── Panel 1: speedup ─────────────────────────────────
        for bw in bandwidths:
            bw_rows = sorted(
                [r for r in wl_data if abs(float(r["link_bw"]) - bw) < 0.01],
                key=lambda r: int(float(r["num_chiplets"]))
            )
            if not bw_rows:
                continue
            xs = [int(float(r["num_chiplets"])) for r in bw_rows]
            ys = [float(r["speedup"]) for r in bw_rows]
            ax_sp.plot(xs, ys, "o-", color=bw_colors.get(bw, "gray"),
                       label=f"B={int(bw)} B/cyc", markersize=5)

        max_P = max(all_Ps)
        ax_sp.plot(all_Ps, all_Ps, "k--", alpha=0.3, label="Ideal")
        ax_sp.set_ylabel("Speedup")
        ax_sp.set_title(f"{wl} — Scaling Efficiency")
        ax_sp.legend(fontsize=8)
        ax_sp.grid(True, alpha=0.3)
        ax_sp.set_xticks(all_Ps)

        # ── Panel 2: efficiency = speedup / P ──────────────────
        for bw in bandwidths:
            bw_rows = sorted(
                [r for r in wl_data if abs(float(r["link_bw"]) - bw) < 0.01],
                key=lambda r: int(float(r["num_chiplets"]))
            )
            if not bw_rows:
                continue
            xs = [int(float(r["num_chiplets"])) for r in bw_rows]
            ys = [float(r["efficiency"]) for r in bw_rows]
            ax_eff.plot(xs, ys, "o-", color=bw_colors.get(bw, "gray"),
                        label=f"B={int(bw)} B/cyc", markersize=5)

        ax_eff.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
        ax_eff.set_ylabel("Efficiency  (speedup / P)")
        ax_eff.set_ylim(0, 1.1)
        ax_eff.legend(fontsize=8)
        ax_eff.grid(True, alpha=0.3)
        ax_eff.set_xticks(all_Ps)

        # ── Panel 3: compute / comm decomposition stacked bars ──
        # Show at B=64 as representative; one group per P value
        bw_ref = 64.0
        decomp_rows = sorted(
            [r for r in wl_data
             if abs(float(r["link_bw"]) - bw_ref) < 0.01
             and int(float(r["num_chiplets"])) > 1],
            key=lambda r: int(float(r["num_chiplets"]))
        )
        if decomp_rows:
            xs = np.arange(len(decomp_rows))
            comp_f = [float(r["compute_fraction"]) for r in decomp_rows]
            comm_f = [float(r["comm_fraction"])    for r in decomp_rows]
            ax_decomp.bar(xs, comp_f, color="#4C72B0", label="Compute", width=0.6)
            ax_decomp.bar(xs, comm_f, bottom=comp_f, color="#DD8452",
                          label="Comm", width=0.6)
            ax_decomp.set_xticks(xs)
            ax_decomp.set_xticklabels(
                [f"P={int(float(r['num_chiplets']))}" for r in decomp_rows])
            ax_decomp.set_ylabel("Time Fraction")
            ax_decomp.set_title(f"Compute vs. Comm  (B={int(bw_ref)} B/cyc)")
            ax_decomp.legend(fontsize=8)
            ax_decomp.set_ylim(0, 1.05)
            ax_decomp.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fname = f"{output_dir}/exp4_scaling_{wl}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Validation MAPE table
# ─────────────────────────────────────────────────────────────────────────────

def plot_validation_mape(validation_csv: str = "validation_results.csv",
                         output_dir: str = "plots"):
    """
    Experiment 5: two figures.

    Figure A – Heatmap: rows = workloads, columns = array_size×dataflow
    combinations, cell = relative error %.

    Figure B – Scatter: analytical_cycles vs. scalesim_cycles for every
    data point (log-log), with a y=x reference line.  Divergence from the
    line quantifies where the analytical model under- or over-predicts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(validation_csv):
        print(f"Skipping validation plot: {validation_csv} not found")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = _load_csv(validation_csv)

    workloads = sorted(set(r["workload"] for r in data))
    configs   = sorted(set(
        f"{int(float(r['array_h']))}x{int(float(r['array_w']))}/{r['dataflow']}"
        for r in data
    ))

    # ── Figure A: heatmap ─────────────────────────────────────
    grid = np.full((len(workloads), len(configs)), np.nan)
    for r in data:
        wi = workloads.index(r["workload"])
        cfg = (f"{int(float(r['array_h']))}x{int(float(r['array_w']))}"
               f"/{r['dataflow']}")
        ci = configs.index(cfg)
        grid[wi, ci] = float(r["rel_error_pct"])

    mape = np.nanmean(grid)

    fig_a, ax_a = plt.subplots(figsize=(max(8, len(configs) * 1.4),
                                         max(4, len(workloads) * 0.7)))
    im = ax_a.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax_a, label="Relative Error (%)")

    for wi in range(len(workloads)):
        for ci in range(len(configs)):
            val = grid[wi, ci]
            if not np.isnan(val):
                ax_a.text(ci, wi, f"{val:.1f}", ha="center", va="center",
                          fontsize=7, color="black" if val < 20 else "white")

    ax_a.set_xticks(range(len(configs)))
    ax_a.set_xticklabels(configs, rotation=35, ha="right", fontsize=8)
    ax_a.set_yticks(range(len(workloads)))
    ax_a.set_yticklabels(workloads, fontsize=8)
    ax_a.set_title(f"Exp 5: Analytical vs. SCALE-Sim  —  MAPE = {mape:.1f}%")

    fname_a = f"{output_dir}/exp5_validation_heatmap.png"
    fig_a.savefig(fname_a, dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved {fname_a}")

    # ── Figure B: scatter analytical vs. scalesim ────────────────
    fig_b, ax_b = plt.subplots(figsize=(7, 6))
    all_ana = [float(r["analytical_cycles"]) for r in data]
    all_sim = [float(r["scalesim_cycles"])   for r in data]

    ax_b.scatter(all_sim, all_ana, s=40, alpha=0.7, color="#4C72B0", zorder=3)
    lo, hi = min(all_sim + all_ana), max(all_sim + all_ana)
    ax_b.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("SCALE-Sim cycles")
    ax_b.set_ylabel("Analytical cycles")
    ax_b.set_title(f"Exp 5: Predicted vs. Simulated  (MAPE = {mape:.1f}%)")
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)

    fname_b = f"{output_dir}/exp5_validation_scatter.png"
    fig_b.savefig(fname_b, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print(f"Saved {fname_b}")

    # Print summary to stdout
    print(f"\nValidation summary  (MAPE = {mape:.2f}%)")
    print(f"  {'Config':<22} MAPE")
    for ci, cfg in enumerate(configs):
        col_mape = np.nanmean(grid[:, ci])
        print(f"  {cfg:<22} {col_mape:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Master dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(sweep_csv: str = "chiplet_sweep_results.csv",
             crossover_csv: str = "crossover_results.csv",
             phase_csv: str = "phase_diagram_results.csv",
             scaling_csv: str = "scaling_results.csv",
             validation_csv: str = "validation_results.csv",
             output_dir: str = "plots"):
    """Generate all plots for all five experiments."""
    if os.path.exists(sweep_csv):
        plot_roofline(sweep_csv, output_dir)
        plot_strategy_map(sweep_csv, output_dir)
        plot_topology_comparison(sweep_csv, output_dir)
    if os.path.exists(crossover_csv):
        plot_crossover(crossover_csv, output_dir)
        plot_crossover_bstar(crossover_csv, output_dir)
    if os.path.exists(phase_csv):
        plot_phase_diagram(phase_csv, output_dir)
    if os.path.exists(scaling_csv):
        plot_scaling_efficiency(scaling_csv, output_dir)
    if os.path.exists(validation_csv):
        plot_validation_mape(validation_csv, output_dir)
