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


def plot_all(sweep_csv: str = "chiplet_sweep_results.csv",
             crossover_csv: str = "crossover_results.csv",
             output_dir: str = "plots"):
    """Generate all plots."""
    if os.path.exists(sweep_csv):
        plot_roofline(sweep_csv, output_dir)
        plot_strategy_map(sweep_csv, output_dir)
    if os.path.exists(crossover_csv):
        plot_crossover(crossover_csv, output_dir)
