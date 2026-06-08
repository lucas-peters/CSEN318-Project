# Chiplet-Based Multi-Systolic Array Simulator

**CSEN 318 Final Project**
Lucas Peters, Ritvik Nayak, Luke Hofstetter

Analytical simulator for modeling the performance limits of multi-chiplet
systolic array DNN accelerators. Jointly models compute, interconnect topology,
and partition strategy to explore the design space of chiplet-based architectures.

---

## Prerequisites

- **Python 3.10+**
- **Git** (required to install the SCALE-Sim dependency)

## Quick Start

```bash
./run.sh
```

This automatically installs all dependencies (if needed) and runs all five
experiments. You can also pass a specific command, e.g. `./run.sh exp1`.

That's it. Results are written to the project root (CSV files) and the `plots/`
directory (PNG figures).

## Manual Setup (Without the Script)

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run all experiments
python run_chiplet_sim.py --all-experiments
```

## Running Individual Experiments

Each experiment can be run independently:

| Experiment | Description | Command |
|---|---|---|
| **Exp 1** | Crossover bandwidth B* and CTC correlation | `./run.sh exp1` |
| **Exp 2** | Topology comparison (Ring/Mesh/All-to-All) at P=8 | `./run.sh exp2` |
| **Exp 3** | Partition strategy phase diagram (P=2-8) | `./run.sh exp3` |
| **Exp 4** | Scaling efficiency vs. chiplet count | `./run.sh exp4` |
| **Exp 5** | Compute model validation against SCALE-Sim | `./run.sh exp5` |

### Other Commands

| Command | Description |
|---|---|
| `./run.sh quick` | Small/fast sweep for quick testing |
| `./run.sh plot-only` | Regenerate plots from existing CSV files (no re-simulation) |
| `./run.sh validate` | Quick analytical vs. SCALE-Sim validation check |
| `./run.sh clean` | Remove virtual environment, CSVs, plots, and caches |

## Project Structure

```
.
├── run_chiplet_sim.py          # Main entry point / CLI driver
├── run.sh                      # Setup & run helper script
├── chiplet_sim/                # Core simulation package
│   ├── __init__.py             #   Public API surface
│   ├── compute.py              #   Analytical + SCALE-Sim compute models
│   ├── interconnect.py         #   Network topology & transfer cost model
│   ├── partition.py            #   GEMM partitioning strategies
│   ├── schedule.py             #   Multi-layer scheduling
│   ├── sweep.py                #   Parameter sweep harness
│   ├── workloads.py            #   ResNet-50 & BERT workload definitions
│   └── plot.py                 #   All plotting routines
├── requirements.txt            #   Python dependencies
├── plots/                      #   Generated plot PNGs (after running)
├── final_report.txt            #   Final written report
├── presentation_draft.txt      #   Presentation notes
├── experiment_writeups.txt     #   Per-experiment analysis
└── plot_reading_guide.txt      #   Guide to interpreting the plots
```

## Output

After running experiments, you will find:

- **CSV files** in the project root (`exp1_crossover_results.csv`, `exp2_sweep_results.csv`, etc.) containing raw numerical results.
- **PNG plots** in `plots/` visualizing each experiment's results (roofline plots, topology comparisons, phase diagrams, scaling curves, and validation charts).

## Dependencies

Installed automatically via `requirements.txt`:

- [SCALE-Sim](https://github.com/lucas-peters/SCALE-Sim) -- Single-chiplet systolic array simulator
- NumPy -- Numerical computation
- SciPy -- Scientific computing utilities
- Matplotlib -- Plot generation
