#!/usr/bin/env bash
set -euo pipefail

VENV=".venv"

# ── Create virtual environment and install dependencies ──────────────────────
setup() {
    if [ ! -d "$VENV" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV"
    fi
    echo "Installing dependencies..."
    "$VENV/bin/pip" install --upgrade pip -q
    "$VENV/bin/pip" install -r requirements.txt -q
    echo "✓ Setup complete."
}

# ── Main ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: ./run.sh <command>

Commands:
  setup            Install dependencies into a virtual environment
  all              Run all five experiments (Exp 1-5)
  exp1             Exp 1: Crossover bandwidth B* and CTC correlation
  exp2             Exp 2: Topology comparison at P=8
  exp3             Exp 3: Partition strategy phase diagram
  exp4             Exp 4: Scaling efficiency vs. chiplet count
  exp5             Exp 5: Compute model validation vs. SCALE-Sim
  quick            Small/fast sweep for quick testing
  plot-only        Regenerate plots from existing CSV files
  validate         Quick analytical vs. SCALE-Sim check
  clean            Remove virtual environment, caches, and outputs
EOF
}

# Ensure venv exists before any run command
ensure_setup() {
    if [ ! -f "$VENV/bin/activate" ]; then
        setup
    fi
}

PY="$VENV/bin/python"

case "${1:-all}" in
    setup)      setup ;;
    all)        ensure_setup; "$PY" run_chiplet_sim.py --all-experiments ;;
    exp1)       ensure_setup; "$PY" run_chiplet_sim.py --exp1 ;;
    exp2)       ensure_setup; "$PY" run_chiplet_sim.py --exp2 ;;
    exp3)       ensure_setup; "$PY" run_chiplet_sim.py --exp3 ;;
    exp4)       ensure_setup; "$PY" run_chiplet_sim.py --exp4 ;;
    exp5)       ensure_setup; "$PY" run_chiplet_sim.py --exp5 ;;
    quick)      ensure_setup; "$PY" run_chiplet_sim.py --quick ;;
    plot-only)  ensure_setup; "$PY" run_chiplet_sim.py --plot-only ;;
    validate)   ensure_setup; "$PY" run_chiplet_sim.py --validate ;;
    clean)      rm -rf "$VENV" __pycache__ chiplet_sim/__pycache__ plots/*.png *.csv
                echo "✓ Cleaned." ;;
    *)          usage ;;
esac
