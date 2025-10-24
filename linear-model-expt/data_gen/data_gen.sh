#!/usr/bin/env bash
# run_all_as.sh
# Simplified launcher: one-line command per tmux session (no embedded newlines).

# Load tmux module if available
module load tmux || true

# Paths
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$BASE_DIR/generate_data.py"
OUTPUT_DIR="data_files"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Parameters
ITERATIONS=5
N_SAMPLES=10000
DIMS=(2 5)
THETA_BOUNDS=(0.1 0.5 1 2 4 5 10 15 20 25 50 75 100 200)
CONDA_ENV="tf_jax_x86"
A_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)

# Launch sessions
for a in "${A_VALUES[@]}"; do
  session="gen_a${a//./_}"
  logfile="$LOG_DIR/${session}.log"
  # Build a single-line command
  cmd="source ~/.bashrc; source \"$(conda info --base)/etc/profile.d/conda.sh\"; conda activate $CONDA_ENV; cd '$BASE_DIR'; python '$SCRIPT' --dims ${DIMS[*]} --theta-bounds ${THETA_BOUNDS[*]} --a-vals $a --iterations $ITERATIONS --n-samples $N_SAMPLES --output-dir '$OUTPUT_DIR' |& tee '$logfile'; exec bash"
  # Start tmux session
  tmux new-session -d -s "$session" bash -lc "$cmd"
  if [ $? -eq 0 ]; then
    echo "Started tmux session: $session (logs -> $logfile)"
  else
    echo "Error: failed to start session $session" >&2
  fi
done

echo "Done. Use 'tmux ls' and 'tmux attach -t <session>' to connect."
