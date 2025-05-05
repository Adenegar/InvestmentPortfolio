#!/bin/bash

# --- Configuration ---
PYTHON_EXE="python"       # Or python3, or path/to/your/venv/bin/python
SCRIPT_NAME="run_simulation.py"
POLICY="Base"           # <<<< Set the desired Policy Type here
NUM_SIMULATIONS=50       # <<<< Set the desired number of simulations per run
SIMULATION_TYPE="monte_carlo" # <<<< Set the simulation type (bootstrap or monte_carlo)

STOCKS=(5 10 15)
DURATIONS=("3m" "6m" "1y" "3y" "5y" "10y")
# --- End Configuration ---

# Get the directory where the script is located to find the python script reliably
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT_PATH"
    exit 1
fi

echo "Starting simulation batch..."
echo "Policy: $POLICY"
echo "Simulations per run: $NUM_SIMULATIONS"
echo "Simulation Type: $SIMULATION_TYPE"
echo "---"

# Loop through all combinations
for stock_count in "${STOCKS[@]}"; do
  for duration in "${DURATIONS[@]}"; do
    echo "==> Running: Stocks=$stock_count, Duration=$duration <=="

    # Construct and execute the command
    command="$PYTHON_EXE \"$PYTHON_SCRIPT_PATH\" \"$POLICY\" --num_stocks $stock_count --simulation_length \"$duration\" --num_simulations $NUM_SIMULATIONS --simulation_type \"$SIMULATION_TYPE\""

    echo "$command" # Print the command being executed
    eval $command    # Execute the command

    # Check the exit code of the last command
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Warning: Command failed with exit code $exit_code for Stocks=$stock_count, Duration=$duration"
        # Decide if you want to stop on error:
        # exit $exit_code
    fi
    echo "---"
  done
done

echo "Simulation batch finished."
