#!/bin/bash

# Dynamic script for running preprocessing experiments
# Usage: ./preprocessing.sh <organism> <method> [additional_args...]
# Example: ./preprocessing.sh kansas_abortion crosscoder pipeline.mode=preprocessing

if [ $# -lt 1 ]; then
    echo "Usage: $0 <organism> [additional_args...]"
    echo "Example: $0 kansas_abortion"
    exit 1
fi

ORGANISM=$1
shift 1  # Remove first two arguments

# Run the command with dynamic arguments
python main.py organism=$ORGANISM pipeline.mode=preprocessing "$@"