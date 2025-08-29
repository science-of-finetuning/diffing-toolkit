#!/bin/bash

# Dynamic script for running diffing experiments
# Usage: ./diffing.sh <organism> <method> [additional_args...]
# Example: ./diffing.sh kansas_abortion crosscoder pipeline.mode=diffing

if [ $# -lt 2 ]; then
    echo "Usage: $0 <organism> <method> [additional_args...]"
    echo "Example: $0 kansas_abortion crosscoder infrastructure=runpod"
    exit 1
fi

ORGANISM=$1
METHOD=$2
shift 2  # Remove first two arguments

# Run the command with dynamic arguments
python main.py diffing/method=$METHOD organism=$ORGANISM pipeline.mode=diffing "$@"

