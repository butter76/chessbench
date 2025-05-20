#!/bin/bash
# Script to run Leela Chess Zero data processing

# Default parameters
OUTPUT_DIR="processed_data"
TEMP_DIR="temp_data"
DOWNLOAD_WORKERS=6
PROCESS_WORKERS=52
BEAM_OUTPUT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --temp_dir)
            TEMP_DIR="$2"
            shift
            shift
            ;;
        --download_workers)
            DOWNLOAD_WORKERS="$2"
            shift
            shift
            ;;
        --process_workers)
            PROCESS_WORKERS="$2"
            shift
            shift
            ;;
        --beam_output)
            BEAM_OUTPUT="$2"
            shift
            shift
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Run test if requested
if [ "$RUN_TEST" = true ]; then
    echo "Running test processing..."
    python test_processing.py \
        --output_dir "$OUTPUT_DIR" \
        --temp_dir "$TEMP_DIR"
    exit $?
fi

# Build command with parameters
CMD="python process_lc0_data.py \
    --output_dir $OUTPUT_DIR \
    --temp_dir $TEMP_DIR \
    --download_workers $DOWNLOAD_WORKERS \
    --process_workers $PROCESS_WORKERS"

# Add beam output if specified
if [ -n "$BEAM_OUTPUT" ]; then
    CMD="$CMD --beam_output $BEAM_OUTPUT"
fi

# Run the command
echo "Running: $CMD"
eval $CMD

echo "Process completed with exit code $?" 