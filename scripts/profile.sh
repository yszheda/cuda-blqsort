#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute profiling script
# Usage: ./scripts/profile.sh <benchmark_binary> [output_dir]

BENCHMARK="${1:-}"
OUTPUT_DIR="${2:-docs/reports}"

if [[ -z "$BENCHMARK" ]]; then
    echo "Usage: $0 <benchmark_binary> [output_dir]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Running Nsight Compute profiler on ${BENCHMARK}..."
echo "Output directory: ${OUTPUT_DIR}"

if ! command -v ncu &> /dev/null; then
    echo "ERROR: ncu (Nsight Compute) not found in PATH"
    echo "Install CUDA toolkit with Nsight Compute to use this script"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/profile_${TIMESTAMP}.ncu-rep"

ncu --set full \
    --target-processes all \
    --import-source yes \
    -o "$REPORT_FILE" \
    "$BENCHMARK"

echo "Profile saved to ${REPORT_FILE}"

TEXT_REPORT="${OUTPUT_DIR}/profile_${TIMESTAMP}.txt"
echo "=== Nsight Compute Summary ===" > "$TEXT_REPORT"
echo "Date: $(date)" >> "$TEXT_REPORT"
echo "" >> "$TEXT_REPORT"

ncu --set full \
    --target-processes all \
    --page raw \
    "$BENCHMARK" >> "$TEXT_REPORT" 2>&1

echo "Text summary saved to ${TEXT_REPORT}"
echo "Done."
