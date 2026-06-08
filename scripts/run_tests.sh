#!/usr/bin/env bash
set -euo pipefail

# Run tests on GPU servers
# Usage: ./scripts/run_tests.sh [thor|a40|all]

declare -A SERVERS=(
    [thor]="shuyua01@10.190.0.91"
    [a40]="shuyua01@szc-td04"
)

declare -A PATHS=(
    [thor]="/home/shuyua01/Development/cuda-blqsort"
    [a40]="/project/ai/npu_sw/shuyua01/Development/cuda-blqsort"
)

TARGET="${1:-all}"

run_tests_on() {
    local server_name="$1"
    local host="${SERVERS[$server_name]}"
    local remote_path="${PATHS[$server_name]}"

    echo "Running tests on ${server_name}..."

    ssh "$host" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
        echo "ERROR: GPU not available on ${server_name}"; return 1
    }

    ssh "$host" "cd ${remote_path}/build && make -j\$(nproc) && ctest -V --output-on-failure"

    echo "Tests passed on ${server_name}."
}

if [[ "$TARGET" == "all" ]]; then
    for server in "${!SERVERS[@]}"; do
        run_tests_on "$server"
    done
else
    if [[ -z "${SERVERS[$TARGET]+x}" ]]; then
        echo "Unknown server: ${TARGET}. Available: ${!SERVERS[*]}"
        exit 1
    fi
    run_tests_on "$TARGET"
fi
