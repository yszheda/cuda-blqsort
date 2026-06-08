#!/usr/bin/env bash
set -euo pipefail

# Deploy cuda-blqsort to GPU servers via SSH
# Usage: ./scripts/deploy.sh [thor|a40|all]

REPO_NAME="cuda-blqsort"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

declare -A SERVERS=(
    [thor]="shuyua01@10.190.0.91:/home/shuyua01/Development/"
    [a40]="shuyua01@szc-td04:/project/ai/npu_sw/shuyua01/Development"
)

TARGET="${1:-all}"

deploy_to() {
    local server_name="$1"
    local server_path="${SERVERS[$server_name]}"
    local host="${server_path%%:*}"
    local remote_path="${server_path#*:}"

    echo "Deploying to ${server_name} (${host})..."
    echo "  Remote path: ${remote_path}"

    ssh "$host" "mkdir -p ${remote_path}"

    rsync -avz --delete \
        --exclude='.git/' \
        --exclude='build/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        "${LOCAL_DIR}/" \
        "${host}:${remote_path}${REPO_NAME}/"

    ssh "$host" "cd ${remote_path} && [ -d blqsort ] || git clone https://github.com/chkas/blqsort.git"

    ssh "$host" "cd ${remote_path}${REPO_NAME} && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"

    echo "Deployed to ${server_name} successfully."
}

if [[ "$TARGET" == "all" ]]; then
    for server in "${!SERVERS[@]}"; do
        deploy_to "$server"
    done
else
    if [[ -z "${SERVERS[$TARGET]+x}" ]]; then
        echo "Unknown server: ${TARGET}. Available: ${!SERVERS[*]}"
        exit 1
    fi
    deploy_to "$TARGET"
fi
