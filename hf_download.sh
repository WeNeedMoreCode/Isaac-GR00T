#!/bin/bash
# 用法: bash hf_download.sh [repo_id] [local_dir]
# 例: bash hf_download.sh                                    # 默认下载 GR00T-N1.7-3B
# 例: bash hf_download.sh nvidia/Cosmos-Reason2-2B ./cosmos  # 指定模型和目录

set -e

REPO_ID="${1:-nvidia/GR00T-N1.7-3B}"
LOCAL_DIR="${2:-./${REPO_ID//\//--}}"

# 自动获取 token
TOKEN=""
for f in ~/.cache/huggingface/token ~/.huggingface/token; do
    if [ -r "$f" ]; then
        TOKEN=$(tr -d '[:space:]' < "$f")
        break
    fi
done

BASE_URL="https://huggingface.co/$REPO_ID/resolve/main"

if [ -n "$TOKEN" ]; then
    AUTH=("--header" "Authorization: Bearer $TOKEN")
else
    AUTH=()
fi

mkdir -p "$LOCAL_DIR"
cd "$LOCAL_DIR"

echo "=== 获取文件列表 ==="
FILES=$(python3 -c "
from huggingface_hub import list_repo_files
for f in list_repo_files('$REPO_ID', token='$TOKEN' if '$TOKEN' else None):
    print(f)
")

for f in $FILES; do
    if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 0 ]; then
        echo "[跳过] $f (已存在)"
        continue
    fi
    echo "[下载] $f"
    wget -c --no-check-certificate "${AUTH[@]}" "$BASE_URL/$f"
done

echo "=== 完成 ==="
ls -lh
