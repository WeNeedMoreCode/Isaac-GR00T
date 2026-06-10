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

python3 -c "
from huggingface_hub import list_repo_files
import os, subprocess, sys

repo = '$REPO_ID'
token = '$TOKEN' if '$TOKEN' else None
base = 'https://huggingface.co/' + repo + '/resolve/main'

auth = ['--header', 'Authorization: Bearer ' + token] if token else []

for f in list_repo_files(repo, token=token):
    if os.path.isfile(f) and os.path.getsize(f) > 0:
        print(f'[跳过] {f} (已存在)')
        continue
    print(f'[下载] {f}')
    url = base + '/' + f
    r = subprocess.run(['wget', '-c'] + auth + [url])
    if r.returncode != 0:
        print(f'[警告] {f} 下载失败，可重新运行脚本续传')
print('=== 完成 ===')
"

echo "=== 完成 ==="
ls -lh
