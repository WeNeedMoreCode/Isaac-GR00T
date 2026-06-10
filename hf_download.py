#!/usr/bin/env python3
"""
HuggingFace 模型断点续传下载工具。
用 wget -c 下载，支持中断后重新运行续传。

用法:
  python hf_download.py                                        # 默认下载 GR00T-N1.7-3B
  python hf_download.py nvidia/Cosmos-Reason2-2B ./cosmos      # 指定模型和目录
"""

import os
import subprocess
import sys

from huggingface_hub import get_token, list_repo_files

DEFAULT_REPO = "nvidia/GR00T-N1.7-3B"


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    local_dir = sys.argv[2] if len(sys.argv) > 2 else repo.replace("/", "--")
    token = get_token()

    base_url = f"https://huggingface.co/{repo}/resolve/main"
    auth = ["--header", f"Authorization: Bearer {token}"] if token else []

    os.makedirs(local_dir, exist_ok=True)
    os.chdir(local_dir)

    print(f"=== 下载 {repo} => {os.getcwd()} ===")
    if token:
        print(f"Token: {token[:8]}...")
    else:
        print("[警告] 未找到 token，gated model 将下载失败")

    files = list_repo_files(repo, token=token)
    for f in files:
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            print(f"[跳过] {f} (已存在)")
            continue
        print(f"[下载] {f}")
        # 部分网络环境下可能需要调整 SSL 验证设置
        url = f"{base_url}/{f}"
        r = subprocess.run(["wget", "-c"] + auth + [url])
        if r.returncode != 0:
            print(f"[警告] {f} 下载失败，可重新运行脚本续传")

    print("=== 完成 ===")


if __name__ == "__main__":
    main()
