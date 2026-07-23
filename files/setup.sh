#!/bin/bash
# 一次性環境建置。在你想放 SLOTHY 的目錄執行。
set -e

git clone https://github.com/slothy-optimizer/slothy.git
python3 -m venv venv
source venv/bin/activate
pip install -r slothy/requirements.txt

# 把 V2 model 放進 target 目錄
cp neoverse_v2.py slothy/slothy/targets/aarch64/

# selftest 需要 LLVM 工具鏈（可選但強烈建議，見 README）
# Ubuntu/Debian:  sudo apt-get install llvm
# macOS:          brew install llvm

echo "完成。之後每次用: source venv/bin/activate && export PYTHONPATH=\$PWD/slothy"
