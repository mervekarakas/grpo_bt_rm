#!/usr/bin/env bash
set -euo pipefail

# Persistent filesystem root on Lambda
FS_ROOT="${FS_ROOT:-/mnt/merve-grpo}"

# Expected layout on Lambda persistent FS:
#   $FS_ROOT/projects/grpo_bt_rm
#   $FS_ROOT/projects/ms-swift
#   $FS_ROOT/venvs/swift
#   $FS_ROOT/hf_home

mkdir -p "$FS_ROOT/projects" \
         "$FS_ROOT/venvs" \
         "$FS_ROOT/hf_home" \
         "$FS_ROOT/hf_home/datasets"

# Hugging Face cache convention (matches your Lin server)
export HF_HOME="$FS_ROOT/hf_home"
export HF_DATASETS_CACHE="$FS_ROOT/hf_home/datasets"
unset TRANSFORMERS_CACHE

# Create a persistent venv named "swift"
if [ ! -d "$FS_ROOT/venvs/swift" ]; then
  python3 -m venv "$FS_ROOT/venvs/swift"
fi

source "$FS_ROOT/venvs/swift/bin/activate"
pip install -U pip wheel setuptools

# Install project dependencies
REQ_FILE="$FS_ROOT/projects/grpo_bt_rm/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
  echo "ERROR: requirements.txt not found at: $REQ_FILE"
  echo "Make sure you have copied / unpacked grpo_bt_rm into $FS_ROOT/projects/grpo_bt_rm"
  exit 1
fi
pip install -r "$REQ_FILE"

# Editable install for ms-swift
MS_SWIFT_DIR="$FS_ROOT/projects/ms-swift"
if [ ! -d "$MS_SWIFT_DIR" ]; then
  echo "ERROR: ms-swift directory not found at: $MS_SWIFT_DIR"
  echo "Make sure you have copied / unpacked ms-swift into $FS_ROOT/projects/ms-swift"
  exit 1
fi
pip install -e "$MS_SWIFT_DIR"

# Quick sanity checks
python -c "import swift; print('swift loaded from:', swift.__file__)"
python -c "import datasets; print('datasets cache:', __import__('os').environ.get('HF_DATASETS_CACHE'))"
echo "OK: environment ready."
