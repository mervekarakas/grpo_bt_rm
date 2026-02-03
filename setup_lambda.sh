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
# ---------------- ms-swift (fork, pinned) ----------------
MS_SWIFT_DIR="$FS_ROOT/projects/ms-swift"
MS_SWIFT_REPO="git@github.com:mervekarakas/ms-swift.git"
MS_SWIFT_COMMIT="ffb9c73a4178a7956cbceee837c2ef11114d2387"

if [ ! -d "$MS_SWIFT_DIR/.git" ]; then
  git clone "$MS_SWIFT_REPO" "$MS_SWIFT_DIR"
fi

cd "$MS_SWIFT_DIR"
git fetch origin
git checkout "$MS_SWIFT_COMMIT"

pip install -e "$MS_SWIFT_DIR"


# Quick sanity checks
python -c "import swift; print('swift loaded from:', swift.__file__)"
python -c "import datasets; print('datasets cache:', __import__('os').environ.get('HF_DATASETS_CACHE'))"
echo "OK: environment ready."
