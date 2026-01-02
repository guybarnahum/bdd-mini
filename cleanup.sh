#!/usr/bin/env bash
set -e

# cleanup.sh: Cleaning utility for bdd-mini.
# Helps remove generated datasets, cached downloads, and virtual environments.

# ------------- Variables -------------
OUTPUT_DIR="output"
DATA_DIR="data"
VENV_DIR="venv"

# ------------- Helper Functions -------------
ask_yes_no() {
  local prompt="$1"
  read -p "$prompt [y/N] " -n 1 -r; echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# --- Header ---
echo "🧹 bdd-mini Cleanup Utility"
echo "============================="

# --- Step 1: Clean Generated Dataset ---
if [[ -d "$OUTPUT_DIR" ]]; then
  if ask_yes_no "🗑️  Delete generated dataset folder ('$OUTPUT_DIR')?"; then
    rm -rf "$OUTPUT_DIR"
    echo "   ✅ Deleted '$OUTPUT_DIR'."
  else
    echo "   ℹ️  Skipped."
  fi
else
  echo "ℹ️  No output directory found."
fi

# --- Step 2: Clean Downloaded Data ---
if [[ -d "$DATA_DIR" ]]; then
  echo ""
  echo "⚠️  The '$DATA_DIR' folder contains downloaded labels and zip caches."
  if ask_yes_no "🗑️  Delete downloaded data ('$DATA_DIR')?"; then
    rm -rf "$DATA_DIR"
    echo "   ✅ Deleted '$DATA_DIR'."
  else
    echo "   ℹ️  Skipped."
  fi
else
  echo "ℹ️  No data directory found."
fi

# --- Step 3: Clean Virtual Environment ---
if [[ -d "$VENV_DIR" ]]; then
  echo ""
  if ask_yes_no "🗑️  Delete virtual environment ('$VENV_DIR')?"; then
    rm -rf "$VENV_DIR"
    echo "   ✅ Deleted '$VENV_DIR'."
  else
    echo "   ℹ️  Skipped."
  fi
fi

# --- Step 4: Clean Pycache (Always safe) ---
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo ""
echo "✨ Cleanup complete (removed .pyc and __pycache__)."
