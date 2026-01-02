#!/usr/bin/env bash
set -e

# cleanup.sh: Cleaning utility for bdd-mini.
# Updated to protect the persistent image cache.

# ------------- Variables -------------
OUTPUT_DIR="output"
DATA_DIR="data"
CACHE_DIR="$DATA_DIR/image_cache"
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

# --- Step 1: Clean Generated Outputs ---
if [[ -d "$OUTPUT_DIR" ]]; then
  echo "👉 Found generated dataset in '$OUTPUT_DIR'"
  if ask_yes_no "   🗑️  Delete generated train/val/test folders?"; then
    rm -rf "$OUTPUT_DIR"
    echo "      ✅ Deleted '$OUTPUT_DIR'."
  else
    echo "      ℹ️  Skipped."
  fi
else
  echo "ℹ️  No output directory found."
fi

echo ""

# --- Step 2: Clean Image Cache (The Safe Guard) ---
if [[ -d "$CACHE_DIR" ]]; then
  # Count files to show user what they are deleting
  FILE_COUNT=$(find "$CACHE_DIR" -type f | wc -l)
  echo "👉 Found Persistent Image Cache in '$CACHE_DIR' ($FILE_COUNT images)"
  echo "   ⚠️  Deleting this will force a re-download next time."
  
  if ask_yes_no "   🗑️  Delete image cache?"; then
    rm -rf "$CACHE_DIR"
    echo "      ✅ Deleted image cache."
  else
    echo "      ℹ️  Kept image cache (Safe)."
  fi
else
  echo "ℹ️  No image cache found."
fi

echo ""

# --- Step 3: Clean Raw Downloads (Labels) ---
# Check for zips or other files in data that aren't the cache
if [[ -d "$DATA_DIR" ]]; then
  echo "👉 Checking for raw label zips..."
  if ask_yes_no "   🗑️  Delete raw downloaded zips/labels?"; then
    # Delete everything in data EXCEPT image_cache
    # using find to avoid complex logic, essentially delete files in data root
    find "$DATA_DIR" -maxdepth 1 -type f -delete
    # Also delete temp folders if they exist
    rm -rf "$DATA_DIR/bdd100k"
    echo "      ✅ Deleted raw zips/labels."
  else
    echo "      ℹ️  Skipped."
  fi
fi

echo ""

# --- Step 4: Clean Virtual Environment ---
if [[ -d "$VENV_DIR" ]]; then
  if ask_yes_no "🗑️  Delete virtual environment ('$VENV_DIR')?"; then
    rm -rf "$VENV_DIR"
    echo "      ✅ Deleted '$VENV_DIR'."
  else
    echo "      ℹ️  Skipped."
  fi
fi

# --- Step 5: Clean Pycache ---
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo ""
echo "✨ Cleanup complete."