#!/usr/bin/env bash
set -e

# setup.sh: Sets up the bdd-mini project environment.

# ------------- Variables -------------
VENV_DIR="venv"
CONFIG_FILE="config.toml"

# ------------- Helper functions -------------
have() { command -v "$1" >/dev/null 2>&1; }

# --- Step 1: Check Prerequisites ---
echo "🔍 Checking system requirements..."
if ! have "python3"; then
  echo "❌ Error: python3 is not installed."
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Found Python $PYTHON_VERSION"

# --- Step 2: Create Virtual Environment ---
# Check if venv dir exists AND if activate script exists
if [[ -d "$VENV_DIR" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "ℹ️  Virtual environment '$VENV_DIR' already exists and looks valid."
else
  if [[ -d "$VENV_DIR" ]]; then
    echo "⚠️  Found broken or incomplete '$VENV_DIR'. Recreating..."
    rm -rf "$VENV_DIR"
  fi
  
  echo "📦 Creating virtual environment in '$VENV_DIR'..."
  
  # Try to create venv. If it fails, print the ubuntu fix hint.
  if ! python3 -m venv "$VENV_DIR"; then
    echo ""
    echo "❌ Failed to create virtual environment."
    echo "👉 On Ubuntu/EC2, you likely need to run:"
    echo "   sudo apt install python3-venv"
    echo ""
    # Clean up the broken folder so we don't loop next time
    rm -rf "$VENV_DIR"
    exit 1
  fi
  echo "✅ Created venv."
fi

# --- Step 3: Install Dependencies ---
echo "⬇️  Installing dependencies..."

# Activate venv for this script execution
source "$VENV_DIR/bin/activate"

# Upgrade pip and install libraries
pip install --upgrade pip > /dev/null
pip install requests tqdm remotezip tomli opencv-python

echo "✅ Dependencies installed."

# --- Step 4: Verify Config Exists ---
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "⚠️  Warning: '$CONFIG_FILE' not found."
    echo "    Please create it before running builder.py."
else
    echo "✅ Found '$CONFIG_FILE'."
fi

# --- Step 5: Final Instructions ---
echo ""
echo "🎉 Setup complete!"
echo "To build your dataset, run:"
echo "  source venv/bin/activate"
echo "  python3 builder.py"