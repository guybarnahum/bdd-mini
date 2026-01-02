#!/usr/bin/env bash
set -e

# setup.sh: Sets up the bdd-mini project environment.
# Creates a venv and installs dependencies.

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
if [[ -d "$VENV_DIR" ]]; then
  echo "ℹ️  Virtual environment '$VENV_DIR' already exists."
else
  echo "📦 Creating virtual environment in '$VENV_DIR'..."
  python3 -m venv "$VENV_DIR"
  echo "✅ Created venv."
fi

# --- Step 3: Install Dependencies ---
echo "⬇️  Installing dependencies..."

# Activate venv for this script execution
source "$VENV_DIR/bin/activate"

# Upgrade pip and install libraries
pip install --upgrade pip > /dev/null
pip install requests tqdm remotezip tomli

echo "✅ Dependencies installed: requests, tqdm, remotezip, tomli"

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