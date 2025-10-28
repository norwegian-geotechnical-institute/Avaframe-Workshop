#!/bin/bash
# qgis-poetry.sh — launch QGIS with Poetry venv dependencies

# -------------------------
# Disable pyenv completely
# -------------------------

unset PYENV_VERSION
unset PYENV_ROOT
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.pyenv' | paste -sd ':' -)

# -------------------------
# Find project root and Poetry venv
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Get Poetry virtual environment path
VENV_PATH="$PROJECT_DIR/.venv"

if [ -z "$VENV_PATH" ]; then
    echo "Error: Could not find Poetry virtualenv for project at $PROJECT_DIR"
    exit 1
fi

# -------------------------
# Find site-packages dynamically
# -------------------------

PYTHON_SITE=$(find "$VENV_PATH/lib/" -maxdepth 1 -type d -name "python3.*" | head -n1)
if [ -z "$PYTHON_SITE" ]; then
    echo "Error: Could not detect Python version in venv $VENV_PATH"
    exit 1
fi

SITE_PACKAGES="$PYTHON_SITE/site-packages"

# -------------------------
# Add venv site-packages to PYTHONPATH
# -------------------------

export PYTHONPATH="$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

# -------------------------
# Create a temporary "python" alias wrapper
# -------------------------

PYTHON_SHIM_DIR="$(mktemp -d)"
echo '#!/bin/bash' > "$PYTHON_SHIM_DIR/python"
echo 'exec /usr/bin/python3 "$@"' >> "$PYTHON_SHIM_DIR/python"
chmod +x "$PYTHON_SHIM_DIR/python"

# Prepend to PATH
export PATH="$PYTHON_SHIM_DIR:$PATH"

# -------------------------
# Launch QGIS
# -------------------------

echo "Launching QGIS with Poetry venv at $VENV_PATH"
echo "Added to PYTHONPATH: $SITE_PACKAGES"
echo "Added python shim at: $PYTHON_SHIM_DIR"

exec /usr/bin/qgis "$@"

