#!/bin/bash

set -euo pipefail

MODULE_NAME="cartographer.py"
PACKAGE_NAME="cartographer3d-plugin"
SCAFFOLDING="from cartographer.extra import *"
DEFAULT_KLIPPER_DIR="$HOME/klipper"
DEFAULT_KLIPPY_ENV="$HOME/klippy-env"

function display_help() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -k, --klipper       Set the Klipper directory (default: $DEFAULT_KLIPPER_DIR)"
  echo "  -e, --klippy-env    Set the Klippy virtual environment directory (default: $DEFAULT_KLIPPY_ENV)"
  echo "  --uninstall         Uninstall the package and remove all scaffolding files"
  echo "  --help              Show this help message and exit"
  echo ""
  echo "The script automatically removes all legacy and scaffolding files before installation."
  exit 0
}

function parse_args() {
  uninstall=false
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
    -k | --klipper)
      klipper_dir="$2"
      shift 2
      ;;
    -e | --klippy-env)
      klippy_env="$2"
      shift 2
      ;;
    --uninstall)
      uninstall=true
      shift
      ;;
    --help)
      display_help
      ;;
    *)
      echo "Unknown option: $1"
      display_help
      ;;
    esac
  done
}

function check_directory_exists() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo "Error: Directory '$dir' does not exist."
    exit 1
  fi
}

function check_virtualenv_exists() {
  if [ ! -d "$klippy_env" ]; then
    echo "Error: Virtual environment directory '$klippy_env' does not exist."
    exit 1
  fi
}

function ensure_numpy() {
  echo "Checking numpy installation in '$klippy_env'..."

  "$klippy_env/bin/python" -c "
import sys
try:
    import numpy
    version = tuple(map(int, numpy.__version__.split('.')[:2]))
    if version >= (1, 16):
        print(f'✓ numpy {numpy.__version__} already installed and >= 1.16')
        sys.exit(0)
    else:
        print(f'numpy {numpy.__version__} found but < 1.16, upgrading...')
        sys.exit(1)
except ImportError:
    print('numpy not found, installing numpy~=1.16...')
    sys.exit(1)
" || "$klippy_env/bin/pip" install "numpy~=1.16"
}

function install_dependencies() {
  ensure_numpy
  echo "Installing or upgrading '$PACKAGE_NAME' into '$klippy_env'..."
  "$klippy_env/bin/pip" install --upgrade "$PACKAGE_NAME"
  echo "'$PACKAGE_NAME' has been successfully installed or upgraded into '$klippy_env'."
}

function uninstall_dependencies() {
  echo "Uninstalling '$PACKAGE_NAME' from '$klippy_env'..."
  "$klippy_env/bin/pip" uninstall -y "$PACKAGE_NAME"
  echo "'$PACKAGE_NAME' has been uninstalled from '$klippy_env'."
}

function create_scaffolding() {
  if [ -d "$klipper_dir/klippy/plugins" ]; then
    scaffolding_dir="$klipper_dir/klippy/plugins"
    use_git_exclude=false
  else
    scaffolding_dir="$klipper_dir/klippy/extras"
    use_git_exclude=true
  fi

  scaffolding_path="$scaffolding_dir/$MODULE_NAME"
  scaffolding_rel_path="${scaffolding_dir#"$klipper_dir"/}/$MODULE_NAME"

  check_directory_exists "$scaffolding_dir"

  if [ -L "$scaffolding_path" ]; then
    local original_target
    original_target=$(readlink "$scaffolding_path")
    echo "Warning: '$scaffolding_path' is a symlink and will be removed."
    echo "If you need to recover it, you can recreate the symlink with:"
    echo "  ln -s \"$original_target\" \"$scaffolding_path\""
    rm "$scaffolding_path"
  fi

  echo "$SCAFFOLDING" >"$scaffolding_path"
  echo "File '$MODULE_NAME' has been created at '$scaffolding_path'."

  if [ "$use_git_exclude" = true ]; then
    local exclude_file="$klipper_dir/.git/info/exclude"
    if [ -d "$klipper_dir/.git" ] && ! grep -qF "$scaffolding_rel_path" "$exclude_file" >/dev/null 2>&1; then
      echo "$scaffolding_rel_path" >>"$exclude_file"
      echo "Added '$scaffolding_rel_path' to git exclude."
    fi
  fi
}

function remove_plugin_files() {
  echo "Cleaning up legacy and scaffolding files..."

  local files=("idm.py" "scanner.py" "cartographer.py")
  local paths=(
    "$klipper_dir/klippy/extras"
    "$klipper_dir/klippy/plugins"
  )

  for dir in "${paths[@]}"; do
    if [ ! -d "$dir" ]; then
      continue # Skip if directory doesn't exist
    fi

    for file in "${files[@]}"; do
      local full_path="$dir/$file"
      local rel_path="${dir#"$klipper_dir"/}/$file"

      if [ -f "$full_path" ] || [ -L "$full_path" ]; then
        if [ -L "$full_path" ]; then
          local original_target
          original_target=$(readlink "$full_path" 2>/dev/null || echo "unknown")
          echo "Removing symlink '$full_path' (was pointing to: $original_target)"
        else
          echo "Removing file '$full_path'"
        fi
        rm "$full_path"

        # Clean up git exclude entries
        local exclude_file="$klipper_dir/.git/info/exclude"
        if [ -f "$exclude_file" ]; then
          sed -i "\|^$rel_path\$|d" "$exclude_file" 2>/dev/null || true
        fi
      fi
    done
  done
  echo "File cleanup completed."
}

function main() {
  klipper_dir="$DEFAULT_KLIPPER_DIR"
  klippy_env="$DEFAULT_KLIPPY_ENV"

  parse_args "$@"

  check_directory_exists "$klipper_dir"
  check_virtualenv_exists

  remove_plugin_files

  if [ "$uninstall" = true ]; then
    uninstall_dependencies
  else
    install_dependencies
    create_scaffolding
  fi
}

main "$@"
