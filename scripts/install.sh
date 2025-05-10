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
  echo "  --uninstall         Uninstall the package and remove the scaffolding"
  echo "  --help              Show this help message and exit"
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

function install_dependencies() {
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

function uninstall_scaffolding_in_path() {
  local target_dir="$1"
  local rel_path="${target_dir#"$klipper_dir"/}/$MODULE_NAME"
  local full_path="$target_dir/$MODULE_NAME"

  if [ -f "$full_path" ]; then
    rm "$full_path"
    echo "Removed file '$full_path'."

    local exclude_file="$klipper_dir/.git/info/exclude"
    if [ -f "$exclude_file" ]; then
      sed -i "\|^$rel_path\$|d" "$exclude_file" && echo "Removed '$rel_path' from git exclude."
    fi
  fi
}

function main() {
  klipper_dir="$DEFAULT_KLIPPER_DIR"
  klippy_env="$DEFAULT_KLIPPY_ENV"

  parse_args "$@"

  check_directory_exists "$klipper_dir"
  check_virtualenv_exists

  if [ "$uninstall" = true ]; then
    uninstall_dependencies
    uninstall_scaffolding_in_path "$klipper_dir/klippy/extras"
    uninstall_scaffolding_in_path "$klipper_dir/klippy/plugins"
  else
    install_dependencies
    create_scaffolding
  fi
}

main "$@"
