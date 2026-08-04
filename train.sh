#!/usr/bin/env bash
# Запуск/продолжение обучения (изнутри WSL Ubuntu или из Windows:
#   wsl -d Ubuntu -- bash train.sh)
# Повторный запуск сам продолжает с последнего чекпоинта.
set -e
cd "$(dirname "$0")"
VENV="${VENV:-$HOME/tgstyle/.venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
exec python train_v2.py "$@" 2>&1 | tee -a train.log
