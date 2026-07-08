#!/usr/bin/env bash
set -euo pipefail

cd /home/elf/segment/seaformer-web

if [[ -e /dev/rknpu ]] && command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  sudo chgrp video /dev/rknpu || true
  sudo chmod g+rw /dev/rknpu || true
fi

exec /userdata/elf-env/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
