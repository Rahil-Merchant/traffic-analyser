#!/bin/bash
# Copy the training code (and optionally the Roboflow export zip) from this Mac to a Windows box over ssh.
#   training/remote/push_to_box.sh <ssh-host> <windows root, e.g. D:/training> [path/to/roboflow.zip]
# Code goes to <root>/code/training, the zip to <root>/data/roboflow.zip (skipped if already there with the
# same size). Nothing in the production repo on the box is touched.
set -euo pipefail
HOST="$1"; ROOT="$2"; ZIP="${3:-}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ps() { ssh "$HOST" 'powershell -NoProfile -Command -' <<< "$1"; }
ps "New-Item -ItemType Directory -Force -Path '$ROOT/code','$ROOT/data' | Out-Null"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
COPYFILE_DISABLE=1 tar --no-mac-metadata -C "$HERE/.." --exclude "._*" --exclude '__pycache__' --exclude '*.pyc' -czf "$STAGE/training.tgz" training
scp -q "$STAGE/training.tgz" "$HOST:$ROOT/code/training.tgz"
ps "Set-Location '$ROOT/code'; if (Test-Path training) { Remove-Item training -Recurse -Force }; tar -xzf training.tgz; Remove-Item training.tgz; (Get-ChildItem training -Recurse -File | Measure-Object).Count"
if [ -n "$ZIP" ]; then
  local_size=$(stat -f %z "$ZIP")
  remote_size=$(ps "if (Test-Path '$ROOT/data/roboflow.zip') { (Get-Item '$ROOT/data/roboflow.zip').Length } else { 0 }" | tr -d '\r')
  if [ "$remote_size" = "$local_size" ]; then echo "zip already on $HOST"; else
    echo "copying $(du -h "$ZIP" | cut -f1) to $HOST:$ROOT/data/roboflow.zip"
    scp -q "$ZIP" "$HOST:$ROOT/data/roboflow.zip"
  fi
fi
echo "pushed to $HOST:$ROOT"
