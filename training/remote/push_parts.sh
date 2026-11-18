#!/bin/bash
# Resumable copy of a large file to a Windows box over a flaky tunnel: split into parts, copy each part with
# retries (skipping parts already there at full size), reassemble on the box and verify SHA-256.
#   training/remote/push_parts.sh <ssh-host> <local file> <remote dir, e.g. D:/training/data> [part size, default 500m]
set -uo pipefail
HOST="$1"; FILE="$2"; RDIR="$3"; SIZE="${4:-500m}"
NAME="$(basename "$FILE")"; WORK="${FILE}.parts"
ps() { ssh -n -o ConnectTimeout=90 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 "$HOST" "powershell -NoProfile -Command \"$1\""; }
[ -f "$FILE.sha256" ] || shasum -a 256 "$FILE" > "$FILE.sha256"
WANT=$(cut -d' ' -f1 "$FILE.sha256" | tr a-f A-F)
if [ ! -d "$WORK" ]; then mkdir -p "$WORK" && (cd "$WORK" && split -b "$SIZE" -a 3 "$FILE" part_); fi
until ps "New-Item -ItemType Directory -Force -Path '$RDIR/$NAME.parts' | Out-Null"; do echo "waiting for $HOST"; sleep 60; done
for p in "$WORK"/part_*; do
  b=$(basename "$p"); want=$(stat -f %z "$p")
  for try in 1 2 3 4 5 6 7 8; do
    have=$(ps "if (Test-Path '$RDIR/$NAME.parts/$b') { (Get-Item '$RDIR/$NAME.parts/$b').Length } else { 0 }" 2>/dev/null | tr -d '\r')
    [ "$have" = "$want" ] && { echo "$b ok"; break; }
    echo "$b: copying (try $try)"
    scp -q -o ConnectTimeout=90 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 "$p" "$HOST:$RDIR/$NAME.parts/$b" || sleep $((30 * try))
  done
done
echo "assembling on $HOST"
cat > "$WORK/assemble.ps1" <<PS
\$ErrorActionPreference = 'Stop'
\$out = [IO.File]::Create('$RDIR/$NAME')
foreach (\$f in Get-ChildItem '$RDIR/$NAME.parts' -Filter 'part_*' | Sort-Object Name) { \$in = [IO.File]::OpenRead(\$f.FullName); \$in.CopyTo(\$out); \$in.Close() }
\$out.Close()
\$h = (Get-FileHash '$RDIR/$NAME' -Algorithm SHA256).Hash
if (\$h -eq '$WANT') { Remove-Item '$RDIR/$NAME.parts' -Recurse -Force; "SHA256 OK \$h" } else { "SHA256 MISMATCH \$h"; exit 1 }
PS
scp -q "$WORK/assemble.ps1" "$HOST:$RDIR/assemble_$NAME.ps1"
ssh -n -o ServerAliveInterval=15 "$HOST" "powershell -NoProfile -ExecutionPolicy Bypass -File $RDIR/assemble_$NAME.ps1; Remove-Item $RDIR/assemble_$NAME.ps1"
