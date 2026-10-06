#!/bin/bash
# Download GLM-5.3-Flash UD-Q5_K_XL (6 shards, ~240 GB) from unsloth; resumable (aria2c -c).
set -u
Q=${Q:-UD-Q5_K_XL}
DIR="$HOME/models/GLM-5.3-Flash-$Q"
mkdir -p "$DIR"; cd "$DIR"
for i in 1 2 3 4 5 6; do
  f="GLM-5.3-Flash-$Q-0000$i-of-00006.gguf"
  echo "[$(date '+%F %T')] start $f"
  until aria2c -c -x16 -s16 -k 10M --file-allocation=none --console-log-level=warn \
      --summary-interval=60 -o "$f" \
      "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/$Q/$f"; do
    echo "[$(date '+%F %T')] retry $f"; sleep 30
  done
  echo "[$(date '+%F %T')] done $f"
done
echo "[$(date '+%F %T')] ALL DONE"
