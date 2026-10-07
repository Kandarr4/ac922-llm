#!/bin/bash
# Download one quant folder of a GGUF repo from Hugging Face; resumable (aria2c -c), retries forever.
# Usage: dl_hf_gguf.sh <repo> <quant-folder> <dest-dir>
#   dl_hf_gguf.sh unsloth/Step-3.7-Flash-GGUF UD-Q4_K_XL ~/models/Step-3.7-Flash-UD-Q4_K_XL
set -u
REPO=$1; Q=$2; DIR=$3
mkdir -p "$DIR"; cd "$DIR"
FILES=$(curl -s "https://huggingface.co/api/models/$REPO/tree/main/$Q" | python3 -c "import json,sys; print(' '.join(f['path'] for f in json.load(sys.stdin) if f['path'].endswith('.gguf')))")
[ -n "$FILES" ] || { echo "no files for $REPO/$Q"; exit 1; }
for p in $FILES; do
  f=$(basename "$p")
  echo "[$(date '+%F %T')] start $f"
  until aria2c -c -x16 -s16 -k 10M --file-allocation=none --console-log-level=warn \
      --summary-interval=300 -o "$f" "https://huggingface.co/$REPO/resolve/main/$p"; do
    echo "[$(date '+%F %T')] retry $f"; sleep 30
  done
  echo "[$(date '+%F %T')] done $f"
done
echo "[$(date '+%F %T')] ALL DONE"
