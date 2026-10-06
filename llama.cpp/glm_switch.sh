#!/bin/bash
# Runs as root (systemd-run). Waits for GLM-5.3-Flash download, then swaps llama-gemma (:8081, GPU 2-3)
# for llama-glm (:8082). If GLM is not healthy within 40 min, rolls back to llama-gemma.
# Log: /home/llm/models/glm_switch.log
LOG=/home/llm/models/glm_switch.log
DIR=/home/llm/models/GLM-5.3-Flash-UD-Q5_K_XL
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "waiting for download"
until grep -q "ALL DONE" /home/llm/models/glm53_dl.log; do sleep 60; done
n=$(ls "$DIR"/*.gguf | wc -l); aria=$(ls "$DIR"/*.aria2 2>/dev/null | wc -l)
log "download done: $n shards, $aria .aria2 leftovers, $(du -sh "$DIR" | cut -f1)"
if [ "$n" -ne 6 ] || [ "$aria" -ne 0 ]; then log "ABORT: download incomplete"; exit 1; fi

log "stopping llama-gemma"
systemctl stop llama-gemma; systemctl disable llama-gemma
log "starting llama-glm"
systemctl start llama-glm
ok=0
for i in $(seq 1 80); do
  sleep 30
  if ! systemctl is-active -q llama-glm && [ "$(systemctl show -p NRestarts --value llama-glm)" -ge 2 ]; then break; fi
  if curl -sf -m 5 http://127.0.0.1:8082/health | grep -q ok; then ok=1; break; fi
done

if [ $ok -eq 1 ]; then
  systemctl enable llama-glm
  log "GLM healthy after ~$((i*30)) s"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >> "$LOG"
  curl -s -m 1800 http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "messages":[{"role":"user","content":"Исправь ошибки в тексте и коротко объясни: «Вчера мы ходили в магазин и купили много продуктоф, но забыли хлеп.»"}],
    "max_tokens":1500}' > /home/llm/models/glm_smoke.json
  python3 -c "import json;d=json.load(open('/home/llm/models/glm_smoke.json'));print('timings',d.get('timings'));print('answer',d['choices'][0]['message'].get('content'))" >> "$LOG" 2>&1
  log "SWITCH OK"
else
  log "GLM NOT healthy -> rollback"
  journalctl -u llama-glm -n 60 --no-pager >> "$LOG"
  systemctl stop llama-glm; systemctl disable llama-glm
  systemctl enable llama-gemma; systemctl start llama-gemma
  log "ROLLBACK DONE"
fi
