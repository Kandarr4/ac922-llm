# Check GLM-5.3-Flash thinking controls on llama-server :8082 (run on the server).
import json, time, urllib.request
URL = "http://127.0.0.1:8082/v1/chat/completions"
Q = "Сколько будет 17 * 23? Ответь одним числом."
def run(name, extra):
    body = {"messages": [{"role": "user", "content": Q}], "max_tokens": 4000, **extra}
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    m = d["choices"][0]["message"]; r = m.get("reasoning_content") or ""
    print(f"{name:34s} reasoning {len(r):5d} chars | tokens {d['usage']['completion_tokens']:4d} | {time.time()-t:5.0f}s | answer {m.get('content','').strip()[:60]!r}", flush=True)
run("default (effort max)", {})
run("enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}})
run("kwargs reasoning_effort=low", {"chat_template_kwargs": {"reasoning_effort": "low"}})
run("top-level reasoning_effort=low", {"reasoning_effort": "low"})
run("kwargs reasoning_effort=high", {"chat_template_kwargs": {"reasoning_effort": "high"}})
