# Quick quality / thinking-control check for a llama-server model (run on the server). Usage: step_quick_test.py [port]
import json, sys, time, urllib.request
URL = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8083}/v1/chat/completions"
def ask(q, **extra):
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": 3000, **extra}
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    m = d["choices"][0]["message"]; tm = d.get("timings", {})
    print(f"--- {q[:60]!r} {extra or ''}\n    reasoning {len(m.get('reasoning_content') or '')} chars | {d['usage']['completion_tokens']} tok | "
          f"{tm.get('predicted_per_second', 0):.1f} t/s | {time.time()-t:.0f}s\n    {(m.get('content') or '').strip()[:500]}", flush=True)
off = {"chat_template_kwargs": {"enable_thinking": False}}
ask("Кто ты? Ответь в двух предложениях.", **off)
ask("Исправь ошибки и коротко объясни: «Вчера мы ходили в магазин и купили много продуктоф, но забыли хлеп.»", **off)
ask("Мына сөйлемдегі қателерді түзет: «Кеше біз дүкенге бардық, көп азық-түлік сатып алдық, бірақ нанды ұмытып кетик.»", **off)
ask("Сколько будет 17 * 23? Ответь одним числом.")
ask("Сколько будет 17 * 23? Ответь одним числом.", reasoning_effort="low", chat_template_kwargs={"reasoning_effort": "low"})
