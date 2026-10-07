# Make the Step-3.7-Flash chat template honour chat_template_kwargs.enable_thinking=false:
# the stock generation prompt always opens <think>; when thinking is off we prefill an empty block,
# the same shape the template renders for past assistant turns without reasoning.
# (GLM-5.3-Flash was patched the same way by hand: tools/glm53_template.jinja.)
# Usage (on the server): patch_think_template.py <port> <orig_out> <patched_out>
import json, sys, urllib.request

port, orig_out, out = sys.argv[1], sys.argv[2], sys.argv[3]
t = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/props"))["chat_template"]
open(orig_out, "w").write(t)

old = r"{{- '<|im_start|>assistant\n<think>\n' }}"
closed = r"{{- '<|im_start|>assistant\n<think>\n\n</think>\n' }}"
assert t.count(old) == 1, f"marker found {t.count(old)} times"
new = "{%- if enable_thinking is defined and enable_thinking is false %}" + closed + "{%- else %}" + old + "{%- endif %}"
open(out, "w").write(t.replace(old, new))
print("patched:", out)
