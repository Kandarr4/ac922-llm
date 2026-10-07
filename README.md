# Large MoE LLMs on IBM Power System AC922 (POWER9 + 4× V100)

Notes and scripts for running modern LLMs on an IBM AC922 (8335-GTH): 2× POWER9 (ppc64le, 160 threads),
512 GB DDR4, 4× Tesla V100 16 GB (SXM2, NVLink 2.0 to the CPU), Ubuntu 22.04 ppc64el.

This is a platform vendors no longer support: no apt repo for NVIDIA on ppc64el, the last driver is
550.54.15 / CUDA 12.4, no PyTorch+CUDA wheels, and Volta lacks bf16/FP8. Everything here was made to work by hand.

> **Status:** GLM-5.3-Flash runs at ~6 tok/s generation, 11–16 tok/s prompt processing (see below).

## What's inside

| Path | What |
| --- | --- |
| [`driver/`](driver/) | Installing NVIDIA 550.54.14 / CUDA 12.4 on Ubuntu 22.04 ppc64el, including the fix for the `rcu_read_unlock_strict` GPL-only symbol error |
| [`llama.cpp/`](llama.cpp/) | Building llama.cpp (unsloth `glm5next` branch) for sm_70 on ppc64le, downloading GLM-5.3-Flash, switching a service over with rollback |
| [`systemd/`](systemd/) | `llama-server` units: a 26B MoE fully on a GPU pair, and a 320B MoE with experts in RAM |

## 1. NVIDIA driver on Ubuntu 22.04 ppc64el

The CUDA 12.4 runfile installs, but the kernel module fails at modpost:

```
ERROR: modpost: GPL-incompatible module nvidia.ko uses GPL-only symbol 'rcu_read_unlock_strict'
```

Ubuntu's ppc64el 5.15 kernel has no `PREEMPT_DYNAMIC`, so `rcu_read_unlock()` inlines an unconditional call to
`rcu_read_unlock_strict()`, which is exported `EXPORT_SYMBOL_GPL`. With `CONFIG_RCU_STRICT_GRACE_PERIOD=n`
the function is an empty stub (16 bytes in `/proc/kallsyms`), so giving `nvidia.ko` and `nvidia-uvm.ko`
their own empty definition is behaviour-neutral. See [`driver/apply_rcu_stub.sh`](driver/apply_rcu_stub.sh).
Patch the extracted driver tree before `nvidia-installer --dkms`, so DKMS rebuilds keep the fix.

Dead ends checked: NVIDIA apt repo (none for ppc64el), Ubuntu archive drivers (none), newer drivers (550.54.15
is the last ppc64le build), open kernel modules (need GSP, Turing+).

## 2. Big MoE models: experts in RAM, the rest on GPU

64 GB of VRAM rules out GPU-only engines (vLLM) for anything past ~70 GB. llama.cpp keeps MoE experts in system
RAM and runs attention / shared layers / KV cache on the GPUs, so token generation is bound by RAM bandwidth.

Measured RAM read bandwidth (OpenMP sum, 40 threads):

| | GB/s |
| --- | --- |
| socket 0, local memory | 127 |
| socket 1, local memory | 141 |
| both sockets, interleaved (80 threads) | 118 |

Rough decode estimate: `tok/s ≈ effective_bw / bytes_of_active_experts_per_token`.

### GLM-5.3-Flash (320B total, 18B active), UD-Q5_K_XL, ~240 GB

- Upstream llama.cpp (v0.6.0, arch `glm5-next`) **builds for sm_70 on ppc64le without patches**
  ([`build_llama_master.sh`](llama.cpp/build_llama_master.sh)). The wmma lightning-indexer kernel is gated
  on Turing+, Volta takes the generic path. Use upstream, not the older unsloth `glm5next` branch: current
  unsloth GGUFs say `glm5-next`, the branch expects `glm5next` (`unknown model architecture`).
- 240 GB does not fit one NUMA node (256 GB each), so weights are interleaved over both sockets:
  `numactl --interleave=all --` + `--load-mode none` (no mmap, anonymous allocation, so the policy applies;
  there is no `--no-mmap` in 0.6).
- `-ngl 99 --override-tensor exps=CPU`: whole graph on the GPUs, only MoE experts in RAM. Plain `--fit`
  put entire layers on the CPU.
- **`--no-op-offload` is essential.** With op offload on, every prompt ubatch copies all ~220 GB of experts to
  the GPU from pageable memory on a single thread: prompt processing drops to ~1.3 tok/s.
- MTP (`--spec-type draft-mtp`) is not implemented for glm5-next yet ("NextN graph not implemented yet").

Unit: [`systemd/llama-glm.service`](systemd/llama-glm.service).

Measured 2026-10-07, UD-Q5_K_XL, experts in RAM (both sockets interleaved), rest on 2× V100, 80 threads,
thinking off, prompts of 90–2188 tokens:

| Config | Prompt processing, tok/s | Generation, tok/s |
| --- | --- | --- |
| `--fit` only | 1.2–1.5 | 5.4–5.6 |
| `-ngl 99 -ot exps=CPU` | 1.3–4.1 | 5.7–6.2 |
| **`-ngl 99 -ot exps=CPU --no-op-offload`** | **10.6–15.6** | **6.0–6.2** |

Load time ~3 min (from page cache), VRAM used ~8.5 + 5.9 GB with 64K context.

## Hardware notes

- GPU 0–1 and GPU 2–3 are NVLink pairs, each pair attached to its own socket; run one model per pair and bind
  it with `numactl --cpunodebind/--membind` to the matching socket.
- V100 has no bf16: vLLM runs fp16/fp32 only.

## License

Scripts: MIT. Model weights and llama.cpp are under their own licenses.
