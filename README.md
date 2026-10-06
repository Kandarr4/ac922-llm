# Large MoE LLMs on IBM Power System AC922 (POWER9 + 4× V100)

Notes and scripts for running modern LLMs on an IBM AC922 (8335-GTH): 2× POWER9 (ppc64le, 160 threads),
512 GB DDR4, 4× Tesla V100 16 GB (SXM2, NVLink 2.0 to the CPU), Ubuntu 22.04 ppc64el.

This is a platform vendors no longer support: no apt repo for NVIDIA on ppc64el, the last driver is
550.54.15 / CUDA 12.4, no PyTorch+CUDA wheels, and Volta lacks bf16/FP8. Everything here was made to work by hand.

> **Status:** work in progress. GLM-5.3-Flash benchmarks below are pending.

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

- `glm5_next` is not in upstream llama.cpp yet; the unsloth branch `glm5next/upstream` **builds for sm_70 on
  ppc64le without patches** ([`build_llama_glm5next.sh`](llama.cpp/build_llama_glm5next.sh)). Its wmma
  lightning-indexer kernel is gated on Turing+, Volta takes the generic path.
- 240 GB does not fit one NUMA node (256 GB each), so weights are interleaved over both sockets:
  `numactl --interleave=all` + `--no-mmap` (anonymous allocation, so the policy applies).
- No `-ngl` / `-ot`: `--fit` (on by default) puts what fits on the GPUs and spills MoE experts to RAM.
- MTP speculative decoding: `--spec-type draft-mtp --spec-draft-n-max 2`.

Unit: [`systemd/llama-glm.service`](systemd/llama-glm.service).

| Config | Prompt processing, tok/s | Generation, tok/s |
| --- | --- | --- |
| Q5_K_XL, 2× V100 + RAM, no MTP | _pending_ | _pending_ |
| Q5_K_XL, 2× V100 + RAM, MTP n=2 | _pending_ | _pending_ |

## Hardware notes

- GPU 0–1 and GPU 2–3 are NVLink pairs, each pair attached to its own socket; run one model per pair and bind
  it with `numactl --cpunodebind/--membind` to the matching socket.
- V100 has no bf16: vLLM runs fp16/fp32 only.

## License

Scripts: MIT. Model weights and llama.cpp are under their own licenses.
