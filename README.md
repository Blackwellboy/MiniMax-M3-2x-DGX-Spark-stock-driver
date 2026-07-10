# MiniMax-M3 on 2× DGX Spark — stock driver (CUDA 13.0)

**Full 428B MiniMax-M3 (no pruning) serving at 34–41 tok/s on two NVIDIA DGX Sparks (GB10) — on the STOCK DGX Spark OS driver.** W4A16 GPTQ weights + KVarN 4-bit-K/2-bit-V KV cache + EAGLE-3 speculative decoding, TP=2, all-Triton attention. No driver swap, no off-channel packages, no kernel gymnastics: this is the deployment that works on **driver 580.159.03 / CUDA 13.0** exactly as DGX Spark OS 7.5.0 ships it.

This repo is a **stock-driver companion** to [tonyd2wild/MiniMax-M3-2x-DGX-Spark-36-tok-s](https://github.com/tonyd2wild/MiniMax-M3-2x-DGX-Spark-36-tok-s), which reproduces [a3refaat/spark-vllm-docker](https://github.com/a3refaat/spark-vllm-docker) branch `minimax-m3-4bit-w4a16`. Their hero lane (b12x nvfp4 attention, ~36.6 tok/s) **does not run on the stock driver** — see [the finding](#the-stock-driver-finding) below. The KVarN lanes here do.

## TL;DR quick start

```bash
# 1. Pull the prebuilt image (skips the 1–2 h source build)
docker pull ghcr.io/blackwellboy/vllm-node-minimax-m3-b12x:cuda13.0-f942437
docker tag  ghcr.io/blackwellboy/vllm-node-minimax-m3-b12x:cuda13.0-f942437 vllm-node-minimax-m3-b12x:latest
# (run on BOTH nodes, or docker save | ssh docker load across your pair's fabric)

# 2. Clone the deployment tooling at the pinned commit + apply the two patches from this repo
git clone -b minimax-m3-4bit-w4a16 https://github.com/a3refaat/spark-vllm-docker.git
cd spark-vllm-docker   # reproduced at commit f942437
git apply ../patches/dockerfile-unreachable-ref-fetch.patch
git apply ../patches/mod-prometheus-graceful-skip.patch
cp ../recipes/minimax-m3-w4a16-gptq-kvarn-eagle3.yaml recipes/

# 3. Stage weights on BOTH nodes (HF-cache layout; containers run HF_HUB_OFFLINE=1)
#    ~211 GB target + 1.6 GB drafter — download once, copy across your pair's fabric
#    Sebesky/MiniMax-M3-W4A16-GPTQ        (snapshot 6696ce6…)
#    Sebesky/MiniMax-M3-EAGLE3-RTN-INT4   (snapshot 86ffe4b…)

# 4. Launch (see Launch prerequisites first!)
VLLM_SPARK_EXTRA_DOCKER_ARGS="--log-opt max-size=100m --log-opt max-file=3" \
./run-recipe.sh --no-ray -d recipes/minimax-m3-w4a16-gptq-kvarn-eagle3.yaml \
  -- --load-format safetensors --max-num-batched-tokens 2048
```

## The stock-driver finding

The b12x **attention** stack (the upstream hero lane: MSA + lightning-indexer + dense attention, nvfp4 KV) is built on **CuTe-DSL kernels that JIT-compile at first inference** — and the indexer's **TMA** op fails to build when the host driver is CUDA 13.0 while the image toolkit is 13.2:

```
b12x/attention/indexer/contiguous_kernel.py → cpasync.tma_partition
  → ValueError: Operation creation failed   (EngineDeadError; first token, every time)
```

The model loads, KV allocates, `/health` returns 200 — then the first real request kills the engine. There is **no sanctioned CUDA-13.2 driver for GB10** as of July 2026 (`nvidia-spark-ota-check is-ota-available` = false on the latest OTA; newer drivers in generic repos are pinned off by NVIDIA), so on a stock DGX Spark the hero lane is unreachable.

**The fix is lane selection, not driver surgery.** The KVarN recipes use stock Triton attention — no runtime TMA JIT — and run perfectly on 13.0. Two other things still work fine on the stock driver: the **WNA16 Marlin int4 MoE** kernels (prebuilt, the checkpoint's `compressed-tensors` W4A16 experts dispatch to `CompressedTensorsWNA16MarlinMoEMethod`) and the **EAGLE-3 drafter** (int4-RTN, rides the same stack). That combination is Lane C.

*(Related: DeepSeek-style b12x **MoE** deployments work on 13.0 because those are prebuilt cubins, not runtime-JIT TMA. Same "b12x" image family, different kernel class — the label isn't the compatibility story.)*

## The three lane modes (one image, one weight set)

| Lane | Recipe | Measured (thinking-off, temp 0, single stream) | Use for |
|---|---|---|---|
| **C — standing** | `kvarn-eagle3` + `--load-format safetensors --max-num-batched-tokens 2048` | **34.0 code / 41.1 structured / 25.5 prose tok/s** (acceptance 0.65 / 0.80 / 0.36); deep: 17.8 tok/s @113K, TTFT ~147 s; KV pool 217,600 tok | code / structured / agent work |
| **B — fallback** | `kvarn` (drafterless) | **~23 tok/s flat** at every shape (22.9 @1K → 20.7 @113K); prefill ~880 tok/s; KV pool 378,496 tok | deep-context prose/summary — **beats C at depth** (20.7 vs 17.8 @113K); simplest lane |
| B + `--max-num-seqs 2` | `kvarn` + override | unbenched (est. ~1.5–1.8× aggregate); pool covers 2 full-context callers | concurrent callers (B/C are single-stream; requests queue) |
| ~~A — hero~~ | b12x nvfp4 + EAGLE-3 (~36.6 tok/s upstream) | **crashes at first token on stock 13.0** (TMA JIT, above) | only after a blessed CUDA-13.2 GB10 OTA |

Notes that generalize: speculative decode is **output-shape-bound** (structured drafts at ~0.8 acceptance, free prose at ~0.36 — at 113K-deep prose the drafter goes net-negative vs Lane B); `max_num_batched_tokens` 1024→2048 bought ~10% prefill (690→767 tok/s at 113K) for ~0.45 GiB of KV pool.

## Hard limits & client rules (Lane C)

- **Never send `repetition_penalty`** — spec-decode illegal-memory crash risk. Permanent rule.
- **Thinking off per request:** `"chat_template_kwargs": {"thinking_mode": "disabled"}` (the recipe's template default is `enabled`; off is faster *and* lifts drafter acceptance).
- **Context = prompt + output ≤ 131,072 exactly** — one token over returns a clean HTTP 400 (no crash). Needle recall verified to 115.7K.
- Single-stream (`max_num_seqs: 1`): concurrent callers serialize.

## Launch prerequisites (all load-bearing)

See [docs/MINIMAX_M3_LANES.md](docs/MINIMAX_M3_LANES.md) for the full runbook. The short version:

1. **Weights on BOTH nodes** at the same HF-cache path (TP=2 shards tensors, not files).
2. **If your head and worker use different usernames:** the launcher mounts the *head's* `$HOME` path into the worker container — create `/home/<head-user> → /home/<worker-user>` as a symlink on the worker, or the worker mounts an empty cache and dies with `LocalEntryNotFoundError`.
3. **Pin the fabric in `.env`** (don't trust autodiscovery blind): `CLUSTER_NODES=<HEAD_FABRIC_IP>,<WORKER_FABRIC_IP>`, `LOCAL_IP=<HEAD_FABRIC_IP>`, `ETH_IF=enp1s0f0np0`, `IB_IF=rocep1s0f0`, `CONTAINER_NCCL_IB_GID_INDEX=<your GID, often 3>`. Stock NCCL otherwise — `NCCL_PROTO=LL` **loses** 5–9% on this 428B MoE.
4. **Worker-before-head** ordering is handled by `run-recipe.sh --no-ray`; don't hand-start the head first.
5. **GMU 0.93** (the free-memory check runs *after* the API server spins up; bracket per your boot state). Never set `num_gpu_blocks_override` above the profiler — GB10 unified memory has no clean OOM, it evicts weights and wedges the node.
6. **Hygiene before every launch (both nodes):** `docker rm -f vllm_node` → `sudo find /dev/shm -name 'psm_*' -delete` → `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`.
7. **Boot gates** to confirm in `docker logs vllm_node`: `CompressedTensorsWNA16MarlinMoEMethod` · `Using KVARN attention backend` · **no b12x/TMA lines** · `LlamaForCausalLMEagle3` · KV pool ≥ 131,072 · `Application startup complete`. Then coherence-test real output — 200 OK is not coherence.

## What's in this repo

| Path | What |
|---|---|
| `recipes/minimax-m3-w4a16-gptq-kvarn-eagle3.yaml` | **Lane C** — KVarN + EAGLE-3 (a3refaat's shelved recipe, recovered by tonyd2wild from git history; byte-exact) |
| `recipes/minimax-m3-w4a16-gptq-kvarn.yaml` | **Lane B** — drafterless KVarN (byte-exact upstream reproduction) |
| `patches/dockerfile-unreachable-ref-fetch.patch` | build fix: the pinned vLLM ref only resolves via `refs/pull/45381/head` (cold caches die without it) |
| `patches/mod-prometheus-graceful-skip.patch` | launch fix: instrumentator ≥8.0.2 already ships the routing fix; without this every launch aborts |
| `docs/MINIMAX_M3_LANES.md` | the full lane runbook / reproduction map |

The image (`ghcr.io/blackwellboy/vllm-node-minimax-m3-b12x`, manifest `sha256:6702fe3a5182…`) = vLLM pinned `979b56a66c96` built from source + the upstream fused-fp8/nvfp4-KV patch + Rust tool parser + vendored b12x kernel lib, from `a3refaat/spark-vllm-docker@f942437` with both patches applied. CUDA 13.2 toolkit inside — fine on a 13.0 host driver for the KVarN lanes (Triton compiles its own cubins; Marlin ships prebuilt).

## Credits — this stands on other people's work

- **[a3refaat](https://github.com/a3refaat/spark-vllm-docker)** — the entire deployment stack: the KVarN backport + MiniMax-M3 extensions, the recipes, the fused KV vLLM patch, the GB10 unified-memory operational rules.
- **[tonyd2wild](https://github.com/tonyd2wild/MiniMax-M3-2x-DGX-Spark-36-tok-s)** — the reproduction this builds on: the two build/ops patches, the recovery of the shelved KVarN+EAGLE-3 recipe, the deployment gotchas.
- **[Sebesky](https://huggingface.co/Sebesky/MiniMax-M3-W4A16-GPTQ)** — the W4A16 GPTQ target and the [int4-RTN EAGLE-3 drafter](https://huggingface.co/Sebesky/MiniMax-M3-EAGLE3-RTN-INT4).
- **huawei-csl / philippebich** — KVarN calibration-free KV quantization ([vLLM PR #46812](https://github.com/vllm-project/vllm/pull/46812)); **Inferact** — the original EAGLE-3 drafter; **eugr** — the upstream spark-vllm-docker tooling; **MiniMax** — the model (MiniMax Community License — non-commercial constraints apply to the weights); **vLLM** — the engine.

What this repo adds: the stock-13.0-driver finding and lane validation, the measured numbers above, the mnbt=2048 tuning, the cross-user mount fix, and the packaging (prebuilt GHCR image + one-place reproduction).

## License

Our original content (README, docs) is **Apache-2.0** ([LICENSE](LICENSE)). The reproduced recipe yamls and the patches (diffs against MIT-licensed upstream files) remain **MIT** with the upstream notice preserved ([LICENSE.upstream-MIT](LICENSE.upstream-MIT)). Model weights are separate — MiniMax Community License + the checkpoint authors' terms.
