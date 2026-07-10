# MiniMax-M3 lane runbook — 2× DGX Spark (GB10), stock CUDA-13.0 driver

*Validated 2026-07-10 on a direct-connected GB10 pair (`spark-head` + `spark-worker`, TP=2 over the QSFP fabric). Standing lane = **Lane C** (KVarN + EAGLE-3, `max_num_batched_tokens=2048`).*

## Image
- `ghcr.io/blackwellboy/vllm-node-minimax-m3-b12x` — tags `:cuda13.0-f942437` and `:latest`, manifest digest `sha256:6702fe3a5182b48a260f95c425e94e82d65daa648b9fa07b44f2ab673ce58807`, ~19.3 GB.
- **Provenance:** `a3refaat/spark-vllm-docker` branch `minimax-m3-4bit-w4a16` @ `f942437`; vLLM pinned `979b56a66c96` (0.22.1rc1.dev535) built from source + the upstream `minimax-m3-fused-fp8-kv.patch` + Rust tool parser + vendored b12x kernel lib; **both patches from this repo applied** (cold-cache vLLM-ref fetch via `refs/pull/45381/head`; prometheus `_resolve_path` graceful-skip).
- **Toolkit inside is CUDA 13.2; host driver 13.0 is fine for the KVarN lanes** (Triton emits its own cubins, Marlin is prebuilt). Only the b12x *attention* lane needs a 13.2-class driver (runtime CuTe-DSL TMA JIT) — parked until a sanctioned GB10 OTA ships one.
- Needed on **both** nodes: `docker pull` twice, or `docker save | ssh <worker> docker load` over the pair's fabric.

## Weights (HF-cache snapshot layout; containers run `HF_HUB_OFFLINE=1`)
- **Target:** `Sebesky/MiniMax-M3-W4A16-GPTQ`, snapshot `6696ce6…` (~211 GB) — verify sha256 against the HF manifest after download.
- **Drafter (Lane C only):** `Sebesky/MiniMax-M3-EAGLE3-RTN-INT4`, snapshot `86ffe4b…` (~1.6 GB, int4-RTN incl. embed/lm_head — do **not** substitute bf16/"slim" variants; the draft-embed-quant mod depends on this quant config).
- Must resolve at the same `~/.cache/huggingface/hub/models--Sebesky--…` path on **both** nodes (flat directories fail under `HF_HUB_OFFLINE=1` — keep the `snapshots/<hash>/ + blobs/ + refs/` layout). Download once on the head, copy over the fabric (`tar | nc` runs at multi-GB/s on the QSFP link), then re-verify checksums on the worker.

## Lane matrix

| Lane | Recipe + overrides | Measured (thinking-off, temp 0, single stream, server `usage` fields) | KV pool | Pick when |
|---|---|---|---|---|
| **C — standing** | `minimax-m3-w4a16-gptq-kvarn-eagle3.yaml` + `-- --load-format safetensors --max-num-batched-tokens 2048` | 34.0 code / 41.1 structured / 25.5 prose tok/s (acc 0.65/0.80/0.36); deep 17.8 @113K, TTFT ~147 s (prefill ~767 tok/s) | 217,600 tok (1.66× @131K) | code / structured / agent work |
| **B — fallback** | `minimax-m3-w4a16-gptq-kvarn.yaml` | ~23 tok/s flat (22.9 @1K → 20.7 @113K); prefill ~880 tok/s @ mnbt 2048 (TTFT ~128 s @113K) | 378,496 tok (2.89× @131K) | deep-context prose/summary — beats C at depth; simplest |
| B + `-- --max-num-seqs 2` | concurrency option | unbenched (~1.5–1.8× aggregate est.); pool covers 2 full-context callers | shared | multiple simultaneous callers |
| ~~A — hero (b12x nvfp4 + EAGLE-3)~~ | upstream `b12x-nvfp4-eagle3` yaml | **crashes at first token on 13.0**: `b12x/attention/indexer/contiguous_kernel.py → cpasync.tma_partition → ValueError: Operation creation failed` | — | only on a 13.2-class driver |

**Mandatory Lane C overrides explained:** the recovered yaml predates upstream's safetensors switch and carries `load_format: instanttensor` (under which any page-cache drop mid-load is fatal) — `--load-format safetensors` appended after `--` wins. `--max-num-batched-tokens 2048` (yaml default 1024) buys ~10% deep prefill for ~0.45 GiB of KV pool.

## Launch prerequisites (all load-bearing)
1. **Cross-user home mount (if head/worker usernames differ):** `launch-cluster.sh` mounts the **head's** `$HOME`-based cache paths into the **worker's** container. If the worker's username differs, create `sudo ln -sfn /home/<worker-user> /home/<head-user>` **on the worker** so the path resolves to its real cache — otherwise the worker mounts an empty directory and dies with `LocalEntryNotFoundError`. Preflight this before every launch.
2. **`.env` in the checkout root** (pin explicitly; autodiscovery can pick the wrong NIC/subnet):
   ```
   CLUSTER_NODES=<HEAD_FABRIC_IP>,<WORKER_FABRIC_IP>
   LOCAL_IP=<HEAD_FABRIC_IP>
   ETH_IF=enp1s0f0np0
   IB_IF=rocep1s0f0
   CONTAINER_NCCL_IB_GID_INDEX=<your GID — check show_gids; often 3>
   ```
   Head↔worker on a direct QSFP cable, same port both ends, MTU 9000 recommended. Stock NCCL otherwise — **do not** set `NCCL_PROTO=LL`/`NCCL_MAX_NCHANNELS` (loses 5–9% on this 428B MoE; that lever helps small dense models, not this).
3. **Passwordless head→worker SSH by bare fabric IP** (the orchestrator SSHes `<WORKER_FABRIC_IP>` directly); if usernames differ add to the head's `~/.ssh/config`:
   ```
   Host <WORKER_FABRIC_IP>
       User <worker-user>
   ```
4. **Worker-before-head** ordering — automatic via `run-recipe.sh --no-ray`; never start the head first by hand.
5. **GMU 0.93** for both KVarN lanes. The free-memory check runs *after* the API server + engine spin up, so it's boot-state-sensitive: too-high fails with the exact GiB shortfall, too-low prints the exact pool size — bracket between those two signals. **Never** set `num_gpu_blocks_override` above the profiler value (GB10 unified memory silently evicts file-backed weights and thrashes the node to a power-cycle).
6. **Hygiene before every launch (both nodes):**
   ```bash
   docker rm -f vllm_node
   sudo find /dev/shm -name 'psm_*' -delete
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```
7. **Log rotation** (vLLM logs every request; an always-on lane grows multi-GB logs): launch with `VLLM_SPARK_EXTRA_DOCKER_ARGS="--log-opt max-size=100m --log-opt max-file=3"`.

## Canonical launch (Lane C standing)
```bash
cd spark-vllm-docker   # a3refaat checkout @ f942437, both patches applied, .env pinned
VLLM_SPARK_EXTRA_DOCKER_ARGS="--log-opt max-size=100m --log-opt max-file=3" \
./run-recipe.sh --no-ray -d recipes/minimax-m3-w4a16-gptq-kvarn-eagle3.yaml \
  -- --load-format safetensors --max-num-batched-tokens 2048
```
**Boot gates** (in `docker logs vllm_node` before trusting the lane): `Using CompressedTensorsWNA16MarlinMoEMethod` · `Using KVARN attention backend` · **zero** b12x/TMA lines · `Resolved architecture: LlamaForCausalLMEagle3` · `GPU KV cache size` ≥ 131,072 · `Application startup complete`. Then **coherence-test actual output** (e.g. bat-and-ball → $0.05; 9.9 vs 9.11 → 9.9) with `{"thinking_mode":"disabled"}` — HTTP 200 is not coherence. The first request pays a ~1-min warmup; never benchmark it.

## Hard limits & client rules
- **Context: `prompt + output ≤ 131,072` exactly.** One token over → graceful HTTP 400, engine unharmed. Needle recall verified to 115.7K.
- **Never send `repetition_penalty`** on the EAGLE-3 lane — spec-decode illegal-memory crash risk.
- **Thinking off per request** (`chat_template_kwargs: {"thinking_mode":"disabled"}`) — faster and dramatically better drafter acceptance; the recipe's template default is `enabled`.
- Single-stream: `max_num_seqs: 1` — concurrent requests queue (aggregate stays flat as concurrency rises). For multi-caller use, run Lane B with `--max-num-seqs 2`.
- Spec-decode throughput is **output-shape-bound**: structured ≈ 0.8 acceptance → 41 tok/s; free prose ≈ 0.36 → 25 tok/s; at 113K-deep prose the drafter goes net-negative vs drafterless Lane B (17.8 vs 20.7). Pick the lane per workload, not by a single benchmark cell.
