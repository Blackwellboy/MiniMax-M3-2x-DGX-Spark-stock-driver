# Recipe provenance

Both yamls are **a3refaat's work** (`a3refaat/spark-vllm-docker` branch `minimax-m3-4bit-w4a16`, MIT — see `../LICENSE.upstream-MIT`), reproduced **byte-exact** via
[tonyd2wild/MiniMax-M3-2x-DGX-Spark-36-tok-s](https://github.com/tonyd2wild/MiniMax-M3-2x-DGX-Spark-36-tok-s):

- `minimax-m3-w4a16-gptq-kvarn.yaml` — branch HEAD `f942437` (Lane B, drafterless).
- `minimax-m3-w4a16-gptq-kvarn-eagle3.yaml` — recovered from a3refaat git history (`git show 27d90ac:recipes/…`), the shelved KVarN+EAGLE-3 recipe (Lane C). **Run it with `-- --load-format safetensors`** (it predates the upstream safetensors switch).

Run them from a checkout of a3refaat's branch (the `mods/` they reference live there); apply the two patches in `../patches/` first.
