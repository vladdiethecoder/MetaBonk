# VLM Model Comparison Report

**Generated**: 2026-01-01T18:43:25

## Results

| Variant | Backend | Model Path | Quant | Hints/min | Scenes/min | Notes |
|---|---|---|---|---:|---:|---|
| `phi3_awq_sglang` | `sglang` | `/models/Phi_3_Vision_128k_Instruct_AWQ_4bit` | `awq_marlin` | 63.0 | 3.2 | errors=10/10 |
| `phi3_full_sglang` | `sglang` | `/models/Phi-3-vision-128k-instruct` | `` | 31.0 | 3.2 | errors=10/10 |
| `phi3_full_transformers` | `transformers` | `/models/Phi-3-vision-128k-instruct` | `` | 58.8 | 1.0 | errors=5/10 |

## Selection

**Selected**: `phi3_awq_sglang`

Selection heuristic: maximize `scenes_per_minute`, tie-break by `hints_per_minute`.

Notes:
- The `*_benchmark.txt` results were collected while the 5 worker stack was running; high contention can cause benchmark timeouts. Treat them as best-effort, not absolute latency measurements.
