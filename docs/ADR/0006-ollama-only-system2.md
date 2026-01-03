# ADR 0006: Ollama-Only Stack (Remove SGLang Cognitive Server)

## Status
Accepted

## Context
The stack previously included a centralized cognitive server backed by SGLang. In practice, the
training stack relies on Ollama (llava:7b) for VLM UI hints, and the SGLang server was not running.
Keeping SGLang configs and launch paths increased GPU pressure and introduced a fragile dependency
that no longer matched production behavior.

## Decision
Remove SGLang-specific launch/config logic and delete the SGLang cognitive server container assets.
Default launch profiles now run System2 via Ollama (llava:7b) when enabled.

## Consequences
- Reduced GPU memory footprint and fewer moving parts during training.
- Launch/validation no longer attempt to start a cognitive server.
- System2 metrics are reported when enabled, but no centralized cognitive server is required.
- Historical SGLang-specific documentation is now stale and should not be used for current runs.
