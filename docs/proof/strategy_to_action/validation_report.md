# Strategy-to-Action Validation Report

**Generated**: 2026-01-01 19:00:34  
**Run**: `run-launch-1767311073`  
**Examples Captured**: 5

## Summary

This report validates that System2 (VLM-driven) strategies are received by workers and influence action selection.

- **Examples with scene_delta > 0**: 5/5 (success rate: 100.0%)
- **Evidence of strategy→action**: action logs show `src=policy+system2` lines for all sampled workers.

## Examples

### Example 1 (worker omega-0)

- Strategy PNG: `docs/proof/strategy_to_action/strategy_example_1.png`
- Text record: `docs/proof/strategy_to_action/example_1.txt`
- Action trace: `docs/proof/strategy_to_action/action_trace_1.txt`

### Example 2 (worker omega-1)

- Strategy PNG: `docs/proof/strategy_to_action/strategy_example_2.png`
- Text record: `docs/proof/strategy_to_action/example_2.txt`
- Action trace: `docs/proof/strategy_to_action/action_trace_2.txt`

### Example 3 (worker omega-2)

- Strategy PNG: `docs/proof/strategy_to_action/strategy_example_3.png`
- Text record: `docs/proof/strategy_to_action/example_3.txt`
- Action trace: `docs/proof/strategy_to_action/action_trace_3.txt`

### Example 4 (worker omega-3)

- Strategy PNG: `docs/proof/strategy_to_action/strategy_example_4.png`
- Text record: `docs/proof/strategy_to_action/example_4.txt`
- Action trace: `docs/proof/strategy_to_action/action_trace_4.txt`

### Example 5 (worker omega-4)

- Strategy PNG: `docs/proof/strategy_to_action/strategy_example_5.png`
- Text record: `docs/proof/strategy_to_action/example_5.txt`
- Action trace: `docs/proof/strategy_to_action/action_trace_5.txt`

## Notes

- Scene deltas over a 10s window are noisy (and may be 0 even when exploration is improving).
- For stronger attribution, correlate `system2_reasoning.last_response_ts` with subsequent `src=policy+system2` action lines.
