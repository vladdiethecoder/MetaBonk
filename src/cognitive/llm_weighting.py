"""LLM-derived weighting and composition helpers.

Goal: move MetaBonk toward a gaming generalist AI by letting an LLM
propose *project-derived* weights for:
  - Federated/Hive Mind policy merges (role weights, merge method, topk)
  - Skill-vector scaling for on-the-fly composition

All calls are best-effort and fall back to safe defaults if the LLM backend
is not configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.llm_clients import LLMConfig, build_llm_fn


@dataclass
class MergeProposal:
    method: str = "ties"  # "ties" | "weighted"
    topk: float = 0.2
    role_weights: Dict[str, float] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "topk": float(self.topk),
            "role_weights": dict(self.role_weights or {}),
        }


class LLMWeightComposer:
    """Proposes weights/configs using a configured LLM backend."""

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig.from_env(default_model="qwen2.5")
        self.llm_fn = build_llm_fn(
            self.cfg,
            system_prompt=(
                "You are a research engineer tuning a generalist game AI. "
                "Return JSON only. Keep weights normalized and stable."
            ),
        )

    def propose_merge(
        self,
        source_policies: List[str],
        target_policy: str,
        metrics_snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> MergeProposal:
        """Propose merge method and weights for hive mind."""
        snap = metrics_snapshot or {}
        compact_metrics = {
            p: {k: v for k, v in (snap.get(p) or {}).items() if k in ("reward", "entropy", "dream_return", "wm_recon")}
            for p in source_policies
        }
        prompt = f"""
We are merging multiple specialized policies into a single generalist agent.

Sources: {source_policies}
Target: {target_policy}
Recent metrics (partial, may be missing): {json.dumps(compact_metrics)}

Task:
1) Choose merge method: "ties" (default) or "weighted".
2) If "weighted", assign a nonnegative weight to each source.
   Weights must sum to 1.0.
3) Choose TIES sparsify topk in [0.05, 0.5] if method=="ties".

Return JSON:
{{
  "method": "ties"|"weighted",
  "topk": 0.2,
  "role_weights": {{"PolicyA": 0.4, "PolicyB": 0.6}}
}}
If unsure, use ties with topk=0.2 and equal weights.
"""
        try:
            raw = self.llm_fn(prompt) or ""
            data = _parse_json_object(raw)
            method = str(data.get("method") or "ties").lower()
            topk = float(data.get("topk") or 0.2)
            weights = data.get("role_weights") or {}
            if not isinstance(weights, dict):
                weights = {}
            # Normalize weights if provided.
            ww: Dict[str, float] = {}
            total = 0.0
            for p in source_policies:
                try:
                    w = float(weights.get(p, 0.0))
                except Exception:
                    w = 0.0
                w = max(0.0, w)
                ww[p] = w
                total += w
            if total <= 1e-8:
                ww = {p: 1.0 / max(1, len(source_policies)) for p in source_policies}
            else:
                ww = {p: w / total for p, w in ww.items()}
            if method not in ("ties", "weighted"):
                method = "ties"
            topk = min(0.5, max(0.05, topk))
            return MergeProposal(method=method, topk=topk, role_weights=ww)
        except Exception:
            return MergeProposal(
                method="ties",
                topk=0.2,
                role_weights={p: 1.0 / max(1, len(source_policies)) for p in source_policies},
            )

    def propose_skill_scales(
        self,
        skills: List[Dict[str, Any]],
        context: str,
    ) -> Dict[str, float]:
        """Propose per-skill scales in [0, 2] for compositional policy soup."""
        prompt = f"""
We are composing a generalist game policy from skill vectors.

Context: {context}
Candidate skills (name,tags,performance,magnitude):
{json.dumps(skills)}

Assign a scale to each skill in [0, 2]. 0 disables a skill.
Prefer high-performing relevant skills; keep total influence stable.

Return JSON: {{"scales": {{"SkillA": 1.0, "SkillB": 0.3}}}}
"""
        try:
            raw = self.llm_fn(prompt) or ""
            data = _parse_json_object(raw)
            scales = data.get("scales") or data.get("role_weights") or {}
            if not isinstance(scales, dict):
                scales = {}
            out: Dict[str, float] = {}
            for s in skills:
                name = str(s.get("name") or "")
                try:
                    v = float(scales.get(name, 1.0))
                except Exception:
                    v = 1.0
                out[name] = min(2.0, max(0.0, v))
            return out
        except Exception:
            return {str(s.get("name") or ""): 1.0 for s in skills}

    def propose_reward_shaping(
        self,
        policy_name: str,
        base_hparams: Dict[str, Any],
        metrics_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Propose intrinsic/reward shaping weights for a policy/sin.

        The returned dict is intended to replace `hparams["reward_shaping"]`.
        Only keys present in the base reward_shaping are preserved.
        """
        base_rs = base_hparams.get("reward_shaping")
        if not isinstance(base_rs, dict):
            base_rs = {}

        allowed_keys = list(base_rs.keys()) or [
            "curiosity_beta",
            "imitation_beta",
            "time_penalty",
            "velocity_penalty",
        ]
        snap = metrics_snapshot or {}
        prompt = f"""
We are running Population-Based Training for a generalist MegaBonk agent.

Policy/Sin role: {policy_name}
Recent metrics (may be partial): {json.dumps(snap)}

Current reward_shaping weights (base):
{json.dumps(base_rs)}

Task:
- Adjust reward_shaping to better fit the role and improve generalist learning.
- Keep values stable; small changes are preferred unless clearly needed.
- For keys containing beta/multiplier/bonus/weight/coef/scale, use nonnegative floats.
- For keys containing penalty, allow negative or positive floats.
- Preserve booleans if present unless you are confident.

Return JSON ONLY:
{{"reward_shaping": {{"key": value, ...}}}}
Include all keys from the base, even if unchanged.
"""
        try:
            raw = self.llm_fn(prompt) or ""
            data = _parse_json_object(raw)
            rs = data.get("reward_shaping") or data.get("scales") or data
            if not isinstance(rs, dict):
                rs = {}

            out: Dict[str, Any] = {}
            for k in allowed_keys:
                base_v = base_rs.get(k)
                proposed = rs.get(k, base_v)
                if isinstance(base_v, bool):
                    out[k] = bool(proposed) if isinstance(proposed, bool) else base_v
                    continue
                try:
                    v = float(proposed) if proposed is not None else float(base_v or 0.0)
                except Exception:
                    v = float(base_v or 0.0)

                lk = k.lower()
                if any(t in lk for t in ("beta", "mult", "bonus", "weight", "coef", "scale")):
                    v = max(0.0, v)
                # Clamp to stable range.
                v = max(-10.0, min(10.0, v))
                out[k] = v

            # Preserve any extra base keys not in allowed_keys.
            for k, v in base_rs.items():
                if k not in out:
                    out[k] = v
            return out
        except Exception:
            return base_rs


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # Try to find first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}
