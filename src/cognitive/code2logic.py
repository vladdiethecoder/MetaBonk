"""Code2Logic: Game Logic Extraction Pipeline.

Synthesizes verifiable reasoning traces from game source code.
Based on Game-RL (Fudan University).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GameLogic:
    """Extracted game logic from decompilation."""
    upgrades: Dict[str, Dict] = field(default_factory=dict)
    enemies: Dict[str, Dict] = field(default_factory=dict)
    formulas: Dict[str, str] = field(default_factory=dict)
    synergies: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class ReasoningScenario:
    """A strategic reasoning scenario for training."""
    current_stats: Dict[str, float]
    upgrade_options: List[Dict]
    context: Dict[str, Any]
    correct_choice: int
    chain_of_thought: str


class Code2LogicExtractor:
    """Extracts game logic from decompiled code."""
    
    def __init__(self, game_path: Optional[Path] = None):
        self.game_path = game_path
        self.logic = GameLogic()
    
    def extract_from_il2cpp(self, dump_path: Path) -> GameLogic:
        """Extract logic from Il2CppDumper output."""
        # Parse dump.cs for class definitions
        if not dump_path.exists():
            raise FileNotFoundError(f"Il2CppDumper output not found: {dump_path}")
        
        content = dump_path.read_text(errors="ignore")
        self._parse_upgrades(content)
        self._parse_enemies(content)
        self._parse_formulas(content)

        if not (self.logic.upgrades or self.logic.enemies or self.logic.formulas):
            raise ValueError(
                "No game logic extracted from Il2CppDumper output. "
                "Update the heuristics in Code2LogicExtractor to match the game's dump format."
            )

        # Best-effort inferred synergies if none were extracted.
        if not self.logic.synergies and self.logic.upgrades:
            self.logic.synergies = self._infer_synergies()
        
        return self.logic

    def _iter_class_blocks(self, content: str) -> List[Tuple[str, str]]:
        """Best-effort extraction of C# class blocks (name, block_text).

        Il2CppDumper output is a single huge C# file. This parser is intentionally
        lightweight and permissive: it uses brace counting and does not attempt
        full C# parsing.
        """
        out: List[Tuple[str, str]] = []
        current_name: Optional[str] = None
        buf: List[str] = []
        depth = 0
        saw_open = False

        class_re = re.compile(r"\bclass\s+([A-Za-z_]\w*)\b")

        for line in content.splitlines():
            if current_name is None:
                m = class_re.search(line)
                if not m:
                    continue
                current_name = m.group(1)
                buf = [line]
                delta = line.count("{") - line.count("}")
                if "{" in line:
                    saw_open = True
                depth += delta
                # If the opening brace is on the next line, keep collecting until we see it.
                if saw_open and depth == 0:
                    out.append((current_name, "\n".join(buf)))
                    current_name = None
                    buf = []
                    saw_open = False
                    depth = 0
                continue

            buf.append(line)
            if "{" in line:
                saw_open = True
            depth += line.count("{") - line.count("}")
            if saw_open and depth <= 0:
                out.append((current_name, "\n".join(buf)))
                current_name = None
                buf = []
                saw_open = False
                depth = 0

        return out

    @staticmethod
    def _snake_case(name: str) -> str:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.replace("__", "_").strip("_").lower()

    @staticmethod
    def _extract_numeric_assignments(block: str) -> Dict[str, float]:
        """Extract simple `field = 1.23f;` style assignments."""
        assigns: Dict[str, float] = {}
        # Example: public float BaseDamage = 10f;
        # Also catch const/static and ints.
        pat = re.compile(
            r"\b([A-Za-z_]\w*)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:f|F|d|D)?\s*;",
        )
        for m in pat.finditer(block):
            key = m.group(1)
            try:
                assigns[key] = float(m.group(2))
            except Exception:
                continue
        return assigns

    def _infer_synergies(self) -> List[Tuple[str, str, float]]:
        """Heuristic synergy inference from upgrade names."""
        names = list(self.logic.upgrades.keys())
        if len(names) < 2:
            return []
        synergies: List[Tuple[str, str, float]] = []

        # Common Megabonk-esque interactions.
        rules: List[Tuple[Tuple[str, ...], Tuple[str, ...], float]] = [
            (("crit", "critical"), ("damage", "dmg"), 1.5),
            (("aoe", "radius"), ("fire", "rate", "speed"), 1.3),
            (("move", "speed"), ("regen", "health"), 1.2),
        ]

        def match_any(n: str, toks: Tuple[str, ...]) -> bool:
            n = n.lower()
            return any(t in n for t in toks)

        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                for left, right, mult in rules:
                    if match_any(a, left) and match_any(b, right):
                        synergies.append((a, b, mult))
                    elif match_any(a, right) and match_any(b, left):
                        synergies.append((a, b, mult))

        # Deduplicate.
        seen = set()
        out: List[Tuple[str, str, float]] = []
        for a, b, m in synergies:
            k = tuple(sorted((a, b))) + (float(m),)
            if k in seen:
                continue
            seen.add(k)
            out.append((a, b, float(m)))
        return out
    
    def _parse_upgrades(self, content: str):
        """Parse upgrade-like classes from Il2CppDumper output.

        This is a best-effort heuristic parser. When it cannot confidently
        extract values it simply leaves the upgrade out.
        """
        upgrades: Dict[str, Dict[str, Any]] = {}

        for class_name, block in self._iter_class_blocks(content):
            assigns = self._extract_numeric_assignments(block)
            name_l = class_name.lower()

            looks_like_upgrade = (
                "upgrade" in name_l
                or any(k.lower() in {"maxlevel", "max_level"} or ("max" in k.lower() and "level" in k.lower()) for k in assigns)
                or any("scal" in k.lower() for k in assigns)
            )
            if not looks_like_upgrade:
                continue

            # Heuristic field selection.
            max_level = None
            scaling = None
            base = None

            for k, v in assigns.items():
                kl = k.lower()
                if max_level is None and ("maxlevel" in kl or (("max" in kl) and ("level" in kl))):
                    max_level = int(round(v))
                if scaling is None and ("scaling" in kl or "scale" in kl or "mult" in kl):
                    scaling = float(v)
                if base is None and ("base" in kl or "start" in kl):
                    base = float(v)

            # If no explicit base, pick a sensible first numeric constant.
            if base is None:
                for k, v in assigns.items():
                    kl = k.lower()
                    if any(tok in kl for tok in ("damage", "rate", "regen", "speed", "radius", "chance")):
                        base = float(v)
                        break
            if base is None and assigns:
                # Deterministic: smallest field name.
                k0 = sorted(assigns.keys())[0]
                base = float(assigns[k0])

            # Reasonable defaults.
            if scaling is None:
                scaling = 1.0
            if max_level is None:
                max_level = 5

            upgrade_key = self._snake_case(class_name)
            upgrade_key = re.sub(r"_?upgrade$", "", upgrade_key)
            upgrade_key = upgrade_key or self._snake_case(class_name)

            upgrades[upgrade_key] = {
                "base": float(base or 0.0),
                "scaling": float(scaling),
                "max_level": int(max(1, max_level)),
                "source_class": class_name,
            }

        if upgrades:
            self.logic.upgrades = upgrades
    
    def _parse_enemies(self, content: str):
        """Parse enemy-like classes from Il2CppDumper output (best-effort)."""
        enemies: Dict[str, Dict[str, Any]] = {}

        for class_name, block in self._iter_class_blocks(content):
            assigns = self._extract_numeric_assignments(block)
            name_l = class_name.lower()

            looks_like_enemy = "enemy" in name_l or any(k.lower() in {"hp", "health", "damage", "speed"} for k in assigns)
            if not looks_like_enemy:
                continue

            hp = None
            damage = None
            speed = None
            for k, v in assigns.items():
                kl = k.lower()
                if hp is None and kl in {"hp", "health", "maxhp", "max_health", "hitpoints"}:
                    hp = float(v)
                if damage is None and "damage" in kl:
                    damage = float(v)
                if speed is None and "speed" in kl:
                    speed = float(v)

            # Only include if we got at least one meaningful field.
            if hp is None and damage is None and speed is None:
                continue

            enemy_key = self._snake_case(class_name)
            enemy_key = re.sub(r"_?enemy$", "", enemy_key)
            enemy_key = enemy_key or self._snake_case(class_name)
            enemies[enemy_key] = {
                "hp": float(hp) if hp is not None else None,
                "damage": float(damage) if damage is not None else None,
                "speed": float(speed) if speed is not None else None,
                "source_class": class_name,
            }

        if enemies:
            self.logic.enemies = enemies
    
    def _parse_formulas(self, content: str):
        """Extract simple arithmetic formulas from method return statements."""
        formulas: Dict[str, str] = {}

        # Method pattern: float Foo(...) { ... return expr; ... }
        meth = re.compile(
            r"\b(?:float|double|int)\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*\{([^{}]*?\breturn\s+[^;]+;[^{}]*?)\}",
            re.DOTALL,
        )
        for m in meth.finditer(content):
            name = m.group(1)
            body = m.group(2)
            r = re.search(r"\breturn\s+([^;]+);", body)
            if not r:
                continue
            expr = r.group(1).strip()
            # Skip trivial returns.
            if expr in ("0", "0f", "0.0", "1", "true", "false"):
                continue
            # Keep only reasonably short, arithmetic-ish formulas.
            if len(expr) > 200:
                continue
            if not any(op in expr for op in ("+", "-", "*", "/", "%")):
                continue
            formulas[self._snake_case(name)] = expr

        if formulas:
            self.logic.formulas = formulas


class TaskTemplateGenerator:
    """Generates reasoning task templates."""
    
    def __init__(self, logic: GameLogic):
        self.logic = logic
    
    def state_prediction(self, stats: Dict, upgrade: str) -> ReasoningScenario:
        """Generate state prediction task."""
        if upgrade not in self.logic.upgrades:
            if not self.logic.upgrades:
                raise ValueError("No upgrades available to generate a state prediction scenario")
            # Deterministic fallback for reproducibility.
            upgrade = sorted(self.logic.upgrades.keys())[0]
        
        upgrade_info = self.logic.upgrades[upgrade]
        new_stats = stats.copy()
        
        if upgrade in new_stats:
            new_stats[upgrade] = new_stats[upgrade] * upgrade_info["scaling"]
        else:
            new_stats[upgrade] = upgrade_info["base"]
        
        cot = f"Applying {upgrade}: base={upgrade_info['base']}, scaling={upgrade_info['scaling']}"
        
        return ReasoningScenario(
            current_stats=stats,
            upgrade_options=[{"name": upgrade, "info": upgrade_info}],
            context={"task": "state_prediction"},
            correct_choice=0,
            chain_of_thought=cot,
        )
    
    def strategic_choice(
        self,
        stats: Dict,
        options: List[str],
        threat_level: str = "medium",
    ) -> ReasoningScenario:
        """Generate strategic choice task."""
        # Calculate best option based on threat
        scores = []
        cot_parts = []
        
        for opt in options:
            if opt not in self.logic.upgrades:
                scores.append(0)
                continue
            
            info = self.logic.upgrades[opt]
            
            # Score based on threat
            if threat_level == "high":
                # Prioritize survival
                score = 2.0 if "health" in opt or "speed" in opt else 1.0
            else:
                # Prioritize damage
                score = 2.0 if "damage" in opt or "crit" in opt else 1.0
            
            # Check synergies
            for s1, s2, mult in self.logic.synergies:
                if opt == s1 and s2 in stats:
                    score *= mult
                elif opt == s2 and s1 in stats:
                    score *= mult
            
            scores.append(score)
            cot_parts.append(f"{opt}: score={score:.2f}")
        
        best_idx = int(np.argmax(scores)) if scores else 0
        
        return ReasoningScenario(
            current_stats=stats,
            upgrade_options=[{"name": o} for o in options],
            context={"threat_level": threat_level, "task": "strategic_choice"},
            correct_choice=best_idx,
            chain_of_thought=f"Threat={threat_level}. " + ", ".join(cot_parts),
        )


class DataEngine:
    """Generates training data from game logic."""
    
    def __init__(self, logic: GameLogic):
        self.logic = logic
        self.template_gen = TaskTemplateGenerator(logic)
    
    def generate_batch(self, size: int = 100) -> List[ReasoningScenario]:
        """Generate a deterministic batch of reasoning scenarios.

        This intentionally avoids synthetic random numbers because those destroy
        troubleshooting signal. Scenarios are generated by enumerating upgrade
        combinations and using plausible stat values derived from the extracted
        upgrade (base, scaling, max_level) parameters.
        """
        upgrade_keys = sorted(self.logic.upgrades.keys())
        if len(upgrade_keys) < 3:
            raise ValueError("Need at least 3 upgrades to generate training scenarios")

        scenarios: List[ReasoningScenario] = []
        threat_cycle = ["low", "medium", "high"]

        for i in range(int(size)):
            # Choose 3 current stats and 3 candidate upgrades deterministically.
            stats_keys = [upgrade_keys[(i + j) % len(upgrade_keys)] for j in range(3)]
            option_keys = [upgrade_keys[(i + 3 + j) % len(upgrade_keys)] for j in range(3)]

            # Derive plausible stat values from base/scaling/max_level.
            stats: Dict[str, float] = {}
            for j, key in enumerate(stats_keys):
                info = self.logic.upgrades.get(key) or {}
                base = float(info.get("base", 0.0) or 0.0)
                scaling = float(info.get("scaling", 1.0) or 1.0)
                max_level = int(info.get("max_level", 1) or 1)
                level = (i + j) % max(1, max_level) + 1
                val = base * (scaling ** max(0, level - 1))
                stats[key] = float(val)

            threat = threat_cycle[i % len(threat_cycle)]
            scenario = self.template_gen.strategic_choice(stats, option_keys, threat)
            scenarios.append(scenario)

        return scenarios
    
    def to_training_format(self, scenarios: List[ReasoningScenario]) -> List[Dict]:
        """Convert to (image, question, cot, answer) format."""
        data = []
        
        for s in scenarios:
            data.append({
                "input": {
                    "stats": s.current_stats,
                    "options": [o["name"] for o in s.upgrade_options],
                    "context": s.context,
                },
                "chain_of_thought": s.chain_of_thought,
                "answer": s.correct_choice,
            })
        
        return data


# Convenience
def create_code2logic_pipeline(game_path: Optional[Path] = None, dump_path: Optional[Path] = None):
    """Create a Code2Logic pipeline from a real Il2CppDumper `dump.cs`.

    Args:
        game_path: Optional game directory (used only to help locate `dump.cs`).
        dump_path: Explicit path to Il2CppDumper `dump.cs`. If omitted, uses:
          1) `METABONK_IL2CPP_DUMP_PATH`
          2) `<game_path>/dump.cs` or `<game_path>/il2cpp/dump.cs`

    Raises:
        ValueError: if no dump path can be resolved.
    """
    import os

    extractor = Code2LogicExtractor(game_path)

    resolved = dump_path
    if resolved is None:
        env = os.environ.get("METABONK_IL2CPP_DUMP_PATH", "").strip()
        if env:
            resolved = Path(env)
    if resolved is None and game_path is not None:
        gp = Path(game_path)
        candidates = [
            gp / "dump.cs",
            gp / "il2cpp" / "dump.cs",
            gp / "Il2CppDumper" / "dump.cs",
        ]
        for c in candidates:
            if c.exists():
                resolved = c
                break
    if resolved is None:
        raise ValueError("create_code2logic_pipeline requires `dump_path` or `METABONK_IL2CPP_DUMP_PATH`")

    logic = extractor.extract_from_il2cpp(Path(resolved))
    return DataEngine(logic)
