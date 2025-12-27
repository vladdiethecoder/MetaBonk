"""System 2: deliberative world-model planning (MCTS).

This is a lightweight MuZero-style scaffold:
  - learned dynamics + reward + value + policy heads
  - Monte Carlo Tree Search over a discrete intent/action space

It is intentionally generic and can be trained offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from src.agent.action_space.hierarchical import Intent


class IntentSpace:
    """Discrete intent space for planning."""

    def __init__(self, intent_names: List[str]) -> None:
        names = [str(n).strip() for n in intent_names if str(n).strip()]
        if not names:
            names = ["idle"]
        self.intent_names = list(dict.fromkeys(names))  # stable unique

    @property
    def size(self) -> int:
        return len(self.intent_names)

    def name_for(self, idx: int) -> str:
        if self.size <= 0:
            return "idle"
        return self.intent_names[int(idx) % self.size]

    def to_intent(self, idx: int) -> Intent:
        return Intent(name=self.name_for(idx), priority=1.0, estimated_steps=1)


@dataclass
class MCTSNode:
    state: "torch.Tensor"  # [state_dim]
    parent: Optional["MCTSNode"] = None
    action: Optional[int] = None
    prior: float = 0.0
    reward: float = 0.0
    children: Dict[int, "MCTSNode"] = None  # type: ignore[assignment]
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}

    @property
    def value(self) -> float:
        return float(self.value_sum) / max(1, int(self.visits))

    def expanded(self) -> bool:
        return bool(self.children)


class DeliberativePlanner(nn.Module):
    """World-model planner with MCTS."""

    def __init__(
        self,
        *,
        state_dim: int,
        intent_space: IntentSpace,
        hidden_dim: int = 512,
        discount: float = 0.99,
        ucb_c: float = 1.5,
    ) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for DeliberativePlanner")
        super().__init__()
        self.state_dim = int(state_dim)
        self.intent_space = intent_space
        self.action_dim = int(intent_space.size)
        self.discount = float(discount)
        self.ucb_c = float(ucb_c)

        ad = self.action_dim
        hd = int(hidden_dim)
        self.dynamics = nn.Sequential(
            nn.Linear(self.state_dim + ad, hd),
            nn.ReLU(),
            nn.Linear(hd, hd),
            nn.ReLU(),
            nn.Linear(hd, self.state_dim),
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.state_dim, hd),
            nn.ReLU(),
            nn.Linear(hd, 1),
        )
        self.value_network = nn.Sequential(
            nn.Linear(self.state_dim, hd),
            nn.ReLU(),
            nn.Linear(hd, 1),
        )
        self.policy_network = nn.Sequential(
            nn.Linear(self.state_dim, hd),
            nn.ReLU(),
            nn.Linear(hd, ad),
        )

    def plan(
        self,
        current_state: "torch.Tensor",
        *,
        num_simulations: int = 64,
        depth: int = 6,
    ) -> List[Intent]:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for DeliberativePlanner")
        state = current_state
        if state.dim() != 1:
            state = state.reshape(-1)
        root = MCTSNode(state=state.detach())

        with torch.no_grad():
            for _ in range(int(num_simulations)):
                node = root
                search_path: List[MCTSNode] = [node]

                # Selection.
                while node.expanded():
                    action, node = self._select_child(node)
                    search_path.append(node)

                # Expansion.
                self._expand(node)

                # Evaluate leaf.
                leaf_value = float(self.value_network(node.state).item())

                # Backprop.
                self._backpropagate(search_path, leaf_value)

        # Extract a greedy plan by visits.
        actions: List[int] = []
        node = root
        for _ in range(int(depth)):
            if not node.children:
                break
            best = max(node.children.items(), key=lambda kv: kv[1].visits)[0]
            actions.append(int(best))
            node = node.children[best]

        return [self.intent_space.to_intent(a) for a in actions] or [Intent(name="idle")]

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        # PUCT-like score.
        best_action = None
        best_score = float("-inf")
        total_visits = max(1, node.visits)
        for a, child in node.children.items():
            q = child.value
            u = self.ucb_c * float(child.prior) * math.sqrt(float(total_visits)) / (1.0 + float(child.visits))
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a
        assert best_action is not None
        return int(best_action), node.children[int(best_action)]

    def _expand(self, node: MCTSNode) -> None:
        if node.expanded():
            return
        logits = self.policy_network(node.state)
        priors = torch.softmax(logits, dim=-1)
        for a in range(self.action_dim):
            one_hot = torch.zeros(self.action_dim, device=node.state.device, dtype=node.state.dtype)
            one_hot[a] = 1.0
            next_state = self.dynamics(torch.cat([node.state, one_hot], dim=-1))
            reward = float(self.reward_predictor(next_state).item())
            node.children[a] = MCTSNode(
                state=next_state,
                parent=node,
                action=a,
                prior=float(priors[a].item()),
                reward=reward,
            )

    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        v = float(value)
        for n in reversed(path):
            n.visits += 1
            n.value_sum += v
            v = float(n.reward) + self.discount * v


__all__ = [
    "DeliberativePlanner",
    "IntentSpace",
]

