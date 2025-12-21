#!/usr/bin/env python3
"""Active Inference Menu Navigator (The "Lobby Agent").

VLM-driven menu navigation using Active Inference principles:
- Maintains probabilistic state graph of menu screens
- Selects actions that minimize Expected Free Energy
- Balances exploitation (reach goal) and exploration (learn new screens)

Usage:
    python -m scripts.menu_navigator --goal "Start Game as Calcium on Desert"
"""

import asyncio
import base64
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

try:
    import httpx
except ImportError:
    httpx = None


@dataclass
class MenuState:
    """Represents a menu screen state."""
    
    name: str
    elements: List[Dict[str, Any]]
    transitions: Dict[str, str] = field(default_factory=dict)  # element -> next_state
    visit_count: int = 0
    
    def to_hash(self) -> str:
        """Generate unique hash for this state."""
        import hashlib
        elements_str = json.dumps(sorted([e.get("text", "") for e in self.elements]))
        return hashlib.md5(elements_str.encode()).hexdigest()[:8]


@dataclass
class NavigatorConfig:
    """Configuration for menu navigator."""
    
    # VLM
    vlm_type: str = "ollama"  # "ollama", "openai", "anthropic"
    vlm_model: str = "llava"
    vlm_url: str = "http://localhost:11434"
    
    # Active Inference
    planning_horizon: int = 5
    exploration_weight: float = 0.3  # Balance exploitation vs exploration
    
    # Input
    screen_capture_method: str = "mss"  # "mss", "pil", "bridge"
    
    # Game window
    window_name: str = "MegaBonk"


class VLMClient:
    """Client for querying Vision-Language Models."""
    
    def __init__(self, cfg: NavigatorConfig):
        self.cfg = cfg
    
    async def query(
        self,
        image: Any,
        prompt: str,
    ) -> str:
        """Query VLM with image and prompt."""
        if self.cfg.vlm_type == "ollama":
            return await self._query_ollama(image, prompt)
        elif self.cfg.vlm_type == "openai":
            return await self._query_openai(image, prompt)
        else:
            raise ValueError(f"Unknown VLM type: {self.cfg.vlm_type}")
    
    async def _query_ollama(self, image: Any, prompt: str) -> str:
        """Query local Ollama instance."""
        if not httpx:
            return '{"elements": []}'
        
        # Convert image to base64
        if isinstance(image, bytes):
            img_b64 = base64.b64encode(image).decode()
        elif Image and hasattr(image, "save"):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
        else:
            img_b64 = str(image)
        
        payload = {
            "model": self.cfg.vlm_model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{self.cfg.vlm_url}/api/generate",
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "")
            except Exception as e:
                print(f"Ollama query error: {e}")
        
        return '{"elements": []}'
    
    async def _query_openai(self, image: Any, prompt: str) -> str:
        """Query OpenAI GPT-4V."""
        if not httpx:
            return '{"elements": []}'
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return '{"elements": []}'
        
        # Convert to base64
        if isinstance(image, bytes):
            img_b64 = base64.b64encode(image).decode()
        elif Image and hasattr(image, "save"):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
        else:
            img_b64 = str(image)
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"OpenAI query error: {e}")
        
        return '{"elements": []}'


class ScreenCapture:
    """Captures game screen."""
    
    def __init__(self, cfg: NavigatorConfig):
        self.cfg = cfg
    
    def capture(self) -> Optional[Any]:
        """Capture current screen."""
        if self.cfg.screen_capture_method == "mss":
            return self._capture_mss()
        elif self.cfg.screen_capture_method == "pil":
            return self._capture_pil()
        return None
    
    def _capture_mss(self) -> Optional[Any]:
        """Capture using mss (fast)."""
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                img = sct.grab(monitor)
                if Image:
                    return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                return img
        except Exception as e:
            print(f"MSS capture error: {e}")
            return None
    
    def _capture_pil(self) -> Optional[Any]:
        """Capture using PIL."""
        if not Image:
            return None
        try:
            from PIL import ImageGrab
            return ImageGrab.grab()
        except Exception as e:
            print(f"PIL capture error: {e}")
            return None


class MenuGraph:
    """Probabilistic graph model of menu screens."""
    
    def __init__(self):
        self.states: Dict[str, MenuState] = {}
        self.current_state: Optional[str] = None
        self.transition_probs: Dict[Tuple[str, str], float] = {}
    
    def add_state(self, state: MenuState) -> str:
        """Add state to graph, return state id."""
        state_id = state.to_hash()
        if state_id not in self.states:
            self.states[state_id] = state
        else:
            # Merge elements
            self.states[state_id].visit_count += 1
        return state_id
    
    def record_transition(self, from_state: str, action: str, to_state: str):
        """Record observed transition."""
        key = (from_state, action)
        
        # Update transition probability
        if key not in self.transition_probs:
            self.transition_probs[key] = 0.0
        self.transition_probs[key] = 0.9 * self.transition_probs[key] + 0.1
        
        # Update state transitions
        if from_state in self.states:
            self.states[from_state].transitions[action] = to_state
    
    def get_unexplored_actions(self, state_id: str) -> List[str]:
        """Get actions not yet explored from this state."""
        if state_id not in self.states:
            return []
        
        state = self.states[state_id]
        explored = set(state.transitions.keys())
        all_actions = [e.get("text", f"element_{i}") for i, e in enumerate(state.elements)]
        
        return [a for a in all_actions if a not in explored]
    
    def compute_path_to_goal(
        self,
        start: str,
        goal_condition: str,
    ) -> Optional[List[str]]:
        """Find path from start to state matching goal."""
        # BFS through known transitions
        from collections import deque
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current in self.states:
                state = self.states[current]
                # Check if goal is satisfied
                for elem in state.elements:
                    if goal_condition.lower() in elem.get("text", "").lower():
                        return path
                
                # Explore transitions
                for action, next_state in state.transitions.items():
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [action]))
        
        return None


class ActiveInferenceNavigator:
    """Active Inference-based menu navigator.
    
    Uses Expected Free Energy to select actions:
    - Pragmatic value: How much does action progress toward goal?
    - Epistemic value: How much will we learn from this action?
    """
    
    def __init__(self, cfg: Optional[NavigatorConfig] = None):
        self.cfg = cfg or NavigatorConfig()
        self.vlm = VLMClient(self.cfg)
        self.capture = ScreenCapture(self.cfg)
        self.graph = MenuGraph()
        self.goal: Optional[str] = None
    
    async def parse_screen(self, image: Any) -> MenuState:
        """Use VLM to parse current screen into menu state."""
        prompt = """Analyze this game menu screen. Return JSON with:
{
  "screen_name": "name of this screen/menu",
  "elements": [
    {"text": "button text", "type": "button/checkbox/slider", "x": 100, "y": 200, "w": 150, "h": 40}
  ]
}
Only include interactive elements. Be precise with text content."""

        response = await self.vlm.query(image, prompt)
        
        # Parse response
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return MenuState(
                    name=data.get("screen_name", "Unknown"),
                    elements=data.get("elements", []),
                )
        except json.JSONDecodeError:
            pass
        
        return MenuState(name="Unknown", elements=[])
    
    def compute_efe(
        self,
        state: MenuState,
        action: str,
        goal: str,
    ) -> float:
        """Compute Expected Free Energy for action.
        
        Lower EFE = better action.
        EFE = Risk (goal distance) + Ambiguity (uncertainty)
        """
        state_id = state.to_hash()
        
        # Pragmatic value (risk): Does action lead toward goal?
        pragmatic = 1.0  # Default high (bad)
        
        # Check if action text matches goal keywords
        goal_words = goal.lower().split()
        action_lower = action.lower()
        
        for word in goal_words:
            if word in action_lower:
                pragmatic -= 0.3  # Reduce risk
        
        # Check known transitions
        if state_id in self.graph.states:
            known_state = self.graph.states[state_id]
            if action in known_state.transitions:
                next_state = known_state.transitions[action]
                if next_state in self.graph.states:
                    next_elements = self.graph.states[next_state].elements
                    for elem in next_elements:
                        for word in goal_words:
                            if word in elem.get("text", "").lower():
                                pragmatic -= 0.5
        
        # Epistemic value (ambiguity): Is this action unexplored?
        ambiguity = 0.0
        unexplored = self.graph.get_unexplored_actions(state_id)
        if action in unexplored:
            ambiguity = -self.cfg.exploration_weight  # Lower EFE (good) for exploration
        
        # EFE = pragmatic risk + epistemic ambiguity
        efe = pragmatic + ambiguity
        
        return efe
    
    def select_action(
        self,
        state: MenuState,
        goal: str,
    ) -> Optional[Dict[str, Any]]:
        """Select best action based on Expected Free Energy."""
        if not state.elements:
            return None
        
        best_action = None
        best_efe = float("inf")
        
        for elem in state.elements:
            action = elem.get("text", "")
            if not action:
                continue
            
            efe = self.compute_efe(state, action, goal)
            
            if efe < best_efe:
                best_efe = efe
                best_action = elem
        
        return best_action
    
    async def click_element(self, element: Dict[str, Any]):
        """Execute click on UI element."""
        x = element.get("x", 0) + element.get("w", 0) // 2
        y = element.get("y", 0) + element.get("h", 0) // 2
        
        print(f"Clicking: '{element.get('text', 'unknown')}' at ({x}, {y})")
        
        # Use xdotool or similar
        try:
            import subprocess
            subprocess.run(["xdotool", "mousemove", str(x), str(y)], check=True)
            subprocess.run(["xdotool", "click", "1"], check=True)
        except Exception as e:
            print(f"Click error: {e}")
    
    async def navigate_to_goal(
        self,
        goal: str,
        max_steps: int = 20,
    ) -> bool:
        """Navigate menus to reach goal state.
        
        Args:
            goal: Natural language goal (e.g., "Start game as Calcium on Desert")
            max_steps: Maximum navigation steps
            
        Returns:
            True if goal reached
        """
        self.goal = goal
        print(f"Navigating to goal: {goal}")
        
        for step in range(max_steps):
            # Capture screen
            image = self.capture.capture()
            if image is None:
                print("Screen capture failed")
                await asyncio.sleep(0.5)
                continue
            
            # Parse screen
            state = await self.parse_screen(image)
            print(f"Step {step + 1}: Screen '{state.name}' with {len(state.elements)} elements")
            
            # Add to graph
            state_id = self.graph.add_state(state)
            prev_state = self.graph.current_state
            self.graph.current_state = state_id
            
            # Check if goal reached
            for elem in state.elements:
                goal_words = goal.lower().split()
                all_match = all(
                    word in elem.get("text", "").lower()
                    for word in goal_words[-2:]  # Check last 2 words
                )
                if all_match:
                    print(f"Goal potentially reached: found '{elem.get('text')}'")
            
            # Select action
            action_elem = self.select_action(state, goal)
            if action_elem is None:
                print("No suitable action found")
                await asyncio.sleep(1.0)
                continue
            
            action_text = action_elem.get("text", "")
            print(f"Selected action: '{action_text}' (EFE: {self.compute_efe(state, action_text, goal):.3f})")
            
            # Execute
            await self.click_element(action_elem)
            
            # Wait for UI to update
            await asyncio.sleep(0.5)
        
        return False


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Active Inference Menu Navigator")
    parser.add_argument("--goal", default="Start Game", help="Navigation goal")
    parser.add_argument("--vlm", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="llava", help="VLM model name")
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    
    cfg = NavigatorConfig(
        vlm_type=args.vlm,
        vlm_model=args.model,
    )
    
    navigator = ActiveInferenceNavigator(cfg)
    
    success = await navigator.navigate_to_goal(args.goal, args.max_steps)
    print(f"Navigation {'succeeded' if success else 'incomplete'}")


if __name__ == "__main__":
    asyncio.run(main())
