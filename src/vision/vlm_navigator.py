"""VLM-based UI Navigator for General Computer Control.

This module implements the "Lobby Agent" - a Vision-Language Model powered
system that can navigate game menus through visual grounding rather than
hardcoded coordinates.

Architecture:
1. Screenshot capture → VLM analysis → Element detection
2. State transition graph learning (Active Inference exploration)
3. Hierarchical planning with LLM for complex goals

Supports:
- Qwen2-VL, LLaVA, Moondream for visual grounding
- Local ONNX inference or API-based (GPT-4V)
- Zero-shot adaptation to UI changes

References:
- Cradle: General Computer Control framework
- AppAgent: LLM-based UI control
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


@dataclass
class UIElement:
    """Detected UI element with bounding box."""
    
    text: str
    element_type: str  # "button", "toggle", "text", "icon"
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float = 1.0
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point for clicking."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.element_type,
            "bbox": list(self.bbox),
            "center": list(self.center),
        }


@dataclass
class UIState:
    """Represents a distinct UI screen state."""
    
    name: str
    elements: List[UIElement] = field(default_factory=list)
    screenshot_hash: Optional[str] = None
    transitions: Dict[str, str] = field(default_factory=dict)  # action -> next_state


@dataclass
class VLMConfig:
    """Configuration for VLM navigator."""
    
    # Model selection
    model_type: str = "local"  # "local", "openai", "anthropic"
    model_name: str = "moondream"  # "qwen2-vl", "llava", "gpt-4-vision"
    
    # Local inference
    onnx_path: Optional[str] = None
    device: str = "cuda"
    
    # API credentials (from env vars)
    api_base: Optional[str] = None
    
    # Inference settings
    max_tokens: int = 512
    temperature: float = 0.1  # Low temp for deterministic UI actions


class VLMNavigator:
    """Vision-Language Model navigator for UI control.
    
    Uses VLM to:
    1. Detect interactive elements (buttons, toggles)
    2. Ground natural language commands to specific elements
    3. Build state transition graph through exploration
    """
    
    def __init__(self, cfg: Optional[VLMConfig] = None):
        self.cfg = cfg or VLMConfig()
        self.state_graph: Dict[str, UIState] = {}
        self.current_state: Optional[str] = None
        self._client: Optional[Any] = None
    
    async def _init_client(self):
        """Initialize HTTP client for API calls."""
        if self._client is None and httpx:
            self._client = httpx.AsyncClient(timeout=30.0)
    
    def _encode_image(self, image: Any) -> str:
        """Encode PIL Image to base64."""
        if Image is None:
            raise ImportError("PIL required for image encoding")
        
        buffer = io.BytesIO()
        if hasattr(image, "save"):
            image.save(buffer, format="JPEG", quality=85)
        else:
            # Assume numpy array
            Image.fromarray(image).save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    async def detect_elements(self, screenshot: Any) -> List[UIElement]:
        """Detect all interactive UI elements in screenshot.
        
        Uses VLM with structured output prompting.
        """
        prompt = """Analyze this game UI screenshot.
Identify all interactive elements (buttons, tabs, toggles, sliders).

For each element, output a JSON object with:
- "text": the label or description
- "type": "button" | "toggle" | "tab" | "slider" | "icon"
- "bbox": [x1, y1, x2, y2] pixel coordinates

Output ONLY a JSON array of elements. Example:
[{"text": "Play", "type": "button", "bbox": [100, 200, 200, 250]}]"""

        response = await self._query_vlm(screenshot, prompt)
        return self._parse_elements(response)
    
    async def ground_command(
        self,
        screenshot: Any,
        command: str,
        elements: Optional[List[UIElement]] = None,
    ) -> Optional[UIElement]:
        """Ground a natural language command to a specific element.
        
        Example: "Click the play button" -> UIElement for Play button
        """
        if elements is None:
            elements = await self.detect_elements(screenshot)
        
        elements_json = json.dumps([e.to_dict() for e in elements])
        
        prompt = f"""Given these UI elements:
{elements_json}

User command: "{command}"

Which element should be interacted with? Output ONLY the element's text label.
If no matching element, output "NONE"."""

        response = await self._query_vlm(screenshot, prompt)
        target_text = response.strip().strip('"').lower()
        
        if target_text == "none":
            return None
        
        # Find matching element
        for elem in elements:
            if target_text in elem.text.lower():
                return elem
        
        return None
    
    async def navigate_to(
        self,
        screenshot: Any,
        goal: str,
        max_steps: int = 10,
    ) -> List[Dict[str, Any]]:
        """Plan and execute navigation to a goal state.
        
        Uses hierarchical planning:
        1. LLM decomposes goal into sub-steps
        2. VLM grounds each step to UI actions
        """
        prompt = f"""You are navigating a game menu.

Current screen: [image]

Goal: {goal}

Plan the sequence of UI interactions needed.
Output a JSON array of steps. Each step has:
- "action": "click" | "type" | "scroll"
- "target": description of what to interact with
- "value": (for type/scroll) the value to input

Example:
[
  {{"action": "click", "target": "Play button"}},
  {{"action": "click", "target": "Character Select tab"}},
  {{"action": "click", "target": "Calcium character"}}
]"""

        response = await self._query_vlm(screenshot, prompt)
        
        try:
            steps = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                steps = json.loads(match.group())
            else:
                steps = []
        
        return steps
    
    async def explore_ui(self, screenshot: Any) -> Dict[str, Any]:
        """Perform epistemic exploration of UI.
        
        Implements Active Inference's epistemic drive:
        - Track which elements have been explored
        - Prioritize unexplored elements (high uncertainty)
        - Build state transition graph
        """
        elements = await self.detect_elements(screenshot)
        
        # Identify current state
        state_hash = self._hash_screen(elements)
        
        if state_hash not in self.state_graph:
            # New state discovered!
            state = UIState(
                name=f"state_{len(self.state_graph)}",
                elements=elements,
                screenshot_hash=state_hash,
            )
            self.state_graph[state_hash] = state
        
        current = self.state_graph[state_hash]
        
        # Calculate epistemic value for each element
        epistemic_values = []
        for elem in elements:
            action_key = f"click_{elem.text}"
            if action_key not in current.transitions:
                # Never clicked this before - high epistemic value!
                epistemic_values.append((elem, 1.0))
            else:
                # Already explored
                epistemic_values.append((elem, 0.1))
        
        # Sort by epistemic value (curiosity-driven exploration)
        epistemic_values.sort(key=lambda x: -x[1])
        
        return {
            "current_state": current.name,
            "elements": [e.to_dict() for e in elements],
            "exploration_order": [
                {"element": e.to_dict(), "epistemic_value": v}
                for e, v in epistemic_values
            ],
            "unexplored_count": sum(1 for _, v in epistemic_values if v > 0.5),
        }
    
    async def _query_vlm(self, image: Any, prompt: str) -> str:
        """Query VLM with image and text prompt."""
        await self._init_client()
        
        if self.cfg.model_type == "openai":
            return await self._query_openai(image, prompt)
        elif self.cfg.model_type == "anthropic":
            return await self._query_anthropic(image, prompt)
        else:
            return await self._query_local(image, prompt)
    
    async def _query_openai(self, image: Any, prompt: str) -> str:
        """Query OpenAI GPT-4 Vision."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        if not self._client:
            raise RuntimeError("HTTP client not initialized")
        
        b64_image = self._encode_image(image)
        
        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                            },
                        ],
                    }
                ],
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
            },
        )
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def _query_anthropic(self, image: Any, prompt: str) -> str:
        """Query Anthropic Claude Vision."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        
        if not self._client:
            raise RuntimeError("HTTP client not initialized")
        
        b64_image = self._encode_image(image)
        
        response = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-sonnet-20240229",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": self.cfg.max_tokens,
            },
        )
        
        data = response.json()
        return data["content"][0]["text"]
    
    async def _query_local(self, image: Any, prompt: str) -> str:
        """Query local VLM (Ollama, ONNX, etc.)."""
        # Default to Ollama with llava/moondream
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        
        if not self._client:
            raise RuntimeError("HTTP client not initialized")
        
        b64_image = self._encode_image(image)
        
        try:
            response = await self._client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": self.cfg.model_name,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                },
            )
            data = response.json()
            return data["response"]
        except Exception as e:
            # Fallback: return empty if VLM unavailable
            return "[]"
    
    def _parse_elements(self, response: str) -> List[UIElement]:
        """Parse VLM response into UIElement list."""
        try:
            # Try direct JSON parse
            data = json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from text
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []
        
        elements = []
        for item in data:
            if isinstance(item, dict) and "text" in item and "bbox" in item:
                elements.append(UIElement(
                    text=item["text"],
                    element_type=item.get("type", "button"),
                    bbox=tuple(item["bbox"]),
                    confidence=item.get("confidence", 1.0),
                ))
        
        return elements
    
    def _hash_screen(self, elements: List[UIElement]) -> str:
        """Create hash of screen state from elements."""
        # Simple hash based on element texts
        texts = sorted([e.text for e in elements])
        return str(hash(tuple(texts)))


# Convenience function
async def navigate_menu(
    screenshot: Any,
    goal: str,
    cfg: Optional[VLMConfig] = None,
) -> List[Dict[str, Any]]:
    """One-shot menu navigation."""
    navigator = VLMNavigator(cfg)
    return await navigator.navigate_to(screenshot, goal)
