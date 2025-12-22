"""Menu/gameplay classifier for MetaBonk UI state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
try:  # torchvision>=0.13
    from torchvision.models import MobileNet_V3_Small_Weights
except Exception:  # pragma: no cover
    MobileNet_V3_Small_Weights = None  # type: ignore

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow is required for menu classifier inference") from e

try:
    from src.worker.ocr import ocr_text
except Exception:  # pragma: no cover
    ocr_text = None  # type: ignore


MENU_CLASSES = ["menu", "combat", "reward", "selection"]
MENU_MODE_SET = {"menu", "reward", "selection"}


def build_menu_model(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    if pretrained:
        if MobileNet_V3_Small_Weights is not None:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
        else:  # pragma: no cover
            model = mobilenet_v3_small(pretrained=True)
    else:
        model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class MenuClassifier:
    model: nn.Module
    classes: List[str]
    device: torch.device
    threshold: float = 0.5
    transform: transforms.Compose = _default_transform()

    def predict(self, image: "Image.Image") -> Dict[str, object]:
        self.model.eval()
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
        cls = self.classes[int(idx)]
        menu_mode = bool(cls in MENU_MODE_SET and float(conf) >= float(self.threshold))
        return {
            "menu_state": cls,
            "menu_conf": float(conf),
            "menu_probs": {self.classes[i]: float(p) for i, p in enumerate(probs.tolist())},
            "menu_mode": menu_mode,
        }


def load_menu_classifier(
    weights_path: str,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> MenuClassifier:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(weights_path, map_location=dev)
    classes = MENU_CLASSES
    state = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("classes"), list):
            classes = [str(c) for c in payload["classes"]]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state = payload["state_dict"]
    model = build_menu_model(num_classes=len(classes))
    model.load_state_dict(state, strict=False)
    model.to(dev)
    return MenuClassifier(model=model, classes=classes, device=dev, threshold=threshold)


def heuristic_menu_metrics(image: "Image.Image") -> Dict[str, object]:
    """Lightweight OCR-based menu detector (fallback when no weights)."""
    if ocr_text is None:
        return {}
    try:
        from PIL import ImageOps

        img = image.convert("RGB")
        # Downscale to speed up OCR while keeping text legible.
        if img.width > 960:
            ratio = 960.0 / float(img.width)
            img = img.resize((960, max(1, int(img.height * ratio))))
        text = ocr_text(img, psm=6).lower()
        if not text:
            text = ocr_text(ImageOps.invert(img.convert("L")), psm=6).lower()
    except Exception:
        return {}
    if not text:
        return {}
    menu_keywords = [
        "play",
        "confirm",
        "settings",
        "exit",
        "map selection",
        "character selection",
        "credits",
        "continue",
        "start",
        "forest",
        "tier",
    ]
    gameplay_keywords = [
        "lvl",
        "level",
        "find the boss",
        "boss portal",
        "hp",
        "health",
        "time",
    ]
    menu_hits = sum(1 for k in menu_keywords if k in text)
    gameplay_hits = sum(1 for k in gameplay_keywords if k in text)
    if menu_hits == 0 and gameplay_hits == 0:
        return {}
    menu_mode = menu_hits >= max(1, gameplay_hits)
    conf = min(1.0, 0.5 + 0.1 * abs(menu_hits - gameplay_hits))
    state = "menu" if menu_mode else "combat"
    return {
        "menu_state": state,
        "menu_conf": float(conf),
        "menu_probs": {"menu": float(conf), "combat": float(1.0 - conf)},
        "menu_mode": bool(menu_mode),
        "menu_source": "heuristic",
    }
