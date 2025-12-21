"""VideoGameBench: Real-Time Evaluation Protocol.

Implements perceptual hashing (dHash), VideoGameBench Lite,
and BALROG-style metrics for game agent evaluation.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class GameEvent(Enum):
    DEATH = auto()
    VICTORY = auto()
    LEVEL_UP = auto()
    BOSS_SPAWN = auto()
    BOSS_DEFEAT = auto()
    UPGRADE_SCREEN = auto()


@dataclass
class HashAnchor:
    event: GameEvent
    dhash: int
    threshold: int = 5
    name: str = ""
    
    def matches(self, other_hash: int) -> bool:
        return bin(self.dhash ^ other_hash).count('1') <= self.threshold


@dataclass
class EvaluationMetrics:
    survival_time_seconds: float = 0.0
    max_level_reached: int = 0
    total_xp_collected: int = 0
    xp_per_minute: float = 0.0
    deaths: int = 0
    victories: int = 0
    progress_percentage: float = 0.0


class PerceptualHasher:
    """dHash perceptual hashing for event detection."""
    
    @staticmethod
    def dhash(image: np.ndarray, hash_size: int = 8) -> int:
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        h, w = gray.shape
        new_h, new_w = hash_size, hash_size + 1
        block_h, block_w = h // new_h, w // new_w
        
        resized = np.zeros((new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                resized[i, j] = np.mean(
                    gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                )
        
        diff = resized[:, 1:] > resized[:, :-1]
        return sum(1 << i for i, bit in enumerate(diff.flatten()) if bit)
    
    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')


class EventDetector:
    def __init__(self, anchors: Optional[List[HashAnchor]] = None):
        self.anchors = anchors or []
        self.detected_events: List[Tuple[float, GameEvent]] = []
    
    def detect(self, frame: np.ndarray, timestamp: float) -> List[GameEvent]:
        frame_hash = PerceptualHasher.dhash(frame)
        detected = [a.event for a in self.anchors if a.matches(frame_hash)]
        for e in detected:
            self.detected_events.append((timestamp, e))
        return detected


class VideoGameBenchEvaluator:
    def __init__(self, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self.detector = EventDetector()
        self.episode_start: Optional[float] = None
        self.metrics = EvaluationMetrics()
        self.results: List[EvaluationMetrics] = []
    
    def start_episode(self):
        self.episode_start = time.time()
        self.metrics = EvaluationMetrics()
        self.detector.detected_events = []
    
    def process_frame(self, frame: np.ndarray, state: Optional[Dict] = None):
        ts = time.time()
        events = self.detector.detect(frame, ts)
        
        if state:
            self.metrics.max_level_reached = max(
                self.metrics.max_level_reached, state.get("level", 0)
            )
            self.metrics.total_xp_collected = state.get("xp", 0)
        
        if self.episode_start:
            self.metrics.survival_time_seconds = ts - self.episode_start
            if self.metrics.survival_time_seconds > 0:
                self.metrics.xp_per_minute = (
                    self.metrics.total_xp_collected / 
                    (self.metrics.survival_time_seconds / 60)
                )
        
        for e in events:
            if e == GameEvent.DEATH:
                self.metrics.deaths += 1
        
        return events
    
    def end_episode(self) -> EvaluationMetrics:
        self.metrics.progress_percentage = min(1.0, 
            0.6 * self.metrics.survival_time_seconds / 1800 +
            0.4 * self.metrics.max_level_reached / 50
        )
        self.results.append(self.metrics)
        return self.metrics
    
    def register_anchor(self, frame: np.ndarray, event: GameEvent, name: str = ""):
        dhash = PerceptualHasher.dhash(frame)
        self.detector.anchors.append(HashAnchor(event, dhash, name=name))
