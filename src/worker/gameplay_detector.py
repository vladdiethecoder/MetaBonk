"""Hybrid Gameplay Detector - Game-Agnostic Implementation.

Pure vision-based gameplay detection using multiple signals.
No game-specific logic - works across all games and platforms.

Usage:
    # Initialize once
    detector = HybridGameplayDetector()
    
    # Update every frame
    gameplay_started = detector.update(
        frame=current_frame_array,
        step_increment=1 if meaningful_step else 0,
        action_taken=True
    )
"""

import numpy as np
from collections import deque
from typing import Dict, Optional


class HybridGameplayDetector:
    """Multi-signal gameplay detection using pure vision.
    
    Detects gameplay start by combining three signals:
    1. Frame variance (primary)   - Menus static, gameplay dynamic
    2. Scene transitions (secondary) - Menus few, gameplay many
    3. Action effectiveness (tertiary) - Menus no effect, gameplay has effect
    
    All methods are game-agnostic and require no game-specific knowledge.
    """
    
    def __init__(
        self,
        variance_threshold: float = 2000.0,
        transition_threshold: float = 50.0,
        min_confidence: float = 0.90,
        window_size: int = 60
    ):
        """Initialize gameplay detector.
        
        Args:
            variance_threshold: Frame variance threshold for gameplay
            transition_threshold: Pixel difference for scene transition
            min_confidence: Minimum confidence for gameplay detection
            window_size: Number of frames to analyze (60 = 1 second at 60 FPS)
        """
        # Configuration
        self.variance_threshold = variance_threshold
        self.transition_threshold = transition_threshold
        self.min_confidence = min_confidence
        self.window_size = window_size
        
        # Signal 1: Frame variance
        self.frame_variances: deque = deque(maxlen=window_size)
        
        # Signal 2: Scene transitions
        self.prev_frame_mean: Optional[float] = None
        self.transitions: deque = deque(maxlen=300)  # 5 seconds at 60 FPS
        
        # Signal 3: Action-step correlation
        self.recent_steps: deque = deque(maxlen=window_size)
        self.recent_actions: deque = deque(maxlen=window_size)
        
        # State
        self.gameplay_started = False
        self.confidence = 0.0
        self.frames_processed = 0
        
        # Debug/monitoring
        self.variance_confidence = 0.0
        self.transition_confidence = 0.0
        self.correlation_confidence = 0.0
    
    def update(
        self,
        frame: np.ndarray,
        step_increment: int = 0,
        action_taken: bool = False
    ) -> bool:
        """Update detector with new frame and game state.
        
        Args:
            frame: Current frame as numpy array (any shape)
            step_increment: 1 if a meaningful step occurred, 0 otherwise
            action_taken: True if an action was executed this frame
            
        Returns:
            bool: True if gameplay has started, False if still in menus
        """
        self.frames_processed += 1
        
        # Once detected, stay detected
        if self.gameplay_started:
            return True
        
        # --- Signal 1: Frame Variance (PRIMARY) ---
        # Menus have low variance (static), gameplay has high variance (dynamic)
        try:
            variance = float(np.var(frame.astype(np.float32)))
        except Exception:
            variance = 0.0
        self.frame_variances.append(variance)
        
        if len(self.frame_variances) >= self.window_size:
            avg_variance = float(np.mean(list(self.frame_variances)))
            # Normalize to 0-1 confidence
            self.variance_confidence = min(1.0, avg_variance / self.variance_threshold)
        else:
            self.variance_confidence = 0.0
        
        # --- Signal 2: Scene Transitions (SECONDARY) ---
        # Menus have few transitions, gameplay has many
        try:
            frame_mean = float(np.mean(frame.astype(np.float32)))
        except Exception:
            frame_mean = 0.0
        
        if self.prev_frame_mean is not None:
            diff = abs(frame_mean - self.prev_frame_mean)
            is_transition = diff > self.transition_threshold
            self.transitions.append(is_transition)
            
            if len(self.transitions) >= self.window_size:
                # Calculate transitions per second
                recent_transitions = sum(list(self.transitions)[-self.window_size:])
                rate = recent_transitions / self.window_size
                # Good gameplay: 0.1-0.3 transitions per frame (at 60 FPS)
                # That's 6-18 transitions per second
                self.transition_confidence = min(1.0, rate / 0.2)
            else:
                self.transition_confidence = 0.0
        else:
            self.transition_confidence = 0.0
        
        self.prev_frame_mean = frame_mean
        
        # --- Signal 3: Action-Step Correlation (TERTIARY) ---
        # In menus, actions don't produce steps
        # In gameplay, actions lead to steps
        self.recent_steps.append(step_increment)
        self.recent_actions.append(1 if action_taken else 0)
        
        if len(self.recent_steps) >= 30:  # Need some history
            steps_sum = sum(self.recent_steps)
            actions_sum = sum(self.recent_actions)
            
            if actions_sum > 0:
                # What fraction of actions lead to steps?
                step_rate = steps_sum / actions_sum
                # In gameplay: typically 0.3-0.7 steps per action
                # In menus: typically 0-0.1 steps per action
                if step_rate > 0.15:
                    self.correlation_confidence = min(1.0, step_rate / 0.5)
                else:
                    self.correlation_confidence = 0.0
            else:
                self.correlation_confidence = 0.0
        else:
            self.correlation_confidence = 0.0
        
        # --- Combine Signals with Weights ---
        self.confidence = (
            self.variance_confidence * 0.5 +       # 50% - Primary signal
            self.transition_confidence * 0.3 +     # 30% - Secondary signal
            self.correlation_confidence * 0.2      # 20% - Tertiary signal
        )
        
        # --- Detection Logic ---
        # Need sustained high confidence (not just a spike)
        if self.confidence >= self.min_confidence:
            # Require confidence to be maintained for at least 2 seconds
            # to avoid false positives from brief animations
            if self.frames_processed > 120:  # 2 seconds at 60 FPS
                self.gameplay_started = True
        
        return self.gameplay_started
    
    def get_status(self) -> Dict:
        """Get detailed status for debugging/monitoring.
        
        Returns:
            dict: Detailed detection status
        """
        return {
            'gameplay_started': self.gameplay_started,
            'confidence': float(self.confidence),
            'frames_processed': self.frames_processed,
            'signals': {
                'variance': {
                    'confidence': float(self.variance_confidence),
                    'current': float(np.mean(list(self.frame_variances))) if self.frame_variances else 0.0,
                    'threshold': float(self.variance_threshold),
                    'weight': 0.5
                },
                'transitions': {
                    'confidence': float(self.transition_confidence),
                    'rate': float(sum(list(self.transitions)[-60:]) / 60.0) if len(self.transitions) >= 60 else 0.0,
                    'threshold': 0.2,
                    'weight': 0.3
                },
                'correlation': {
                    'confidence': float(self.correlation_confidence),
                    'recent_steps': int(sum(self.recent_steps)) if self.recent_steps else 0,
                    'recent_actions': int(sum(self.recent_actions)) if self.recent_actions else 0,
                    'ratio': float(sum(self.recent_steps) / sum(self.recent_actions)) if sum(self.recent_actions) > 0 else 0.0,
                    'weight': 0.2
                }
            }
        }
    
    def reset(self):
        """Reset detector state.
        
        Call this when starting a new episode or game.
        """
        self.frame_variances.clear()
        self.transitions.clear()
        self.recent_steps.clear()
        self.recent_actions.clear()
        
        self.prev_frame_mean = None
        self.gameplay_started = False
        self.confidence = 0.0
        self.frames_processed = 0
        
        self.variance_confidence = 0.0
        self.transition_confidence = 0.0
        self.correlation_confidence = 0.0
