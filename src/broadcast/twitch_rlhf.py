"""Twitch RLHF: Reinforcement Learning from Human Feedback via Chat.

Implements:
- Weighted voting based on subscriber status and loyalty
- Binary preference queries for trajectory selection
- Real-time reward shaping from crowd input
- Channel points integration for environmental perturbation

References:
- Neuromorphic Broadcast Engine specification
- TwitchIO library
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from twitchio.ext import commands
    from twitchio import Channel, Message, User
    HAS_TWITCHIO = True
except ImportError:
    HAS_TWITCHIO = False
    # Keep type hints importable without installing twitchio.
    Channel = Any  # type: ignore
    Message = Any  # type: ignore
    User = Any  # type: ignore
    commands = None  # type: ignore


class VoteType(Enum):
    """Types of votes/interactions."""
    
    PREFERENCE = "preference"   # A vs B choice
    AGENT_BET = "agent_bet"    # Bet on agent
    ACTION_VOTE = "action_vote" # Vote for action
    SPAWN_REQUEST = "spawn"     # Channel point spawn


@dataclass
class UserReputation:
    """Tracks user reputation for weighted voting."""
    
    user_id: str
    display_name: str
    
    # Status
    is_subscriber: bool = False
    is_vip: bool = False
    is_moderator: bool = False
    
    # Loyalty
    loyalty_points: int = 0
    total_votes: int = 0
    correct_predictions: int = 0
    
    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    def compute_weight(
        self,
        alpha: float = 0.5,  # Subscriber bonus
        beta: float = 0.2,   # Loyalty multiplier
    ) -> float:
        """Compute vote weight based on reputation.
        
        w_u = 1 + (α × IsSubscriber) + (β × log(LoyaltyPoints))
        """
        import math
        
        weight = 1.0
        
        if self.is_subscriber:
            weight += alpha
        if self.is_vip:
            weight += alpha * 0.5
        if self.is_moderator:
            weight += alpha * 0.3
        
        if self.loyalty_points > 1:
            weight += beta * math.log(self.loyalty_points)
        
        return weight


@dataclass
class VotingSession:
    """A voting session for binary preferences."""
    
    session_id: str
    question: str
    options: List[str]  # e.g., ["Attack Boss", "Retreat and Heal"]
    
    # Timing
    start_time: float = field(default_factory=time.time)
    duration: float = 10.0  # Seconds
    
    # Votes
    votes: Dict[str, int] = field(default_factory=dict)  # user_id -> option_idx
    weighted_votes: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    
    # State
    active: bool = True
    result: Optional[int] = None
    
    @property
    def time_remaining(self) -> float:
        return max(0, self.start_time + self.duration - time.time())
    
    def add_vote(
        self,
        user_id: str,
        option_idx: int,
        weight: float = 1.0,
    ):
        """Add or update a vote."""
        if not self.active:
            return
        
        # Remove old vote if exists
        if user_id in self.votes:
            old_idx = self.votes[user_id]
            self.weighted_votes[old_idx] -= weight
        
        # Add new vote
        self.votes[user_id] = option_idx
        self.weighted_votes[option_idx] += weight
    
    def finalize(self) -> int:
        """Finalize voting and return winning option."""
        self.active = False
        
        if not self.weighted_votes:
            self.result = 0
        else:
            self.result = max(self.weighted_votes, key=self.weighted_votes.get)
        
        return self.result
    
    def get_results(self) -> Dict[str, Any]:
        """Get voting results."""
        total = sum(self.weighted_votes.values())
        
        return {
            "question": self.question,
            "options": self.options,
            "votes": dict(self.weighted_votes),
            "percentages": {
                idx: (v / total * 100) if total > 0 else 0
                for idx, v in self.weighted_votes.items()
            },
            "winner": self.result,
            "total_voters": len(self.votes),
        }


@dataclass
class RLHFConfig:
    """Configuration for Twitch RLHF."""
    
    # Twitch connection
    twitch_token: str = ""
    twitch_channel: str = ""
    
    # Voting
    vote_duration: float = 10.0
    vote_cooldown: float = 60.0
    
    # Weights
    subscriber_weight_alpha: float = 0.5
    loyalty_weight_beta: float = 0.2
    
    # Reward shaping
    positive_reinforcement: float = 1.0
    negative_reinforcement: float = -0.5
    
    # Anti-spam
    vote_ratelimit: float = 1.0  # Seconds between votes per user


class CrowdWill:
    """Aggregates crowd sentiment over a sliding window."""
    
    def __init__(self, window_seconds: float = 10.0):
        self.window = window_seconds
        self.signals: List[Tuple[float, float, float]] = []  # (timestamp, value, weight)
    
    def add_signal(self, value: float, weight: float = 1.0):
        """Add a sentiment signal."""
        self.signals.append((time.time(), value, weight))
        self._prune()
    
    def _prune(self):
        """Remove old signals."""
        cutoff = time.time() - self.window
        self.signals = [(t, v, w) for t, v, w in self.signals if t > cutoff]
    
    def get_sentiment(self) -> float:
        """Get weighted average sentiment in [-1, 1]."""
        self._prune()
        
        if not self.signals:
            return 0.0
        
        total_weighted = sum(v * w for _, v, w in self.signals)
        total_weight = sum(w for _, _, w in self.signals)
        
        return total_weighted / total_weight if total_weight > 0 else 0.0
    
    def get_activity(self) -> float:
        """Get activity level (messages per second)."""
        self._prune()
        return len(self.signals) / self.window


class TwitchRLHF:
    """Main Twitch RLHF integration.
    
    Processes chat for:
    - Voting on agent actions
    - Betting on agent performance
    - Channel point spawns
    - General sentiment
    """
    
    def __init__(self, cfg: RLHFConfig):
        self.cfg = cfg
        
        # User tracking
        self.users: Dict[str, UserReputation] = {}
        
        # Voting
        self.current_vote: Optional[VotingSession] = None
        self.vote_history: List[VotingSession] = []
        self.last_vote_time: float = 0
        
        # Agent bets
        self.agent_bets: Dict[int, Dict[str, float]] = defaultdict(dict)  # agent_id -> {user_id: amount}
        
        # Sentiment
        self.crowd_will = CrowdWill(window_seconds=10.0)
        
        # Callbacks
        self.spawn_callback: Optional[Callable[[str], None]] = None
        self.reward_callback: Optional[Callable[[int, float], None]] = None
        
        # Rate limiting
        self.user_last_vote: Dict[str, float] = {}
    
    def get_or_create_user(
        self,
        user_id: str,
        display_name: str,
        is_subscriber: bool = False,
        is_vip: bool = False,
        is_moderator: bool = False,
    ) -> UserReputation:
        """Get or create user reputation."""
        if user_id not in self.users:
            self.users[user_id] = UserReputation(
                user_id=user_id,
                display_name=display_name,
                is_subscriber=is_subscriber,
                is_vip=is_vip,
                is_moderator=is_moderator,
            )
        else:
            # Update status
            user = self.users[user_id]
            user.is_subscriber = is_subscriber
            user.is_vip = is_vip
            user.is_moderator = is_moderator
            user.last_active = time.time()
        
        return self.users[user_id]
    
    def start_vote(
        self,
        question: str,
        options: List[str],
        duration: Optional[float] = None,
    ) -> Optional[VotingSession]:
        """Start a new voting session."""
        # Check cooldown
        if time.time() - self.last_vote_time < self.cfg.vote_cooldown:
            return None
        
        # End any active vote
        if self.current_vote and self.current_vote.active:
            self.current_vote.finalize()
        
        # Create new vote
        session_id = hashlib.md5(f"{question}{time.time()}".encode()).hexdigest()[:8]
        
        self.current_vote = VotingSession(
            session_id=session_id,
            question=question,
            options=options,
            duration=duration or self.cfg.vote_duration,
        )
        
        self.last_vote_time = time.time()
        
        return self.current_vote
    
    def process_chat_message(
        self,
        user_id: str,
        display_name: str,
        message: str,
        is_subscriber: bool = False,
        is_vip: bool = False,
        is_moderator: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Process a chat message for RLHF signals."""
        user = self.get_or_create_user(
            user_id, display_name, is_subscriber, is_vip, is_moderator
        )
        user.total_votes += 1
        
        weight = user.compute_weight(
            self.cfg.subscriber_weight_alpha,
            self.cfg.loyalty_weight_beta,
        )
        
        msg_lower = message.lower().strip()
        result = None
        
        # Check for vote
        if self.current_vote and self.current_vote.active:
            # Check rate limit
            if time.time() - self.user_last_vote.get(user_id, 0) < self.cfg.vote_ratelimit:
                return None
            
            # Parse vote
            if msg_lower in ("a", "1", "attack", "yes"):
                self.current_vote.add_vote(user_id, 0, weight)
                self.user_last_vote[user_id] = time.time()
                result = {"type": "vote", "option": 0}
                
            elif msg_lower in ("b", "2", "retreat", "no"):
                self.current_vote.add_vote(user_id, 1, weight)
                self.user_last_vote[user_id] = time.time()
                result = {"type": "vote", "option": 1}
        
        # Check for sentiment
        positive_words = {"pog", "poggers", "nice", "sick", "clean", "gg", "ez"}
        negative_words = {"rip", "f", "sadge", "unlucky", "bad", "throw"}
        
        sentiment = 0.0
        if any(w in msg_lower for w in positive_words):
            sentiment = 1.0
        elif any(w in msg_lower for w in negative_words):
            sentiment = -1.0
        
        if sentiment != 0:
            self.crowd_will.add_signal(sentiment, weight)
            result = result or {"type": "sentiment", "value": sentiment}
        
        # Check for agent bet (!bet 7 100)
        if msg_lower.startswith("!bet "):
            parts = msg_lower.split()
            if len(parts) >= 3:
                try:
                    agent_id = int(parts[1])
                    amount = int(parts[2])
                    self.agent_bets[agent_id][user_id] = amount
                    result = {"type": "bet", "agent": agent_id, "amount": amount}
                except ValueError:
                    pass
        
        return result
    
    def finalize_current_vote(self) -> Optional[Dict[str, Any]]:
        """Finalize the current vote and return results."""
        if not self.current_vote:
            return None
        
        winner = self.current_vote.finalize()
        results = self.current_vote.get_results()
        
        self.vote_history.append(self.current_vote)
        self.current_vote = None
        
        return results
    
    def get_crowd_reward(self) -> float:
        """Get reward signal from crowd sentiment."""
        sentiment = self.crowd_will.get_sentiment()
        
        if sentiment > 0:
            return sentiment * self.cfg.positive_reinforcement
        else:
            return sentiment * abs(self.cfg.negative_reinforcement)
    
    def get_agent_crowd_affinity(self, agent_id: int) -> float:
        """Get crowd affinity for an agent based on bets."""
        if agent_id not in self.agent_bets:
            return 0.0
        
        total_bet = sum(self.agent_bets[agent_id].values())
        all_bets = sum(
            sum(bets.values())
            for bets in self.agent_bets.values()
        )
        
        return total_bet / max(all_bets, 1)
    
    def settle_bets(self, winning_agent: int):
        """Settle bets when an agent wins."""
        for user_id, amount in self.agent_bets.get(winning_agent, {}).items():
            if user_id in self.users:
                # Award loyalty points
                self.users[user_id].loyalty_points += amount * 2
                self.users[user_id].correct_predictions += 1
        
        # Clear bets
        self.agent_bets.clear()
    
    def get_rlhf_state(self) -> Dict[str, Any]:
        """Get current RLHF state."""
        return {
            "current_vote": self.current_vote.get_results() if self.current_vote else None,
            "crowd_sentiment": self.crowd_will.get_sentiment(),
            "crowd_activity": self.crowd_will.get_activity(),
            "total_users": len(self.users),
            "active_bets": {
                agent_id: sum(bets.values())
                for agent_id, bets in self.agent_bets.items()
            },
        }


class ChannelPointHandler:
    """Handles channel point redemptions for environmental perturbation."""
    
    def __init__(self):
        self.spawn_queue: List[Tuple[str, str, float]] = []  # (user, entity, timestamp)
        
        # Redemption mappings
        self.redemptions = {
            "spawn_elite": "elite_enemy",
            "spawn_boss": "boss_enemy",
            "give_weapon": "random_weapon",
            "heal_agent": "heal",
            "spawn_swarm": "enemy_swarm",
        }
    
    def handle_redemption(
        self,
        user: str,
        redemption_id: str,
    ) -> Optional[str]:
        """Handle a channel point redemption."""
        if redemption_id in self.redemptions:
            entity = self.redemptions[redemption_id]
            self.spawn_queue.append((user, entity, time.time()))
            return entity
        return None
    
    def get_pending_spawns(self) -> List[Tuple[str, str]]:
        """Get and clear pending spawn requests."""
        # Clean old requests (> 30 seconds)
        cutoff = time.time() - 30
        self.spawn_queue = [(u, e, t) for u, e, t in self.spawn_queue if t > cutoff]
        
        pending = [(u, e) for u, e, _ in self.spawn_queue]
        self.spawn_queue.clear()
        
        return pending
