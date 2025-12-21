"""Betting Ecosystem: Virtual Currency and Predictions.

Salty Bet-style betting for spectator engagement:
- Virtual currency (Bonk Bucks)
- Binary predictions
- Live odds updates
- Whale leaderboard

References:
- Salty Bet model
- Prediction markets
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Bet:
    """A single bet."""
    
    bet_id: str
    user_id: str
    username: str
    
    # Bet details
    prediction: str  # "success" or "failure"
    amount: int
    
    # Odds at time of bet
    odds: float
    
    # Result
    resolved: bool = False
    won: bool = False
    payout: int = 0


@dataclass
class BettingRound:
    """A single betting round."""
    
    round_id: int
    
    # Question
    question: str
    options: List[str]  # e.g., ["success", "failure"]
    
    # State
    status: str = "open"  # "open", "locked", "resolved"
    start_time: float = field(default_factory=time.time)
    lock_time: Optional[float] = None
    
    # Bets
    bets: List[Bet] = field(default_factory=list)
    pool_by_option: Dict[str, int] = field(default_factory=dict)
    
    # Result
    winning_option: Optional[str] = None
    
    @property
    def total_pool(self) -> int:
        return sum(self.pool_by_option.values())
    
    def get_odds(self) -> Dict[str, float]:
        """Get current odds for each option."""
        total = self.total_pool
        if total == 0:
            return {opt: 2.0 for opt in self.options}
        
        odds = {}
        for opt in self.options:
            opt_pool = self.pool_by_option.get(opt, 0)
            if opt_pool > 0:
                odds[opt] = total / opt_pool
            else:
                odds[opt] = float('inf')  # No bets on this option
        
        return odds


@dataclass
class UserWallet:
    """A user's virtual currency wallet."""
    
    user_id: str
    username: str
    
    # Currency
    balance: int = 1000  # Starting balance
    
    # Stats
    total_won: int = 0
    total_lost: int = 0
    bets_placed: int = 0
    bets_won: int = 0
    
    # Rankings
    all_time_high: int = 1000
    
    @property
    def win_rate(self) -> float:
        if self.bets_placed == 0:
            return 0.0
        return self.bets_won / self.bets_placed


@dataclass
class BettingConfig:
    """Configuration for the betting system."""
    
    # Currency
    starting_balance: int = 1000
    min_bet: int = 10
    max_bet: int = 10000
    
    # Timing
    betting_window_s: float = 30.0
    
    # Odds
    min_odds: float = 1.1
    max_odds: float = 100.0
    
    # Welfare
    daily_allowance: int = 500
    bankrupt_bailout: int = 100


class BettingEcosystem:
    """Complete betting system for spectator engagement."""
    
    def __init__(self, cfg: Optional[BettingConfig] = None):
        self.cfg = cfg or BettingConfig()
        
        # Users
        self.wallets: Dict[str, UserWallet] = {}
        
        # Current round
        self.current_round: Optional[BettingRound] = None
        self.round_counter = 0
        
        # History
        self.completed_rounds: List[BettingRound] = []
    
    # ═══════════════════════════════════════════════════════════
    # WALLET MANAGEMENT
    # ═══════════════════════════════════════════════════════════
    
    def get_or_create_wallet(self, user_id: str, username: str) -> UserWallet:
        """Get or create a user wallet."""
        if user_id not in self.wallets:
            self.wallets[user_id] = UserWallet(
                user_id=user_id,
                username=username,
                balance=self.cfg.starting_balance,
            )
        return self.wallets[user_id]
    
    def get_balance(self, user_id: str) -> int:
        """Get user balance."""
        if user_id in self.wallets:
            return self.wallets[user_id].balance
        return self.cfg.starting_balance
    
    def claim_daily(self, user_id: str, username: str) -> int:
        """Claim daily allowance."""
        wallet = self.get_or_create_wallet(user_id, username)
        wallet.balance += self.cfg.daily_allowance
        return self.cfg.daily_allowance
    
    # ═══════════════════════════════════════════════════════════
    # BETTING ROUNDS
    # ═══════════════════════════════════════════════════════════
    
    def start_round(
        self,
        question: str,
        options: List[str] = None,
    ) -> BettingRound:
        """Start a new betting round."""
        if self.current_round and self.current_round.status == "open":
            self.lock_round()
        
        self.round_counter += 1
        options = options or ["success", "failure"]
        
        self.current_round = BettingRound(
            round_id=self.round_counter,
            question=question,
            options=options,
            pool_by_option={opt: 0 for opt in options},
        )
        
        return self.current_round
    
    def lock_round(self):
        """Lock bets for current round (no more betting)."""
        if self.current_round:
            self.current_round.status = "locked"
            self.current_round.lock_time = time.time()
    
    def place_bet(
        self,
        user_id: str,
        username: str,
        prediction: str,
        amount: int,
    ) -> Tuple[bool, str]:
        """Place a bet."""
        # Validate round
        if self.current_round is None:
            return False, "No active betting round"
        
        if self.current_round.status != "open":
            return False, "Betting is closed"
        
        if prediction not in self.current_round.options:
            return False, f"Invalid option. Choose: {self.current_round.options}"
        
        # Validate amount
        if amount < self.cfg.min_bet:
            return False, f"Minimum bet is {self.cfg.min_bet}"
        
        if amount > self.cfg.max_bet:
            return False, f"Maximum bet is {self.cfg.max_bet}"
        
        # Check balance
        wallet = self.get_or_create_wallet(user_id, username)
        if wallet.balance < amount:
            return False, f"Insufficient balance ({wallet.balance})"
        
        # Create bet
        bet = Bet(
            bet_id=f"{self.current_round.round_id}_{user_id}",
            user_id=user_id,
            username=username,
            prediction=prediction,
            amount=amount,
            odds=self.current_round.get_odds()[prediction],
        )
        
        # Deduct from wallet
        wallet.balance -= amount
        wallet.bets_placed += 1
        
        # Add to round
        self.current_round.bets.append(bet)
        self.current_round.pool_by_option[prediction] += amount
        
        return True, f"Bet placed: {amount} on {prediction}"
    
    def resolve_round(self, winning_option: str) -> List[Tuple[str, int]]:
        """Resolve the current round and distribute payouts."""
        if self.current_round is None:
            return []
        
        round_ = self.current_round
        round_.status = "resolved"
        round_.winning_option = winning_option
        
        payouts = []
        
        # Calculate payouts
        for bet in round_.bets:
            bet.resolved = True
            
            if bet.prediction == winning_option:
                # Winner!
                bet.won = True
                odds = round_.get_odds()[winning_option]
                bet.payout = int(bet.amount * min(odds, self.cfg.max_odds))
                
                # Credit wallet
                wallet = self.wallets.get(bet.user_id)
                if wallet:
                    wallet.balance += bet.payout
                    wallet.total_won += bet.payout - bet.amount
                    wallet.bets_won += 1
                    wallet.all_time_high = max(wallet.all_time_high, wallet.balance)
                
                payouts.append((bet.username, bet.payout))
            else:
                # Loser
                bet.won = False
                wallet = self.wallets.get(bet.user_id)
                if wallet:
                    wallet.total_lost += bet.amount
                    
                    # Bailout for bankrupt users
                    if wallet.balance < self.cfg.bankrupt_bailout:
                        wallet.balance = self.cfg.bankrupt_bailout
        
        # Archive
        self.completed_rounds.append(round_)
        self.current_round = None
        
        return payouts
    
    # ═══════════════════════════════════════════════════════════
    # LEADERBOARDS
    # ═══════════════════════════════════════════════════════════
    
    def get_whale_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get richest users leaderboard."""
        sorted_wallets = sorted(
            self.wallets.values(),
            key=lambda w: w.balance,
            reverse=True,
        )[:top_n]
        
        return [
            {
                "rank": i + 1,
                "username": w.username,
                "balance": w.balance,
                "win_rate": w.win_rate,
                "all_time_high": w.all_time_high,
            }
            for i, w in enumerate(sorted_wallets)
        ]
    
    def get_win_rate_leaderboard(self, min_bets: int = 10, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get best win rate leaderboard."""
        qualified = [w for w in self.wallets.values() if w.bets_placed >= min_bets]
        
        sorted_wallets = sorted(
            qualified,
            key=lambda w: w.win_rate,
            reverse=True,
        )[:top_n]
        
        return [
            {
                "rank": i + 1,
                "username": w.username,
                "win_rate": f"{w.win_rate * 100:.1f}%",
                "bets": w.bets_placed,
            }
            for i, w in enumerate(sorted_wallets)
        ]
    
    # ═══════════════════════════════════════════════════════════
    # SENTIMENT
    # ═══════════════════════════════════════════════════════════
    
    def get_believer_ratio(self) -> Dict[str, Any]:
        """Get the % of chat betting on success vs failure."""
        if self.current_round is None:
            return {"ratio": 0.5, "believers": 0, "doubters": 0}
        
        believers = self.current_round.pool_by_option.get("success", 0)
        doubters = self.current_round.pool_by_option.get("failure", 0)
        total = believers + doubters
        
        if total == 0:
            ratio = 0.5
        else:
            ratio = believers / total
        
        return {
            "ratio": ratio,
            "believers": believers,
            "doubters": doubters,
            "color": "#00FF88" if ratio > 0.5 else "#FF4444",
        }
    
    def get_round_status(self) -> Dict[str, Any]:
        """Get current round status for UI."""
        if self.current_round is None:
            return {"active": False}
        
        round_ = self.current_round
        odds = round_.get_odds()
        
        return {
            "active": True,
            "round_id": round_.round_id,
            "question": round_.question,
            "status": round_.status,
            "total_pool": round_.total_pool,
            "bets_count": len(round_.bets),
            "options": [
                {
                    "name": opt,
                    "pool": round_.pool_by_option.get(opt, 0),
                    "odds": f"{min(odds.get(opt, 0), 99.9):.1f}x",
                }
                for opt in round_.options
            ],
            "believer_ratio": self.get_believer_ratio(),
        }


class Bounty:
    """A viewer-set bounty/prediction for future milestones."""
    
    def __init__(
        self,
        bounty_id: str,
        creator: str,
        target_value: float,
        target_metric: str,  # e.g., "survival_time", "score"
        reward_pool: int,
    ):
        self.bounty_id = bounty_id
        self.creator = creator
        self.target_value = target_value
        self.target_metric = target_metric
        self.reward_pool = reward_pool
        
        # Contributions
        self.contributors: Dict[str, int] = {creator: reward_pool}
        
        # State
        self.active = True
        self.claimed_by: Optional[str] = None
    
    def contribute(self, user: str, amount: int):
        """Add to the bounty pool."""
        if user in self.contributors:
            self.contributors[user] += amount
        else:
            self.contributors[user] = amount
        self.reward_pool += amount
    
    def check_and_claim(self, current_value: float, run_id: int) -> bool:
        """Check if bounty is hit."""
        if current_value >= self.target_value:
            self.active = False
            self.claimed_by = f"Run_{run_id}"
            return True
        return False


class BountyBoard:
    """Manages viewer-set bounties."""
    
    def __init__(self):
        self.bounties: List[Bounty] = []
        self.bounty_counter = 0
    
    def create_bounty(
        self,
        creator: str,
        target_metric: str,
        target_value: float,
        initial_pool: int,
    ) -> Bounty:
        """Create a new bounty."""
        self.bounty_counter += 1
        
        bounty = Bounty(
            bounty_id=f"bounty_{self.bounty_counter}",
            creator=creator,
            target_value=target_value,
            target_metric=target_metric,
            reward_pool=initial_pool,
        )
        
        self.bounties.append(bounty)
        return bounty
    
    def check_bounties(
        self,
        metrics: Dict[str, float],
        run_id: int,
    ) -> List[Bounty]:
        """Check if any bounties are hit."""
        claimed = []
        
        for bounty in self.bounties:
            if not bounty.active:
                continue
            
            value = metrics.get(bounty.target_metric, 0)
            if bounty.check_and_claim(value, run_id):
                claimed.append(bounty)
        
        return claimed
    
    def get_active_bounties(self) -> List[Dict[str, Any]]:
        """Get active bounties for UI."""
        return [
            {
                "id": b.bounty_id,
                "creator": b.creator,
                "target": f"{b.target_metric} >= {b.target_value}",
                "pool": b.reward_pool,
                "contributors": len(b.contributors),
            }
            for b in self.bounties if b.active
        ]
