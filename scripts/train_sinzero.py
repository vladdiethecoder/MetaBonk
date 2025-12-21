#!/usr/bin/env python3
"""SinZero prototype training loop (toy environment).

This script exercises the SinZero architecture on a lightweight vectorized
environment to validate:
  - Sin-specific intrinsic rewards (Lust=RND, Sloth action penalty, etc.)
  - League leader selection + Envy distillation
  - (Optional) MoE gate instantiation

It is not a replacement for MegaBonk integration; it is a feasibility
prototype that runs anywhere with PyTorch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.sins import Sin
from src.learner.sinzero import MoEPolicy, SinAgent, SinLeague


@dataclass
class ToyEnv:
    n_envs: int
    state_dim: int
    action_dim: int
    device: str

    def __post_init__(self):
        self.states = torch.zeros(self.n_envs, self.state_dim, device=self.device)
        self.dones = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        self.ep_returns = torch.zeros(self.n_envs, device=self.device)
        self.ep_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        self.states[env_ids] = torch.randn(len(env_ids), self.state_dim, device=self.device) * 0.1
        self.dones[env_ids] = False
        self.ep_returns[env_ids] = 0.0
        self.ep_lengths[env_ids] = 0
        return self.states[env_ids]

    def step(self, actions: torch.Tensor):
        # Simple stochastic linear-ish dynamics
        noise = torch.randn_like(self.states) * 0.02
        if actions.shape[1] < self.state_dim:
            actions = torch.nn.functional.pad(actions, (0, self.state_dim - actions.shape[1]))
        else:
            actions = actions[:, : self.state_dim]
        self.states = self.states * 0.995 + actions * 0.05 + noise

        # Extrinsic reward: stay near origin (survival proxy)
        rewards = -self.states[:, :2].pow(2).sum(dim=1).sqrt()

        self.ep_lengths += 1
        self.ep_returns += rewards

        # Random terminations with length bias
        term_prob = torch.clamp(self.ep_lengths.float() / 500.0, 0, 0.1)
        terminated = torch.rand(self.n_envs, device=self.device) < term_prob
        self.dones = terminated

        return self.states.clone(), rewards.clone(), self.dones.clone()

    def pop_completed(self):
        mask = self.dones
        if not mask.any():
            return []
        returns = self.ep_returns[mask].tolist()
        env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
        self.reset(env_ids)
        return returns


def main() -> int:
    p = argparse.ArgumentParser(description="SinZero toy prototype")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--envs-per-sin", type=int, default=8)
    p.add_argument("--state-dim", type=int, default=64)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--moe-every", type=int, default=5000)
    args = p.parse_args()

    league = SinLeague()
    envs: Dict[Sin, ToyEnv] = {}
    buffers: Dict[Sin, Dict[str, List[torch.Tensor]]] = {}

    for sin in Sin:
        agent = SinAgent(
            sin=sin,
            obs_dim=args.state_dim,
            action_dim=args.action_dim,
            device=args.device,
        )
        league.add_agent(agent)
        envs[sin] = ToyEnv(args.envs_per_sin, args.state_dim, args.action_dim, args.device)
        envs[sin].reset()
        buffers[sin] = {"obs": [], "act": [], "rew": [], "done": []}

    moe: MoEPolicy | None = None

    for step in range(1, args.steps + 1):
        for sin, agent in league.agents.items():
            env = envs[sin]
            obs = env.states

            with torch.no_grad():
                cont_dist, _, values = agent.learner.net.dist_and_value(obs)
                actions = cont_dist.sample()

            next_obs, extr_rew, dones = env.step(actions)
            intr_rew = agent.intrinsic_reward(obs, actions)
            total_rew = extr_rew + agent.bias.intrinsic_coef * intr_rew

            # Update RND predictor online for Lust.
            agent.update_rnd(obs)

            # Store rollout.
            buffers[sin]["obs"].append(obs.detach())
            buffers[sin]["act"].append(actions.detach())
            buffers[sin]["rew"].append(total_rew.detach())
            buffers[sin]["done"].append(dones.detach())

            # PPO update on fixed horizon.
            if len(buffers[sin]["obs"]) >= agent.learner.cfg.time_horizon:
                obs_b = torch.cat(buffers[sin]["obs"], dim=0)
                act_b = torch.cat(buffers[sin]["act"], dim=0)
                rew_b = torch.cat(buffers[sin]["rew"], dim=0)
                done_b = torch.cat(buffers[sin]["done"], dim=0)

                # No discrete actions in toy env.
                disc_b = torch.zeros(obs_b.shape[0], 0, dtype=torch.long, device=args.device)

                agent.learner.update_from_rollout(
                    obs=obs_b,
                    actions_cont=act_b,
                    actions_disc=disc_b,
                    rewards=rew_b,
                    dones=done_b,
                )
                buffers[sin] = {"obs": [], "act": [], "rew": [], "done": []}

            # Episode accounting for league.
            completed = env.pop_completed()
            for r in completed:
                agent.update_return_ema(r)

        # Envy distillation every 256 steps.
        if step % 256 == 0:
            all_obs = torch.cat([env.states for env in envs.values()], dim=0)
            league.envy_distill(all_obs)

        if step % 1000 == 0:
            lead = league.leader()
            print(
                f"[sinzero] step={step} leader={lead.sin.value if lead else 'none'} "
                f"ema={lead.episode_return_ema if lead else 0:.2f}"
            )

        if step % args.moe_every == 0:
            moe = MoEPolicy(league, device=args.device)
            print(f"[sinzero] MoE gate initialized at step {step}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
