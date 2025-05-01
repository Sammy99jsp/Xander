import os
import sys
from typing import Any, Optional

import numpy as np
import wandb

from wandb.sdk.wandb_run import Run
import gymnasium as gym
import torch
import tqdm # type: ignore

from RL.env.duel import XanderDuelEnvConfig, XanderDuelEnv
from RL.algorithms._types import Agent
from RL.algorithms import ALGORITHMS

def count(lst: list[bool] | bool) -> int:
    """
    Count the number of True values in a list of booleans.
    """
    if isinstance(lst, bool):
        return int(lst)
    else:
        return sum(int(x) for x in lst)
    
def maybe_sum(lst: list[int] | int) -> int:
    """
    Sum the values in a list of floats.
    """
    return np.sum(lst)
    
type MaybeList[T] = list[T] | T

def maybe_get[T](lst: MaybeList[dict[str, Any]], key: str, default: T) -> list[T]:
    if isinstance(lst, dict):
        return [lst.get(key, default)]
    else:
        return [x.get(key, default) for x in lst]

def test(agent: Agent[Any, gym.Env], run: Run, steps: int,):
    """
    Test the agent in the environment.
    """

    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    hp_at_end: list[float] = []
    wins = 0
    losses = 0
    damage_dealt = 0
    illegal_actions = 0

    with torch.no_grad():
        info: MaybeList[dict[str, Any]]
        tmp = agent.env.reset()
        if agent.num_envs > 1:
            state = tmp
            info: MaybeList[dict[str, Any]] = [{} for _ in range(agent.num_envs)]
        else:
            state, info = tmp

        episode_length = 0
        episode_reward = 0.0
        for i in range(steps):
            action = agent.predict(state)

            state, reward, terminated, truncated, info = agent.env.step(action)
            done = terminated | truncated

            episode_length += 1
            episode_reward += reward

            # TODO: Collect semantic metrics
            illegal_actions += int(info.get("illegal", False))

            damage_dealt += info.get("damage", 0)

            wins += int(info.get("won", False))
            losses += int(info.get("lost", False))
            
            if "hp_left" in info.keys():
                hp_at_end.append(info["hp_left"])
                
            if done:
                state, info = agent.env.reset()
                
                episode_lengths.append(episode_length)
                episode_length = 0
                episode_rewards.append(episode_reward)
                episode_reward = 0


    if len(episode_lengths) > 0:
        run.log({
            "test/episode_lengths_mean": np.mean(episode_lengths),
            "test/episode_rewards_mean": np.mean(episode_rewards),
            "test/wins": wins / len(episode_lengths),
            "test/losses": losses / len(episode_lengths),
            "test/hp_left_mean": np.mean(hp_at_end),
        })
    
    run.log({
        "test/illegal_actions": illegal_actions / steps,
        "test/damage_dealt": damage_dealt,
    })



def train(cfg: XanderDuelEnvConfig):
    """Train and test the agent."""

    env = XanderDuelEnv(cfg)
    algo = ALGORITHMS[env.learner.algorithm]
    
    if cfg.wandb.sync_tensorboard: # type: ignore
        # Sync the wandb run with tensorboard
        wandb.tensorboard.patch(root_logdir=cfg.training.save_path) # type: ignore
    
    # Log scnario to W&B
    cfg_dict = cfg.model_dump()
    cfg_dict.pop("wandb")
    wandb_config = cfg.wandb.model_dump()
    wandb_config["config"] = cfg_dict

    print(wandb_config)

    run = wandb.init(
        **wandb_config
    )

    agent: Agent[Any, gym.Env] = algo(env, env.learner.hyperparameters, run)
    
    if cfg.training.max_steps == 0:
        # Train 'indefinitely'
        cfg.training.max_steps = 1_000_000_000_000

    # Make the save directory
    os.makedirs(cfg.training.save_path, exist_ok=True)

    # Test the thingy at the start.
    test(agent, run=run, steps=cfg.training.test_steps)

    # Stable Baselines3 is annoying and doesn't allow
    # for multiple .train() calls
    if not agent.is_training_divisible:
        agent.train(cfg.training.max_steps)
        agent.save(os.path.join(cfg.training.save_path, "final.pt"))
        run.finish()
        sys.exit(0)

    total_steps = 0

    pbar = tqdm.tqdm(
        range(0, cfg.training.max_steps // cfg.training.test_every),
        unit_scale=cfg.training.test_every,
    )
    
    for _ in pbar:
        print(f"Training for {cfg.training.test_every}")
        agent.train(cfg.training.test_every)
        total_steps += cfg.training.test_every

        print(f"Testing for {cfg.training.test_steps}")
        test(agent, run=run, steps=cfg.training.test_steps)

        print("Checkpointing...")
        agent.save(os.path.join(cfg.training.save_path, f"{total_steps}.pt"))

    print(f"Final test for {cfg.training.test_steps}")
    test(agent, run=run, steps=cfg.training.test_steps)

    print("Saving final model")
    agent.save(os.path.join(cfg.training.save_path, "final.pt"))

    print("Done!")

