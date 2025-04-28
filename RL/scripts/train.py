import os
import sys
from typing import Any, Optional

import numpy as np
import wandb

from wandb.sdk.wandb_run import Run
import gymnasium as gym
import torch
import tqdm

from RL.env2.duel import XanderDuelEnvConfig, XanderDuelEnv
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

    episode_lengths = []
    rewards = np.zeros(steps * agent.num_envs, dtype=np.float32)
    wins = 0
    losses = 0
    damage_dealt = 0
    illegal_actions = 0
    hp_at_end = []

    with torch.no_grad():
        info: MaybeList[dict[str, Any]]
        tmp = agent.env.reset()
        if agent.num_envs > 1:
            state = tmp
            info: MaybeList[dict[str, Any]] = [{} for _ in range(agent.num_envs)]
        else:
            state, info = tmp

        episode_length = 0
        for i in range(steps):
            action = agent.predict(state)

            if agent.num_envs > 1:
                state, reward, done, info = agent.env.step(action)
            else:
                state, reward, terminated, truncated, info = agent.env.step(action)
                done = terminated | truncated

            episode_length += 1
            rewards[i*agent.num_envs:(i+1)*agent.num_envs] = reward

            # TODO: Collect semantic metrics
            illegal_action: list[bool] = maybe_get(info, "illegal", False)
            illegal_actions += count(illegal_action)

            damage = maybe_get(info, "damage", 0)
            damage_dealt += maybe_sum(damage)

            if np.any(done):
                if agent.num_envs == 1:
                    state, info = agent.env.reset()
                
                episode_lengths.append(episode_length)

                win = maybe_get(info, "won", False)
                wins += count(win)
                
                loss = maybe_get(info, "lost", False)
                losses += count(loss)

                hp_left = maybe_get(info, "hp_left", 0)
                hp_at_end.extend(hp_left)


    run.log({
        "test/episode_length": np.mean(episode_lengths) if len(episode_lengths) > 0 else None,
        "test/wins": wins / len(episode_lengths) if len(episode_lengths) > 0 else None,
        "test/losses": losses / len(episode_lengths) if len(episode_lengths) > 0 else None,
        "test/illegal_actions": illegal_actions / steps,
        "test/damage_dealt": damage_dealt,
        "test/reward_mean": np.mean(rewards),
        "test/hp_left_mean": np.mean(hp_at_end) if len(episode_lengths) > 0 else None,
    })

if __name__ == "__main__":
    # Test!

    with open(sys.argv[1], "r") as f:
        cfg = XanderDuelEnvConfig.model_validate_json(f.read())

    env = XanderDuelEnv(cfg)
    algo = ALGORITHMS[env.learner.algorithm]
    
    if cfg.wandb.sync_tensorboard:
        # Sync the wandb run with tensorboard
        wandb.tensorboard.patch(root_logdir=cfg.training.save_path)
    
    run = wandb.init(
        **cfg.wandb.model_dump()
    )


    agent: Agent[Any, gym.Env] = algo(env, env.learner.hyperparameters, run)
    
    if cfg.training.max_steps == 0:
        # Train 'indefinitely'
        cfg.training.max_steps = 1_000_000_000_000

    # Make the save directory
    os.makedirs(cfg.training.save_path, exist_ok=True)

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
        test(agent, run=run, steps=cfg.training.test_every)

        print("Checkpointing...")
        agent.save(os.path.join(cfg.training.save_path, f"{total_steps}.pt"))

    print(f"Final test for {cfg.training.test_steps}")
    test(agent, run=run, steps=cfg.training.test_steps)

    print("Saving final model")
    agent.save(os.path.join(cfg.training.save_path, "final.pt"))

    print("Done!")

