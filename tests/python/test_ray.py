import argparse
from ray.rllib.algorithms.ppo import PPOConfig

from tests.python.rl.env import Duel, parse_config

def main(args: argparse.Namespace) -> None:
    config: PPOConfig = (
        PPOConfig()
            .environment(Duel, env_config=parse_config(args.config))
    )

    algo = config.build()
    print(algo.train())


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    
    p.add_argument("--config", type=str, required=True)

    args = p.parse_args()
    main(args)