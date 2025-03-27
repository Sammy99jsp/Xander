import argparse

import torch
from dqn import train
from rl.env import Duel, parse_config


def main(args: argparse.Namespace) -> None:
    config = parse_config(args.config)
    env = Duel(config)
    train(env, config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    
    p.add_argument("--config", type=str, required=True)

    args = p.parse_args()
    main(args)