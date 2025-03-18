import argparse

import torch
from tests.python.dqn import train
from tests.python.rl.env import Duel, parse_config


def main(args: argparse.Namespace) -> None:
    env = Duel(parse_config(args.config))
    train(env, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    
    p.add_argument("--config", type=str, required=True)

    args = p.parse_args()
    main(args)