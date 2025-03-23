import torch
import argparse

from dqn import DQN
from rl.env import observation_size, parse_config

n_actions = 8
n_dirs = 3

def main(args: argparse.Namespace) -> None:
    path = args.model
    cfg  = parse_config(args.config)
    model = DQN(n_observations=observation_size(cfg), n_actions=n_actions, n_dirs=n_dirs)
    model.load_state_dict(torch.load(path))
    torch.onnx.export(model, torch.zeros(1, observation_size(cfg)), args.out, input_names=["input"], output_names=["output"])

    print(f"Model saved to {args.out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    main(p.parse_args())