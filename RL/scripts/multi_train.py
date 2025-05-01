from copy import deepcopy
import math
import multiprocessing
import os
import random
import sys
import time
from RL.env.duel import XanderDuelEnvConfig
from RL.scripts.train import train
import argparse

def config_with_seed(cfg: XanderDuelEnvConfig, i: int, seed: int) -> XanderDuelEnvConfig:
    cfg = deepcopy(cfg)
    cfg.seed = seed
    cfg.training.save_path = os.path.join(cfg.training.save_path, f"{i}")
    return cfg

def multi_train(cfg_path: str, n_proc: int):
    print(f"--- Attempting to train with {n_proc} processes! ---")

    with open(cfg_path, "r") as f:
        cfg = XanderDuelEnvConfig.model_validate_json(f.read())

    match n_proc:
        case n if n <= 0:
            print("ERROR: At least one process")
            sys.exit(1)
        case 1:
            train(cfg)
            return
        case n:
            pass
    
    cfgs = [ config_with_seed(cfg, i+1, cfg.seed + random.randint(0, 2**31 -1)) for i in range(n_proc) ]

    procs: list[multiprocessing.Process] = []
    for i, cfg in enumerate(cfgs):
        print(f"\t-- Starting worker {i+1}, seed = {cfg.seed} --")
        proc = multiprocessing.Process(target=train, args=(cfg,))
        proc.start()
        procs.append(proc)

    while any(proc.is_alive() for proc in procs):
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(("multi_train.py"))
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-j", "--procs", type=int, required=False, default=1)

    args = parser.parse_args()
    multi_train(args.config, args.procs)
    

