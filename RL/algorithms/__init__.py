from RL.algorithms._types import Agent
from RL.algorithms.ppo import PPOAgent
from RL.algorithms.rainbow import RainbowAgent

ALGORITHMS: dict[str, type[Agent]] = {
    "rainbow": RainbowAgent,
    "ppo": PPOAgent,
}