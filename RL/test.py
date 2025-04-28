import sys
from RL.env2.util import XanderEnvConfig, actions_available
from RL.models import ActionHeads, map_action_block
from xander.engine.actors import Stats
from xander.engine.combat import Combat
from xander.engine.combat.arena import Simple
from RL.algorithms.rainbow.network import Network as RainbowNetwork
import torch


cfg: XanderEnvConfig
combat: Combat
with open(sys.argv[1], "r") as f:
    cfg = XanderEnvConfig.model_validate_json(f.read())
    combat = cfg.build()


print(cfg)
a = actions_available(combat, cfg, "RAX")
print(a)

# actions_available(combat, )
# print(attacks)

# a = ActionHeads(128, (8, 8), attacks)
# print(map_action_block(a(torch.rand(8, 128)), lambda x: x.shape))


# support = torch.linspace(-20, 20, 51)
# net = RainbowNetwork(8, 8 + 8 + 1, 51, support.to("cuda:0")).to("cuda:0")
# dqn = net(torch.tensor([0, 1, 2, 3, 0, 2, 3.0, 1.0], dtype=torch.float32).to("cuda:0"))
# print("Value:", dqn)