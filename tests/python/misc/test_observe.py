from xander.engine.actors import Stats
from xander.engine.combat import Combat
from xander.engine.combat.arena import Simple


import numpy as np

combat = Combat(Simple(25, 25))
ai = combat.join(Stats.from_json("../rat.json"), "AI", (0, 0, 0))
rando1 = combat.join(Stats.from_json("../rat.json"), "Rando1", (0, 10, 0))
rando2 = combat.join(Stats.from_json("../rat.json"), "Rando2", (10, 10, 0))

space = np.array(ai.observe()).reshape((5, 5))
print(space)