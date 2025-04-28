from xander.engine.combat.turn import Turn # type: ignore
from xander.engine.combat.arena import Simple # type: ignore
from xander.engine.actors import Stats # type: ignore
import xander.engine as X

def always_end(t: "Turn"):
    t.end()

ra1 = Stats.from_json("tests/rat.json")
ra2 = Stats.from_json("tests/rat.json")

def direction(s: str):
    match s:
        case "n":
            return (0, 5, 0)
        case "ne":
            return (5, 5, 0)
        case "e":
            return (5, 0, 0)
        case "se":
            return (5, -5, 0)
        case "s":
            return (0, -5, 0)
        case "sw":
            return (-5, -5, 0)
        case "w":
            return (-5, 0, 0)
        case "nw":
            return (-5, 5, 0)

def action_type(stats: Stats, t: "Turn", s: str):
    r_ty = s[0:1]

    match r_ty:
        case "A":
            i = int(s[1:2])
            return t.attack(stats.actions[i].as_attack(), eval(s[2:])), False # type: ignore
        case "M":
            return t.move(eval(s[1:])), False
        case "E":
            return t.end(), True
        

AGENTS: dict[Stats, X.combat.Combatant] = {}

def interactive(stats: Stats):
    def __inner(t: "Turn"):
        combatant = AGENTS[stats]
        while True:
            print(f"{combatant.name} {combatant.position} -- {combatant.stats.hp}/{combatant.stats.max_hp} HP")
            res, br = action_type(combatant.stats, t, input("> "))
            print(res)
            if br: break
        print()

    return __inner
        

combat = X.combat.Combat(Simple(40, 40))
__ra1 = combat.join(ra1, "RA1", (0, 0, 0), interactive(ra1))
__ra2 = combat.join(ra2, "RA2", (0, 10, 0), interactive(ra2))

AGENTS = {
    ra1: __ra1,
    ra2: __ra2
}

is_any_dead = lambda: any([ra1.dead, ra2.dead])

while not is_any_dead():
    combat.step()