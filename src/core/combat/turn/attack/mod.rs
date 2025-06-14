pub mod melee;
pub mod ranged;
pub mod roll;

use std::sync::{Arc, Weak};

pub use melee::MeleeAttackAction;
pub use ranged::RangedAttackAction;
use roll::{AttackRoll, Criticality};
use serde::Deserialize;

use crate::{
    core::{
        combat::Combatant,
        dice::{DExpr, D20},
        geom::{self, Coord, P3},
        stats::damage::{Damage, DamageCause, DamagePart, DamageTypeMeta},
    },
    utils::legality::{self, Legality},
};

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "attack_type")]
pub enum AttackAction {
    Melee(MeleeAttackAction),
    Ranged(RangedAttackAction),
}

impl AttackAction {
    pub fn name(&self) -> &str {
        match self {
            AttackAction::Melee(melee_attack_action) => &melee_attack_action.name,
            AttackAction::Ranged(ranged_attack_action) => &ranged_attack_action.name,
        }
    }

    pub fn damage(&self) -> &[(DExpr, &'static DamageTypeMeta)] {
        match self {
            AttackAction::Melee(melee_attack_action) => melee_attack_action.damage.as_slice(),
            AttackAction::Ranged(ranged_attack_action) => ranged_attack_action.damage.as_slice(),
        }
    }

    pub fn range(&self) -> Range {
        match self {
            AttackAction::Melee(melee_attack_action) => melee_attack_action.range(),
            AttackAction::Ranged(ranged_attack_action) => ranged_attack_action.range(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AttackResult {
    NoHit {
        attacker: Weak<Combatant>,
        target: Weak<Combatant>,
        attack: AttackAction,
        to_hit: AttackRoll,
    },
    Hit {
        attacker: Weak<Combatant>,
        attack: AttackAction,
        target: Weak<Combatant>,
        to_hit: AttackRoll,
        damage: Damage,
    },
}

pub const NO_ONE_TO_TARGET: legality::Reason = legality::Reason {
    id: "NO_ONE_TO_TARGET",
};

pub const OUT_OF_RANGE: legality::Reason = legality::Reason { id: "OUT_OF_RANGE" };

impl AttackAction {
    fn to_hit_mods(&self, _me: &Combatant) -> DExpr {
        // TODO: Check in with the stats.
        //       Replace with stats.attack_roll(self)

        match self {
            AttackAction::Melee(melee) => melee.to_hit.clone(),
            AttackAction::Ranged(ranged) => ranged.to_hit.clone(),
        }
    }

    pub fn make_attack(&self, me: &Arc<Combatant>, position: P3) -> Legality<AttackResult> {
        let combat = me.combat.upgrade().unwrap();
        let arena = combat.arena.as_ref();

        let target_square = position;

        let distance_to_target =
            geom::resolve_distance(P3::from(target_square.coords - me.position.load().coords));

        let base_attack_roll = match (distance_to_target, self.range()) {
            (d, Range::Reach) => {
                // TODO: Check if this weapon has the "Reach" property for +5ft range.
                if d > 5.0 {
                    return Legality::Illegal(OUT_OF_RANGE);
                }

                DExpr::from(D20)
            }
            (d, Range::Single(normal)) => match d {
                ..0.0 => unimplemented!("Negative distance?!"),
                d if (0.0..=normal).contains(&d) => DExpr::from(D20),
                d if (normal..).contains(&d) => return Legality::Illegal(OUT_OF_RANGE),
                _ => unreachable!(),
            },
            (d, Range::Long(LongRange(normal, long))) => match d {
                ..0.0 => unimplemented!("Negative distance?!"),
                d if (0.0..=5.0).contains(&d) => DExpr::disadvantage(D20.into()),
                d if (5.0..=normal).contains(&d) => DExpr::from(D20),
                d if (normal..=long).contains(&d) => DExpr::advantage(D20.into()),
                d if (long..).contains(&d) => return Legality::Illegal(OUT_OF_RANGE),
                _ => unreachable!(),
            },
        };

        let a = arena.at(target_square);

        // TODO: Choose another heuristic rather than 'first in the list'.
        let target = match a.combatants.first() {
            Some(t) => t,

            // QUESTION: Should this be illegal? It probably shouldn't.
            None => return Legality::Illegal(NO_ONE_TO_TARGET),
        };

        // Use our attack here. It's debatable putting it here...

        // Do our attack roll.
        let to_hit = AttackRoll((D20 + self.to_hit_mods(me)).evaluate());
        let criticality = to_hit.criticality();

        // (Extra)  If the d20 roll for an attack is a 1, the attack misses regardless
        //          of any modifiers or the target's AC.
        if !target.stats.ac.does_hit(&to_hit) || criticality == Some(Criticality::Failure) {
            return Legality::Legal(AttackResult::NoHit {
                attack: self.clone(),
                attacker: Arc::downgrade(me),
                target: Arc::downgrade(target),
                to_hit,
            });
        }

        let mut damage = self.damage().to_vec();

        // Double the damage dice.
        // "When you score a critical hit, you get to roll extra dice for the attack’s damage
        // against the target. Roll all of the attack's damage dice twice and add them together."
        // 5.1E SRD pg. 96-97
        if criticality == Some(Criticality::Success) {
            for (damage, _) in &mut damage {
                damage.apply_to_dice(|die| {
                    die.0 *= 2;
                });
            }
        }

        let damage: Damage = damage
            .into_iter()
            .map(|(amount, damage_type)| DamagePart {
                damage_type,
                amount: Box::new(amount.evaluate()),
                cause: DamageCause {
                    actor: crate::core::stats::damage::DamageActor::Entity(Arc::downgrade(me)),
                    source: crate::core::stats::damage::DamageSource,
                },
                handling: Default::default(),
            })
            .collect();

        // TODO: Decide whether to return damage taken,
        //       or just damage dealt?
        let _taken = target.stats.damage(damage.clone());

        Legality::Legal(AttackResult::Hit {
            attack: self.clone(),
            attacker: Arc::downgrade(me),
            target: Arc::downgrade(target),
            to_hit,
            damage,
        })
    }
}

#[derive(Debug, Default, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Targeting {
    /// Targeting a single creature,
    #[default]
    #[serde(alias = "one")]
    Single,
}

impl std::fmt::Display for Targeting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Targeting::Single => write!(f, "one target"),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct LongRange(Coord, Coord);

impl std::fmt::Display for LongRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "range {}ft. / {}ft.", self.0, self.1)
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Range {
    Reach,
    #[serde(untagged)]
    Long(LongRange),
    #[serde(untagged)]
    Single(Coord),
}

impl std::fmt::Display for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // TODO: This is not always true!
            Range::Reach => write!(f, "Reach 5ft."),
            Range::Long(LongRange(a, b)) => write!(f, "range {a}/{b} ft."),
            Range::Single(a) => write!(f, "range {a} ft."),
        }
    }
}

impl Default for Range {
    fn default() -> Self {
        Self::Reach
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::core::combat::turn::action::Action;

    #[test]
    fn parse_action() {
        let value = json!({
            "type": "attack",
            "attack_type": "melee",
            "name": "Bite",
            "to_hit": 0,
            "target": "single",
            "range": "reach",
            "damage": [
                [
                    "1",
                    "piercing"
                ]
            ],
            "description": ""
        });

        let val: Action = serde_json::from_value(value).expect("Valid parse!");
        println!("{val:?}");
    }
}
