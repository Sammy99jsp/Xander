use std::sync::{Arc, Weak};

use action::ActionCtx;
use attack::{AttackAction, AttackResult, Range};

use crate::{
    core::{
        geom::{DIRECTIONS, P3},
        stats::monster::speed::SpeedTypeMeta,
    },
    utils::legality::Legality,
};

use super::{movement::MovementCtx, Combatant};

pub mod action;
pub mod attack;

#[derive(Debug)]
pub struct TurnCtx {
    pub movement: MovementCtx,
    pub actions: ActionCtx,
}

impl TurnCtx {
    pub fn new(weak: Weak<Combatant>) -> Self {
        Self {
            movement: MovementCtx::new(weak.clone()),
            actions: ActionCtx::new(weak),
        }
    }

    pub fn combatant(&self) -> &Weak<Combatant> {
        &self.movement.combatant
    }

    // EXPORTED METHODS

    pub fn is_combat_active(&self) -> bool {
        self.combatant().upgrade().is_some()
    }

    #[inline]
    pub fn try_move(&self, delta: P3, mode: &'static SpeedTypeMeta) -> Legality<()> {
        self.movement.try_move(mode, delta)
    }

    pub fn attack(&self, attack: AttackAction, target_delta: P3) -> Legality<AttackResult> {
        let me = self.combatant().upgrade().unwrap();
        // Check if we've already used all actions for this turn, and return early if so.
        // If not, update the used action count by 1, and continue.
        let () = self.actions.can_use()?;

        attack.make_attack(&me, target_delta).map(|res| {
            self.actions.mark_used();
            res
        })
    }

    pub fn end(&self) -> Legality<()> {
        let me = self.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        combat.initiative.advance_turn();

        Legality::Legal(())
    }

    pub fn movement_directions(&self, mode: &'static SpeedTypeMeta) -> Legality<Vec<P3>> {
        // Is there any movement left? If not, return early, stating it's illegal.
        let () = self.movement.any_movement_left(mode)?;

        let me = self.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();

        Legality::Legal(
            DIRECTIONS
                .map(|(x, y, z)| P3::new(x, y, z))
                .into_iter()
                .filter(|p| {
                    combat
                        .arena
                        .is_passable(
                            P3::from(me.position.load().coords + p.coords),
                            me.stats.size,
                        )
                        .is_legal()
                })
                .collect::<Vec<_>>(),
        )
    }

    pub fn movement_directions_one_hot(&self, mode: &'static SpeedTypeMeta) -> [f32; 8] {
        // Is there any movement left? If not, return early.
        if let Legality::Illegal(_) = self.movement.any_movement_left(mode) {
            return [0.0; 8];
        }

        let me = self.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();

        let directions = DIRECTIONS.map(|(x, y, z)| P3::new(x, y, z));

        directions.map(|p| {
            if combat
                .arena
                .is_passable(
                    P3::from(me.position.load().coords + p.coords),
                    me.stats.size,
                )
                .is_legal()
            {
                1.0
            } else {
                0.0
            }
        })
    }

    pub fn attack_directions<'a>(
        &self,
        attack: AttackAction,
        filter: impl Fn(Vec<Arc<Combatant>>) -> bool + 'a,
    ) -> Legality<Vec<P3>> {
        let me = self.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        let range = attack.range();
        match range {
            Range::Reach => {
                // Assume reach is 5 ft. (this isn't always the case)
                // TODO: Look this up from the stats of the weapon + combatant.
                // TODO: Add a "grid search" struct to handle this.
                Legality::Legal(
                    DIRECTIONS
                        .map(|(x, y, z)| P3::new(x, y, z))
                        .into_iter()
                        .filter_map(|p| {
                            let combatants = combat
                                .arena
                                .at(P3::from(me.position.load().coords + p.coords))
                                .combatants;

                            if filter(combatants) {
                                return Some(P3::from(p));
                            }

                            None
                        })
                        .collect::<Vec<_>>(),
                )
            }
            _ => todo!(),
        }
    }

    pub fn attack_directions_one_hot(
        &self,
        attack: AttackAction,
        filter: impl Fn(Vec<Arc<Combatant>>) -> bool,
    ) -> [f32; 8] {
        let me = self.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        let range = attack.range();
        let ret = match range {
            Range::Reach => {
                DIRECTIONS
                    .map(|(x, y, z)| P3::new(x, y, z))
                    .into_iter()
                    .map(|p| {
                        let combatants = combat
                            .arena
                            .at(P3::from(me.position.load().coords + p.coords))
                            .combatants;

                        if filter(combatants) {
                            return 1.0;
                        }

                        0.0
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            }

            _ => todo!(),
        };

        ret
    }
}
