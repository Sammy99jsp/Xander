//! Random Agent
//!
//! Plays legal moves at random.

use std::sync::Arc;

use rand::{thread_rng, Rng};

use crate::core::{
    combat::{arena::SQUARE_LENGTH, turn::TurnCtx, CombatHook},
    geom::{Coord, P3},
    stats::monster::speed::Walking,
};

pub struct RandomAgent;

const M: Coord = SQUARE_LENGTH;

/// N-NE-E-SE-S-SW-W-NW
const DIRECTIONS: [P3; 8] = [
    P3::new(0.0, M, 0.0),
    P3::new(M, M, 0.0),
    P3::new(M, 0.0, 0.0),
    P3::new(M, -M, 0.0),
    P3::new(0.0, -M, 0.0),
    P3::new(-M, -M, 0.0),
    P3::new(-M, 0.0, 0.0),
    P3::new(-M, M, 0.0),
];

impl CombatHook for RandomAgent {
    fn turn(&self, turn: Arc<TurnCtx>) {
        let me = turn.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        let mut rng = thread_rng();

        let can_attack = turn.actions.can_use().is_legal();
        let can_move = turn.movement.any_movement_left(&Walking).is_legal();

        let moveable_dirs = if can_move {
            DIRECTIONS
                .as_slice()
                .iter()
                .filter(|p| {
                    combat
                        .arena
                        .is_passable(
                            P3::from(me.position.load().coords + p.coords),
                            me.stats.size,
                        )
                        .is_legal()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let attackable_dirs = if can_attack {
            DIRECTIONS
                .as_slice()
                .iter()
                .filter(|p| {
                    let combatants = combat
                        .arena
                        .at(P3::from(me.position.load().coords + p.coords))
                        .combatants;

                    combatants.iter().any(|c| !c.stats.is_dead())
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let options = 1 + moveable_dirs.len() + attackable_dirs.len();

        let choice = rng.gen_range(0..options);

        match choice {
            0 => {
                combat.initiative.advance_turn();
            }
            i if (1..(1 + moveable_dirs.len())).contains(&i) => {
                let l = turn.movement.try_move(&Walking, *moveable_dirs[choice - 1]);
                debug_assert!(l.is_legal());
            }
            i if (1 + moveable_dirs.len()..).contains(&i) => {
                let delta = *attackable_dirs[choice - 1];

                let attack = me
                    .stats
                    .actions
                    .get()
                    .into_iter()
                    .map(|action| match action {
                        crate::core::combat::turn::action::Action::Attack(attack_action) => {
                            attack_action
                        }
                    })
                    .next()
                    .unwrap_or_else(|| panic!("No attack action found for {}", me.stats.name));

                let l = attack.make_attack(&me, delta).map(|res| {
                    turn.actions.mark_used();
                    res
                });

                debug_assert!(l.is_legal());
            }
            _ => unreachable!(),
        }
    }
}
