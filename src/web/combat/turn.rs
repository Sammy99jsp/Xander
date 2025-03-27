use std::sync::Arc;

use js_sys::TypeError;
use wasm_bindgen::{prelude::wasm_bindgen, JsCast, JsValue};

use crate::web::utils::{js_array, JsResult, OneHotDirection};

mod rs {
    pub use crate::core::combat::turn::TurnCtx;
}

mod web {
    pub use crate::web::{
        combat::{attack::*, speed::*, Combatant},
        legality::*,
        utils::P3,
    };
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "(t: Combatant) => bool")]
    pub type AttackFilterPredicate;
}

#[wasm_bindgen]
pub struct Turn(pub(in crate::web) Arc<rs::TurnCtx>);

#[wasm_bindgen]
impl Turn {
    pub fn is_combat_active(&self) -> bool {
        self.0.is_combat_active()
    }

    /// Tries to move the combatant by a delta of [x, y, z].
    ///
    /// @param delta -- Delta in [x, y, z]
    ///
    /// @returns the {@link Legality} of the movement
    #[wasm_bindgen(js_name = "move")]
    pub fn try_move(&self, delta: web::P3, mode: web::SpeedType) -> JsResult<web::Legality> {
        Ok(self
            .0
            .try_move(delta.into_rust()?, mode.0)
            .map(|()| JsValue::null())
            .into())
    }

    /// Tries to attack the square with a delta of [x, y, z].
    ///
    /// @param attack -- the attack to use.
    /// @param delta -- A 3-length {@link Float32Array} for [x, y, z]
    ///
    /// @returns the {@link Legality} of the attack
    pub fn attack(&self, attack: web::Attack, delta: web::P3) -> JsResult<web::Legality> {
        Ok(self
            .0
            .attack(attack.0, delta.into_rust()?)
            .map(web::AttackResult)
            .into())
    }

    /// End this combatant's turn.
    ///
    /// @returns a {@link Legality} which is never illegal.
    pub fn end(&self) -> web::Legality {
        let me = self.0.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        combat.initiative.advance_turn();

        web::Legality::legal(JsValue::null())
    }

    pub fn movement_directions(&self, mode: web::SpeedType) -> JsResult<web::Legality> {
        Ok(self
            .0
            .movement_directions(mode.0)
            .map(|v| {
                v.into_iter()
                    .map(|p3| js_array!(p3.x, p3.y, p3.z))
                    .collect::<Vec<_>>()
            })
            .into())
    }

    pub fn movement_directions_one_hot(&self, mode: web::SpeedType) -> OneHotDirection {
        OneHotDirection::from(self.0.movement_directions_one_hot(mode.0))
    }

    pub fn attack_direction(
        &self,
        attack: web::Attack,
        filter: AttackFilterPredicate,
    ) -> JsResult<web::Legality> {
        self.0
            .attack_directions(attack.0, |combatants| {
                let filter = filter.obj.dyn_ref::<js_sys::Function>().ok_or_else(|| {
                    TypeError::new(
                        "Expected predicate function `(combatant: Combatant) => bool` here!",
                    )
                })?;

                combatants.into_iter().map(web::Combatant).try_fold(
                    false,
                    |acc, combatant: web::Combatant| {
                        filter
                            .call1(&JsValue::null(), &JsValue::from(combatant))
                            .map(|res| acc || res.is_truthy())
                    },
                )
            })
            .map(|legality| {
                legality.map(|points| {
                    points
                        .into_iter()
                        .map(web::P3::from)
                        .collect::<js_sys::Array>()
                })
            })
            .map(web::Legality::from)
    }

    pub fn attack_directions_one_hot(
        &self,
        attack: web::Attack,
        filter: AttackFilterPredicate,
    ) -> JsResult<OneHotDirection> {
        self.0
            .attack_directions_one_hot(attack.0, |combatants| {
                let filter = filter.obj.dyn_ref::<js_sys::Function>().ok_or_else(|| {
                    TypeError::new(
                        "Expected predicate function `(combatant: Combatant) => bool` here!",
                    )
                })?;
                
                combatants.into_iter().map(web::Combatant).try_fold(
                    false,
                    |acc, combatant: web::Combatant| {
                        filter
                            .call1(&JsValue::null(), &JsValue::from(combatant))
                            .map(|res| acc || res.is_truthy())
                    },
                )
            })
            .map(OneHotDirection::from)
    }

    pub fn movement_left(&self, mode: web::SpeedType) -> u32 {
        self.0.movement_left(mode.0)
    }

    #[wasm_bindgen(getter)]
    pub fn actions_left(&self) -> u32 {
        self.0.actions_left()
    }

    #[wasm_bindgen(getter)]
    pub fn max_actions(&self) -> u32 {
        self.0.max_actions()
    }
}
