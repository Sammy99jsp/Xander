use std::sync::Arc;

use js_sys::TypeError;
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

mod rs {
    pub use crate::core::{combat::turn::TurnCtx, geom::P3};
}

mod web {
    pub use crate::web::{
        combat::{attack::*, speed::*},
        legality::*,
    };
}

type JsResult<T> = Result<T, JsValue>;

fn js_vec_to_p3(delta: Vec<f32>) -> JsResult<rs::P3> {
    let Ok([x, y, z]): Result<[f32; 3], _> = delta.try_into() else {
        return Err(JsValue::from(TypeError::new(
            "Expected [number, number, number] for `delta`",
        )));
    };

    Ok(rs::P3::new(x, y, z))
}

#[wasm_bindgen]
pub struct Turn(pub(in crate::web) Arc<rs::TurnCtx>);

#[wasm_bindgen]
impl Turn {
    /// Tries to move the combatant by a delta of [x, y, z].
    ///
    /// @param delta -- A 3-length {@link Float32Array} for [x, y, z]
    ///
    /// @returns the {@link Legality} of the movement
    #[wasm_bindgen(js_name = "move")]
    pub fn try_move(&self, delta: Vec<f32>, mode: web::SpeedType) -> JsResult<web::Legality> {
        Ok(self
            .0
            .try_move(js_vec_to_p3(delta)?, mode.0)
            .map(|()| JsValue::null())
            .into())
    }

    /// Tries to attack the square with a delta of [x, y, z].
    ///
    /// @param attack -- the attack to use.
    /// @param delta -- A 3-length {@link Float32Array} for [x, y, z]
    ///
    /// @returns the {@link Legality} of the attack
    pub fn attack(&self, attack: web::Attack, delta: Vec<f32>) -> JsResult<web::Legality> {
        Ok(self
            .0
            .attack(attack.0, js_vec_to_p3(delta)?)
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

    pub fn movement_directions_one_hot(&self, mode: web::SpeedType) -> Vec<f32> {
        Vec::from(self.0.movement_directions_one_hot(mode.0))
    }
}
