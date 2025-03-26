use wasm_bindgen::prelude::wasm_bindgen;

mod rs {
    pub use crate::core::{
        combat::turn::attack::{roll::*, *},
        stats::damage::pretty_damage,
    };
}

mod web {
    pub use crate::web::damage::*;
}

#[wasm_bindgen]
pub struct Attack(pub(in crate::web) rs::AttackAction);

#[wasm_bindgen]
impl Attack {
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        match &self.0 {
            rs::AttackAction::Melee(rs::MeleeAttackAction {
                name,
                to_hit,
                range,
                target,
                damage,
                ..
            }) => owo_colors::with_override(false, || {
                format!(
                    "{name}. Melee Weapon Attack: {to_hit} to hit, {range}, {target}. Hit: {}.",
                    rs::pretty_damage(damage.as_slice())
                )
            }),
        }
    }
}

#[wasm_bindgen]
pub struct AttackResult(pub(in crate::web) rs::AttackResult);

#[wasm_bindgen]
impl AttackResult {
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        owo_colors::with_override(false, || format!("{:?}", self.0))
    }

    #[wasm_bindgen(getter)]
    pub fn successful(&self) -> bool {
        match self.0 {
            rs::AttackResult::Hit { .. } => true,
            rs::AttackResult::NoHit { .. } => false,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn to_hit(&self) -> AttackRoll {
        match &self.0 {
            rs::AttackResult::Hit { to_hit, .. } => AttackRoll(to_hit.clone()),
            rs::AttackResult::NoHit { to_hit, .. } => AttackRoll(to_hit.clone()),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn damage(&self) -> Option<web::Damage> {
        match &self.0 {
            rs::AttackResult::Hit { damage, .. } => Some(web::Damage(damage.clone())),
            _ => None,
        }
    }
}

#[wasm_bindgen]
pub struct AttackRoll(pub(in crate::web) rs::AttackRoll);

#[wasm_bindgen]
impl AttackRoll {
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        owo_colors::with_override(false, || format!("{}", self.0))
    }
}
