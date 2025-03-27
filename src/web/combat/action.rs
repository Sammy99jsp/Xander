use wasm_bindgen::prelude::wasm_bindgen;

mod rs {
    pub use crate::core::combat::turn::action::Action;
}

mod web {
    pub use crate::web::combat::attack::Attack;
}

#[wasm_bindgen]
pub struct Action(pub(in crate::web) rs::Action);

#[wasm_bindgen]
impl Action {
    #[allow(unreachable_patterns)]
    pub fn as_attack(&self) -> Option<web::Attack> {
        match &self.0 {
            rs::Action::Attack(attack) => Some(web::Attack(attack.clone())),
            _ => None,
        }
    }
}
