use std::sync::Arc;

use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

mod rs {
    pub use crate::core::{creature::Monster, stats::stat_block::StatBlock};
}

mod web {
    pub use crate::web::combat::{action::Action, speed::SpeedType};
}


#[wasm_bindgen]
pub struct Stats(pub(in crate::web) Arc<rs::StatBlock>);

#[wasm_bindgen]
impl Stats {
    #[wasm_bindgen(constructor)]
    pub fn new(value: JsValue) -> Result<Self, String> {
        let rs::Monster(stats) =
            serde_wasm_bindgen::from_value(value).map_err(|err| ToString::to_string(&err))?;

        Ok(Self(stats))
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        let this = self.0.as_ref();
        format!("<{} {}/{} HP>", this.name, this.hp(), this.max_hp())
    }

    pub fn speed(&self, mode: web::SpeedType) -> Option<u32> {
        self.0.speed(mode.0)
    }

    #[wasm_bindgen(getter)]
    pub fn hp(&self) -> u32 {
        self.0.hp()
    }

    #[wasm_bindgen(getter)]
    pub fn max_hp(&self) -> u32 {
        self.0.max_hp()
    }

    #[wasm_bindgen(getter)]
    pub fn temp_hp(&self) -> Option<u32> {
        self.0.temp_hp()
    }

    #[wasm_bindgen(getter)]
    pub fn actions(&self) -> Vec<web::Action> {
        self.0.actions().map(web::Action).collect()
    }

    #[wasm_bindgen(getter)]
    pub fn dead(&self) -> bool {
        self.0.is_dead()
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.0.name.clone()
    }
}
