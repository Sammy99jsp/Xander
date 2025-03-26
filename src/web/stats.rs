use std::sync::Arc;

use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

mod rs {
    pub use crate::core::{creature::Monster, stats::stat_block::StatBlock};
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
}
