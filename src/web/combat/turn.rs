use std::sync::Arc;

use wasm_bindgen::prelude::wasm_bindgen;

mod rs {
    pub use crate::core::combat::turn::TurnCtx;
}

#[wasm_bindgen]
pub struct Turn(pub(in crate::web) Arc<rs::TurnCtx>);
