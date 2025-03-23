use std::sync::Weak;

use wasm_bindgen::prelude::wasm_bindgen;
use web_sys::CanvasRenderingContext2d;

use super::super::canvas::ArenaCanvas;

mod rs {
    pub use crate::core::combat::Arena;
}

#[wasm_bindgen]
pub struct Arena(pub(in crate::web) Weak<dyn rs::Arena>);

#[wasm_bindgen]
impl Arena {
    pub fn draw(&self, ctx: CanvasRenderingContext2d) -> Result<(), String> {
        let Some(this) = self.0.upgrade() else {
            return Err("Combat no longer exists!".to_string());
        };

        this.draw(ctx);

        Ok(())
    }
}
