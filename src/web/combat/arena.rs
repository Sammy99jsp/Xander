use std::sync::Weak;

use js_sys::Error;
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};
use web_sys::CanvasRenderingContext2d;

use super::super::canvas::ArenaCanvas;

mod rs {
    pub use crate::core::combat::Arena;
}

#[wasm_bindgen(typescript_custom_section)]
const DRAW_PARAMS: &'static str = r#"
type DrawParams = {
    token: (combatant: Combatant) => HTMLImageElement;
} | null;

export interface Token {
    kind: "combatant" | "other";
    position: [number, number, number];
    name: string;
}
type Tokens = Token[];
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "DrawParams")]
    pub type DrawParams;

    #[wasm_bindgen(typescript_type = "Tokens")]
    pub type Tokens;
}

#[wasm_bindgen]
pub struct Arena(pub(in crate::web) Weak<dyn rs::Arena>);

#[wasm_bindgen]
impl Arena {
    pub fn draw(&self, ctx: CanvasRenderingContext2d, params: DrawParams) -> Result<(), JsValue> {
        let Some(this) = self.0.upgrade() else {
            return Err(Error::new("Combat no longer exists!").into());
        };

        this.draw(ctx, params.obj)?;

        Ok(())
    }

    pub fn tokens(&self) -> Result<Tokens, JsValue> {
        use js_sys::Reflect::set;
        let tokens = js_sys::Array::new();

        let combat = self
            .0
            .upgrade()
            .ok_or_else(|| JsValue::from(Error::new("Combat not running!")))?
            .combat();

        for member in combat.initiative.as_vec() {
            let token = js_sys::Object::new();
            set(&token, &"kind".into(), &"combatant".into())?;
            set(&token, &"name".into(), &member.name.as_str().into())?;

            let position = js_sys::Array::new();
            let p = member.position.load();
            position.push(&(p.x).into());
            position.push(&(p.y).into());
            position.push(&(p.z).into());

            set(&token, &"position".into(), &position)?;
            tokens.push(&token);
        }

        Ok(Tokens { obj: tokens.into() })
    }

    #[wasm_bindgen(getter)]
    pub fn grid_size(&self) -> Result<js_sys::Array, JsValue> {
        let arena = self
            .0
            .upgrade()
            .ok_or_else(|| Error::new("Combat ended!"))?;

        let (width, height) = arena.grid_size();
        let size = js_sys::Array::new();
        size.push(&width.into());
        size.push(&height.into());

        Ok(size)
    }
}
