pub mod agents;
pub mod arena;
pub mod turn;

use crossbeam_utils::atomic::AtomicCell;
use itertools::Position;
use js_sys::{Error, TypeError};
use web_sys::console;
use std::{ops::Deref, sync::Arc};
use wasm_bindgen::{
    convert::{FromWasmAbi, TryFromJsValue},
    prelude::wasm_bindgen,
    JsCast, JsValue,
};

mod rs {
    pub use crate::core::{
        combat::{
            arena::{SimpleArena, SimpleArenaParams},
            Combat, CombatHook, Combatant,
        },
        combat::{turn::TurnCtx, InitiativeRoll},
        geom::P3,
    };
}

#[wasm_bindgen]
pub struct Combat(#[wasm_bindgen(skip)] Arc<rs::Combat>);

#[wasm_bindgen]
impl Combat {
    #[allow(clippy::new_without_default)]
    #[wasm_bindgen(constructor)]
    pub fn new(width: f32, height: f32) -> Combat {
        Combat(rs::Combat::new(None, |this| {
            rs::SimpleArena::new(this, rs::SimpleArenaParams::new(width, height))
        }))
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        let this: &rs::Combat = self.0.deref();
        format!(
            "Combat({} members, Arena{:?})",
            this.initiative.len(),
            this.arena.grid_size()
        )
    }

    #[wasm_bindgen(getter)]
    pub fn arena(&self) -> arena::Arena {
        arena::Arena(Arc::downgrade(&self.0.arena))
    }
}

#[wasm_bindgen(typescript_custom_section)]
const COMBAT_PARAMS: &'static str = r#"
interface CombatantParams {
    stats: Stats;
    name: string;
    position: [number, number, number];
    hook: CombatHook;
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "COMBAT_PARAMS")]
    pub type CombatParams;
}

#[wasm_bindgen]
pub struct Combatant(#[wasm_bindgen(skip)] Arc<rs::Combatant>);

#[wasm_bindgen]
impl Combat {
    pub fn join(&self, params: CombatParams) -> Result<Combatant, JsValue> {
        let params: JsValue = params.into();

        let stats = super::stats::Stats::try_from_js_value(js_sys::Reflect::get(
            &params,
            &"stats".into(),
        )?)?;

        console::log_1(&format!("Stats: {:?}", stats.0).into());

        let name = js_sys::Reflect::get(&params, &"name".into())?
            .as_string()
            .ok_or_else(|| TypeError::new("Expected 'name' to be a string"))?;
        let arr: js_sys::Array = js_sys::Reflect::get(&params, &"position".into())?.dyn_into()?;
        let position = rs::P3::new(
            arr.get(0)
                .as_f64()
                .ok_or_else(|| TypeError::new("Expected 'position' to be a 3-tuple"))?
                as f32,
            arr.get(1)
                .as_f64()
                .ok_or_else(|| TypeError::new("Expected 'position' to be a 3-tuple"))?
                as f32,
            arr.get(2)
                .as_f64()
                .ok_or_else(|| TypeError::new("Expected 'position' to be a 3-tuple"))?
                as f32,
        );

        let hook = js_sys::Reflect::get(&params, &"hook".into())?;
        let hook: Box<dyn rs::CombatHook> = 'a: {
            match hook.dyn_into::<js_sys::Function>() {
                Ok(hook) => {
                    break 'a Result::<_, JsValue>::Ok(Box::new(UnsafeJSFunction(hook)));
                }
                Err(_) => return Err(TypeError::new("Expected 'hook' to be a function").into()),
            };
        }?;

        let combatant = Arc::new(rs::Combatant {
            combat: Arc::downgrade(&self.0),
            position: AtomicCell::new(position),
            name,
            initiative: rs::InitiativeRoll(stats.0.initiative.get().result()),
            stats: stats.0,
            hook,
        });

        Ok(Combatant(combatant))
    }
}

#[wasm_bindgen]
impl Combatant {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn position(&self) -> js_sys::Array {
        let pos = self.0.position.load();
        js_sys::Array::of3(&pos.x.into(), &pos.y.into(), &pos.z.into())
    }

    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> super::stats::Stats {
        super::stats::Stats(self.0.stats.clone())
    }
}

pub struct UnsafeJSFunction(js_sys::Function);

unsafe impl Send for UnsafeJSFunction {}
unsafe impl Sync for UnsafeJSFunction {}

impl rs::CombatHook for UnsafeJSFunction {
    fn turn(&self, turn: Arc<rs::TurnCtx>) {
        self.0
            .call1(&self.0.clone().into(), &self::turn::Turn(turn).into())
            .expect("Cannot call the hook function");
    }
}
