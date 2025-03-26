use std::str::FromStr;

use js_sys::{Error, TypeError};
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

mod rs {
    pub use crate::core::{dice::*, stats::damage::*};
}

use rs::DamageType as _;

#[wasm_bindgen]
pub struct DamageType(&'static rs::DamageTypeMeta);

#[wasm_bindgen]
impl DamageType {
    pub fn name(&self) -> String {
        self.0.name().to_string()
    }

    pub fn description(&self) -> String {
        self.0.description.to_string()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        self.0.name().to_string()
    }

    /// Create some amount of this type of damage.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(&self, amount: JsValue, cause: DamageCause) -> Result<Damage, JsValue> {
        let amount = ('a: {
            if let Some(amount) = amount.as_f64() {
                if amount.fract() > f64::EPSILON {
                    break 'a Err(JsValue::from(TypeError::new(
                        "Expected an integer here, not a decimal.",
                    )));
                }

                // TODO: Check to see if > i32::MAX and < i32::MIN
                let amount = amount as i32;

                break 'a Ok(Box::new(rs::DEvalTree::Modifier(amount)));
            }

            if let Some(amount) = amount.as_string() {
                let Ok(amount) = rs::DExpr::from_str(&amount) else {
                    break 'a Err(JsValue::from(Error::new("couldn't parse dice notation")));
                };

                break 'a Ok(Box::new(amount.evaluate()));
            }

            // TODO: DExpr

            Err(JsValue::from(TypeError::new(
                "Expected either integer, or dice notation string.",
            )))
        })?;

        Ok(Damage(rs::Damage(vec![rs::DamagePart {
            damage_type: self.0,
            amount,
            cause: cause.0,
            handling: Default::default(),
        }])))
    }
}

#[wasm_bindgen]
pub struct DamageCause(pub(in crate::web) rs::DamageCause);

#[wasm_bindgen]
impl DamageCause {
    #[wasm_bindgen(js_name = "UNKNOWN")]
    pub fn unknown() -> DamageCause {
        Self(rs::DamageCause::UNKNOWN)
    }
}

#[wasm_bindgen]
pub struct Damage(pub(in crate::web) rs::Damage);

#[wasm_bindgen]
impl Damage {
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    #[wasm_bindgen(getter, js_name = "Acid")]
    pub fn acid() -> DamageType {
        DamageType(rs::Acid::META)
    }

    #[wasm_bindgen(getter, js_name = "Bludgeoning")]
    pub fn bludgeoning() -> DamageType {
        DamageType(rs::Bludgeoning::META)
    }

    #[wasm_bindgen(getter, js_name = "Cold")]
    pub fn cold() -> DamageType {
        DamageType(rs::Cold::META)
    }

    #[wasm_bindgen(getter, js_name = "Fire")]
    pub fn fire() -> DamageType {
        DamageType(rs::Fire::META)
    }

    #[wasm_bindgen(getter, js_name = "Force")]
    pub fn force() -> DamageType {
        DamageType(rs::Force::META)
    }

    #[wasm_bindgen(getter, js_name = "Lightning")]
    pub fn lightning() -> DamageType {
        DamageType(rs::Lightning::META)
    }

    #[wasm_bindgen(getter, js_name = "Necrotic")]
    pub fn necrotic() -> DamageType {
        DamageType(rs::Necrotic::META)
    }

    #[wasm_bindgen(getter, js_name = "Piercing")]
    pub fn piercing() -> DamageType {
        DamageType(rs::Piercing::META)
    }

    #[wasm_bindgen(getter, js_name = "Poison")]
    pub fn poison() -> DamageType {
        DamageType(rs::Poison::META)
    }

    #[wasm_bindgen(getter, js_name = "Psychic")]
    pub fn psychic() -> DamageType {
        DamageType(rs::Psychic::META)
    }

    #[wasm_bindgen(getter, js_name = "Radiant")]
    pub fn radiant() -> DamageType {
        DamageType(rs::Radiant::META)
    }

    #[wasm_bindgen(getter, js_name = "Slashing")]
    pub fn slashing() -> DamageType {
        DamageType(rs::Slashing::META)
    }

    #[wasm_bindgen(getter, js_name = "Thunder")]
    pub fn thunder() -> DamageType {
        DamageType(rs::Thunder::META)
    }
}
