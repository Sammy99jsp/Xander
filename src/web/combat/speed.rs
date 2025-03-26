mod rs {
    pub(crate) use crate::core::stats::monster::speed::*;
}

use rs::SpeedType as _;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub struct SpeedType(pub(in crate::web) &'static rs::SpeedTypeMeta);

#[wasm_bindgen]
impl SpeedType {
    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        self.0.name.to_string()
    }

    #[wasm_bindgen(getter, js_name="WALKING")]
    pub fn walking() -> SpeedType {
        SpeedType(rs::Walking::META)
    }

    #[wasm_bindgen(getter, js_name="BURROWING")]
    pub fn burrowing() -> SpeedType {
        SpeedType(rs::Burrowing::META)
    }

    #[wasm_bindgen(getter, js_name="CLIMBING")]
    pub fn climbing() -> SpeedType {
        SpeedType(rs::Climbing::META)
    }

    #[wasm_bindgen(getter, js_name="FLYING")]
    pub fn flying() -> SpeedType {
        SpeedType(rs::Flying::META)
    }

    #[wasm_bindgen(getter, js_name="SWIMMING")]
    pub fn swimming() -> SpeedType {
        SpeedType(rs::Swimming::META)
    }
    
    #[wasm_bindgen(getter, js_name="CRAWLING")]
    pub fn crawling() -> SpeedType {
        SpeedType(rs::Crawling::META)
    }
}