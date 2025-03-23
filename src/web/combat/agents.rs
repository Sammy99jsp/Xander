use wasm_bindgen::prelude::wasm_bindgen;

mod rs {
    pub use crate::agents::random::RandomAgent;
}

#[wasm_bindgen]
pub struct RandomAgent(pub(in crate::web) rs::RandomAgent);

#[wasm_bindgen]
impl RandomAgent {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RandomAgent {
        RandomAgent(rs::RandomAgent)
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        "RandomAgent".to_string()
    }
}
