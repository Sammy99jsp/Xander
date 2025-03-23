#![allow(clippy::inherent_to_string, clippy::new_without_default)]
use wasm_bindgen::prelude::wasm_bindgen;

pub mod canvas;
pub mod combat;
pub mod stats;
// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}
