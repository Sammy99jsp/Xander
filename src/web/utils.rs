use std::array;

use js_sys::TypeError;
use wasm_bindgen::{prelude::wasm_bindgen, JsCast, JsValue};

mod rs {
    pub use crate::core::geom::P3;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "[number, number, number]")]
    pub type P3;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(
        typescript_type = "[number, number, number, number, number, number, number,  number]"
    )]
    pub type OneHotDirection;
}

impl P3 {
    pub fn into_rust(self) -> JsResult<rs::P3> {
        let arr: js_sys::Array = self.obj.dyn_into().map_err(|_| {
            JsValue::from(TypeError::new("Expected [number, number, number] for `P3`"))
        })?;

        if arr.length() != 3 {
            return Err(JsValue::from(TypeError::new(
                "Expected [number, number, number] for `P3`",
            )));
        }

        array::try_from_fn(|i| {
            arr.get(i as u32).as_f64().map(|x| x as f32).ok_or_else(|| {
                JsValue::from(TypeError::new("Expected [number, number, number] for `P3`"))
            })
        })
        .map(rs::P3::from)
    }
}

impl From<rs::P3> for P3 {
    fn from(p3: rs::P3) -> Self {
        let arr = js_array!(p3.x, p3.y, p3.z);
        Self { obj: arr.into() }
    }
}

impl From<[f32; 8]> for OneHotDirection {
    fn from(arr: [f32; 8]) -> Self {
        let arr = js_array!(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);
        Self { obj: arr.into() }
    }
}

pub macro js_array($($v: expr),*) {
    {
        let __tmp = js_sys::Array::new();
        $(
            __tmp.push(&wasm_bindgen::JsValue::from($v));
        )*

        __tmp
    }
}

pub type JsResult<T> = Result<T, JsValue>;
