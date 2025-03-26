use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

mod rs {
    pub(crate) use crate::utils::legality::*;
}

#[wasm_bindgen]
pub struct Legality {
    pub(in crate::web) reason: Option<&'static str>,
    pub(in crate::web) obj: Option<JsValue>,
}

impl Legality {
    pub(in crate::web) fn illegal(reason: &'static str) -> Self {
        Self {
            reason: Some(reason),
            obj: None,
        }
    }

    pub(in crate::web) fn legal<T>(obj: T) -> Self
    where
        JsValue: From<T>,
    {
        Self {
            reason: None,
            obj: Some(JsValue::from(obj)),
        }
    }
}

#[wasm_bindgen]
impl Legality {
    pub fn is_legal(&self) -> bool {
        self.obj.is_some()
    }

    pub fn is_illegal(&self) -> bool {
        self.obj.is_none()
    }

    pub fn inner(&self) -> JsValue {
        match self.obj {
            None => JsValue::null(),
            Some(ref t) => t.clone(),
        }
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string(&self) -> String {
        match self {
            Self {
                reason: Some(r),
                obj: None,
            } => format!("Illegal: {}", r),
            Self {
                reason: None,
                obj: Some(_),
            } => "<Something>".to_string(),
            _ => unreachable!(),
        }
    }
}

impl<T> From<rs::Legality<T>> for Legality
where
    JsValue: From<T>,
{
    fn from(value: rs::Legality<T>) -> Self {
        match value {
            rs::Legality::Legal(obj) => Self {
                reason: None,
                obj: Some(JsValue::from(obj)),
            },
            rs::Legality::Illegal(reason) => Self {
                reason: Some(reason.id),
                obj: None,
            },
        }
    }
}
