use itertools::Itertools;
use js_sys::TypeError;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::HtmlImageElement;

use crate::core::{
    combat::{arena::SQUARE_LENGTH, Arena},
    geom::P3,
};

mod web {
    pub use crate::web::combat::Combatant;
}

pub fn square_floor(square_len: f32, pos: P3) -> (f32, f32) {
    let sq = square_len;
    (
        pos.x.div_euclid(SQUARE_LENGTH) * sq,
        pos.y.div_euclid(SQUARE_LENGTH) * sq,
    )
}

pub trait ArenaCanvas: Arena {
    fn draw(&self, ctx: web_sys::CanvasRenderingContext2d, params: JsValue) -> Result<(), JsValue> {
        let canvas = ctx.canvas().unwrap();

        let width = canvas.width() as f64;
        let height = canvas.height() as f64;
        ctx.clear_rect(0.0, 0.0, width, height);

        let grid = self.grid_size();
        let (grid_w, grid_h) = (grid.0 as f64, grid.0 as f64);

        let sq = width.div_euclid(grid_w).min(height.div_euclid(grid_h));

        let start = (width.rem_euclid(sq) / 2.0, height.rem_euclid(sq) / 2.0);

        for ((x, y), darker) in (0..grid.0).cartesian_product(0..grid.1).map(|(x, y)| {
            (
                (start.0 + x as f64 * sq, start.1 + y as f64 * sq),
                (x + y) % 2 == 0,
            )
        }) {
            if darker {
                ctx.set_fill_style_str("rgba(10, 10, 10, 0.9)");
            } else {
                ctx.set_fill_style_str("rgba(150, 150, 150, 0.9)");
            }

            ctx.fill_rect(x, y, sq, sq);
        }

        if params.is_falsy() {
            return Ok(());
        }

        let params = params.dyn_into::<js_sys::Object>().unwrap();
        let token = js_sys::Reflect::get(&params, &JsValue::from("token"))?;

        let token_fn = token.dyn_into::<js_sys::Function>()?;

        self.combat()
            .initiative
            .as_vec()
            .into_iter()
            .filter(|combatant| !combatant.stats.is_dead())
            .try_for_each(|combatant| -> Result<(), JsValue> {
                let pos = combatant.position.load();
                // console::log_1(&pos.to_string().into());
                let res = token_fn.call1(&JsValue::null(), &web::Combatant(combatant).into())?;

                if res.is_falsy() {
                    return Ok(());
                }

                if res.is_instance_of::<HtmlImageElement>() {
                    let image: HtmlImageElement = res.into();

                    let (x, y) = square_floor(sq as f32, pos);
                    let (dx, dy, dw, dh) =
                        (x as f64, canvas.height() as f64 - (y as f64 + sq), sq, sq);

                    ctx.draw_image_with_html_image_element_and_dw_and_dh(&image, dx, dy, dw, dh)?;

                    return Ok(());
                }

                Err(TypeError::new("Expected function to return an HTMLImageElement").into())
            })?;

        Ok(())
    }
}

impl<A: Arena + ?Sized> ArenaCanvas for A {}
