use itertools::Itertools;

use crate::core::combat::Arena;

pub trait ArenaCanvas: Arena {
    fn draw(&self, ctx: web_sys::CanvasRenderingContext2d) {
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
    }
}

impl<A: Arena + ?Sized> ArenaCanvas for A {}
