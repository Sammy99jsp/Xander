use crate::core::combat::arena::{ArenaR, SimpleArena};

pub trait VisArena: ArenaR {
    fn visualize(&self) -> crate::vis::Image;
}

impl VisArena for SimpleArena {
    fn visualize(&self) -> crate::vis::Image {
        use crate::vis::{
            color::{rgba, TOKEN_COLORS},
            FindReplace, Grid, Root, Token,
        };

        const IMG: &[u8] = include_bytes!("../../tests/python/img/rat.png");

        // SAFETY: IMG is &'static [u8], so it will outlast us all.
        let rat =
            skia::Image::from_encoded(unsafe { skia::Data::new_bytes(IMG) }).expect("Load rat!");

        let width = self.dimensions.0 as u32;
        let height = self.dimensions.1 as u32;

        let scale = 40;
        let mut root = Root::new(width, height, scale, 3);
        let c = self.weak.upgrade().expect("Not dead");

        root.add_child(Grid {
            square_len: 5,
            width,
            height,
            tokens: {
                c.initiative
                    .as_vec()
                    .into_iter()
                    .zip(TOKEN_COLORS.iter().copied().cycle())
                    .map(|(combatant, color)| {
                        Token::new(
                            combatant.position.load(),
                            rat.clone(),
                            FindReplace {
                                find: rgba(0x000000FF),
                                replace: {
                                    if combatant.stats.is_dead() {
                                        rgba(0x5E5757AA)
                                    } else {
                                        color
                                    }
                                },
                            },
                        )
                    })
                    .collect()
            },
        });

        root.render()
    }
}
