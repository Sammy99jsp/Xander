use serde::Deserialize;

use crate::core::{dice::DExpr, geom::Coord, stats::damage::DamageTypeMeta};

use super::{LongRange, Range, Targeting};


#[derive(Debug, Deserialize, Clone)]
#[serde(try_from = "crate::serde::combat::attack::RangedAttackRaw")]
pub struct RangedAttackAction {
    pub name: String,
    pub description: String,
    pub to_hit: DExpr,
    pub range: LongRange,
    pub target: Targeting,
    pub damage: Vec<(DExpr, &'static DamageTypeMeta)>,
}

impl RangedAttackAction {
    pub const fn range(&self) -> Range {
        Range::Long(self.range)
    }
}