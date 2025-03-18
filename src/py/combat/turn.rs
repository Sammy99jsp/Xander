use std::sync::Arc;

use pyo3::{pyclass, pymethods, PyObject, PyResult, Python};

mod py {
    pub(crate) use crate::py::{
        combat::{
            attack::{Attack, AttackResult},
            speed::{SpeedType, WALKING},
            Combatant,
        },
        legality::Legality,
    };
}

mod rs {
    pub(crate) use crate::{
        core::{
            combat::turn::{attack::Range, TurnCtx},
            geom::P3,
        },
        utils::legality::Legality,
    };
}

const DIRECTIONS: [(f32, f32, f32); 8] = [
    (0.0, 5.0, 0.0),
    (5.0, 5.0, 0.0),
    (5.0, 0.0, 0.0),
    (5.0, -5.0, 0.0),
    (0.0, -5.0, 0.0),
    (-5.0, -5.0, 0.0),
    (-5.0, 0.0, 0.0),
    (-5.0, 5.0, 0.0),
];

#[pyclass]
pub struct Turn(pub(super) Arc<rs::TurnCtx>);

#[pymethods]
impl Turn {
    #[pyo3(name = "move")]
    #[pyo3(signature = (delta, mode = py::WALKING))]
    fn try_move(&self, delta: (f32, f32, f32), mode: py::SpeedType) -> PyResult<py::Legality> {
        let (x, y, z) = delta;
        self.0
            .movement
            .try_move(mode.0, rs::P3::new(x, y, z))
            .try_into()
    }

    fn attack(&self, attack: py::Attack, target: (f32, f32, f32)) -> PyResult<py::Legality> {
        let me = self.0.combatant().upgrade().unwrap();
        let (x, y, z) = target;

        // Check if we've already used all actions for this turn, and return early if so.
        // If not, update the used action count by 1, and continue.
        if let l @ rs::Legality::Illegal(_) = self.0.actions.can_use() {
            return l.try_into();
        }

        attack
            .0
            .make_attack(&me, rs::P3::new(x, y, z))
            .map(py::AttackResult)
            .map(|res| {
                self.0.actions.mark_used();
                res
            })
            .try_into()
    }

    fn end(&self) -> PyResult<py::Legality> {
        let me = self.0.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        combat.initiative.advance_turn();

        py::Legality::void_success()
    }

    #[pyo3(signature = (mode = py::WALKING))]
    fn movement_directions(&self, mode: py::SpeedType) -> PyResult<py::Legality> {
        // Is there any movement left? If not, return early, stating it's illegal.
        if let l @ rs::Legality::Illegal(_) = self.0.movement.any_movement_left(mode.0) {
            return l.try_into();
        }

        let me = self.0.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();

        rs::Legality::Legal(
            DIRECTIONS
                .map(|(x, y, z)| rs::P3::new(x, y, z))
                .into_iter()
                .filter_map(|p| {
                    combat
                        .arena
                        .is_passable(
                            rs::P3::from(me.position.load().coords + p.coords),
                            me.stats.size,
                        )
                        .is_legal()
                        .then_some((p.x, p.y, p.z))
                })
                .collect::<Vec<_>>(),
        )
        .try_into()
    }

    fn is_combat_active(&self) -> bool {
        self.0.combatant().upgrade().is_some()
    }

    #[pyo3(signature = (mode = py::WALKING))]
    fn movement_directions_one_hot(&self, mode: py::SpeedType) -> [f32; 8] {
        // Is there any movement left? If not, return early.
        if let rs::Legality::Illegal(_) = self.0.movement.any_movement_left(mode.0) {
            return [0.0; 8];
        }

        let me = self.0.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();

        let directions = DIRECTIONS.map(|(x, y, z)| rs::P3::new(x, y, z));

        directions.map(|p| {
            if combat
                .arena
                .is_passable(
                    rs::P3::from(me.position.load().coords + p.coords),
                    me.stats.size,
                )
                .is_legal()
            {
                1.0
            } else {
                0.0
            }
        })
    }

    fn attack_directions(&self, attack: py::Attack, filter: PyObject) -> PyResult<py::Legality> {
        let me = self.0.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        let range = attack.0.range();
        match range {
            rs::Range::Reach => {
                // Assume reach is 5 ft. (this isn't always the case)
                // TODO: Look this up from the stats of the weapon + combatant.
                // TODO: Add a "grid search" struct to handle this.
                rs::Legality::Legal(
                    DIRECTIONS
                        .map(|(x, y, z)| rs::P3::new(x, y, z))
                        .into_iter()
                        .map(|p| -> PyResult<_> {
                            let combatants = combat
                                .arena
                                .at(rs::P3::from(me.position.load().coords + p.coords))
                                .combatants;

                            // Very ugly looking filtering via Python callback.
                            if Python::with_gil(|py| -> PyResult<_> {
                                combatants
                                    .into_iter()
                                    .map(|combatant| {
                                        filter
                                            .call1(py, (py::Combatant(combatant),))?
                                            .extract::<bool>(py)
                                    })
                                    .try_fold(0, |acc, res| res.map(|res| acc + res as usize))
                            })? > 0
                            {
                                return Ok(Some((p.x, p.y, p.z)));
                            }

                            Ok(None)
                        })
                        .collect::<PyResult<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<(f32, f32, f32)>>(),
                )
                .try_into()
            }
            _ => todo!(),
        }
    }

    fn attack_directions_one_hot(
        &self,
        attack: py::Attack,
        filter: PyObject,
    ) -> PyResult<[f32; 8]> {
        let me = self.0.combatant().upgrade().unwrap();
        let combat = me.combat.upgrade().unwrap();
        let range = attack.0.range();
        let ret = match range {
            rs::Range::Reach => {
                DIRECTIONS
                    .map(|(x, y, z)| rs::P3::new(x, y, z))
                    .into_iter()
                    .map(|p| -> PyResult<_> {
                        let combatants = combat
                            .arena
                            .at(rs::P3::from(me.position.load().coords + p.coords))
                            .combatants;

                        // Very ugly looking filtering via Python callback.
                        if Python::with_gil(|py| -> PyResult<_> {
                            combatants
                                .into_iter()
                                .map(|combatant| {
                                    filter
                                        .call1(py, (py::Combatant(combatant),))?
                                        .extract::<bool>(py)
                                })
                                .try_fold(0, |acc, res| res.map(|res| acc + res as usize))
                        })? > 0
                        {
                            return Ok(1.0);
                        }

                        Ok(0.0)
                    })
                    .collect::<PyResult<Vec<_>>>()?
                    .try_into()
                    .unwrap()
            }

            _ => todo!(),
        };

        Ok(ret)
    }

    #[pyo3(signature = (mode = py::WALKING))]
    fn movement_left(&self, mode: py::SpeedType) -> u32 {
        self.0
            .combatant()
            .upgrade()
            .unwrap()
            .stats
            .speeds
            .of_type(mode.0)
            .unwrap_or(0)
            .saturating_sub(self.0.movement.used())
    }

    #[getter]
    fn actions_left(&self) -> u32 {
        self.0.actions.max() - self.0.actions.used()
    }

    #[getter]
    fn max_actions(&self) -> u32 {
        self.0.actions.max()
    }

    fn __repr__(&self) -> String {
        format!(
            "Turn(movement_used = {}, actions_used = {})",
            self.0.movement.used(),
            self.0.actions.used()
        )
    }
}
