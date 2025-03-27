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
    pub(crate) use crate::core::{combat::turn::TurnCtx, geom::P3};
}

#[pyclass]
pub struct Turn(pub(super) Arc<rs::TurnCtx>);

#[pymethods]
impl Turn {
    fn is_combat_active(&self) -> bool {
        self.0.is_combat_active()
    }

    #[pyo3(name = "move")]
    #[pyo3(signature = (delta, mode = py::WALKING))]
    fn try_move(&self, delta: (f32, f32, f32), mode: py::SpeedType) -> PyResult<py::Legality> {
        let (x, y, z) = delta;
        self.0.try_move(rs::P3::new(x, y, z), mode.0).try_into()
    }

    fn attack(&self, attack: py::Attack, target: (f32, f32, f32)) -> PyResult<py::Legality> {
        let (x, y, z) = target;
        self.0
            .attack(attack.0, rs::P3::new(x, y, z))
            .map(py::AttackResult)
            .try_into()
    }

    fn end(&self) -> PyResult<py::Legality> {
        self.0.end().try_into()
    }

    #[pyo3(signature = (mode = py::WALKING))]
    fn movement_directions(&self, mode: py::SpeedType) -> PyResult<py::Legality> {
        self.0
            .movement_directions(mode.0)
            .map(|v| {
                v.into_iter()
                    .map(|p3| (p3.x, p3.y, p3.z))
                    .collect::<Vec<_>>()
            })
            .try_into()
    }

    #[pyo3(signature = (mode = py::WALKING))]
    fn movement_directions_one_hot(&self, mode: py::SpeedType) -> [f32; 8] {
        self.0.movement_directions_one_hot(mode.0)
    }

    fn attack_directions(&self, attack: py::Attack, filter: PyObject) -> PyResult<py::Legality> {
        self.0
            .attack_directions(attack.0, |combatants| -> PyResult<bool> {
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
                    return Ok(true);
                }

                Ok(false)
            })?
            .map(|dirs| {
                dirs.into_iter()
                    .map(|p| (p.x, p.y, p.z))
                    .collect::<Vec<_>>()
            })
            .try_into()
    }

    fn attack_directions_one_hot(
        &self,
        attack: py::Attack,
        filter: PyObject,
    ) -> PyResult<[f32; 8]> {
        self.0
            .attack_directions_one_hot(attack.0, |combatants| -> PyResult<_> {
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
                    return Ok(true);
                }

                Ok(false)
            })
    }

    #[pyo3(signature = (mode = py::WALKING))]
    fn movement_left(&self, mode: py::SpeedType) -> u32 {
       self.0.movement_left(mode.0)
    }

    #[getter]
    fn actions_left(&self) -> u32 {
        self.0.actions_left()
    }

    #[getter]
    fn max_actions(&self) -> u32 {
        self.0.max_actions()
    }

    fn __repr__(&self) -> String {
        format!(
            "Turn(movement_used = {}, actions_used = {})",
            self.0.movement.used(),
            self.0.actions.used()
        )
    }
}
