use std::{
    any::Any,
    collections::HashMap,
    fmt::Write,
    sync::{Arc, RwLock},
};

use crossbeam_utils::atomic::AtomicCell;
use pyo3::{
    exceptions::{PyException, PyTypeError, PyValueError},
    pyclass, pymethods, PyObject, PyResult, Python,
};
use rs::arena::{grid_round_p, SQUARE_LENGTH};

pub mod action;
pub mod arena;
pub mod attack;
pub mod speed;
pub mod turn;

mod rs {
    pub(crate) use crate::core::{
        combat::{arena, turn::TurnCtx, Combat, CombatHook, Combatant, InitiativeRoll},
        geom::P3,
    };
}

impl rs::CombatHook for PyObject {
    fn turn(&self, turn: Arc<rs::TurnCtx>) {
        Python::with_gil(|py| {
            let res = self.call1(py, (turn::Turn(turn),));
            if res.is_err() {
                PyTypeError::new_err("expected a Callable[[Turn], None] as a hook").restore(py);
            }
        });
    }
}

#[pyclass]
pub struct Combat(Arc<rs::Combat>);

#[pymethods]
impl Combat {
    #[new]
    fn __init__(arena: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            if let Ok(arena::Simple(simple)) = arena.extract::<arena::Simple>(py) {
                return Ok(Self(rs::Combat::new(None, |weak| {
                    rs::arena::SimpleArena::new(weak, simple)
                })));
            }

            Err(PyTypeError::new_err("Expected an Arena here"))
        })
    }

    #[getter]
    fn arena(&self) -> arena::Arena {
        arena::Arena(Arc::downgrade(&self.0.arena))
    }

    fn join(
        &mut self,
        monster: super::Stats,
        name: String,
        position: (f32, f32, f32),
        hook: PyObject,
    ) -> Combatant {
        let (x, y, z) = position;
        let combatant = Arc::new(rs::Combatant {
            combat: Arc::downgrade(&self.0),
            name,
            initiative: rs::InitiativeRoll(monster.0.initiative.get().result()),
            stats: monster.0,
            position: AtomicCell::new(rs::P3::new(x, y, z)),
            hook: Box::new(hook),
        });

        self.0.initiative.add(combatant.clone());
        Combatant(combatant)
    }

    fn _repr_html_(&self) -> String {
        let mut s = r#"<div style="display: grid; grid-template-columns: max-content max-content max-content max-content; gap: 4px 1em; margin: 0 25vw;">
        <span style="font-weight: bold;"></span>
        <span style="font-weight: bold;">Initiative</span>
        <span style="font-weight: bold;">Name</span>
        <span style="font-weight: bold;">Health</span>"#.to_string();

        let current = self.0.initiative.current();
        for combatant in self.0.initiative.as_vec() {
            write!(
                s,
                r#"<span style="font-weight: bold;">{}</span>
                <span>{}</span>
                <span>{}</span>
                <span>{} / {} HP</span>"#,
                if Arc::ptr_eq(&combatant, &current) {
                    "★"
                } else if combatant.stats.is_dead() {
                    "💀"
                } else {
                    ""
                },
                combatant.initiative.0,
                &combatant.name,
                combatant.stats.hp(),
                combatant.stats.max_hp()
            )
            .unwrap();
        }

        s.write_str("</div>").unwrap();
        s
    }

    fn __repr__(&self) -> String {
        format!("Combat({} members)", self.0.len())
    }

    fn step(&self) {
        // Update internal state.
        self.0.step();
    }

    #[getter]
    fn current(&self) -> Combatant {
        Combatant(self.0.initiative.current().clone())
    }

    #[getter]
    fn rounds(&self) -> usize {
        self.0.initiative.rounds.load()
    }

    #[getter]
    fn combatants(&self) -> Vec<Combatant> {
        self.0
            .initiative
            .as_vec()
            .iter()
            .map(|c| Combatant(c.clone()))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Combatant(pub(in crate::py) Arc<rs::Combatant>);

#[pymethods]
impl Combatant {
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    #[getter]
    fn position(&self) -> (f32, f32, f32) {
        let p = self.0.position.load();
        (p.x, p.y, p.z)
    }

    #[getter]
    fn stats(&self) -> super::Stats {
        super::Stats(self.0.stats.clone())
    }

    #[getter]
    fn initiative(&self) -> i32 {
        self.0.initiative.0
    }

    fn observe(&self) -> Vec<i32> {
        let combat = self.0.combat.upgrade().unwrap();
        let (width, height) = combat.arena.grid_size();
        let mut ret = vec![0; (width * height) as usize];

        combat
            .initiative
            .members
            .read()
            .unwrap()
            .iter()
            .for_each(|combatant| {
                let p = grid_round_p(combatant.position.load()) / SQUARE_LENGTH;
                let idx = (p.x as u32 + p.y as u32 * width) as usize;
                // println!(" {p:?} -> {idx}");
                ret[idx] = match combatant {
                    c if Arc::ptr_eq(c, &self.0) => 0,
                    c if c.stats.is_dead() => 0,
                    _ => 1,
                }
            });

        ret
    }

    fn __repr__(&self) -> String {
        format!(
            "<{} {}/{} HP>",
            self.0.name,
            self.0.stats.hp(),
            self.0.stats.max_hp()
        )
    }

    fn __eq__(&self, other: &Combatant) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
