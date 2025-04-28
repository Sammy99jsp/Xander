use pyo3::{pyclass, pymethods};

mod py {
    pub(crate) use crate::py::combat::attack::Attack;
}

mod rs {
    pub(crate) use crate::core::combat::turn::{action::Action, attack::AttackAction};
}

#[pyclass]
pub struct Action(pub(in crate::py) rs::Action);

#[pymethods]
impl Action {
    fn __repr__(&self) -> String {
        match &self.0 {
            rs::Action::Attack(attack) => py::Attack(attack.clone()).to_string(),
            _ => todo!(),
        }
    }

    fn _repr_html_(&self) -> String {
        match &self.0 {
            rs::Action::Attack(attack) => py::Attack(attack.clone()).to_html_string(),
            _ => todo!(),
        }
    }

    #[allow(unreachable_patterns)]
    fn as_attack(&self) -> Option<py::Attack> {
        match &self.0 {
            rs::Action::Attack(attack) => Some(py::Attack(attack.clone())),
            _ => None,
        }
    }
}
