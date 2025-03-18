use std::{fs, sync::Weak};

use pyo3::{pyclass, pymethods, PyResult};

mod rs {
    pub(crate) use crate::core::combat::arena::{Arena, SimpleArenaParams};
}

#[pyclass]
pub struct Arena(pub(super) Weak<dyn rs::Arena>);

#[pymethods]
impl Arena {
    fn __repr__(&self) -> String {
        "Arena".to_string()
    }

    #[cfg(feature = "vis")]
    fn _repr_html_(&self) -> String {
        self.0.upgrade().unwrap().as_ref().visualize().as_html()
    }

    #[cfg(feature = "vis")]
    fn save_image(&self, path: String) -> PyResult<()> {
        use pyo3::exceptions::PyValueError;


        if !path.ends_with(".png") {
            return Err(PyValueError::new_err("only can export .png files!"));
        }

        let vis = self.0.upgrade().unwrap().as_ref().visualize();
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        vis.write_to(&mut file)?;

        Ok(())
    }

    #[getter]
    fn grid_dimensions(&self) -> (u32, u32) {
        self.0.upgrade().unwrap().as_ref().grid_size()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Simple(pub(super) rs::SimpleArenaParams);

#[pymethods]
impl Simple {
    #[new]
    fn __init__(width: u32, height: u32) -> Self {
        Self(rs::SimpleArenaParams::new(width as f32, height as f32))
    }
}
