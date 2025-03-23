#![feature(
    rustc_attrs,
    int_roundings,
    ptr_as_ref_unchecked,
    const_option,
    box_patterns,
    let_chains,
    ptr_fn_addr_eq,
    assert_matches,
    fn_traits,
    trait_alias,
    debug_closure_helpers,
    iter_intersperse,
    get_mut_unchecked,
    cell_update,
    decl_macro,
    try_trait_v2,
    pointer_is_aligned_to,
    lock_value_accessors
)]
#![allow(internal_features)]

pub mod core;
pub mod utils;

pub mod serde;


#[cfg(feature = "web")]
pub mod web;

// PYTHON //
#[cfg(all(feature = "vis", feature = "py"))]
pub mod vis;

#[cfg(feature = "py")]
#[doc(hidden)]
pub(crate) mod py;
pub mod agents;

#[cfg(feature = "py")]
use pyo3::prelude::*;

#[cfg(feature = "py")]
#[pymodule]
mod xander {
    use pyo3::pymodule;

    #[pymodule]
    mod engine {
        use pyo3::prelude::*;

        #[pymodule_init]
        fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            Python::with_gil(|py| {
                py.import("sys")?
                    .getattr("modules")?
                    .set_item("xander.engine", m)
            })
        }

        #[pymodule]
        mod dice {
            use pyo3::prelude::*;

            use crate::{core::dice::*, py};

            #[pymodule_init]
            fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                Python::with_gil(|py| {
                    py.import("sys")?
                        .getattr("modules")?
                        .set_item("xander.engine.dice", m)
                })?;

                m.add_class::<py::Die>()?;
                m.add_class::<py::DExpr>()?;
                m.add_class::<py::DEvalTree>()?;

                // TODO: Move this out!
                m.add("D4", py::Die(D4))?;
                m.add("D6", py::Die(D6))?;
                m.add("D8", py::Die(D8))?;
                m.add("D10", py::Die(D10))?;
                m.add("D12", py::Die(D12))?;
                m.add("D20", py::Die(D20))?;
                m.add("D100", py::Die(D100))?;

                m.add_function(wrap_pyfunction!(crate::py::dice::set_seed, m)?)?;
                m.add_function(wrap_pyfunction!(crate::py::dice::random_seed, m)?)?;

                Ok(())
            }
        }
        #[pymodule]
        mod actors {
            use pyo3::{
                types::{PyAnyMethods, PyModule, PyModuleMethods},
                Bound, PyResult, Python,
            };

            use crate::py;

            #[pymodule_init]
            fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                Python::with_gil(|py| {
                    py.import("sys")?
                        .getattr("modules")?
                        .set_item("xander.engine.actors", m)
                })?;

                m.add_class::<py::Stats>()?;

                Ok(())
            }
        }

        #[pymodule]
        mod damage {
            use pyo3::{
                types::{PyAnyMethods, PyModule, PyModuleMethods},
                Bound, PyResult, Python,
            };

            use crate::core::stats::damage as rs;
            use crate::py::damage as py;

            #[pymodule_init]
            fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                Python::with_gil(|py| {
                    py.import("sys")?
                        .getattr("modules")?
                        .set_item("xander.engine.damage", m)
                })?;

                m.add_class::<py::DamageType>()?;
                m.add_class::<py::Damage>()?;

                m.add("Acid", py::DamageType::of(rs::Acid))?;
                m.add("Bludgeoning", py::DamageType::of(rs::Bludgeoning))?;
                m.add("Cold", py::DamageType::of(rs::Cold))?;
                m.add("Fire", py::DamageType::of(rs::Fire))?;
                m.add("Force", py::DamageType::of(rs::Force))?;
                m.add("Lightning", py::DamageType::of(rs::Lightning))?;
                m.add("Necrotic", py::DamageType::of(rs::Necrotic))?;
                m.add("Piercing", py::DamageType::of(rs::Piercing))?;
                m.add("Poison", py::DamageType::of(rs::Poison))?;
                m.add("Psychic", py::DamageType::of(rs::Psychic))?;
                m.add("Radiant", py::DamageType::of(rs::Radiant))?;
                m.add("Slashing", py::DamageType::of(rs::Slashing))?;
                m.add("Thunder", py::DamageType::of(rs::Thunder))?;

                m.add_class::<py::DamageCause>()?;
                m.add_class::<py::DamageType>()?;

                Ok(())
            }
        }

        #[pymodule]
        mod combat {
            use pyo3::{
                pymodule,
                types::{PyAnyMethods, PyModule, PyModuleMethods},
                Bound, PyResult, Python,
            };

            use crate::py::combat as py;

            #[pymodule_init]
            fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                Python::with_gil(|py| {
                    py.import("sys")?
                        .getattr("modules")?
                        .set_item("xander.engine.combat", m)
                })?;

                m.add_class::<py::Combat>()?;
                m.add_class::<py::Combatant>()?;

                Ok(())
            }

            #[pymodule]
            mod speed {
                use pyo3::{
                    types::{PyAnyMethods, PyModule, PyModuleMethods},
                    Bound, PyResult, Python,
                };

                use crate::py::combat::speed as py;

                #[pymodule_init]
                fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                    Python::with_gil(|py| {
                        py.import("sys")?
                            .getattr("modules")?
                            .set_item("xander.engine.combat.speed", m)
                    })?;

                    m.add_class::<py::SpeedType>()?;

                    m.add("Walking", py::WALKING)?;
                    m.add("Burrowing", py::BURROWING)?;
                    m.add("Climbing", py::CLIMBING)?;
                    m.add("Flying", py::FLYING)?;
                    m.add("Swimming", py::SWIMMING)?;
                    m.add("Crawling", py::CRAWLING)?;

                    Ok(())
                }
            }

            #[pymodule]
            mod turn {
                use pyo3::{
                    pymodule,
                    types::{PyAnyMethods, PyModule, PyModuleMethods},
                    Bound, PyResult, Python,
                };

                use crate::py::combat::turn as py;

                #[pymodule_init]
                fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                    Python::with_gil(|py| {
                        py.import("sys")?
                            .getattr("modules")?
                            .set_item("xander.engine.combat.turn", m)
                    })?;

                    m.add_class::<py::Turn>()?;

                    Ok(())
                }
            }

            #[pymodule]
            mod action {
                use pyo3::{
                    pymodule,
                    types::{PyAnyMethods, PyModule, PyModuleMethods},
                    Bound, PyResult, Python,
                };

                use crate::py::combat::action as py;

                #[pymodule_init]
                fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                    Python::with_gil(|py| {
                        py.import("sys")?
                            .getattr("modules")?
                            .set_item("xander.engine.combat.action", m)
                    })?;

                    m.add_class::<py::Action>()?;

                    Ok(())
                }
                #[pymodule]
                mod attack {
                    use pyo3::{
                        pymodule,
                        types::{PyAnyMethods, PyModule, PyModuleMethods},
                        Bound, PyResult, Python,
                    };

                    use crate::py::combat::attack as py;

                    #[pymodule_init]
                    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                        Python::with_gil(|py| {
                            py.import("sys")?
                                .getattr("modules")?
                                .set_item("xander.engine.combat.action.attack", m)
                        })?;

                        m.add_class::<py::Attack>()?;
                        m.add_class::<py::AttackResult>()?;
                        m.add_class::<py::AttackRoll>()?;

                        Ok(())
                    }
                }
            }

            #[pymodule]
            mod arena {
                use crate::py::combat::arena as py;
                use pyo3::{
                    types::{PyAnyMethods, PyModule, PyModuleMethods},
                    Bound, PyResult, Python,
                };

                #[pymodule_init]
                fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                    Python::with_gil(|py| {
                        py.import("sys")?
                            .getattr("modules")?
                            .set_item("xander.engine.combat.arena", m)
                    })?;

                    m.add_class::<py::Arena>()?;
                    m.add_class::<py::Simple>()?;

                    Ok(())
                }
            }
        }

        #[pymodule]
        mod legality {
            use crate::py::legality as py;
            use pyo3::{
                types::{PyAnyMethods, PyModule, PyModuleMethods},
                Bound, PyResult, Python,
            };

            #[pymodule_init]
            fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
                Python::with_gil(|py| {
                    py.import("sys")?
                        .getattr("modules")?
                        .set_item("xander.engine.legality", m)
                })?;

                m.add_class::<py::Legality>()?;

                Ok(())
            }
        }
    }
}

