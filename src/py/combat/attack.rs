use pyo3::{pyclass, pymethods};
use rs::pretty_damage;

#[cfg(feature = "vis")]
use crate::vis::rich::RichFormatting;

mod rs {
    pub(crate) use crate::core::{
        combat::turn::attack::{
            melee::MeleeAttackAction, ranged::RangedAttackAction, roll::AttackRoll, AttackAction,
            AttackResult,
        },
        stats::damage::pretty_damage,
    };
}

mod py {
    pub(crate) use crate::py::damage::Damage;
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Attack(pub(super) rs::AttackAction);

impl Attack {
    pub fn to_string(&self) -> String {
        match &self.0 {
            rs::AttackAction::Melee(
                act @ rs::MeleeAttackAction {
                    name,
                    to_hit,
                    target,
                    damage,
                    ..
                },
            ) => owo_colors::with_override(false, || {
                format!(
                    "{name}. Melee Weapon Attack: {to_hit} to hit, {}, {target}. Hit: {}.",
                    act.range(),
                    rs::pretty_damage(damage.as_slice())
                )
            }),
            rs::AttackAction::Ranged(
                act @ rs::RangedAttackAction {
                    name,
                    to_hit,
                    target,
                    damage,
                    ..
                },
            ) => owo_colors::with_override(false, || {
                format!(
                    "{name}. Ranged Weapon Attack: {to_hit} to hit, {}, {target}. Hit: {}.",
                    act.range(),
                    rs::pretty_damage(damage.as_slice())
                )
            }),
        }
    }

    pub fn to_html_string(&self) -> String {
        match &self.0 {
            rs::AttackAction::Melee(rs::MeleeAttackAction {
                name,
                to_hit,
                range,
                target,
                damage,
                ..
            }) => owo_colors::with_override(false, || {
                format!(
                    r#"<div class="attack"><strong>{name}.</strong> <em>Melee Weapon Attack:</em> {to_hit} to hit, {range}, {target}. <em>Hit</em>: {} damage.</div>"#,
                    rs::pretty_damage(damage.as_slice())
                )
            }),
            rs::AttackAction::Ranged(rs::RangedAttackAction {
                name,
                to_hit,
                range,
                target,
                damage,
                ..
            }) => owo_colors::with_override(false, || {
                format!(
                    r#"<div class="attack"><strong>{name}.</strong> <em>Ranged Weapon Attack:</em> {to_hit} to hit, {range}, {target}. <em>Hit</em>: {} damage.</div>"#,
                    rs::pretty_damage(damage.as_slice())
                )
            }),
        }
    }
}

#[pymethods]
impl Attack {
    #[getter]
    fn name(&self) -> String {
        match &self.0 {
            rs::AttackAction::Melee(act) => act.name.clone(),
            rs::AttackAction::Ranged(act) => act.name.clone(),
        }
    }

    #[getter]
    #[pyo3(name = "type")]
    fn _type(&self) -> String {
        match &self.0 {
            rs::AttackAction::Melee(_) => "melee".to_string(),
            rs::AttackAction::Ranged(_) => "ranged".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn _repr_html_(&self) -> String {
        self.to_html_string()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct AttackResult(pub(super) rs::AttackResult);

#[pymethods]
impl AttackResult {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn _repr_html_(&self) -> String {
        owo_colors::with_override(false, || match &self.0 {
            rs::AttackResult::NoHit {
                attack,
                attacker,
                target,
                to_hit,
            } => {
                let attacker = &attacker.upgrade().unwrap().name;
                let target = &target.upgrade().unwrap().name;
                let name = attack.name();
                let to_hit = to_hit.html();

                format!(
                    r#"<div style="border: 1px solid black;">
                            <div class="header" style="padding: 0.25em 1em 0.5em 1em; border-bottom: 1px solid black; font-weight: bold;">{attacker} attacks {target}</em></div>
                            <div class="attack-body" style="padding: 1em; display: grid; grid-template-columns: max-content 1fr; gap: 0.5em 1em; font-size: 80%;">
                                <span>Weapon</span><span>{name}</span>
                                <span>To Hit</span><span>{to_hit} Miss</span>
                            </div>
                        </div>"#
                )
            }

            rs::AttackResult::Hit {
                attacker,
                attack,
                target,
                to_hit,
                damage,
            } => {
                let attacker = &attacker.upgrade().unwrap().name;
                let target = &target.upgrade().unwrap().name;
                let name = attack.name();
                let attack_dmg = pretty_damage(attack.damage());
                let to_hit = to_hit.html();
                let damage_dealt = damage.to_string();

                format!(
                    r#"<div style="border: 1px solid black;">
                            <div class="header" style="padding: 0.25em 1em 0.5em 1em; border-bottom: 1px solid black; font-weight: bold;">{attacker} attacks {target}</em></div>
                            <div class="attack-body" style="padding: 1em; display: grid; grid-template-columns: max-content 1fr; gap: 0.5em 1em; font-size: 80%;">
                                <span>Weapon</span><span>{name}</span>
                                <span>To Hit</span><span>{to_hit} Hit</span>
                                <span>Damage</span><span>{attack_dmg} &rarr; {damage_dealt}</span>
                            </div>
                        </div>"#,
                )
            }
        })
    }

    #[getter]
    fn successful(&self) -> bool {
        match self.0 {
            rs::AttackResult::Hit { .. } => true,
            rs::AttackResult::NoHit { .. } => false,
        }
    }

    #[getter]
    fn to_hit(&self) -> AttackRoll {
        match &self.0 {
            rs::AttackResult::Hit { to_hit, .. } => AttackRoll(to_hit.clone()),
            rs::AttackResult::NoHit { to_hit, .. } => AttackRoll(to_hit.clone()),
        }
    }

    #[getter]
    fn damage(&self) -> Option<py::Damage> {
        match &self.0 {
            rs::AttackResult::Hit { damage, .. } => Some(py::Damage(damage.clone())),
            _ => None,
        }
    }
}

#[pyclass]
pub struct AttackRoll(pub(in crate::py) rs::AttackRoll);

#[pymethods]
impl AttackRoll {
    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }
}
