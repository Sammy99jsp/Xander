//!
//! ## Dice Expressions.
//! 
//! The main type here is [DExpr].
//! 
//! ### Example
//! 
//! ```
//! use xander_engine::core::dice::{self, D20};
//! dice::random_seed();
//! 
//! let to_hit = D20.advantage() + 6;
//! let eval = to_hit.evaluate();
//! println!("{to_hit} => {eval} == {}", eval.result());
//! ```
//!

use std::{ops::{Add, Div, Mul, Sub}, ptr::fn_addr_eq as fn_eq};

use owo_colors::OwoColorize;

use crate::utils::prettify;

use super::Die;

///
/// Represents a valid dice expression,
/// which can contain:
/// * Dice rolls (with advantage, disadvantage, or neither)
/// * Operations: `+`, `-`, `*`, `/`
/// 
/// This can then be evaluated into a [DEvalTree]
/// through [DExpr::evaluate]
/// 
#[derive(Debug, Clone, PartialEq)]
pub enum DExpr {
    Modifier(i32),
    Die {
        die: Die,
        /// Has this die received both advantage and disadvantage to it
        both_adv_dis: bool,
    },
    Advantage(Die),
    Disadvantage(Die),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
}

impl DExpr {
    ///
    /// Apply advantage to **ALL THE DICE ROLLS** in this expression.
    ///
    pub fn advantage(self) -> Self {
        match self {
            k @ Self::Modifier(_) => k,
            Self::Die {
                die,
                both_adv_dis: false,
            } => Self::Advantage(die),

            // "If multiple situations affect a roll and each one
            // grants advantage or imposes disadvantage on it, you
            // don’t roll more than one additional d20. If two
            // favorable situations grant advantage, for example,
            // you still roll only one additional d20."
            adv @ Self::Advantage(_) => adv,

            // "If circumstances cause a roll to have both
            // advantage and disadvantage, you are considered to
            // have neither of them, and you roll one d20...
            Self::Disadvantage(die) => Self::Die {
                die,
                both_adv_dis: true,
            },

            // ...This is true even if multiple circumstances impose
            // disadvantage and only one grants advantage or vice
            // versa. In such a situation, you have neither
            // advantage nor disadvantage."
            d @ Self::Die {
                both_adv_dis: true,
                ..
            } => d,

            Self::Add(box left, box right) => {
                Self::Add(Box::new(left.advantage()), Box::new(right.advantage()))
            }
            Self::Sub(box left, box right) => {
                Self::Sub(Box::new(left.advantage()), Box::new(right.advantage()))
            }
            Self::Mul(box left, box right) => {
                Self::Mul(Box::new(left.advantage()), Box::new(right.advantage()))
            }
            Self::Div(box left, box right) => {
                Self::Div(Box::new(left.advantage()), Box::new(right.advantage()))
            }
        }
    }

    ///
    /// Apply disadvantage to  **ALL THE DICE ROLLS** in this expression.
    ///
    pub fn disadvantage(self) -> Self {
        match self {
            k @ Self::Modifier(_) => k,
            Self::Die {
                die,
                both_adv_dis: false,
            } => Self::Disadvantage(die),

            // "If circumstances cause a roll to have both
            // advantage and disadvantage, you are considered to
            // have neither of them, and you roll one d20...
            Self::Advantage(die) => Self::Die {
                die,
                both_adv_dis: true,
            },

            // ...This is true even if multiple circumstances impose
            // disadvantage and only one grants advantage or vice
            // versa. In such a situation, you have neither
            // advantage nor disadvantage."
            d @ Self::Die {
                both_adv_dis: true,
                ..
            } => d,

            // "If multiple situations affect a roll and each one
            // grants advantage or imposes disadvantage on it, you
            // don’t roll more than one additional d20. If two
            // favorable situations grant advantage, for example,
            // you still roll only one additional d20."
            dis @ Self::Disadvantage(_) => dis,
            Self::Add(box left, box right) => {
                Self::Add(Box::new(left.advantage()), Box::new(right.advantage()))
            }
            Self::Sub(box left, box right) => {
                Self::Sub(Box::new(left.advantage()), Box::new(right.advantage()))
            }
            Self::Mul(box left, box right) => {
                Self::Mul(Box::new(left.advantage()), Box::new(right.advantage()))
            }
            Self::Div(box left, box right) => {
                Self::Div(Box::new(left.advantage()), Box::new(right.advantage()))
            }
        }
    }

    ///
    /// Evaluate this expression into a [DEvalTree],
    /// which can be used to find the end result ([DEvalTree::result]).
    /// 
    pub fn evaluate(&self) -> DEvalTree {
        match self {
            DExpr::Modifier(c) => DEvalTree::Modifier(*c),
            DExpr::Die { die, .. } => DEvalTree::Roll(die.roll()),
            DExpr::Advantage(die) => DEvalTree::Advantage(die.roll(), die.roll()),
            DExpr::Disadvantage(die) => DEvalTree::Disadvantage(die.roll(), die.roll()),
            DExpr::Add(left, right) => {
                DEvalTree::Add(Box::new(left.evaluate()), Box::new(right.evaluate()))
            }
            DExpr::Sub(left, right) => {
                DEvalTree::Sub(Box::new(left.evaluate()), Box::new(right.evaluate()))
            }
            DExpr::Mul(left, right) => {
                DEvalTree::Mul(Box::new(left.evaluate()), Box::new(right.evaluate()))
            }
            DExpr::Div(left, right) => {
                DEvalTree::Div(Box::new(left.evaluate()), Box::new(right.evaluate()))
            }
        }
    }
}

///
/// Represents an evaluated [DExpr] with the result of dice rolls,
/// but an the same tree structure.
/// 
/// You may find the numerical result using [DEvalTree::result].
/// 
#[derive(Debug, Clone, PartialEq)]
pub enum DEvalTree {
    Modifier(i32),
    Roll(i32),
    Advantage(i32, i32),
    Disadvantage(i32, i32),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
}

impl DEvalTree {
    ///
    /// Return the numerical result of the dice
    /// rolls and calculations in this tree.
    /// 
    pub fn result(&self) -> i32 {
        match self {
            DEvalTree::Modifier(modifier) => *modifier,
            DEvalTree::Roll(roll) => *roll,
            DEvalTree::Advantage(r1, r2) => *r1.max(r2),
            DEvalTree::Disadvantage(r1, r2) => *r1.min(r2),
            DEvalTree::Add(t1, t2) => t1.result() + t2.result(),
            DEvalTree::Sub(t1, t2) => t1.result() - t2.result(),
            DEvalTree::Mul(t1, t2) => t1.result() * t2.result(),
            DEvalTree::Div(t1, t2) => t1.result() / t2.result(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right
}

type BinaryOp<T> = fn(Box<T>, Box<T>) -> T;

///
/// Internal utility crate to handle proper
/// use of parentheses (only where necessary)
/// of [DExpr] and [DEvalTree]. 
/// 
trait Bracketed: std::fmt::Debug + std::fmt::Display {
    /// + and - operations.
    const ADD_SUB: [BinaryOp<Self>; 2];
    /// * and / operations.
    const MUL_DIV: [BinaryOp<Self>; 2];

    /// + - * /
    const ALL: [BinaryOp<Self>; 4] = [
        Self::ADD_SUB[0],
        Self::ADD_SUB[1],
        Self::MUL_DIV[0],
        Self::MUL_DIV[1],
    ];

    ///
    /// Is this term a unary expression?
    /// 
    fn is_unary(&self) -> bool;

    ///
    /// Is this term an `op` binary expression?
    /// 
    /// Note: use the enum variant itself as the function,
    /// such as [DExpr::Add].
    /// 
    fn is_binary(&self, op: BinaryOp<Self>) -> bool;

    ///
    /// Should this expression be bracketed given the binary operation `op` will
    /// be applied to it?
    ///
    /// Returns [Some] if the operation is Add, Sub, Mul, or Div;
    /// [None] otherwise.
    ///
    #[rustfmt::skip]
    fn should_bracket_for(&self, op: BinaryOp<Self>, side: Side) -> Option<bool> {
        if self.is_unary() {
            return Some(false);
        }

        fn any<const K: usize, T: PartialEq<T> + Copy, F: Fn(T) -> bool>(
            arr: [T; K],
            f: F,
        ) -> bool {
            arr.iter().copied().any(f)
        }

        match (self, op) {
            // (Self::Add | Self::SUB, SUB)
            (s, op)
                if any(Self::ADD_SUB, |el| s.is_binary(el))
                && fn_eq(Self::ADD_SUB[1], op)
            => Some(side == Side::Right), // Apply this only for the right side.

            // (Self::Add | Self::Sub, ADD)
            (s, op)
                if any(Self::ADD_SUB, |el| s.is_binary(el))
                && fn_eq(Self::ADD_SUB[0], op)
                => Some(false),
            
            // (Self::Add | Self::Sub, MUL | DIV)
            (s, op)
                if any(Self::ADD_SUB, |el| s.is_binary(el))
                && any(Self::MUL_DIV, |el| fn_eq(el, op))
                => Some(true),
            
            // (Self::Mul | Self::DIV, DIV)
            (s, op)
                if any(Self::MUL_DIV, |el| s.is_binary(el))
                && fn_eq(Self::MUL_DIV[1], op)
            => Some(side == Side::Right),

            // (Self::Mul | Self::Div, ADD | SUB | MUL | DIV*) *should be covered by above arm.
            (s, op)
                if any(Self::MUL_DIV, |el| s.is_binary(el))
                && any(Self::ALL, |el| fn_eq(el, op))
                => Some(false),

            _ => None,
        }
    }

    ///
    /// Write this term with parentheses (brackets)
    /// if necessary.
    /// 
    fn write_bracketed(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        op: BinaryOp<Self>,
        side: Side,
    ) -> std::fmt::Result {
        let parens = self.should_bracket_for(op, side).unwrap();
        if parens {
            write!(f, "(")?;
        }

        write!(f, "{self}")?;

        if parens {
            write!(f, ")")?;
        }

        Ok(())
    }
}

impl Bracketed for DExpr {
    const ADD_SUB: [BinaryOp<Self>; 2] = [Self::Add, Self::Sub];
    const MUL_DIV: [BinaryOp<Self>; 2] = [Self::Mul, Self::Div];

    fn is_binary(&self, op: BinaryOp<Self>) -> bool {
        if let Self::Add(_, _) = self && fn_eq(op, Self::Add as BinaryOp<Self>) {
            return true;
        }
        if let Self::Sub(_, _) = self && fn_eq(op, Self::Sub as BinaryOp<Self>) {
            return true;
        }
        if let Self::Mul(_, _) = self && fn_eq(op, Self::Mul as BinaryOp<Self>) {
            return true;
        }
        if let Self::Div(_, _) = self && fn_eq(op, Self::Div as BinaryOp<Self>) {
            return true;
        }
         
        false
    }
    
    fn is_unary(&self) -> bool {
        match self {
            Self::Modifier(_) | Self::Die {.. } | Self::Advantage(..) | Self::Disadvantage(..) => true,
            Self::Add(..) | Self::Sub(..) | Self::Mul(..) | Self::Div(..) => false,
        }
    }
}

impl std::fmt::Display for DExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Modifier(modifier) => write!(f, "{}", prettify::modifier(modifier)),
            Self::Die { die, .. } => write!(f, "{}", prettify::die(die)),
            Self::Advantage(die) => {
                write!(f, "Adv({})", prettify::die(die))
            }
            Self::Disadvantage(die) => {
                write!(f, "Dis({})", prettify::die(die))
            }
            Self::Add(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Add, Side::Left)?;
                write!(f, " + ")?;
                arg1.write_bracketed(f, Self::Add, Side::Right)
            }
            Self::Sub(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Sub, Side::Left)?;
                write!(f, " - ")?;
                arg1.write_bracketed(f, Self::Sub, Side::Right)
            }
            Self::Mul(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Mul, Side::Left)?;
                write!(f, " * ")?;
                arg1.write_bracketed(f, Self::Mul, Side::Right)
            }
            Self::Div(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Div, Side::Left)?;
                write!(f, " / ")?;
                arg1.write_bracketed(f, Self::Div, Side::Right)
            }
        }
    }
}

impl Bracketed for DEvalTree {
    const ADD_SUB: [BinaryOp<Self>; 2] = [Self::Add, Self::Sub];
    const MUL_DIV: [BinaryOp<Self>; 2] = [Self::Mul, Self::Div];

    fn is_binary(&self, op: BinaryOp<Self>) -> bool {
        if let Self::Add(_, _) = self && fn_eq(op, Self::Add as BinaryOp<Self>) {
            return true;
        }
        if let Self::Sub(_, _) = self && fn_eq(op, Self::Sub as BinaryOp<Self>) {
            return true;
        }
        if let Self::Mul(_, _) = self && fn_eq(op, Self::Mul as BinaryOp<Self>) {
            return true;
        }
        if let Self::Div(_, _) = self && fn_eq(op, Self::Div as BinaryOp<Self>) {
            return true;
        }
         
        false
    }

    fn is_unary(&self) -> bool {
        match self {
            Self::Modifier(_) | Self::Roll(..) | Self::Advantage(..) | Self::Disadvantage(..) => true,
            Self::Add(..) | Self::Sub(..) | Self::Mul(..) | Self::Div(..) => false,
        }
    }
}

impl std::fmt::Display for DEvalTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Modifier(modifier) => write!(f, "{}", prettify::modifier(modifier)),
            Self::Roll(roll) => write!(f, "{}", prettify::die(roll)),
            Self::Advantage(roll1, roll2) => {
                write!(f, "Adv(")?;
                match roll1 >= roll2 {
                    true => {
                        write!(f, "{}, {}", prettify::die(roll1).bold(), prettify::strikethrough(&prettify::die(roll2)))?
                    }
                    false => {
                        write!(f, "{}, {}", prettify::strikethrough(&prettify::die(roll1)), prettify::die(roll2).bold())?
                    }
                };

                write!(f, ")")
            }
            Self::Disadvantage(roll1, roll2) => {
                write!(f, "Dis(")?;
                match roll1 <= roll2 {
                    true => {
                        write!(f, "{}, {}", prettify::die(roll1).bold(), prettify::strikethrough(&prettify::die(roll2)))?
                    }
                    false => {
                        write!(f, "{}, {}", prettify::strikethrough(&prettify::die(roll1)), prettify::die(roll2).bold())?
                    }
                };

                write!(f, ")")
            }
            Self::Add(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Add, Side::Left)?;
                write!(f, " + ")?;
                arg1.write_bracketed(f, Self::Add, Side::Right)
            }
            Self::Sub(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Sub, Side::Left)?;
                write!(f, " - ")?;
                arg1.write_bracketed(f, Self::Sub, Side::Right)
            }
            Self::Mul(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Mul, Side::Left)?;
                write!(f, " * ")?;
                arg1.write_bracketed(f, Self::Mul, Side::Right)
            }
            Self::Div(arg0, arg1) => {
                arg0.write_bracketed(f, Self::Div, Side::Left)?;
                write!(f, " / ")?;
                arg1.write_bracketed(f, Self::Div, Side::Right)
            }
        }
    }
}

impl From<Die> for DExpr {
    fn from(die: Die) -> Self {
        Self::Die {
            die,
            both_adv_dis: false,
        }
    }
}

impl Die {
    pub fn advantage(self) -> DExpr {
        DExpr::Advantage(self)
    }

    pub fn disadvantage(self) -> DExpr {
        DExpr::Disadvantage(self)
    }
}

impl From<i32> for DExpr {
    fn from(die: i32) -> Self {
        Self::Modifier(die)
    }
}

// Macro put into another namespace to avoid annoying namespace collision
// between the internal `inner2!` and `inner1!` macros.
#[doc(hidden)]
mod macros {
    macro_rules! impl_bin_ops {
        ([$($this: ty),*], [$(($tr: ident, $method: ident)),*], [$($rhs: ty),*]) => {
            macro_rules! inner2 {
                ($this1: ty, $tr1: ident, $method1: ident) => {
                    $(
                        impl $tr1<$rhs> for $this1 {
                            type Output = DExpr;
                            fn $method1(self, rhs: $rhs) -> Self::Output {
                                DExpr::$tr1(Box::new(self.into()), Box::new(rhs.into()))
                            }
                        }
                    )*
                }
            }
    
            macro_rules! inner1 {
                ($this1: ty) => {
                    $(
                        inner2!($this1, $tr, $method);
                    )*
                }
            }
    
    
            $(
                inner1!($this);
            )*
        };
    }

    #[doc(hidden)]
    pub(super) use impl_bin_ops;
}

/*
    Implementations (rows = LHS, columns = RHS)
                i32     DExp    Die 
        i32             +-*÷    +-*÷  
        DExp    +-*÷    +-      +-  
        Die     +-*÷    +-      +-  

*/
macros::impl_bin_ops!([DExpr, Die], [(Add, add), (Sub, sub)                        ], [DExpr, Die]);
macros::impl_bin_ops!([i32       ], [(Add, add), (Sub, sub), (Mul, mul), (Div, div)], [DExpr, Die]);
macros::impl_bin_ops!([DExpr, Die], [(Add, add), (Sub, sub), (Mul, mul), (Div, div)], [i32       ]);

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use crate::core::dice::{self, expr::{DEvalTree, Side}, D10, D20, D6};

    use super::{Bracketed, DExpr};


    impl From<i32> for DEvalTree {
        fn from(value: i32) -> Self {
            Self::Roll(value)
        }
    }

    #[test]
    fn test_op_equality() {
        assert!(DExpr::Add(Box::new(1.into()), Box::new(1.into())).is_binary(DExpr::Add));
        assert!(DExpr::Sub(Box::new(1.into()), Box::new(1.into())).is_binary(DExpr::Sub));
        assert!(DExpr::Mul(Box::new(1.into()), Box::new(1.into())).is_binary(DExpr::Mul));
        assert!(DExpr::Div(Box::new(1.into()), Box::new(1.into())).is_binary(DExpr::Div));

        assert!(DEvalTree::Add(Box::new(DEvalTree::Roll(1)), Box::new(DEvalTree::Roll(1))).is_binary(DEvalTree::Add));
        assert!(DEvalTree::Sub(Box::new(DEvalTree::Roll(1)), Box::new(DEvalTree::Roll(1))).is_binary(DEvalTree::Sub));
        assert!(DEvalTree::Mul(Box::new(DEvalTree::Roll(1)), Box::new(DEvalTree::Roll(1))).is_binary(DEvalTree::Mul));
        assert!(DEvalTree::Div(Box::new(DEvalTree::Roll(1)), Box::new(DEvalTree::Roll(1))).is_binary(DEvalTree::Div));
    }

    #[test]
    fn test_op_bracketing() {
        #[rustfmt::skip]
        const ANSWERS: [[(bool, bool); 4]; 4] = [
            /* <exp>       L   +   R       L   -   R       L   *   R       L   /   R     */
            /*   +   */ [(false, false), (false, true),  (true,  true),  (true,  true)   ],
            /*   -   */ [(false, false), (false, true),  (true,  true),  (true,  true)   ],
            /*   *   */ [(false, false), (false, false), (false, false), (false, true)   ],
            /*   /   */ [(false, false), (false, false), (false, false), (false, true)   ],
        ];

        fn test_for<T: Bracketed + From<i32>>(arg: i32) -> [[(bool, bool); 4]; 4] {
            T::ALL.map(|op| op(Box::new(arg.into()), Box::new(arg.into())))
                .map(|expr| {
                    T::ALL.map(|op| {
                        (
                            expr.should_bracket_for(op, Side::Left).unwrap(),
                            expr.should_bracket_for(op, Side::Right).unwrap(),
                        )
                    })
                })
        }

        assert_eq!(ANSWERS, test_for::<DExpr>(1));
        assert_eq!(ANSWERS, test_for::<DEvalTree>(1));    
    }

    #[test]
    fn test_expr() {
        dice::set_seed(0); // DO NOT CHANGE THIS SEED!
        let expr = D6 + D10 - 2 * 10 + D10 * 13;
        assert_matches!(expr, DExpr::Add(box DExpr::Sub(box DExpr::Add(box DExpr::Die { die: D6, both_adv_dis: false }, box DExpr::Die { die: D10, both_adv_dis: false} ), box DExpr::Modifier(20)), box DExpr::Mul(box DExpr::Die { die: D10, both_adv_dis: false }, box DExpr::Modifier(13))));
        let eval = expr.evaluate();
        assert_matches!(eval, DEvalTree::Add(box DEvalTree::Sub(box DEvalTree::Add(box DEvalTree::Roll(5), box DEvalTree::Roll(6)), box DEvalTree::Modifier(20)), box DEvalTree::Mul(box DEvalTree::Roll(9), box DEvalTree::Modifier(13))));
        assert_eq!(eval.result(), 108);
        
        owo_colors::with_override(false, || {
            let st = eval.to_string();
            assert_eq!(st, "5 + 6 - 20 + 9 * 13");
        });
    }

    #[test]
    fn test_adv_dis_stacking() {
        let adv = D20.advantage();
        let adv_adv = D20.advantage().advantage();
        assert_eq!(adv, adv_adv);

        let d20 = DExpr::from(D20);
        let adv_dis = D20.advantage().disadvantage();
        assert_matches!((d20, adv_dis), (DExpr::Die { die: D20, .. }, DExpr::Die { die: D20, both_adv_dis: true }));
        
        let d20 = DExpr::from(D20);
        let dis_adv = D20.disadvantage().advantage();
        assert_matches!((d20, dis_adv), (DExpr::Die { die: D20, .. }, DExpr::Die { die: D20, both_adv_dis: true }));
        
        let d20 = DExpr::from(D20);
        let dis_dis_adv = D20.disadvantage().disadvantage().advantage();
        assert_matches!((d20, dis_dis_adv), (DExpr::Die { die: D20, .. }, DExpr::Die { die: D20, both_adv_dis: true }));
        
        let d20 = DExpr::from(D20);
        let dis_adv_adv = D20.disadvantage().advantage().advantage();
        assert_matches!((d20, dis_adv_adv), (DExpr::Die { die: D20, .. }, DExpr::Die { die: D20, both_adv_dis: true }));
    }
}