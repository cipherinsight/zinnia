//! The source graph: where every value of $\Delta$ came from
//! (`implementation.md` §5a and §8, "The source graph").
//!
//! An entry of $\Delta$ says what a value *is*. A [`Producer`] says how it was
//! made: the rules applied to it, in application order, and the keys of the
//! values they read. `definition.md` §3.3 defers provenance, alignment and
//! order to "the source graph"; these edges are it, and phase 1 builds its
//! need graph on them (`../README.md` §5.1).
//!
//! [`ValueStack`] is how the walker collects them. One frame is open per value
//! being evaluated: [`crate::phase0::walk`] pushes the key of every
//! sub-expression into the enclosing frame's operand list, and every rule that
//! runs appends its name to the innermost frame
//! ([`crate::phase0::rules::RuleCtx::applied`]). The innermost frame's key is
//! also the *subject* of every check the rule records: the operation instance
//! the condition belongs to.

use crate::phase0::desc::{Desc, ValueKey};
use crate::phase0::rules::Rule;
use crate::phase0::term::BoolId;

/// How one value of $\Delta$ was produced.
#[derive(Debug, Clone)]
pub enum Producer {
    /// A circuit input (`E-Input-*`, §5.1).
    Input,
    /// A reference to a bound name: the same value under another key.
    Alias(ValueKey),
    /// A value that carries no descriptor content: a dtype or module name
    /// ([`Desc::Class`]) or [`Desc::None`].
    Opaque,
    /// The result of applying transfer rules to operand values.
    Applied {
        /// The rules applied to produce this value, in application order. A
        /// subscript with a slice and an index applies two, a coercion adds
        /// its own, and a binding that merges records the merge rule after
        /// `S-Assign`.
        rules: Vec<Rule>,
        /// The values the rules read, in evaluation order.
        operands: Vec<ValueKey>,
        /// The condition a merge selected on, when this value is a merge
        /// (§6.1): the `b` of an `if`, the liveness of a guarded binding.
        condition: Option<BoolId>,
    },
}

impl Producer {
    /// The producer of a value built by `rules` from `operands`.
    ///
    /// A value with no descriptor content is [`Producer::Opaque`] whatever
    /// ran: the dtype names bound in every scope are merged at the end of an
    /// `if` like any other binding (§6.1), and that merge applies no rule.
    pub fn of(
        desc: &Desc,
        rules: Vec<Rule>,
        operands: Vec<ValueKey>,
        condition: Option<BoolId>,
    ) -> Producer {
        match desc {
            Desc::Class(_) | Desc::None => Producer::Opaque,
            _ => Producer::Applied {
                rules,
                operands,
                condition,
            },
        }
    }

    /// The rules this producer names, for the invariants of
    /// `implementation.md` §5a.
    pub fn rules(&self) -> &[Rule] {
        match self {
            Producer::Applied { rules, .. } => rules,
            _ => &[],
        }
    }

    /// The value this producer only re-labels, when it does: a use of a name
    /// under a new key, an `E-Copy`, which returns the descriptor of its
    /// first operand unchanged (§5.11), or a binding `S-Assign` of one value
    /// that merges nothing. Following these edges gives the value a key
    /// names, which is what `Count` is keyed by (§5.5).
    pub fn relabels(&self) -> Option<&ValueKey> {
        match self {
            Producer::Alias(key) => Some(key),
            Producer::Applied {
                rules, operands, ..
            } if matches!(rules[..], [Rule::ECopy] | [Rule::SAssign])
                && operands.len() == 1 =>
            {
                operands.first()
            }
            _ => None,
        }
    }

    /// Every key this value was built from: the operands of an
    /// [`Producer::Applied`], the target of an [`Producer::Alias`].
    pub fn inputs(&self) -> Vec<&ValueKey> {
        match self {
            Producer::Alias(key) => vec![key],
            Producer::Applied { operands, .. } => operands.iter().collect(),
            _ => Vec::new(),
        }
    }

    /// The JSON of [`crate::phase0::export`]'s `producer` field.
    ///
    /// A reference to another value is its canonical key string
    /// ([`ValueKey::encode`]), not the whole key object: a producer names one
    /// value per operand and Δ has an entry per key, so the object would be
    /// the bulk of the record and says nothing the string does not.
    pub fn to_json(&self) -> serde_json::Value {
        let mut value = match self {
            Producer::Input => serde_json::json!({"kind": "input"}),
            Producer::Alias(key) => {
                serde_json::json!({"kind": "alias", "alias": key.encode()})
            }
            Producer::Opaque => serde_json::json!({"kind": "opaque"}),
            Producer::Applied {
                rules,
                operands,
                condition,
            } => serde_json::json!({
                "kind": "applied",
                "rules": rules.iter().map(|r| r.name()).collect::<Vec<_>>(),
                "operands": operands.iter().map(|k| k.encode()).collect::<Vec<_>>(),
                "condition": condition.map(|b| b.index()),
            }),
        };
        // Every producer carries the same four fields, so a consumer can read
        // one shape whatever the kind.
        let object = value.as_object_mut().expect("json! built an object");
        for field in ["alias", "rules", "operands", "condition"] {
            object.entry(field).or_insert(serde_json::Value::Null);
        }
        value
    }
}

/// One value under construction: what the walker has seen while evaluating it.
#[derive(Debug)]
pub(crate) struct ValueFrame {
    /// The value's key, which is the subject of every check the rules record.
    key: ValueKey,
    rules: Vec<Rule>,
    operands: Vec<ValueKey>,
    /// Set when the value is another value under a new key, so that the
    /// producer is an [`Producer::Alias`] rather than a rule application.
    alias: Option<ValueKey>,
    /// How many intermediates of this value have been closed off already
    /// (`definition.md` §3.3): the step index the next one takes.
    steps: u32,
}

impl ValueFrame {
    /// The producer of a value the frame was opened for: an alias when the
    /// handler named one, [`Producer::Opaque`] for a value with no descriptor
    /// content, and the collected rules and operands otherwise.
    pub fn into_producer(self, desc: &Desc) -> Producer {
        match self.alias {
            Some(key) => Producer::Alias(key),
            None => Producer::of(desc, self.rules, self.operands, None),
        }
    }

    /// The rules the frame collected, for a statement-level site that names
    /// its own rule and operands (`implementation.md` §5a).
    pub fn into_rules(self) -> Vec<Rule> {
        self.rules
    }

    /// The key the frame was opened for.
    pub fn key(&self) -> &ValueKey {
        &self.key
    }
}

/// The open value frames, innermost last.
#[derive(Debug, Default)]
pub(crate) struct ValueStack {
    frames: Vec<ValueFrame>,
}

impl ValueStack {
    /// Open a frame for the value at `key`.
    pub fn open(&mut self, key: &ValueKey) {
        self.frames.push(ValueFrame {
            key: key.clone(),
            rules: Vec::new(),
            operands: Vec::new(),
            alias: None,
            steps: 0,
        });
    }

    /// Open the frame of an *intermediate* of the value being built
    /// (`definition.md` §3.3) and return its key.
    ///
    /// The intermediate takes the value's key with the next step index. Its
    /// own frame collects the rules that run until it is closed, and its key
    /// is the subject of the checks they record, exactly as for a value the
    /// source wrote.
    pub fn open_step(&mut self) -> ValueKey {
        let key = match self.frames.last_mut() {
            Some(frame) => {
                let key = frame.key.intermediate(frame.steps);
                frame.steps += 1;
                key
            }
            None => unreachable!("an intermediate belongs to a value under construction"),
        };
        self.open(&key);
        key
    }

    /// Close the innermost frame. Paired with [`ValueStack::open`] by
    /// [`crate::phase0::walk::Walker::in_value`], which closes it even when
    /// the body rejects.
    pub fn close(&mut self) -> ValueFrame {
        match self.frames.pop() {
            Some(frame) => frame,
            None => unreachable!("open and close are paired"),
        }
    }

    /// Whether every frame that was opened has been closed, which the end of
    /// the pass asserts.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Record that `key` is an operand of the value being built. Outside any
    /// frame — at statement level — there is no value to be an operand of.
    pub fn operand(&mut self, key: &ValueKey) {
        if let Some(frame) = self.frames.last_mut() {
            frame.operands.push(key.clone());
        }
    }

    /// Record that `rule` was applied to the value being built.
    pub fn applied(&mut self, rule: Rule) {
        if let Some(frame) = self.frames.last_mut() {
            frame.rules.push(rule);
        }
    }

    /// Record that the value being built is the value at `key` under a new
    /// key: a name reference, or an operator that returns its operand.
    pub fn alias(&mut self, key: &ValueKey) {
        if let Some(frame) = self.frames.last_mut() {
            frame.alias = Some(key.clone());
        }
    }

    /// The key of the value being built: the subject of every check a rule
    /// records now (`implementation.md` §5a).
    pub fn subject(&self) -> &ValueKey {
        match self.frames.last() {
            Some(frame) => &frame.key,
            // A `RuleCtx` is built only where a value is under construction:
            // the walker opens a frame around every expression and around
            // each statement-level rule application, and the two pre-walk
            // sites that have no value (the shape symbols and the
            // `scalar_requires` facts) take no `RuleCtx`.
            None => unreachable!("a rule runs inside an open value frame"),
        }
    }
}
