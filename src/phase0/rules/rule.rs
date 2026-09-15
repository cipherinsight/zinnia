//! The rule names phase 0 records, as a closed set.
//!
//! Every constraint, obligation and fact of $\Phi$ carries the name of the
//! rule that produced it ([`crate::phase0::phi::Constraint::provenance`]), and
//! that name is what makes the exported record diffable against the
//! specification. A free string cannot be diffed — it can name a rule that
//! does not exist — so the name is this enum instead, and
//! [`Rule::name`] is the only place a rule name becomes text.
//!
//! The variants are exactly the rules of `definition.md` §5, §6.1 and §6.2,
//! plus the additions of `implementation.md` §8. A rule the walker needs and
//! no document states is not added here without the matching one-line
//! statement in §8; that is the whole point of the enum.
//!
//! A rule that records more than one condition names which one with
//! [`Rule::detail`], and the two print as `"E-Reshape: the sizes agree"`.

/// One rule of `definition.md` §5, §6.1, §6.2 or `implementation.md` §8.
///
/// Some variants name a rule the walker realises without building a
/// descriptor through [`super::RuleCtx`] — `S-Assign`, `S-Call`, `S-If`,
/// `S-Unpack`, `S-For-Unrolled`, `S-For-Elems` — and so appear here as the
/// rule list rather than as a constructor argument. The list is meant to be
/// read against §5 and §6 line by line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rule {
    // §5.1 Inputs and constants
    EInputFixed,
    EInputVarLen,
    EConst,
    EInputSymbolic,
    EInputComposite,
    EInputHashed,
    // §5.2 Scalars
    ELen,
    EShape,
    EArithShape,
    EArithData,
    ECmp,
    EElem,
    EReduceScalar,
    // §5.3 Elementwise operations and broadcasting
    EMap,
    EZip,
    // §5.4 Indexing and slicing
    EIndex,
    ESlice,
    ETake,
    /// `implementation.md` §8, "Added in step 5": `E-Take` for a subscript
    /// with two or more advanced entries.
    ETakeBroadcast,
    // §5.5 Boolean indexing and counts
    EFilter1D,
    EFilterAxis0,
    EFilterSame,
    ENonzero,
    EWhere3,
    // §5.6 Shape transformations
    ETranspose,
    EMoveaxis,
    EReshape,
    EConcat,
    EStack,
    // §5.7 Reductions, matmul, sorting
    EReduceAxis,
    EMatmul,
    ESort,
    ECumsum,
    EReverse,
    EUnique,
    // §5.8 Constructors
    EArange,
    EZeros,
    EOnes,
    ERepeat,
    // §5.9 Lengths that depend on element values
    EValueSized,
    // §5.10 Assignment and calls
    SAssign,
    SCall,
    // §5.11 Composites, in-place writes, and the remaining forms
    EComposite,
    ECompositeIndex,
    ECompositeBinop,
    SListMethod,
    SSetItem,
    SUnpack,
    ECondExp,
    EComp,
    EExternal,
    EProperty,
    // §6.1 Branches
    SIf,
    /// §6.1's "Array merge". `implementation.md` §8 names the string.
    MergeArray,
    /// §6.1's "Scalar merge". `implementation.md` §8 names the string.
    MergeScalar,
    // §6.2 Loops
    SForUnrolled,
    SForElems,
    SForDyn,
    SWhileBounded,
    /// `implementation.md` §8, "Added in step 2b".
    SForRangeBounded,
    // `implementation.md` §8, "Added in step 5"
    EToList,
    // `implementation.md` §8, "Rule names the walker needs"
    ENewAxis,
    ESqueeze,
    EBroadcast,
    EPad,
    ETile,
    EDelete,
    EInsert,
    EDiff,
    EDiag,
    ECorrelate,
    ECorrcoef,
    EOuter,
    ECopy,
    SAssert,
}

impl Rule {
    /// The rule's name, exactly as the document spells it. This is the string
    /// the JSON export carries and the only place a rule becomes text.
    pub fn name(self) -> &'static str {
        match self {
            Rule::EInputFixed => "E-Input-Fixed",
            Rule::EInputVarLen => "E-Input-VarLen",
            Rule::EConst => "E-Const",
            Rule::EInputSymbolic => "E-Input-Symbolic",
            Rule::EInputComposite => "E-Input-Composite",
            Rule::EInputHashed => "E-Input-Hashed",
            Rule::ELen => "E-Len",
            Rule::EShape => "E-Shape",
            Rule::EArithShape => "E-Arith-Shape",
            Rule::EArithData => "E-Arith-Data",
            Rule::ECmp => "E-Cmp",
            Rule::EElem => "E-Elem",
            Rule::EReduceScalar => "E-Reduce-Scalar",
            Rule::EMap => "E-Map",
            Rule::EZip => "E-Zip",
            Rule::EIndex => "E-Index",
            Rule::ESlice => "E-Slice",
            Rule::ETake => "E-Take",
            Rule::ETakeBroadcast => "E-Take-Broadcast",
            Rule::EFilter1D => "E-Filter-1D",
            Rule::EFilterAxis0 => "E-Filter-Axis0",
            Rule::EFilterSame => "E-Filter-Same",
            Rule::ENonzero => "E-Nonzero",
            Rule::EWhere3 => "E-Where3",
            Rule::ETranspose => "E-Transpose",
            Rule::EMoveaxis => "E-Moveaxis",
            Rule::EReshape => "E-Reshape",
            Rule::EConcat => "E-Concat",
            Rule::EStack => "E-Stack",
            Rule::EReduceAxis => "E-Reduce-Axis",
            Rule::EMatmul => "E-Matmul",
            Rule::ESort => "E-Sort",
            Rule::ECumsum => "E-Cumsum",
            Rule::EReverse => "E-Reverse",
            Rule::EUnique => "E-Unique",
            Rule::EArange => "E-Arange",
            Rule::EZeros => "E-Zeros",
            Rule::EOnes => "E-Ones",
            Rule::ERepeat => "E-Repeat",
            Rule::EValueSized => "E-ValueSized",
            Rule::SAssign => "S-Assign",
            Rule::SCall => "S-Call",
            Rule::EComposite => "E-Composite",
            Rule::ECompositeIndex => "E-Composite-Index",
            Rule::ECompositeBinop => "E-Composite-Binop",
            Rule::SListMethod => "S-ListMethod",
            Rule::SSetItem => "S-SetItem",
            Rule::SUnpack => "S-Unpack",
            Rule::ECondExp => "E-CondExp",
            Rule::EComp => "E-Comp",
            Rule::EExternal => "E-External",
            Rule::EProperty => "E-Property",
            Rule::SIf => "S-If",
            Rule::MergeArray => "merge-array",
            Rule::MergeScalar => "merge-scalar",
            Rule::SForUnrolled => "S-For-Unrolled",
            Rule::SForElems => "S-For-Elems",
            Rule::SForDyn => "S-For-Dyn",
            Rule::SWhileBounded => "S-While-Bounded",
            Rule::SForRangeBounded => "S-For-Range-Bounded",
            Rule::EToList => "E-ToList",
            Rule::ENewAxis => "E-NewAxis",
            Rule::ESqueeze => "E-Squeeze",
            Rule::EBroadcast => "E-Broadcast",
            Rule::EPad => "E-Pad",
            Rule::ETile => "E-Tile",
            Rule::EDelete => "E-Delete",
            Rule::EInsert => "E-Insert",
            Rule::EDiff => "E-Diff",
            Rule::EDiag => "E-Diag",
            Rule::ECorrelate => "E-Correlate",
            Rule::ECorrcoef => "E-Corrcoef",
            Rule::EOuter => "E-Outer",
            Rule::ECopy => "E-Copy",
            Rule::SAssert => "S-Assert",
        }
    }

    /// This rule, naming which of its conditions the constraint is: the
    /// provenance prints as `"<rule>: <detail>"`.
    pub fn detail(self, detail: &'static str) -> Provenance {
        Provenance {
            rule: self,
            detail: Some(detail),
        }
    }
}

impl std::fmt::Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A rule, optionally naming which of its conditions a constraint records.
///
/// A rule with one condition passes the [`Rule`] itself; the `impl
/// From<Rule>` is what lets both spellings reach the same constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Provenance {
    pub rule: Rule,
    pub detail: Option<&'static str>,
}

impl From<Rule> for Provenance {
    fn from(rule: Rule) -> Self {
        Provenance { rule, detail: None }
    }
}

impl std::fmt::Display for Provenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.detail {
            Some(detail) => write!(f, "{}: {}", self.rule.name(), detail),
            None => f.write_str(self.rule.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two rules that print alike would make the export ambiguous, and a
    /// name that is not the document's spelling would make it undiffable.
    #[test]
    fn every_rule_name_is_distinct_and_spelled_as_the_document_spells_it() {
        let rules = [
            Rule::EInputFixed,
            Rule::EInputVarLen,
            Rule::EConst,
            Rule::EInputSymbolic,
            Rule::EInputComposite,
            Rule::EInputHashed,
            Rule::ELen,
            Rule::EShape,
            Rule::EArithShape,
            Rule::EArithData,
            Rule::ECmp,
            Rule::EElem,
            Rule::EReduceScalar,
            Rule::EMap,
            Rule::EZip,
            Rule::EIndex,
            Rule::ESlice,
            Rule::ETake,
            Rule::ETakeBroadcast,
            Rule::EFilter1D,
            Rule::EFilterAxis0,
            Rule::EFilterSame,
            Rule::ENonzero,
            Rule::EWhere3,
            Rule::ETranspose,
            Rule::EMoveaxis,
            Rule::EReshape,
            Rule::EConcat,
            Rule::EStack,
            Rule::EReduceAxis,
            Rule::EMatmul,
            Rule::ESort,
            Rule::ECumsum,
            Rule::EReverse,
            Rule::EUnique,
            Rule::EArange,
            Rule::EZeros,
            Rule::EOnes,
            Rule::ERepeat,
            Rule::EValueSized,
            Rule::SAssign,
            Rule::SCall,
            Rule::EComposite,
            Rule::ECompositeIndex,
            Rule::ECompositeBinop,
            Rule::SListMethod,
            Rule::SSetItem,
            Rule::SUnpack,
            Rule::ECondExp,
            Rule::EComp,
            Rule::EExternal,
            Rule::EProperty,
            Rule::SIf,
            Rule::MergeArray,
            Rule::MergeScalar,
            Rule::SForUnrolled,
            Rule::SForElems,
            Rule::SForDyn,
            Rule::SWhileBounded,
            Rule::SForRangeBounded,
            Rule::EToList,
            Rule::ENewAxis,
            Rule::ESqueeze,
            Rule::EBroadcast,
            Rule::EPad,
            Rule::ETile,
            Rule::EDelete,
            Rule::EInsert,
            Rule::EDiff,
            Rule::EDiag,
            Rule::ECorrelate,
            Rule::ECorrcoef,
            Rule::EOuter,
            Rule::ECopy,
            Rule::SAssert,
        ];
        let names: std::collections::HashSet<&str> = rules.iter().map(|r| r.name()).collect();
        assert_eq!(names.len(), rules.len(), "rule names are not distinct");
        for rule in rules {
            let name = rule.name();
            assert!(
                name.starts_with("E-") || name.starts_with("S-") || name.starts_with("merge-"),
                "{} is not spelled like a rule of §5, §6 or §6.1's merges",
                name
            );
        }
    }

    #[test]
    fn a_detail_prints_after_the_rule_name() {
        assert_eq!(Rule::ESlice.to_string(), "E-Slice");
        assert_eq!(
            Rule::EReshape.detail("the sizes agree").to_string(),
            "E-Reshape: the sizes agree"
        );
    }
}
