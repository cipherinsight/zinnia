//! Read-only memory by a dynamic lookup: the tables of `IR::ReadTable`.
//!
//! A table is a memory segment's slots at the time of its reads, tagged by a
//! number unique per (segment, version). Its rows `(tag, address, value)`
//! have the tag and the address in fixed columns and the value in an advice
//! column copy-constrained to the slot's cell. A read is a row
//! `(tag, index, value)` of a key column set: the tag fixed, the index and
//! the value advice copy-constrained to the program's index cell and to the
//! read's (witness) value cell. halo2's lookup argument (`lookup_any`)
//! proves that every enabled key row equals an enabled table row:
//!
//! ```text
//! (k_en, k_tag, k_idx, k_val)  in  {(t_en, t_tag, t_addr, t_val)}
//! ```
//!
//! Tables are packed, whole and in tag order, into *table groups* of at most
//! `usable - 1` rows (one table column set each, with its own key sets and
//! lookup arguments), so the tables never force a larger `k` than one table
//! needs.
//!
//! **Soundness.** An enabled key row (`k_en = 1`) matches only enabled table
//! rows (`t_en = 1`) of its group with the same tag; that tag's rows are
//! exactly the addresses `0..n` of one table (fixed at key generation) with
//! the slots' cells as values. So the index is one of `0..n` and the read
//! value is the slot's value: there is no witness for an index outside the
//! segment and no other value for the read. The tag is its own tuple
//! component (folding it into the address, `tag·2^32 + i`, would let an
//! out-of-range index name another table's row). The lookup argument draws
//! its challenges (θ, β, γ) after the advice is committed: the same
//! Fiat-Shamir assumption as the range-check lookups every circuit already
//! uses. Key rows a set does not use have `k_en = 0` and match a disabled
//! table row (`t_en = 0`; each group keeps one free row); their advice cells
//! are copy-constrained to nothing.
//!
//! **Completeness.** The honest prover writes the slot's value, which is in
//! the table at its address.
//!
//! **Cost.** Per table slot one advice cell (the value); per read two
//! advice cells (index, value copies) besides the read's witness cell. Per
//! table group: one value column, two advice columns per key set and one
//! lookup argument per key set (`estimate_params` counts them).

use std::collections::BTreeMap;

use halo2_base::halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::bn256::Fr as Fp,
    plonk::{Advice, Column, ConstraintSystem, Error, Fixed, VirtualCells},
    poly::Rotation,
};
use halo2_base::utils::halo2::{constrain_virtual_equals_external, raw_assign_advice, raw_assign_fixed};
use halo2_base::virtual_region::copy_constraints::SharedCopyConstraintManager;
use halo2_base::AssignedValue;

/// The tables and reads of one circuit, in registration order.
#[derive(Clone, Default)]
pub struct Tables {
    /// `(tag, address, value cell)`, each table's rows together.
    pub rows: Vec<(u64, u64, AssignedValue<Fp>)>,
    /// `(tag, index cell, value cell)`.
    pub reads: Vec<(u64, AssignedValue<Fp>, AssignedValue<Fp>)>,
}

/// Where each table goes: per group, its rows, its reads and its key sets,
/// for `usable` rows per column.
pub struct Packing {
    /// The group of each tag.
    pub group_of: BTreeMap<u64, usize>,
    /// Per group: (table rows, reads).
    pub groups: Vec<(usize, usize)>,
    pub usable: usize,
}

impl Packing {
    /// Key sets per group (the circuit's `table_layout`).
    pub fn key_sets(&self) -> Vec<usize> {
        key_sets(&self.groups, self.usable)
    }
}

impl Tables {
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
    }

    /// The rows of each table, by tag.
    fn table_sizes(&self) -> BTreeMap<u64, usize> {
        let mut sizes = BTreeMap::new();
        for (tag, _, _) in &self.rows {
            *sizes.entry(*tag).or_insert(0) += 1;
        }
        sizes
    }

    /// The largest table plus the free row its group keeps: the rows `k`
    /// must provide.
    pub fn rows_needed(&self) -> usize {
        if self.is_empty() {
            return 0;
        }
        self.table_sizes().values().max().map_or(0, |n| n + 1)
    }

    /// `(rows, reads)` of each table, in tag order (what the packing and the
    /// cost estimate need).
    pub fn shape(&self) -> Vec<(usize, usize)> {
        let mut reads: BTreeMap<u64, usize> = BTreeMap::new();
        for (tag, _, _) in &self.reads {
            *reads.entry(*tag).or_insert(0) += 1;
        }
        self.table_sizes().into_iter().map(|(tag, rows)| (rows, reads.get(&tag).copied().unwrap_or(0))).collect()
    }

    /// Tables packed in tag order into groups of at most `usable - 1` rows
    /// (`usable >= rows_needed()`).
    pub fn packing(&self, usable: usize) -> Packing {
        let tags: Vec<u64> = self.table_sizes().into_keys().collect();
        let (groups, group_of_table) = pack(&self.shape(), usable);
        Packing { group_of: tags.into_iter().zip(group_of_table).collect(), groups, usable }
    }

    /// Advice cells: one per slot, two per read.
    pub fn advice_cells(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.rows.len() + 2 * self.reads.len()
        }
    }
}

/// One table group's columns.
#[derive(Clone, Debug)]
struct GroupConfig {
    t_en: Column<Fixed>,
    t_tag: Column<Fixed>,
    t_addr: Column<Fixed>,
    t_val: Column<Advice>,
    /// Per key set: `(k_en, k_tag, k_idx, k_val)`.
    keys: Vec<(Column<Fixed>, Column<Fixed>, Column<Advice>, Column<Advice>)>,
}

/// Packs tables `(rows, reads)` in order into groups of at most `usable - 1`
/// rows: per group `(rows, reads)`, and each table's group.
pub fn pack(tables: &[(usize, usize)], usable: usize) -> (Vec<(usize, usize)>, Vec<usize>) {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut group_of = Vec::with_capacity(tables.len());
    for &(size, reads) in tables {
        match groups.last_mut() {
            Some((rows, group_reads)) if *rows + size < usable => {
                *rows += size;
                *group_reads += reads;
            }
            _ => groups.push((size, reads)),
        }
        group_of.push(groups.len() - 1);
    }
    (groups, group_of)
}

/// Key sets per group for `usable` rows per column.
pub fn key_sets(groups: &[(usize, usize)], usable: usize) -> Vec<usize> {
    groups.iter().map(|&(_, reads)| reads.div_ceil(usable).max(1)).collect()
}

/// The columns and lookup arguments of the tables.
#[derive(Clone, Debug)]
pub struct TableConfig {
    groups: Vec<GroupConfig>,
}

impl TableConfig {
    /// `layout[g]`: the key sets of table group `g`.
    pub fn configure(meta: &mut ConstraintSystem<Fp>, layout: &[usize]) -> Self {
        let groups = layout.iter().map(|&sets| GroupConfig::configure(meta, sets)).collect();
        TableConfig { groups }
    }

    /// Lays out the tables and the reads by `packing`. With `copy` (keygen
    /// and mock: the first phase's copy manager, after the first phase is
    /// laid out) the value, index and read cells are constrained to the
    /// program's cells; the witness-only prover writes values only.
    pub fn assign(
        &self,
        tables: &Tables,
        packing: &Packing,
        layouter: &mut impl Layouter<Fp>,
        copy: Option<&SharedCopyConstraintManager<Fp>>,
    ) -> Result<(), Error> {
        assert_eq!(packing.key_sets(), self.groups.iter().map(|g| g.keys.len()).collect::<Vec<_>>(), "the pinned table layout");
        layouter.assign_region(
            || "memory tables",
            |mut region| {
                let mut copy = copy.map(|c| c.lock().unwrap());
                let mut place = |region: &mut _, column, row, cell: &AssignedValue<Fp>| {
                    let raw = raw_assign_advice(region, column, row, Value::known(*cell.value()));
                    if let Some(copy) = copy.as_mut() {
                        constrain_virtual_equals_external(region, *cell, raw.cell(), copy);
                    }
                };
                let mut table_row = vec![0usize; self.groups.len()];
                for (tag, address, value) in &tables.rows {
                    let g = packing.group_of[tag];
                    let (group, row) = (&self.groups[g], table_row[g]);
                    raw_assign_fixed(&mut region, group.t_en, row, Fp::one());
                    raw_assign_fixed(&mut region, group.t_tag, row, Fp::from(*tag));
                    raw_assign_fixed(&mut region, group.t_addr, row, Fp::from(*address));
                    place(&mut region, group.t_val, row, value);
                    table_row[g] += 1;
                }
                let mut read_no = vec![0usize; self.groups.len()];
                for (tag, index, value) in &tables.reads {
                    let g = packing.group_of[tag];
                    let group = &self.groups[g];
                    let (k_en, k_tag, k_idx, k_val) = group.keys[read_no[g] % group.keys.len()];
                    let row = read_no[g] / group.keys.len();
                    raw_assign_fixed(&mut region, k_en, row, Fp::one());
                    raw_assign_fixed(&mut region, k_tag, row, Fp::from(*tag));
                    place(&mut region, k_idx, row, index);
                    place(&mut region, k_val, row, value);
                    read_no[g] += 1;
                }
                Ok(())
            },
        )
    }
}

impl GroupConfig {
    fn configure(meta: &mut ConstraintSystem<Fp>, sets: usize) -> Self {
        let (t_en, t_tag, t_addr) = (meta.fixed_column(), meta.fixed_column(), meta.fixed_column());
        let t_val = meta.advice_column();
        meta.enable_equality(t_val);
        let keys: Vec<_> = (0..sets)
            .map(|_| {
                let (k_en, k_tag) = (meta.fixed_column(), meta.fixed_column());
                let (k_idx, k_val) = (meta.advice_column(), meta.advice_column());
                meta.enable_equality(k_idx);
                meta.enable_equality(k_val);
                (k_en, k_tag, k_idx, k_val)
            })
            .collect();
        for &(k_en, k_tag, k_idx, k_val) in &keys {
            meta.lookup_any("memory table", |m: &mut VirtualCells<Fp>| {
                let fixed = |m: &mut VirtualCells<Fp>, c| m.query_fixed(c, Rotation::cur());
                let advice = |m: &mut VirtualCells<Fp>, c| m.query_advice(c, Rotation::cur());
                vec![
                    (fixed(m, k_en), fixed(m, t_en)),
                    (fixed(m, k_tag), fixed(m, t_tag)),
                    (advice(m, k_idx), fixed(m, t_addr)),
                    (advice(m, k_val), advice(m, t_val)),
                ]
            });
        }
        GroupConfig { t_en, t_tag, t_addr, t_val, keys }
    }
}
