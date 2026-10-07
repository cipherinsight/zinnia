//! The layer report (measurement only): `ZINNIA_LAYER_REPORT=<path>` appends
//! one JSON line per lowering round to `<path>`, with, for every group, its
//! family, the constructions applicable to it, the forced construction (if
//! any) and the constructions selected for its lanes. The ablation scripts read it
//! (`benchmarking/ablations/`); the compiled program does not depend on it.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

use serde_json::json;

use super::construction::Target;
use super::emit::{NodeEntry, Offer, Options};
use super::graph::{Group, GroupId};

/// Append the report of one lowering to `path`.
///
/// # Panics
/// When the file cannot be written (a measurement must not lose its record
/// silently).
#[allow(clippy::too_many_arguments)]
pub(super) fn append(
    path: &Path,
    target: Target,
    options: &Options,
    groups: &[Group],
    nodes: &[NodeEntry],
    selected: &[usize],
    offers: &[Offer],
    released: &BTreeSet<GroupId>,
) {
    let names = |pick: &dyn Fn(&NodeEntry) -> bool, among: &mut dyn Iterator<Item = usize>| -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();
        for i in among {
            if pick(&nodes[i]) && !out.contains(&nodes[i].name) {
                out.push(nodes[i].name);
            }
        }
        out
    };
    let rows: Vec<serde_json::Value> = groups
        .iter()
        .map(|group| {
            let own = |n: &NodeEntry| n.group == Some(group.id);
            let produces_lane = |n: &NodeEntry| n.lanes().any(|s| group.lanes.iter().any(|l| l.stmt == s));
            // A group without lanes (a memory group whose reads are all at
            // runtime positions) is decided by the nodes offered for it.
            let chosen = if group.lanes.is_empty() {
                names(&own, &mut selected.iter().copied())
            } else {
                names(&produces_lane, &mut selected.iter().copied())
            };
            json!({
                "group": group.id,
                "family": group.kind.family(),
                "kind": group.kind.name(),
                "lanes": group.lanes.len(),
                "applicable": offers[group.id].applicable,
                "forced": offers[group.id].forced,
                "released": released.contains(&group.id),
                // A family a named configuration fixes (a group of it that
                // keeps the layer's choice is a fallback).
                "fixed_family": options.forcing.fixes_family(group.kind.family()),
                "chosen": chosen,
            })
        })
        .collect();
    let line = json!({
        "target": match target {
            Target::Halo2Base => "halo2",
            Target::Noir => "noir",
        },
        "local": options.local,
        "no_facts": options.no_facts,
        "forcing": options.forcing.names(),
        "estimated_cost": selected.iter().map(|&i| nodes[i].node.cost).sum::<u64>(),
        "groups": rows,
    });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("ZINNIA_LAYER_REPORT: cannot open {}: {e}", path.display()));
    // One write per line, so that processes appending to one file do not
    // interleave within a line.
    file.write_all(format!("{line}\n").as_bytes())
        .unwrap_or_else(|e| panic!("ZINNIA_LAYER_REPORT: cannot write {}: {e}", path.display()));
}
