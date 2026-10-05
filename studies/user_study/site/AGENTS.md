# Participant link handling

Explicit user instructions (updated 2026-09-24): P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, and P11 are already assigned and protected.

- Never invalidate or rotate these participants' links during a general request to invalidate links.
- Never include their links in general link lists or responses unless the user specifically asks for those participants' links.
- Read runtime/link-distribution.json before rotating, renumbering, or presenting participant links. The registry is keyed by token so protection follows the participant if labels change. Do not print the registry wholesale: it contains capability tokens.
- Preserve the registry and these restrictions when regenerating link lists or renumbering participants. A general request for all links is not a specific request for these protected participants' links.
- Marked assigned here means distributed to a person, distinct from the task/language schedule assignment. Do not reset study progress when changing distribution metadata.

P9 was randomly selected, assigned, and frozen on 2026-09-24.

## Completed-result protection

Explicit user instruction (2026-09-23): P0, P2, and P8 are completed; their results must not be reset or cleared.

- Preserve all study results, attempts, drafts, timing, and practice records for these participants. Exclude them from general or previously authorized reset operations.
- Read runtime/link-distribution.json before any participant reset; preserve_results_on_reset tracks this protection by token. Only a new explicit instruction overriding this protection for a named participant can authorize clearing their results.
- P0 is the completed pilot, not an available unassigned survey; preserve its link during general invalidation. Existing frozen-link disclosure rules for P2 and P8 still apply.

On 2026-09-24, P1 was marked unassigned before a random draw from available surveys. The draw selected P9, which is now assigned and frozen. Links and progress were preserved.

On 2026-09-24, P1, P6, P7, and P10 were assigned and frozen for distribution of the remaining surveys. P3 and P11 were already assigned. Defer comparative study claims until the remaining six are collected.

Assignment remark (2026-09-24): P3 and P11 are assigned to zhaoyu.
