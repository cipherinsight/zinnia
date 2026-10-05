// DS-1000 case 0400 -- Noir baseline. np.array([]) -- empty result.
// Noir requires non-empty public output for a meaningful circuit; emit
// a single sentinel (matches Zinnia's "no IR stmts" minimality).

fn main() -> pub Field {
    0
}
