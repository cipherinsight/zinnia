from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_circuit
def scrabble_fun_score(word: str, scoretable: Float):
    return sum([scoretable.get(x, 0) for x in word])

@zk_circuit
def scrabble_imp_score(word: str, scoretable: Float[Array, 'N<=1024']):
    score = 0
    for letter in word:
        if letter in scoretable:
            score += scoretable[letter]
    return score
