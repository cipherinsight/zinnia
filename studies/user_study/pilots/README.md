# Independent implementation pilots

Round 1 samples ten fresh-context GPT-5.6 agents, each implementing all six task specifications: 60 submissions, ten per task. Four agents use GPT-5.6 Sol, three use Terra, and three use Luna, all at medium reasoning effort. Model allocation, task count, and validation inputs were specified before compiler evaluation.

Every agent receives the same [brief](round1/brief.md), apart from its identifier and output directory. Agents are instructed to inspect only that brief and their own files, to avoid the compiler and curated inventory, and to avoid other agents' submissions. They may test their own solutions with NumPy before submitting. This is isolation by fresh context and instructions within a shared filesystem, not a security sandbox. No agent receives compiler feedback or revises a solution in response to it.

The [protocol](round1/protocol.json) records model allocation and the initial build. Once all agents finish, [frozen.json](round1/frozen.json) hashes all 60 programs before compiler testing begins. Every submitted implementation is retained, including duplicates; none is replaced to improve diversity or pass rates. LLM generation seeds and sampling parameters beyond the requested model/reasoning setting were not controlled.

The evaluator supplies the required input type annotations and specializes the declared public `R` configuration line. It makes no algorithm changes. Original functions compile separately from correctness-instrumented versions. All reference programs run in NumPy against scalar oracles. Seven predefined fixtures plus twelve independently seeded random fixtures run at each of the existing shape profiles, for 190 implementation/profile combinations and 3,610 correct-output cases. Each combination also receives an incorrect-output control. A separate Halo2 run generates and verifies one proof per submission on profile 0, fixture 6.

See the [round 1 report](round1/REPORT.md) for results, source overlap, and build fingerprints.

```sh
# Run from zinnia-src. Submissions are already frozen; do not overwrite them.
.venv/bin/python -m studies.user_study.pilots.validate --output /tmp/pilot-validation.json
.venv/bin/python -m studies.user_study.pilots.validate --backend halo2 --profile 0 --fixture 6 --output /tmp/pilot-proofs.json
.venv/bin/python -m pytest -q testing/lang/test_user_study_pilot_harness.py
```

This pilot measures compatibility with the sampled solutions. Agents from the same model family may converge on the same code; ten submissions do not imply ten algorithms or ten statistically independent human observations. These results do not measure user effort or establish a usability advantage over Rust/RISC Zero. Human piloting remains necessary for session length, task difficulty, training, and the eventual user study.
