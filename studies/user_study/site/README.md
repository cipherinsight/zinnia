> **Current study protocol:** [12 participants × 6 tasks crossover](CROSSOVER.md). Each participant completes each task once, three in each environment, after both practice sessions.

> New assignments use [revision 4](../revision4/README.md): six composed array tasks with versioned contracts. Original pilot links retain their original task definitions and data.

# Interactive array-programming study

A local, working study site modeled on `ComfyTy/study-site` in `~/Projects/comfy-ui`. Every participant completes **all six tasks in one assigned language**: Zinnia/Python or Rust/RISC Zero. English/中文 switches the interface and task text; it cannot change programming condition. The researcher can create separate previews of either condition.

The participant flow is consent → background questions → language-specific tutorial → untimed practice → six timed tasks → completion. The CodeMirror editor supports highlighting, undo/redo (Ctrl+Z or Cmd+Z, Ctrl+Shift+Z or Cmd+Shift+Z, and Ctrl+Y), search, four-space indentation, autosave, local recovery, source download, example tests, custom JSON inputs, and a diagnostics console. Undo history is retained through interface-language changes and reloads in the same browser tab when the restored code still matches; each participant/task has separate history. Each task includes an expandable implementation hint and links to a searchable English/Chinese programming manual. The manual covers shapes, reductions, conditional aggregation, slicing and copies, broadcasting, reshaping, allocation, clamping, and repeated updates, with examples for the participant’s environment. Hint openings and manual access are recorded. There is no breakpoint debugger or interactive shell. No external CDN is used.

## Start on this workstation

From the repository root:

```sh
.venv/bin/python -m studies.user_study.site.server --host 127.0.0.1 --port 8877
```

Open `http://127.0.0.1:8877`. The server prints a private researcher URL. Its key is also in `studies/user_study/site/runtime/config.json`. Keep this file and participant links private: the links grant access to their sessions.

The dashboard creates personal links, displays progress and final grades, and opens source histories, execution logs, ratings, and interaction timelines. It can pause/resume a participant's current task and retry an interrupted frozen grade. Download participant links as CSV and study records as JSON. Preview links are marked **DEMO** and excluded from the research export. Participant detail exports can include previews.

The server must run in a context where `bwrap --unshare-all` is permitted. If launching from a coding-agent sandbox that prohibits nested namespaces, run the server outside that outer sandbox. The app still isolates each submitted program with bubblewrap.

## Rebuild / install dependencies

Tested on Linux x86-64 with Python 3.14.7, NumPy 2.2.2, Rust host 1.96.1, RISC Zero Rust toolchain 1.97.0, RISC Zero SDK/VM 3.0.3, ndarray 0.16.1, Node 26.8.2, and bubblewrap. Cargo and npm lockfiles pin the dependency trees. The Python worker currently expects the repository's Python 3.14 virtual environment and `/usr/bin/python3.14`.

1. Build this checkout's Zinnia extension using the repository development instructions. The study depends on the compiler fixes from the six-task compatibility pilot. For this existing checkout: `PYTHON=.venv/bin/python scripts/develop.sh --offline`.
2. Install bubblewrap and util-linux (`prlimit`) through the operating system, plus Node/npm and rustup/stable Rust.
3. Install `rzup` using the official RISC Zero installation instructions, then install the tested components:

   ```sh
   rzup install rust 1.97.0
   rzup install r0vm 3.0.3
   ```

4. Build browser assets and both Rust projects, then run real execution preflights:

   ```sh
   .venv/bin/python -m studies.user_study.site.prepare
   # If all packages are cached:
   .venv/bin/python -m studies.user_study.site.prepare --offline
   ```

Preparation saves `runtime/build-manifest.json`, including compiler/library source hashes, the native Zinnia extension hash, executable hash, and versions. Archive this with each study export. Freeze the checkout, tests, toolchains, and task content before enrollment. No Cargo or npm network access occurs while a participant runs code.

## Study settings and interpretation

- Assignments use randomly ordered pairs balanced across the two conditions. With the default cyclic order, the nth person in each condition receives the same rotation of the six tasks; twelve links cover all six positions once in each condition. This is positional counterbalancing, not full carryover counterbalancing. The dashboard also supports easy-to-hard order for everyone. Use the same ordering policy throughout a cohort.
- The default budgets are **5, 7, 9, 12, 12, and 15 minutes** (60 total), plus tutorial and practice. These are provisional, not human-pilot estimates. The researcher can change budgets for newly created links; existing assignments keep their original budgets.
- Task descriptions are revealed when the participant starts that task's timer. The server clock continues during compilation/execution. Participants can pause and resume a task; an opaque overlay hides task material and disables interaction while paused. Closing the page pauses timing, and reopening the personal link resumes automatic pauses. Explicit participant pauses require Resume; researcher pauses require the researcher to resume them. A two-second heartbeat detects missing close notifications after ten seconds and pauses back to the last confirmed heartbeat. All pauses are excluded from elapsed task time and recorded with their reason.
- On expiry, the latest **server-saved** draft is frozen and graded. Autosave debounces edits by 750 ms; unsent/offline text cannot be graded by the server. The browser retains unsaved text locally and offers source download. Only one active tab should be used per link; conflicting saves are rejected rather than silently overwriting code.
- Participants must pass practice examples with their current source. They can submit an imperfect task solution; correctness feedback does not prevent submission. Opening the submission form pauses the task clock and locks the saved draft. Rating time is excluded from task duration; cancelling the form resumes the same task. The paused form and unfinished ratings survive reloads. Difficulty and confidence each use seven visible, descriptive choices (stored as 1–7 for analysis). Expired tasks have no required ratings.
- Both languages use the same scalar correctness oracles and fixtures, including additional shapes for final grading. Participants see three examples, their actual outputs, and expected outputs. Custom input is validated and scored too. Hidden grading results are available only to the researcher. Passing examples is not a proof of correctness.
- The online smoothing task fixes **R=3 for all cases**. This is explicit in both language templates and descriptions, including tests on shapes that used other R values in the earlier compiler pilot.
- Rust uses ordinary `ndarray` arrays of `i64`; the project supplies the guest entry point, input/output protocol, and imports. Zinnia supplies NumPy as `np` and injects the shape annotations. Algorithms are not prescribed. The editor is a prepared function-level project; adding arbitrary packages is not part of either condition.
- The interface uses neutral wording and the same workflow for both groups. Correctness, source histories, compiler failures, run counts, wall-clock timing, experience, and ratings can support either language doing better. Automated checks below are engineering tests, not human usability results.

The consent text is a working draft. Replace it with the study's approved participant information, researcher contact, retention policy, and any compensation terms. Confirm the tutorial, reference material, time allowances, permitted assistance, and final analysis plan in a human pilot before recruitment.

## Execution semantics

**Zinnia:** compile the submitted function with input-shape annotations, then compile an output-instrumented version and execute its IR with Zinnia's Rust mock evaluator. Read the returned shape and integer cells from ordered compiled public outputs. A constrained zero input preserves constant cells in that output layout. Participant source is never executed by Python/NumPy; Python only hosts the compiler and marshals input/output data. The trusted parent compares compiled results with the scalar oracle. This does not generate a cryptographic proof. The compiled evaluator currently does not emit `print` output.

**Rust:** compile with the RISC Zero Rust toolchain for `riscv32im-risc0-zkvm-elf`, package the ELF with the v1 compatibility kernel, and execute it with `r0vm`. The trusted host reads the guest's committed output and compares it with the same oracle. This uses the VM executor, not proof generation. Native Rust execution and `--native-preview` are disabled for all sessions, including practice and previews.

New execution records use `zinnia-ir-mock` or `risc0-zkvm`. Historical `zinnia-compiled-mock` records used Python/NumPy result extraction followed by a compiled consistency check; they remain distinguishable in exports.

Each run has a private temporary work directory. The sandbox mounts only the required read-only toolchains/packages, has no network or home-directory access, drops capabilities, and uses private process namespaces. The database, test oracles, other participants' source, and reference solutions are never mounted. Jobs have wall/CPU/address-space/file/output limits; the RISC Zero executor also has a 32-million-cycle limit. The execution pool has two workers and a bounded queue. Runtime directories are private and ignored by Git.

For remote participants, place this service behind an HTTPS reverse proxy; bind it to loopback and use a dedicated unprivileged account. Add service-level memory/process limits (for example systemd `MemoryMax=8G`, `TasksMax=128`) because per-process `prlimit` is not an aggregate cgroup limit. This implementation is intended for a controlled study deployment, not an unrestricted public code-execution service. The current page has no deployment or public URL until you configure hosting.

## Recorded data

SQLite uses WAL transactions. Tables record assignments/background, per-task timestamps and ratings, source snapshots, execution requests and exact inputs/source hashes, results, and interaction events. Events include focus/visibility changes, paste events (not clipboard contents), reference access, saves, runs, pauses, and transitions. No unrelated browser/application content is recorded. The server uses its own timestamps; it does not trust participant-supplied elapsed times.

Final grading uses the frozen source even after the participant moves to the next task. Restarted jobs become infrastructure interruptions, not participant errors. A researcher retry preserves the old attempt and uses its frozen source and test inputs. Treat infrastructure failures separately from compilation or incorrect-output outcomes in analysis. Execution timestamps let analysis distinguish queue, execution, and participant time; the UI timer itself includes all three.

JSON exports contain sensitive participant code and capability tokens; keep raw exports private and remove tokens before sharing analysis data. Use the dashboard export for a consistent snapshot. For a restorable database backup, stop the server first or use SQLite's online backup API; do not copy only a live `.sqlite3` file while ignoring its WAL. Withdrawal stops the session and retains already recorded data for researcher handling; it is not automatic erasure.

## Verification

The [six-task Astra audit](../adversarial/README.md) checks 148 additional plausible Zinnia implementations, including helper functions, imports, filtered loops, NumPy reductions, and writable views. All programs pass both the compiler/oracle gate and this site's actual worker. Its result manifest identifies the verified compiler build. This is finite compatibility coverage, not a guarantee for every possible program.

From the repository root, outside an outer sandbox that blocks bubblewrap:

```sh
.venv/bin/python -m unittest studies.user_study.site.tests.test_store studies.user_study.site.tests.test_api studies.user_study.site.tests.test_execution -v
cd studies/user_study/site
npm run test:browser
```

The execution suite checks practice and all six tasks in both real engines against visible and hidden cases; it also checks wrong answers, diagnostics, print output, denied filesystem/network access, output limits, and timeout. Persistence tests cover assignments, practice gating, conflicting drafts, task binding, pause/restart/expiry, frozen grading, hidden-result privacy, retries, and completion.

The Playwright test starts an isolated temporary server/database, enrolls two synthetic participants, completes onboarding/practice and all six tasks in each language through the UI, and checks autosave/reload, EN/中文, custom input, pause/resume, API access controls, researcher inspection, and export. It writes screenshots and a report under ignored `test-results/`. Use `PLAYWRIGHT_CHROMIUM_EXECUTABLE=/path/to/chromium` to select an installed browser, or `npx playwright install chromium` to install the matching browser.

Tests never create records in your real study runtime. Their reference implementations live only in `tests/reference.py`; no endpoint serves them.
