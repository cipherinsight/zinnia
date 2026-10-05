# Local verification — 2026-09-21

All checks below passed on the workstation described in README.md. These are synthetic engineering checks, not participant observations.

- **10 Python tests passed**: language/order assignment, current-code practice gate, draft conflicts and task binding, pause/restart/expiry, independent timer submission while disconnected, frozen grading/retries, access controls, and sandboxed execution.
- **Both execution engines passed all seven reference programs** (practice + six tasks), including every visible and hidden case: 116 case evaluations total. Rust ran inside RISC Zero 3.0.3, and Zinnia compiled and evaluated its constraint checks.
- Negative checks detected incorrect outputs, invalid source, a Rust panic, output limits, denied sandbox filesystem/network access, and forced timeout. Rust print output reached the console.
- **Two complete Chromium sessions passed**, one per language: onboarding, practice, all 12 timed task submissions, 16 example runs (including deliberate syntax errors), custom input, autosave/reload, EN/中文, server pause/resume, API privacy, researcher source history, and JSON download. All 12 final hidden grades passed. No browser JavaScript errors occurred.
- Desktop and mobile screenshots were inspected. The narrow header was adjusted so both interface-language buttons remain usable.
- Offline dependency preparation and both real execution preflights passed. The running local preview has two demo links and zero real participant assignments.

Reproduce using the commands in README.md. Browser evidence and synthetic exports are under ignored `test-results/`. `runtime/build-manifest.json` identifies the current build and is included in researcher exports. The site remains a local preview; task budgets and participant materials need human-pilot review before recruitment.
