# Public study deployment

Public entry: https://zhantong.me/zinnia-study/
Researcher entry: https://zhantong.me/zinnia-study/admin
Use the existing researcher key. Do not put keys or participant tokens in logs.

The VPS terminates HTTPS and proxies only /zinnia-study/ to its loopback port
18877. A persistent SSH reverse tunnel connects that port to 10.0.0.180:8877.
The existing local backend runs the compiled Zinnia and RISC Zero engines,
using the original runtime/study.sqlite3. Public and LAN links therefore
share drafts, timing, attempts and completion. P010 remains completed.
The machine at 10.0.0.180 must remain online; this is not a standalone VPS
installation of the compilers.

Local user services:

- ~/.config/systemd/user/zinnia-study.service
- ~/.config/systemd/user/zinnia-study-tunnel.service

Both are enabled and restart automatically. User lingering is enabled so
services do not require an interactive login. Inspect with:

```sh
systemctl --user status zinnia-study.service zinnia-study-tunnel.service
journalctl --user -u zinnia-study.service -u zinnia-study-tunnel.service
```

The SSH destination is the existing study-vps alias. The remote listener is
127.0.0.1:18877, not a public TCP listener. The nginx location is in
/etc/nginx/sites-available/comfy-ty-study; its original configuration was
backed up before modification. Existing ComfyTy routes remain intact.
Rate limits are in /etc/nginx/conf.d/zinnia-study-ratelimit.conf. Access
logging is disabled for this capability-token-bearing route.

The backend accepts both root paths and /zinnia-study/ paths. HTML injects
the matching path into assets, API requests, close beacons, researcher
links and CSV exports. Public link lists are in runtime/crossover-12x6-links
(JSON/CSV); CSV also retains lan_url. Database backups before deployment
are under ../study-backups relative to the repository root's parent.

Validation: original API tests and the full prefixed browser crossover flow
(2 complementary participants, both practices, all six tasks, both engines).
Public HTTPS checks cover every existing link, engine readiness and P010
completion. To repeat the isolated prefixed browser test:

```sh
STUDY_TEST_BASE_PATH=/zinnia-study npm run test:crossover --prefix studies/user_study/site
```
