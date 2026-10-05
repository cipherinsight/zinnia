"""Transactional assignment, timing, drafts, runs, and event persistence."""
import contextlib
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time

from .content import ORDER, BUDGETS, TITLES, starter, task_payload, cases, is_practice
from .design import DESIGN_ID, schedules


class StudyError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Store:
    HEARTBEAT_TIMEOUT = 10

    def __init__(self, root, clock=time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.clock = clock
        self.lock = threading.RLock()
        config = self.root / "config.json"
        if not config.exists():
            config.write_text(json.dumps({"admin_key": secrets.token_urlsafe(32), "budgets": BUDGETS}, indent=2))
            config.chmod(0o600)
        self.config = json.loads(config.read_text())
        if set(self.config["budgets"]) != set(ORDER):
            # Assignment defaults only: existing participants retain their own
            # saved order and budgets, and legacy task contracts still resolve.
            self.config["budgets"] = BUDGETS.copy()
            config.write_text(json.dumps(self.config, indent=2))
        self.path = self.root / "study.sqlite3"
        with self.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS participants(
              token TEXT PRIMARY KEY, label TEXT UNIQUE NOT NULL, condition TEXT NOT NULL CHECK(condition IN ('zinnia','rust')),
              preview INTEGER NOT NULL DEFAULT 0, locale TEXT NOT NULL DEFAULT 'en', task_order TEXT NOT NULL,
              budgets TEXT NOT NULL, stage TEXT NOT NULL DEFAULT 'consent', task_index INTEGER NOT NULL DEFAULT 0,
              created REAL NOT NULL, last_seen REAL, completed REAL, background TEXT);
            CREATE TABLE IF NOT EXISTS sessions(
              token TEXT NOT NULL, task TEXT NOT NULL, code TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
              started REAL NOT NULL, budget REAL, paused_at REAL, paused_total REAL NOT NULL DEFAULT 0,
              finished REAL, outcome TEXT, response TEXT, PRIMARY KEY(token,task));
            CREATE TABLE IF NOT EXISTS snapshots(
              id INTEGER PRIMARY KEY, token TEXT NOT NULL, task TEXT NOT NULL, revision INTEGER NOT NULL,
              at REAL NOT NULL, code TEXT NOT NULL, sha256 TEXT NOT NULL, UNIQUE(token,task,revision));
            CREATE TABLE IF NOT EXISTS runs(
              id TEXT PRIMARY KEY, token TEXT NOT NULL, task TEXT NOT NULL, purpose TEXT NOT NULL,
              code TEXT NOT NULL, source_sha256 TEXT NOT NULL, requested REAL NOT NULL,
              started REAL, finished REAL, status TEXT NOT NULL, result TEXT, inputs_json TEXT);
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY, token TEXT NOT NULL, task TEXT, at REAL NOT NULL, kind TEXT NOT NULL, payload TEXT);
            CREATE INDEX IF NOT EXISTS runs_token ON runs(token,requested);
            CREATE INDEX IF NOT EXISTS events_token ON events(token,at);
            """)
            participant_columns={r[1] for r in db.execute("PRAGMA table_info(participants)")}
            for name,kind in (("design","TEXT NOT NULL DEFAULT 'between_subjects'"),("cohort_id","TEXT"),("schedule_slot","INTEGER"),("task_conditions","TEXT"),("practice_order","TEXT"),("practice_index","INTEGER NOT NULL DEFAULT 0"),("practice_version","INTEGER NOT NULL DEFAULT 1")):
                if name not in participant_columns:db.execute(f"ALTER TABLE participants ADD COLUMN {name} {kind}")
            for table in ("sessions","runs"):
                fields={r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                for name,kind in (("condition","TEXT"),("period","INTEGER")):
                    if name not in fields:db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
                db.execute(f"UPDATE {table} SET condition=(SELECT condition FROM participants WHERE participants.token={table}.token) WHERE condition IS NULL")
            if "inputs_json" not in {r[1] for r in db.execute("PRAGMA table_info(runs)")}:
                db.execute("ALTER TABLE runs ADD COLUMN inputs_json TEXT")
            run_columns = {r[1] for r in db.execute("PRAGMA table_info(runs)")}
            for name, kind in (("revision", "INTEGER"), ("parent_run_id", "TEXT"), ("retry_of", "TEXT"), ("active_elapsed", "REAL"), ("evaluation_policy", "TEXT"), ("attempt_index", "INTEGER"), ("interval_active", "REAL")):
                if name not in run_columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {name} {kind}")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS attempt_capture ON runs(parent_run_id) WHERE purpose='attempt_grade' AND retry_of IS NULL")
            columns = {r[1] for r in db.execute("PRAGMA table_info(sessions)")}
            for name, kind in (("last_active", "REAL"), ("pause_reason", "TEXT"), ("presence_id", "TEXT")):
                if name not in columns:
                    db.execute(f"ALTER TABLE sessions ADD COLUMN {name} {kind}")
            db.execute("UPDATE sessions SET last_active=COALESCE((SELECT last_seen FROM participants WHERE participants.token=sessions.token),started) WHERE last_active IS NULL")
            db.execute("UPDATE sessions SET pause_reason='researcher' WHERE paused_at IS NOT NULL AND pause_reason IS NULL")
            # Server downtime must not consume participant time.
            db.execute("UPDATE sessions SET paused_at=last_active,pause_reason='disconnected' WHERE finished IS NULL AND paused_at IS NULL AND budget IS NOT NULL")
            # A restart cannot resurrect a subprocess. Preserve it as an infrastructure interruption.
            db.execute("UPDATE runs SET status='interrupted',finished=?,result=? WHERE status IN ('queued','running')",
                       (clock(), json.dumps({"status": "infrastructure_error", "message": "Server restarted during execution; run again."})))

    @contextlib.contextmanager
    def db(self):
        with self.lock:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def event(self, db, token, task, kind, payload=None):
        db.execute("INSERT INTO events(token,task,at,kind,payload) VALUES(?,?,?,?,?)",
                   (token, task, self.clock(), kind, json.dumps(payload or {})))

    def participant(self, db, token):
        row = db.execute("SELECT * FROM participants WHERE token=?", (token,)).fetchone()
        if not row:
            raise StudyError("Participant link not found", 404)
        return dict(row)

    def session(self, db, participant):
        stage = participant["stage"]
        if stage not in ("practice", "task"):
            raise StudyError("No active programming task", 409)
        task = self.practice_task(participant) if stage == "practice" else json.loads(participant["task_order"])[participant["task_index"]]
        return dict(db.execute("SELECT * FROM sessions WHERE token=? AND task=?", (participant["token"], task)).fetchone())

    @staticmethod
    def practice_task(p):
        if p.get("design")==DESIGN_ID:
            return "practice_"+json.loads(p["practice_order"])[p["practice_index"]]+("_v2" if p.get("practice_version",1)==2 else "")
        return "practice"

    @staticmethod
    def condition_for(p,task=None):
        if p.get("design")!=DESIGN_ID:return p["condition"]
        if task and not is_practice(task):return json.loads(p["task_conditions"])[task]
        if task and task.startswith("practice_"):return task.removeprefix("practice_").removesuffix("_v2")
        return json.loads(p["practice_order"])[min(p["practice_index"],1)]

    def create_crossover(self, count=12, preview=False, budgets=None, slot=1):
        if type(count) is not int or count!=(1 if preview else 12):
            raise StudyError("Create exactly 12 research links, or one crossover preview")
        if type(slot) is not int or not 1<=slot<=12:raise StudyError("Invalid schedule slot")
        budgets=budgets or self.config["budgets"]
        if set(budgets)!=set(ORDER) or any(type(v) is not int or not 60<=v<=7200 for v in budgets.values()):raise StudyError("Invalid task budgets")
        plans=schedules(ORDER)
        if preview:plans=[plans[slot-1]]
        else:secrets.SystemRandom().shuffle(plans)
        cohort="preview-"+secrets.token_hex(6) if preview else "crossover-"+secrets.token_hex(6)
        result=[]
        with self.db() as db:
            offset=db.execute("SELECT COUNT(*) FROM participants").fetchone()[0]
            for i,plan in enumerate(plans):
                token=secrets.token_urlsafe(24);label=f'{"DEMO" if preview else "P"}{offset+i+1:03d}'
                db.execute("""INSERT INTO participants(token,label,condition,preview,task_order,budgets,created,design,cohort_id,schedule_slot,task_conditions,practice_order,practice_version)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,2)""",(token,label,plan['practice_order'][0],int(preview),json.dumps(plan['order']),json.dumps(budgets),self.clock(),DESIGN_ID,cohort,plan['slot'],json.dumps(plan['task_conditions']),json.dumps(plan['practice_order'])))
                record={'token':token,'label':label,'condition':plan['practice_order'][0],'preview':preview,'design':DESIGN_ID,'cohort_id':cohort,'practice_version':2,**plan}
                result.append(record)
                self.event(db,token,None,'assigned',record)
        return result

    def remaining(self, session):
        if session["budget"] is None:
            return None
        end = session["finished"] or session["paused_at"] or self.clock()
        return max(0, session["budget"] - (end - session["started"] - session["paused_total"]))

    def editable(self, session):
        if session["finished"] or session["paused_at"]:
            raise StudyError("This task is closed or paused", 409)
        if self.remaining(session) == 0:
            raise StudyError("The task time has ended; the last saved draft will be submitted", 409)

    def create(self, count=2, order="balanced", preview=False, condition=None, budgets=None):
        if type(count) is not int or not 1 <= count <= 200 or order not in ("balanced", "progressive"):
            raise StudyError("Choose 1–200 participants and a valid task order")
        if condition not in (None, "zinnia", "rust") or (condition and not preview):
            raise StudyError("Real participants use balanced between-subjects assignments")
        budgets = budgets or self.config["budgets"]
        if set(budgets) != set(ORDER) or any(type(v) is not int or not 60 <= v <= 7200 for v in budgets.values()):
            raise StudyError("Each task budget must be an integer between 60 and 7,200 seconds")
        result = []
        with self.db() as db:
            counts = {c: db.execute("SELECT COUNT(*) FROM participants WHERE condition=? AND preview=0", (c,)).fetchone()[0] for c in ("zinnia", "rust")}
            offset = db.execute("SELECT COUNT(*) FROM participants").fetchone()[0]
            for i in range(count):
                chosen = condition or (secrets.choice(["zinnia", "rust"]) if counts["zinnia"] == counts["rust"] else min(counts, key=counts.get))
                # The nth participant in each condition receives the same cyclic order.
                shift = counts[chosen] % 6 if order == "balanced" else 0
                tasks = ORDER[shift:] + ORDER[:shift]
                token, label = secrets.token_urlsafe(24), f'{"DEMO" if preview else "P"}{offset+i+1:03d}'
                db.execute("INSERT INTO participants(token,label,condition,preview,task_order,budgets,created) VALUES(?,?,?,?,?,?,?)",
                           (token, label, chosen, int(preview), json.dumps(tasks), json.dumps(budgets), self.clock()))
                self.event(db, token, None, "assigned", {"condition": chosen, "order": tasks, "preview": preview})
                result.append({"token": token, "label": label, "condition": chosen, "order": tasks, "preview": preview})
                counts[chosen] += 1
        return result

    def start_session(self, db, p, task):
        condition=self.condition_for(p,task)
        code = starter(task, condition)
        budget = None if is_practice(task) else json.loads(p["budgets"])[task]
        period=None if is_practice(task) else p["task_index"]+1
        db.execute("INSERT INTO sessions(token,task,code,started,budget,last_active,condition,period) VALUES(?,?,?,?,?,?,?,?)", (p["token"],task,code,self.clock(),budget,self.clock(),condition,period))
        db.execute("INSERT INTO snapshots(token,task,revision,at,code,sha256) VALUES(?,?,?,?,?,?)",
                   (p["token"], task, 0, self.clock(), code, hashlib.sha256(code.encode()).hexdigest()))
        self.event(db, p["token"], task, "task_started", {"budget": budget})

    def state(self, token):
        with self.db() as db:
            p = self.participant(db, token)
            db.execute("UPDATE participants SET last_seen=? WHERE token=?", (self.clock(), token))
            output = {k: p[k] for k in ("label", "condition", "preview", "locale", "stage", "task_index", "design", "practice_index")}
            output.update(server_time=self.clock(), order=json.loads(p["task_order"]), budgets=json.loads(p["budgets"]))
            output["practice_total"]=2 if p["design"]==DESIGN_ID else 1
            output["condition"]=self.condition_for(p)
            if p["stage"] in ("practice", "task"):
                session = self.session(db, p)
                output.update(session=session, task=task_payload(session["task"], session["condition"]), condition=session["condition"], remaining=self.remaining(session))
                runs = db.execute("SELECT id,status,purpose,source_sha256,result FROM runs WHERE token=? AND task=? AND purpose IN ('examples','custom') ORDER BY requested DESC LIMIT 8", (token, session["task"])).fetchall()
                output["runs"] = [self.public_run(dict(r)) for r in runs if r["purpose"] != "grade"]
            elif p["stage"] == "ready":
                task = output["order"][p["task_index"]]
                output["condition"]=self.condition_for(p,task)
                output["next_task"] = {"id": task, "titles": TITLES[task], "condition":output["condition"]}
            return output

    def action(self, token, action, payload):
        with self.db() as db:
            p = self.participant(db, token)
            if "condition" in payload:
                raise StudyError("Programming language is fixed by the study assignment", 403)
            stage = p["stage"]
            target = None
            if action == "locale":
                if payload.get("locale") not in ("en", "zh"):
                    raise StudyError("Unsupported interface language")
                db.execute("UPDATE participants SET locale=? WHERE token=?", (payload["locale"], token))
            elif action == "withdraw" and stage not in ("complete", "withdrawn"):
                db.execute("UPDATE sessions SET finished=?,outcome='withdrawn' WHERE token=? AND finished IS NULL", (self.clock(), token))
                target = "withdrawn"
            elif action == "consent" and stage == "consent" and payload.get("agree") is True:
                target = "background"
            elif action == "background" and stage == "background":
                answers = payload.get("answers", {})
                if any(answers.get(k) not in ("none", "basic", "regular", "expert") for k in ("python", "rust", "arrays", "zk")):
                    raise StudyError("Please answer all four background questions")
                db.execute("UPDATE participants SET background=? WHERE token=?", (json.dumps(answers), token))
                target = "tutorial"
            elif action == "tutorial_done" and stage == "tutorial":
                self.start_session(db, p, self.practice_task(p))
                target = "practice"
            elif action == "practice_done" and stage == "practice":
                session = self.session(db, p)
                digest = hashlib.sha256(session["code"].encode()).hexdigest()
                passed = db.execute("SELECT result FROM runs WHERE token=? AND task=? AND purpose='examples' AND source_sha256=? AND status='complete'", (token, session["task"], digest)).fetchall()
                if not any(json.loads(r[0]).get("status") == "passed" for r in passed):
                    raise StudyError("Pass the practice examples with your current code before continuing", 409)
                db.execute("UPDATE sessions SET finished=?,outcome='practice_complete' WHERE token=? AND task=?", (self.clock(), token, session["task"]))
                if p["design"]==DESIGN_ID and p["practice_index"]==0:
                    db.execute("UPDATE participants SET practice_index=1 WHERE token=?",(token,))
                    target="tutorial"
                else:target="ready"
            elif action == "start_task" and stage == "ready":
                self.start_session(db, p, json.loads(p["task_order"])[p["task_index"]])
                target = "task"
            else:
                if action != "locale":
                    raise StudyError("This action is not available at the current study step", 409)
            if target:
                db.execute("UPDATE participants SET stage=? WHERE token=?", (target, token))
            self.event(db, token, None, action, payload)

    def save(self, token, code, revision, task):
        if not isinstance(code, str) or len(code.encode()) > 64_000 or type(revision) is not int:
            raise StudyError("Invalid draft or draft is larger than 64 KB")
        with self.db() as db:
            p = self.participant(db, token)
            s = self.session(db, p)
            self.editable(s)
            if task != s["task"]:
                raise StudyError("This draft belongs to a different task. Download it before reloading.", 409)
            if revision != s["revision"]:
                raise StudyError("Another tab changed this draft. Download your local copy, then reload.", 409)
            if code == s["code"]:
                return revision
            revision += 1
            db.execute("UPDATE sessions SET code=?,revision=? WHERE token=? AND task=?", (code, revision, token, s["task"]))
            digest = hashlib.sha256(code.encode()).hexdigest()
            db.execute("INSERT INTO snapshots(token,task,revision,at,code,sha256) VALUES(?,?,?,?,?,?)", (token, s["task"], revision, self.clock(), code, digest))
            self.event(db, token, s["task"], "draft_saved", {"revision": revision, "sha256": digest, "characters": len(code)})
            return revision

    def retry_grade(self, ident):
        with self.db() as db:
            row = db.execute("SELECT * FROM runs WHERE id=? AND purpose IN ('grade','attempt_grade')", (ident,)).fetchone()
            if not row or row["status"] in ("queued", "running", "deferred"):
                raise StudyError("No finished grading run to retry", 409)
            if db.execute("SELECT 1 FROM runs WHERE token=? AND task=? AND purpose=? AND status IN ('queued','running')", (row["token"], row["task"], row["purpose"])).fetchone():
                raise StudyError("A grading attempt is already in progress", 409)
            p = self.participant(db, row["token"])
            job = self.insert_run(db, p, dict(row), row["purpose"], json.loads(row["inputs_json"]) if row["inputs_json"] else None, retry_of=ident)
            self.event(db, row["token"], row["task"], "researcher_retry_grade", {"previous_run_id": ident, "run_id": job["id"]})
            return job

    def new_run(self, token, purpose, inputs=None):
        if purpose not in ("examples", "custom"):
            raise StudyError("Unsupported execution mode")
        with self.db() as db:
            p = self.participant(db, token)
            s = self.session(db, p)
            self.editable(s)
            active = db.execute("SELECT 1 FROM runs WHERE token=? AND purpose IN ('examples','custom') AND status IN ('queued','running')", (token,)).fetchone()
            if active:
                raise StudyError("An execution is already in progress", 409)
            previous = db.execute("SELECT active_elapsed FROM runs WHERE token=? AND task=? AND purpose IN ('examples','custom') ORDER BY requested DESC,rowid DESC LIMIT 1",(token,s['task'])).fetchone()
            count = db.execute("SELECT COUNT(*) FROM runs WHERE token=? AND task=? AND purpose IN ('examples','custom')",(token,s['task'])).fetchone()[0]
            elapsed = s['budget']-self.remaining(s) if s['budget'] is not None else None
            s.update(attempt_index=count+1, active_elapsed=elapsed,
                     interval_active=(elapsed-(previous['active_elapsed'] or 0) if elapsed is not None and (not previous or previous['active_elapsed'] is not None) else None) if previous else elapsed)
            job = self.insert_run(db, p, s, purpose, inputs)
            if not is_practice(s["task"]):
                frozen = dict(s, parent_run_id=job["id"], evaluation_policy="attempt-trail-hidden-v1")
                self.insert_run(db, p, frozen, "attempt_grade", status="deferred")
            return job

    def claim_attempt_grades(self, limit=2):
        """Score every frozen attempt after the participant finishes/leaves.

        Delay evaluation, not capture: no extra execution in that participant's
        timed session. Interrupted evaluations remain visible and retryable.
        """
        with self.db() as db:
            rows = db.execute("""SELECT r.*,p.condition AS fallback_condition FROM runs r JOIN participants p ON p.token=r.token
                WHERE r.purpose='attempt_grade' AND r.status='deferred'
                AND p.stage IN ('complete','withdrawn')
                AND NOT EXISTS (SELECT 1 FROM runs live WHERE live.token=r.token AND live.status IN ('queued','running'))
                ORDER BY r.requested,r.id LIMIT ?""", (limit,)).fetchall()
            jobs=[]
            for row in rows:
                db.execute("UPDATE runs SET status='queued' WHERE id=? AND status='deferred'",(row['id'],))
                jobs.append({**dict(row), "condition":row["condition"] or row["fallback_condition"], "inputs":json.loads(row["inputs_json"])})
            return jobs

    def insert_run(self, db, p, s, purpose, inputs=None, status="queued", retry_of=None):
        ident = secrets.token_urlsafe(18)
        inputs = inputs if inputs is not None else cases(s["task"], hidden=purpose in ("grade", "attempt_grade"))
        run = {"id": ident, "token": p["token"], "task": s["task"], "condition": s.get("condition") or self.condition_for(p,s["task"]), "purpose": purpose, "code": s["code"], "inputs": inputs}
        db.execute("""INSERT INTO runs(id,token,task,purpose,code,source_sha256,requested,status,inputs_json,
            revision,parent_run_id,retry_of,active_elapsed,evaluation_policy,attempt_index,interval_active,condition,period) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ident,p["token"],s["task"],purpose,s["code"],hashlib.sha256(s["code"].encode()).hexdigest(),self.clock(),status,json.dumps(inputs),
             s.get("revision"),s.get("parent_run_id"),retry_of,s.get("active_elapsed"),s.get("evaluation_policy"),s.get("attempt_index"),s.get("interval_active"),run["condition"],s.get("period")))
        self.event(db, p["token"], s["task"], "run_requested", {"id": ident, "purpose": purpose})
        return run

    def finish_task(self, token, timed_out=False, response=None):
        with self.db() as db:
            p = self.participant(db, token)
            if p["stage"] != "task":
                return None
            s = self.session(db, p)
            self._expire_presence(db, s)
            s = self.session(db, p)
            expired = self.remaining(s) == 0
            # Only a live heartbeat may confirm that the deadline was reached.
            if timed_out and s["last_active"] < s["started"] + s["paused_total"] + s["budget"]:
                return None
            if timed_out and not expired:
                return None
            if s["paused_at"] and s["pause_reason"] != "submission":
                raise StudyError("This task is paused", 409)
            response = response or {}
            if not expired and any(type(response.get(k)) is not int or not 1 <= response[k] <= 7 for k in ("difficulty", "confidence")):
                raise StudyError("Please answer the two task rating questions")
            ended = s["started"] + s["paused_total"] + s["budget"] if expired else (s["paused_at"] or self.clock())
            db.execute("UPDATE sessions SET finished=?,outcome=?,response=? WHERE token=? AND task=?", (ended, "timeout" if expired else "submitted", json.dumps(response), token, s["task"]))
            s["active_elapsed"] = s["budget"]-self.remaining(s)
            previous = db.execute("SELECT active_elapsed FROM runs WHERE token=? AND task=? AND purpose IN ('examples','custom') ORDER BY requested DESC,rowid DESC LIMIT 1",(token,s["task"])).fetchone()
            s["interval_active"] = s["active_elapsed"]-(previous[0] or 0) if previous and previous[0] is not None else (s["active_elapsed"] if not previous else None)
            run = self.insert_run(db, p, s, "grade")
            index = p["task_index"] + 1
            order = json.loads(p["task_order"])
            # Selective researcher resets retain completed tasks and their periods.
            while index < len(order) and db.execute(
                "SELECT 1 FROM sessions WHERE token=? AND task=? AND finished IS NOT NULL",
                (token, order[index]),
            ).fetchone():
                index += 1
            last = index == len(order)
            db.execute("UPDATE participants SET task_index=?,stage=?,completed=? WHERE token=?", (index, "complete" if last else "ready", self.clock() if last else None, token))
            self.event(db, token, s["task"], "task_finished", {"timeout": expired, "final_run": run["id"], "revision": s["revision"]})
            return run

    def update_run(self, ident, status, result=None):
        with self.db() as db:
            if status == "running":
                db.execute("UPDATE runs SET status='running',started=? WHERE id=?", (self.clock(), ident))
            else:
                db.execute("UPDATE runs SET status=?,finished=?,result=? WHERE id=?", (status, self.clock(), json.dumps(result), ident))
                r = db.execute("SELECT token,task FROM runs WHERE id=?", (ident,)).fetchone()
                self.event(db, r["token"], r["task"], "run_finished", {"id": ident, "status": result.get("status")})

    @staticmethod
    def public_run(run):
        output = {k: run[k] for k in ("id", "status", "purpose", "source_sha256") if k in run}
        if run["purpose"] in ("examples", "custom") and run.get("result"):
            output["result"] = json.loads(run["result"])
        return output

    def get_run(self, token, ident):
        with self.db() as db:
            row = db.execute("SELECT * FROM runs WHERE token=? AND id=? AND purpose!='attempt_grade'", (token, ident)).fetchone()
            if not row:
                raise StudyError("Run not found", 404)
            return self.public_run(dict(row))

    def _set_pause(self, db, s, reason, at=None):
        at = self.clock() if at is None else at
        if not s["paused_at"]:
            db.execute("UPDATE sessions SET paused_at=?,pause_reason=? WHERE token=? AND task=?", (at, reason, s["token"], s["task"]))
            self.event(db, s["token"], s["task"], "task_paused", {"reason": reason, "effective_at": at})

    def _expire_presence(self, db, s):
        if (s["budget"] is not None and not s["paused_at"]
                and s["last_active"] < s["started"] + s["paused_total"] + s["budget"]
                and self.clock() - s["last_active"] > self.HEARTBEAT_TIMEOUT):
            self._set_pause(db, s, "disconnected", s["last_active"])

    def _resume(self, db, s):
        if s["paused_at"]:
            db.execute("UPDATE sessions SET paused_total=paused_total+?,paused_at=NULL,pause_reason=NULL,last_active=? WHERE token=? AND task=?", (self.clock()-s["paused_at"], self.clock(), s["token"], s["task"]))
            self.event(db, s["token"], s["task"], "task_resumed", {"reason": s["pause_reason"]})

    def presence(self, token, action, page_id, task=None):
        if action not in ("open", "heartbeat", "close", "pause", "resume", "review") or not isinstance(page_id, str) or not 8 <= len(page_id) <= 80:
            raise StudyError("Invalid presence request")
        with self.db() as db:
            p = self.participant(db, token)
            if p["stage"] != "task":
                return
            s = self.session(db, p)
            if task != s["task"]:
                return
            if action != "open" and s["presence_id"] != page_id:
                return
            self._expire_presence(db, s)
            s = self.session(db, p)
            if action == "open":
                db.execute("UPDATE sessions SET presence_id=? WHERE token=? AND task=?", (page_id, token, s["task"]))
            if action in ("open", "heartbeat", "resume") and s["pause_reason"] in ("disconnected", "closed"):
                self._resume(db, s)
            elif action == "resume" and s["pause_reason"] in ("participant", "submission"):
                self._resume(db, s)
            s = self.session(db, p)
            if not s["paused_at"]:
                db.execute("UPDATE sessions SET last_active=? WHERE token=? AND task=?", (self.clock(), token, s["task"]))
                # An already exhausted task must still be graded.
                if action in ("close", "pause", "review") and self.remaining(s) > 0:
                    self._set_pause(db, s, {"close":"closed", "pause":"participant", "review":"submission"}[action])
                if action == "close":
                    db.execute("UPDATE sessions SET presence_id=NULL WHERE token=? AND task=?", (token, s["task"]))

    def pause(self, token, paused):
        with self.db() as db:
            p = self.participant(db, token)
            s = self.session(db, p)
            self._expire_presence(db, s)
            s = self.session(db, p)
            if s["finished"] or self.remaining(s) == 0:
                raise StudyError("Cannot pause a finished task", 409)
            if paused:
                self._set_pause(db, s, "researcher")
                db.execute("UPDATE sessions SET pause_reason='researcher' WHERE token=? AND task=?", (token, s["task"]))
            else:
                self._resume(db, s)
            self.event(db, token, s["task"], "researcher_pause" if paused else "researcher_resume")

    def log(self, token, kind, payload):
        if kind not in ("focus", "blur", "reference_open", "reference_close", "hint_open", "editor_paste", "connection_restored"):
            raise StudyError("Unsupported event")
        with self.db() as db:
            p = self.participant(db, token)
            task = self.session(db, p)["task"] if p["stage"] in ("practice", "task") else None
            self.event(db, token, task, kind, payload)

    def export(self, include_preview=False, token=None, cohort=None):
        with self.db() as db:
            tokens = [r[0] for r in db.execute("SELECT token FROM participants WHERE (? OR preview=0) AND (? IS NULL OR token=?) AND (? IS NULL OR cohort_id=?)", (int(include_preview), token, token, cohort, cohort))]
            result = {"exported_at": self.clock(), "schema_version": 3, "design": None, "participants": [], "sessions": [], "runs": [], "snapshots": [], "events": []}
            manifest = self.root / "build-manifest.json"
            result["build_manifest"] = json.loads(manifest.read_text()) if manifest.exists() else None
            for table in ("participants", "sessions", "runs", "snapshots", "events"):
                for t in tokens:
                    order = " ORDER BY requested,rowid" if table == "runs" else ""
                    result[table].extend(dict(r) for r in db.execute(f"SELECT * FROM {table} WHERE token=?"+order, (t,)))
            designs={p["design"] for p in result["participants"]}
            result["design"]=next(iter(designs)) if len(designs)==1 else ("mixed" if designs else DESIGN_ID)
            result["attempt_trail"] = self.attempt_trail(result["runs"])
            return result

    @staticmethod
    def attempt_trail(runs):
        """Research-only joined view; original execution rows remain in export."""
        scores={}
        for run in runs:
            if run['purpose']=='attempt_grade':scores[run['parent_run_id']]=run
        final={}
        for run in runs:
            if run['purpose']=='grade':final[(run['token'],run['task'])]=run['id']
        rows=[]
        for run in runs:
            if is_practice(run['task']) or run['purpose'] not in ('examples','custom','grade'):continue
            if run['purpose']=='grade' and final[(run['token'],run['task'])]!=run['id']:continue
            score=run if run['purpose']=='grade' else scores.get(run['id'])
            result=json.loads(score['result']) if score and score['result'] else None
            cases=result.get('cases',[]) if result else []
            rows.append({**{k:run.get(k) for k in ('id','token','task','purpose','condition','period','attempt_index','revision','source_sha256','requested','active_elapsed','interval_active')},
                         'execution_seconds':run['finished']-run['started'] if run['purpose']!='grade' and run['finished'] is not None and run['started'] is not None else None,
                         'hidden_run_id':score['id'] if score else None,
                         'hidden_status':result.get('status') if result else (score['status'] if score else 'not_recorded'),
                         'hidden_passed':sum(bool(c.get('passed')) for c in cases) if cases else None,
                         'hidden_total':len(cases) if cases else None})
        return rows
