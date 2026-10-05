"""Run with: .venv/bin/python -m studies.user_study.site.server"""
import argparse
import csv
import hmac
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit, parse_qs

from . import content
from .execution import HERE, Runner
from .store import Store, StudyError
from .design import DESIGN_ID


class Application:
    def __init__(self, runtime, rust_mode="risc0"):
        self.store = Store(runtime)
        self.runner = Runner(runtime, rust_mode=rust_mode)
        self.stopping = threading.Event()

    def start_clock(self):
        def sweep():
            while not self.stopping.wait(1):
                with self.store.db() as db:
                    tokens = [r[0] for r in db.execute("SELECT token FROM participants WHERE stage='task'")]
                for token in tokens:
                    try:
                        self.runner.schedule(self.store, self.store.finish_task(token, timed_out=True))
                    except Exception as exc:
                        print(f"Timer service: {type(exc).__name__}: {exc}", flush=True)
                try:
                    for job in self.store.claim_attempt_grades():
                        self.runner.schedule(self.store, job)
                except Exception as exc:
                    print(f"Attempt evaluation service: {type(exc).__name__}: {exc}", flush=True)
        threading.Thread(target=sweep, daemon=True, name="study-clock").start()

    def dashboard(self):
        with self.store.db() as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM participants ORDER BY created")]
            for p in rows:
                p["task_order"] = json.loads(p["task_order"])
                p["task_conditions"] = json.loads(p["task_conditions"]) if p["task_conditions"] else {t:p["condition"] for t in p["task_order"]}
                p["practice_order"] = json.loads(p["practice_order"]) if p["practice_order"] else [p["condition"]]
                p["sessions"] = []
                for row in db.execute("SELECT * FROM sessions WHERE token=? ORDER BY started", (p["token"],)):
                    session = dict(row)
                    session.pop("code")
                    session["remaining"] = self.store.remaining(session)
                    grade = db.execute("SELECT status,result FROM runs WHERE token=? AND task=? AND purpose='grade' ORDER BY requested DESC LIMIT 1", (p["token"], session["task"])).fetchone()
                    session["grade"] = ({"job_status": grade[0], "result": json.loads(grade[1]) if grade[1] else None} if grade else None)
                    attempts = db.execute("SELECT r.id,r.status,r.result,r.parent_run_id,r.revision,r.active_elapsed,r.attempt_index FROM runs r WHERE r.token=? AND r.task=? AND r.purpose='attempt_grade' AND NOT EXISTS (SELECT 1 FROM runs retry WHERE retry.retry_of=r.id) ORDER BY r.attempt_index", (p["token"],session["task"])).fetchall()
                    session["attempt_scores"] = [{**dict(r),"result":json.loads(r["result"]) if r["result"] else None} for r in attempts]
                    p["sessions"].append(session)
                p["runs"] = db.execute("SELECT COUNT(*) FROM runs WHERE token=? AND purpose IN ('examples','custom')", (p["token"],)).fetchone()[0]
                p["edits"] = db.execute("SELECT COUNT(*) FROM snapshots WHERE token=? AND revision>0", (p["token"],)).fetchone()[0]
            return {"participants": rows, "server_time": self.store.clock(), "readiness": self.runner.readiness(), "budgets": self.store.config["budgets"]}


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Study/1"

        def log_message(self, fmt, *args):
            # Capability tokens never enter access logs.
            message = re.sub(r"/p/[A-Za-z0-9_-]+", "/p/[participant]", fmt % args)
            print(message, flush=True)

        def send(self, payload, status=200, mime="application/json; charset=utf-8", filename=None):
            data = json.dumps(payload).encode() if mime.startswith("application/json") else (payload.encode() if isinstance(payload, str) else payload)
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def admin(self):
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, "Bearer " + app.store.config["admin_key"]):
                raise StudyError("Researcher authentication required", 401)

        def body(self):
            if self.headers.get_content_type() != "application/json":
                raise StudyError("Send JSON", 415)
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > 100_000:
                raise StudyError("Request is too large", 413)
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise StudyError("Cross-origin requests are not accepted", 403)
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise StudyError("Cross-site requests are not accepted", 403)
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise StudyError("Expected a JSON object")
            return value

        def do_GET(self):
            try:
                self.get()
            except StudyError as exc:
                self.send({"error": str(exc)}, exc.status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"GET failed: {type(exc).__name__}: {exc}", flush=True)
                self.send({"error": "Server error; your saved work is retained"}, 500)

        def route_path(self):
            path = urlsplit(self.path).path
            self.base_path = "/zinnia-study" if path == "/zinnia-study" or path.startswith("/zinnia-study/") else ""
            return path[len(self.base_path):] or "/"

        def get(self):
            path = self.route_path()
            if path == "/health":
                return self.send({"ok": True})
            if path.startswith("/api/admin"):
                self.admin()
                if path == "/api/admin":
                    return self.send(app.dashboard())
                if path == "/api/admin/export":
                    return self.send(app.store.export(cohort=parse_qs(urlsplit(self.path).query).get("cohort",[None])[0]), filename="study-export.json")
                if path == "/api/admin/assignments.csv":
                    out = io.StringIO()
                    writer = csv.writer(out)
                    writer.writerow(["participant","design","cohort","schedule_slot","path","practice_order","task_order","language_order"])
                    cohort=parse_qs(urlsplit(self.path).query).get("cohort",[None])[0]
                    for p in app.store.export(cohort=cohort)["participants"]:
                        order=json.loads(p["task_order"])
                        conditions=json.loads(p["task_conditions"]) if p["task_conditions"] else dict.fromkeys(order,p["condition"])
                        practice=json.loads(p["practice_order"]) if p["practice_order"] else [p["condition"]]
                        writer.writerow([p["label"],p["design"],p["cohort_id"],p["schedule_slot"],self.base_path+"/p/"+p["token"]," ".join(practice)," ".join(order)," ".join(conditions[t] for t in order)])
                    return self.send(out.getvalue(), mime="text/csv; charset=utf-8", filename="participant-links.csv")
                match = re.fullmatch(r"/api/admin/participant/([\w-]{20,64})", path)
                if match:
                    return self.send(app.store.export(include_preview=True, token=match[1]))
            match = re.fullmatch(r"/api/p/([\w-]{20,64})/(state|runs/[\w-]+)", path)
            if match:
                token, action = match.groups()
                if action == "state":
                    app.runner.schedule(app.store, app.store.finish_task(token, timed_out=True))
                    return self.send(app.store.state(token))
                return self.send(app.store.get_run(token, action.split("/")[1]))
            static = {"/bundle.js": ("bundle.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in static:
                file, mime = static[path]
                return self.send((HERE/"static"/file).read_bytes(), mime=mime)
            if path in ("/", "/admin") or re.fullmatch(r"/p/[\w-]{20,64}", path):
                return self.send((HERE/"static/index.html").read_text().replace("__STUDY_BASE__", self.base_path), mime="text/html; charset=utf-8")
            raise StudyError("Not found", 404)

        def do_POST(self):
            try:
                self.post()
            except StudyError as exc:
                self.send({"error": str(exc)}, exc.status)
            except (ValueError, TypeError, KeyError) as exc:
                self.send({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f"POST failed: {type(exc).__name__}: {exc}", flush=True)
                self.send({"error": "Server error; your saved work is retained"}, 500)

        def post(self):
            path = self.route_path()
            data = self.body()
            if path.startswith("/api/admin/"):
                self.admin()
                if path == "/api/admin/create":
                    if app.runner.rust_mode != "risc0" and not data.get("preview"):
                        raise StudyError("Native preview mode cannot create research participants", 409)
                    if data.get("design")==DESIGN_ID:
                        if data.get("condition") is not None:raise StudyError("Crossover languages are assigned per task")
                        return self.send({"participants":app.store.create_crossover(count=data.get("count",12),preview=bool(data.get("preview")),budgets=data.get("budgets"),slot=data.get("slot",1))})
                    if data.get("design") not in (None,"between_subjects"):raise StudyError("Unknown study design")
                    return self.send({"participants": app.store.create(count=data.get("count", 2), order=data.get("order", "balanced"), preview=bool(data.get("preview")), condition=data.get("condition"), budgets=data.get("budgets"))})
                if path == "/api/admin/pause":
                    if type(data.get("paused")) is not bool:
                        raise StudyError("paused must be a Boolean")
                    app.store.pause(data["token"], data["paused"])
                    return self.send({"ok": True})
                if path == "/api/admin/retry-grade":
                    job = app.store.retry_grade(data["run_id"])
                    app.runner.schedule(app.store, job)
                    return self.send({"ok": True})
                raise StudyError("Not found", 404)
            match = re.fullmatch(r"/api/p/([\w-]{20,64})/(action|save|run|submit|event|presence|review)", path)
            if not match:
                raise StudyError("Not found", 404)
            token, action = match.groups()
            if "condition" in data:
                raise StudyError("Programming language is fixed by assignment", 403)
            if action == "review":
                with app.store.lock:
                    app.store.save(token, data["code"], data["revision"], data["task"])
                    app.store.presence(token, "review", data.get("page_id"), data["task"])
                    state = app.store.state(token)
                    if state["session"]["pause_reason"] != "submission":
                        raise StudyError("Could not pause for submission; reload this task", 409)
                return self.send(state)
            if action == "presence":
                app.store.presence(token, data.get("action"), data.get("page_id"), data.get("task"))
                return self.send(app.store.state(token))
            if action == "action":
                if data.get("action") in ("start_task", "tutorial_done"):
                    state = app.store.state(token)
                    if not app.runner.readiness()[state["condition"]]:
                        raise StudyError("The execution environment is unavailable. Please contact the researcher before starting.", 503)
                app.store.action(token, data.get("action"), data)
                return self.send(app.store.state(token))
            if action == "save":
                return self.send({"revision": app.store.save(token, data["code"], data["revision"], data["task"])})
            if action == "run":
                purpose = data.get("purpose", "examples")
                if purpose not in ("examples", "custom"):
                    raise StudyError("Unsupported execution mode")
                # Serialize the save and frozen run against other tabs and the timer.
                with app.store.lock:
                    state = app.store.state(token)
                    task = state.get("task", {}).get("id")
                    inputs = None
                    if purpose == "custom":
                        content.validate_input(task, data["input"])
                        inputs = [data["input"]]
                    revision = app.store.save(token, data["code"], data["revision"], data["task"])
                    job = app.store.new_run(token, purpose, inputs)
                app.runner.schedule(app.store, job, inputs)
                return self.send({"run_id": job["id"], "revision": revision}, 202)
            if action == "submit":
                response = {k: data.get(k) for k in ("difficulty", "confidence")}
                response["comment"] = str(data.get("comment", ""))[:1000]
                with app.store.lock:
                    if not data.get("timeout"):
                        current = app.store.state(token)
                        session = current.get("session", {})
                        if session.get("pause_reason") == "submission":
                            if (data["code"] != session["code"] or data["revision"] != session["revision"] or data["task"] != session["task"]):
                                raise StudyError("Submission differs from the draft saved before rating", 409)
                        else:
                            app.store.save(token, data["code"], data["revision"], data["task"])
                    job = app.store.finish_task(token, timed_out=bool(data.get("timeout")), response=response)
                app.runner.schedule(app.store, job)
                return self.send(app.store.state(token))
            app.store.log(token, data["kind"], data.get("payload", {}))
            return self.send({"ok": True})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--runtime", type=Path, default=HERE/"runtime")
    args = parser.parse_args()
    app = Application(args.runtime)
    server = ThreadingHTTPServer((args.host, args.port), handler(app))
    app.start_clock()
    print(f"Study: http://{args.host}:{args.port}", flush=True)
    print(f"Researcher: http://{args.host}:{args.port}/admin#key={app.store.config['admin_key']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stopping.set()
        server.server_close()
        app.runner.pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
