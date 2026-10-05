"""Immutable full trails, active-time accounting, private scoring and recovery."""
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from ..store import Store, StudyError
from ..content import ORDER,cases

class AttemptTrailTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.now=1000.
        self.store=Store(self.tmp.name,clock=lambda:self.now)
        self.token=self.store.create(1,order='progressive',preview=True,condition='zinnia')[0]['token']
        self.task=ORDER[0]
        with self.store.db() as db:
            db.execute("UPDATE participants SET stage='task' WHERE token=?",(self.token,))
            self.store.start_session(db,self.store.participant(db,self.token),self.task)
        self.store.presence(self.token,'open','test-page',self.task)
    def tearDown(self):self.tmp.cleanup()
    def tick(self,n):
        for _ in range(n):
            self.now+=1;self.store.presence(self.token,'heartbeat','test-page',self.task)
    def attempt(self,code,revision,purpose='examples'):
        revision=self.store.save(self.token,code,revision,self.task)
        job=self.store.new_run(self.token,purpose,cases(self.task)[:1] if purpose=='custom' else None)
        self.store.update_run(job['id'],'running')
        self.tick(2)
        self.store.update_run(job['id'],'complete',{'status':'compile_error' if code=='broken' else 'passed'})
        return job,revision
    def export(self):return self.store.export(include_preview=True,token=self.token)

    def test_every_attempt_frozen_with_active_intervals_and_final(self):
        self.tick(10)
        first,rev=self.attempt('broken',0)
        self.store.presence(self.token,'pause','test-page',self.task)
        self.now+=100
        self.store.presence(self.token,'resume','test-page',self.task)
        self.tick(5)
        second,rev=self.attempt('corrected',rev,'custom')
        self.tick(3)
        third,rev=self.attempt('corrected',rev)
        data=self.export();captured=[r for r in data['runs'] if r['purpose']=='attempt_grade']
        self.assertEqual(len(captured),3)
        self.assertEqual([r['code'] for r in captured],['broken','corrected','corrected'])
        self.assertEqual([r['attempt_index'] for r in captured],[1,2,3])
        self.assertEqual([r['active_elapsed'] for r in captured],[10,17,22])
        self.assertEqual([r['interval_active'] for r in captured],[10,7,5])
        self.assertEqual([r['parent_run_id'] for r in captured],[first['id'],second['id'],third['id']])
        self.assertTrue(all(json.loads(r['inputs_json'])==cases(self.task,True) for r in captured))
        self.assertEqual(self.store.claim_attempt_grades(),[])
        self.assertEqual(len(self.store.state(self.token)['runs']),3)
        for r in captured:
            with self.assertRaises(StudyError):self.store.get_run(self.token,r['id'])
        self.store.presence(self.token,'review','test-page',self.task)
        self.now+=200
        grade=self.store.finish_task(self.token,response={'difficulty':4,'confidence':4})
        self.store.update_run(grade['id'],'complete',{'status':'passed','cases':[{'passed':True}]})
        # Completing a task does not start extra execution during later tasks.
        self.assertEqual(self.store.claim_attempt_grades(),[])
        self.store.action(self.token,'withdraw',{})
        jobs=self.store.claim_attempt_grades(10)
        self.assertEqual(len(jobs),3)
        self.assertEqual(self.store.claim_attempt_grades(10),[])
        for i,job in enumerate(jobs):
            self.store.update_run(job['id'],'complete',{'status':'compile_error'} if i==0 else {'status':'passed','cases':[{'passed':True}]})
        trail=self.export()['attempt_trail']
        self.assertEqual(len(trail),4)
        self.assertEqual([r['hidden_status'] for r in trail],['compile_error','passed','passed','passed'])
        self.assertEqual(trail[-1]['active_elapsed'],24)
        self.assertEqual(trail[-1]['interval_active'],2)
        self.assertEqual(trail[0]['execution_seconds'],2)

    def test_concurrent_run_and_restart_retry_preserve_sources(self):
        self.store.save(self.token,'frozen',0,self.task)
        def request():
            try:return self.store.new_run(self.token,'examples')
            except StudyError:return None
        with ThreadPoolExecutor(max_workers=2) as pool:jobs=list(pool.map(lambda _:request(),range(2)))
        self.assertEqual(sum(j is not None for j in jobs),1)
        public=next(j for j in jobs if j)
        self.store.update_run(public['id'],'complete',{'status':'passed'})
        self.store.action(self.token,'withdraw',{})
        hidden=self.store.claim_attempt_grades()[0]
        self.store.update_run(hidden['id'],'running')
        self.store=Store(self.tmp.name,clock=lambda:self.now)
        data=self.export();interrupted=next(r for r in data['runs'] if r['id']==hidden['id'])
        self.assertEqual(interrupted['status'],'interrupted')
        retry=self.store.retry_grade(hidden['id'])
        self.assertEqual(retry['code'],'frozen')
        row=next(r for r in self.export()['runs'] if r['id']==retry['id'])
        self.assertEqual(row['parent_run_id'],public['id'])
        self.assertEqual(row['source_sha256'],hashlib.sha256(b'frozen').hexdigest())
        self.assertEqual(row['attempt_index'],1)
        self.assertEqual(len(self.export()['attempt_trail']),1)
        self.assertEqual(self.export()['attempt_trail'][0]['hidden_run_id'],retry['id'])

    def test_hidden_jobs_never_become_public_feedback(self):
        public,_=self.attempt('broken',0)
        private=next(r for r in self.export()['runs'] if r['purpose']=='attempt_grade')
        self.store.update_run(private['id'],'complete',{'status':'failed','cases':[{'input':'SECRET','expected':'SECRET','passed':False}]})
        state=json.dumps(self.store.state(self.token))
        self.assertNotIn('SECRET',state);self.assertNotIn(private['id'],state)
        self.assertEqual(self.store.get_run(self.token,public['id'])['result']['status'],'compile_error')
        self.assertNotIn('result',self.store.public_run(next(r for r in self.export()['runs'] if r['id']==private['id'])))
        with self.assertRaises(StudyError):self.store.new_run(self.token,'attempt_grade')
