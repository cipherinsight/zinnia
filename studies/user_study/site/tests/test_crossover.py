"""Allocation balance, dual-practice gating, and immutable per-run languages."""
import json
import tempfile
import unittest
from collections import Counter
from ..content import ORDER,is_practice
from ..design import DESIGN_ID,schedules
from ..store import Store,StudyError
from .reference import PYTHON,RUST

class CrossoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_selective_redo_skips_completed_tasks_and_preserves_periods(self):
        p = self.store.create_crossover(preview=True, count=1)[0]
        token, order = p['token'], p['order']
        redo = [t for t in order if p['task_conditions'][t] == 'zinnia']
        with self.store.db() as db:
            participant = self.store.participant(db, token)
            for index, task in enumerate(order):
                if task not in redo:
                    participant['task_index'] = index
                    self.store.start_session(db, participant, task)
                    db.execute("UPDATE sessions SET finished=?,outcome='submitted' WHERE token=? AND task=?",
                               (self.store.clock(), token, task))
            before = [tuple(r) for r in db.execute("SELECT * FROM sessions WHERE token=? ORDER BY task", (token,))]
            db.execute("UPDATE participants SET stage='ready',task_index=? WHERE token=?", (order.index(redo[0]), token))
        for task in redo:
            state = self.store.state(token)
            self.assertEqual(state['next_task']['id'], task)
            self.assertEqual(state['order'], order)
            self.store.action(token, 'start_task', {})
            state = self.store.state(token)
            self.assertEqual(state['session']['period'], order.index(task) + 1)
            self.assertEqual(state['session']['budget'], 600)
            self.store.finish_task(token, response={'difficulty': 3, 'confidence': 3})
        self.assertEqual(self.store.state(token)['stage'], 'complete')
        with self.store.db() as db:
            after = [tuple(r) for r in db.execute("SELECT * FROM sessions WHERE token=? AND condition='rust' ORDER BY task", (token,))]
        self.assertEqual(before, after)

    def test_complete_cohort_balances_people_tasks_positions_and_order(self):
        people=self.store.create_crossover()
        self.assertEqual(len(people),12)
        self.assertEqual({p['slot'] for p in people},set(range(1,13)))
        self.assertEqual(len({p['cohort_id'] for p in people}),1)
        for p in people:
            self.assertEqual(len(set(p['order'])),6)
            self.assertEqual(Counter(p['task_conditions'].values()),{'zinnia':3,'rust':3})
        for task in ORDER:
            self.assertEqual(Counter(p['task_conditions'][task] for p in people),{'zinnia':6,'rust':6})
            self.assertEqual(Counter(p['order'].index(task) for p in people),dict.fromkeys(range(6),2))
        for pos in range(6):
            self.assertEqual(Counter(p['task_conditions'][p['order'][pos]] for p in people),{'zinnia':6,'rust':6})
        pairs=Counter((a,b) for p in people for a,b in zip(p['order'],p['order'][1:]))
        self.assertEqual(len(pairs),30);self.assertEqual(set(pairs.values()),{2})
        training=Counter((p['practice_order'][0],p['task_conditions'][p['order'][0]]) for p in people)
        self.assertEqual(set(training.values()),{3});self.assertEqual(len(training),4)
        with self.assertRaises(StudyError):self.store.create_crossover(count=6)
        # Preview assignments cannot perturb a later balanced real cohort.
        self.store.create_crossover(count=1,preview=True)
        self.assertEqual(len(self.store.export(cohort=people[0]['cohort_id'])['participants']),12)
        self.assertEqual(self.store.export(cohort=people[0]['cohort_id'])['design'],DESIGN_ID)

    def complete_practice(self,token):
        self.store.action(token,'consent',{'agree':True})
        self.store.action(token,'background',{'answers':dict.fromkeys(['python','rust','arrays','zk'],'basic')})
        for index in range(2):
            state=self.store.state(token);self.assertEqual(state['stage'],'tutorial')
            condition=state['condition'];self.assertEqual(state['practice_index'],index)
            self.store.action(token,'tutorial_done',{})
            state=self.store.state(token);task=state['task']['id']
            self.assertEqual(task,'practice_'+condition+'_v2')
            with self.assertRaises(StudyError):self.store.action(token,'practice_done',{})
            code=(PYTHON if condition=='zinnia' else RUST)[task]
            self.store.save(token,code,0,task)
            run=self.store.new_run(token,'examples')
            self.assertEqual(run['condition'],condition)
            self.store.update_run(run['id'],'complete',{'status':'passed'})
            self.store.action(token,'practice_done',{})
        self.assertEqual(self.store.state(token)['stage'],'ready')

    def test_both_practices_switching_and_hidden_replay_use_frozen_language(self):
        p=self.store.create_crossover(count=1,preview=True)[0];token=p['token']
        self.complete_practice(token)
        final_ids=[]
        for index,task in enumerate(p['order']):
            condition=p['task_conditions'][task]
            self.assertEqual(self.store.state(token)['condition'],condition)
            self.store.action(token,'start_task',{})
            state=self.store.state(token)
            self.assertEqual(state['session']['condition'],condition)
            self.assertEqual(state['session']['period'],index+1)
            code=(PYTHON if condition=='zinnia' else RUST)[task]
            self.store.save(token,code,0,task)
            run=self.store.new_run(token,'examples')
            self.assertEqual(run['condition'],condition)
            self.store.update_run(run['id'],'complete',{'status':'passed'})
            grade=self.store.finish_task(token,response={'difficulty':4,'confidence':6})
            self.assertEqual(grade['condition'],condition);final_ids.append(grade['id'])
            self.store.update_run(grade['id'],'complete',{'status':'passed'})
        self.assertEqual(self.store.state(token)['stage'],'complete')
        replay=[]
        while jobs:=self.store.claim_attempt_grades():
            for job in jobs:
                replay.append(job)
                self.assertEqual(job['condition'],p['task_conditions'][job['task']])
                self.assertEqual(job['code'],(PYTHON if job['condition']=='zinnia' else RUST)[job['task']])
                self.store.update_run(job['id'],'complete',{'status':'passed'})
        self.assertEqual(Counter(j['condition'] for j in replay),{'zinnia':3,'rust':3})
        retry=self.store.retry_grade(final_ids[-1])
        self.assertEqual(retry['condition'],p['task_conditions'][p['order'][-1]])
        self.store.update_run(retry['id'],'complete',{'status':'passed'})
        exported=self.store.export(include_preview=True,token=token)
        self.assertEqual(len(exported['attempt_trail']),12)
        for row in exported['attempt_trail']:
            self.assertEqual(row['condition'],p['task_conditions'][row['task']])
            self.assertEqual(row['period'],p['order'].index(row['task'])+1)
        self.assertEqual(len([s for s in exported['sessions'] if is_practice(s['task'])]),2)
        # Reopening the DB cannot alter schedules or replay languages.
        reopened=Store(self.tmp.name).export(include_preview=True,token=token)
        self.assertEqual(json.loads(reopened['participants'][0]['task_conditions']),p['task_conditions'])

    def test_new_cohort_does_not_change_existing_single_environment_assignment(self):
        old=self.store.create(1,preview=True,condition='rust',order='progressive')[0]
        self.store.create_crossover()
        state=self.store.state(old['token'])
        self.assertEqual(state['design'],'between_subjects')
        self.assertEqual(state['condition'],'rust');self.assertEqual(state['practice_total'],1)
