import json
import tempfile
import unittest
from collections import Counter

from ..content import ORDER, BUDGETS
from ..store import Store, StudyError


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = 1000.
        self.store = Store(self.temp.name, clock=lambda: self.now)
        self.token = self.store.create(count=1, order='progressive')[0]['token']

    def tearDown(self):
        self.temp.cleanup()

    def practice(self):
        self.store.action(self.token, 'consent', {'agree': True})
        self.store.action(self.token, 'background', {'answers': dict.fromkeys(['python','rust','arrays','zk'], 'basic')})
        self.store.action(self.token, 'tutorial_done', {})

    def ready(self):
        self.practice()
        run = self.store.new_run(self.token, 'examples')
        self.store.update_run(run['id'], 'complete', {'status': 'passed'})
        self.store.action(self.token, 'practice_done', {})

    def test_assignments_balance_languages_and_orders(self):
        self.store.create(count=23)
        rows = self.store.export()['participants']
        self.assertEqual(Counter(p['condition'] for p in rows), {'rust':12,'zinnia':12})
        self.assertTrue(all(sorted(json.loads(p['task_order'])) == sorted(ORDER) for p in rows))
        with self.assertRaises(StudyError):
            self.store.action(self.token, 'locale', {'locale':'en', 'condition':'rust'})
        demo = self.store.create(count=1, preview=True, condition='rust')[0]
        self.assertNotIn(demo['token'], [p['token'] for p in self.store.export()['participants']])
        with self.assertRaises(StudyError):
            self.store.create(count=1,condition='zinnia')

    def test_practice_gate_requires_current_example_code(self):
        with self.assertRaises(StudyError):
            self.store.action(self.token, 'start_task', {})
        self.practice()
        custom = self.store.new_run(self.token, 'custom')
        self.store.update_run(custom['id'], 'complete', {'status':'passed'})
        with self.assertRaises(StudyError):
            self.store.action(self.token, 'practice_done', {})
        run = self.store.new_run(self.token, 'examples')
        self.store.update_run(run['id'], 'complete', {'status':'passed'})
        self.store.save(self.token, 'different code', 0, 'practice')
        with self.assertRaises(StudyError):
            self.store.action(self.token, 'practice_done', {})

    def heartbeat(self, seconds=0):
        for _ in range(seconds):
            self.now += 1
            self.store.presence(self.token, "heartbeat", "test-page-1", ORDER[0])

    def start_timed(self):
        self.ready()
        self.store.action(self.token, 'start_task', {})
        self.store.presence(self.token, "open", "test-page-1", ORDER[0])
        self.store.save(self.token, 'latest draft', 0, ORDER[0])

    def test_pause_restart_timeout_and_latest_source(self):
        self.start_timed()
        self.heartbeat(20)
        self.store.pause(self.token, True)
        self.now += 100
        self.assertEqual(self.store.state(self.token)['remaining'], BUDGETS[ORDER[0]]-20)
        self.store.presence(self.token, "resume", "test-page-1", ORDER[0])
        self.assertEqual(self.store.state(self.token)['session']['pause_reason'], 'researcher')
        with self.assertRaises(StudyError):
            self.store.save(self.token,'paused edit',1,ORDER[0])
        self.store.pause(self.token, False)
        self.store = Store(self.temp.name, clock=lambda:self.now)
        self.now += 100
        self.store.presence(self.token, "open", "test-page-1", ORDER[0])
        self.assertEqual(self.store.state(self.token)['remaining'], BUDGETS[ORDER[0]]-20)
        self.heartbeat(BUDGETS[ORDER[0]]-20)
        run = self.store.finish_task(self.token, timed_out=True)
        self.assertEqual(run['code'], 'latest draft')
        self.assertEqual(self.store.state(self.token)['stage'], 'ready')
        self.assertIsNone(self.store.finish_task(self.token, timed_out=True))
        self.assertNotIn('result',self.store.get_run(self.token,run['id']))

    def test_participant_pause_close_and_resume(self):
        self.start_timed()
        self.heartbeat(5)
        self.store.presence(self.token, "pause", "test-page-1", ORDER[0])
        self.now += 100
        self.store.presence(self.token, "open", "test-page-2", ORDER[0])
        self.assertEqual(self.store.state(self.token)['session']['pause_reason'], 'participant')
        self.assertEqual(self.store.state(self.token)['remaining'], BUDGETS[ORDER[0]]-5)
        with self.assertRaises(StudyError):
            self.store.new_run(self.token, 'examples')
        with self.assertRaises(StudyError):
            self.store.finish_task(self.token, response={'difficulty':1,'confidence':1})
        self.store.presence(self.token, "resume", "test-page-2", ORDER[0])
        self.now += 2
        self.store.presence(self.token, "close", "test-page-2", ORDER[0])
        self.now += 200
        self.store.presence(self.token, "heartbeat", "test-page-2", ORDER[0])
        self.assertEqual(self.store.state(self.token)['remaining'], BUDGETS[ORDER[0]]-7)
        self.store.presence(self.token, "open", "test-page-3", ORDER[0])
        self.store.presence(self.token, "close", "test-page-2", ORDER[0])
        self.assertIsNone(self.store.state(self.token)['session']['paused_at'])
        self.assertEqual(self.store.state(self.token)['session']['code'], 'latest draft')

    def test_lost_close_notification_does_not_expire_task(self):
        self.start_timed()
        self.heartbeat(BUDGETS[ORDER[0]]-2)
        self.now += 3
        self.assertIsNone(self.store.finish_task(self.token, timed_out=True))
        self.now += 1000
        self.assertIsNone(self.store.finish_task(self.token, timed_out=True))
        self.assertEqual(self.store.state(self.token)['remaining'], 2)
        self.store.presence(self.token, "open", "test-page-2", ORDER[0])
        self.now += 2
        self.store.presence(self.token, "heartbeat", "test-page-2", ORDER[0])
        self.assertIsNotNone(self.store.finish_task(self.token, timed_out=True))

    def test_submission_rating_time_is_excluded(self):
        self.start_timed()
        self.heartbeat(10)
        self.store.presence(self.token, "review", "test-page-1", ORDER[0])
        remaining = self.store.state(self.token)['remaining']
        self.now += 1000
        self.assertIsNone(self.store.finish_task(self.token, timed_out=True))
        self.store.presence(self.token, "open", "test-page-2", ORDER[0])
        self.assertEqual(self.store.state(self.token)['remaining'], remaining)
        self.assertEqual(self.store.state(self.token)['session']['pause_reason'], 'submission')
        with self.assertRaises(StudyError):
            self.store.save(self.token, 'changed while rating', 1, ORDER[0])
        run = self.store.finish_task(self.token, response={'difficulty':3,'confidence':6})
        self.assertEqual(run['code'], 'latest draft')
        session = next(s for s in self.store.export()['sessions'] if s['task']==ORDER[0])
        self.assertEqual(session['finished']-session['started']-session['paused_total'], 10)

    def test_cancel_submission_resumes_same_budget(self):
        self.start_timed()
        self.heartbeat(10)
        self.store.presence(self.token, 'review', 'test-page-1', ORDER[0])
        self.now += 200
        self.store.presence(self.token, 'resume', 'test-page-1', ORDER[0])
        self.assertIsNone(self.store.state(self.token)['session']['paused_at'])
        self.heartbeat(3)
        self.assertEqual(self.store.state(self.token)['remaining'], BUDGETS[ORDER[0]]-13)

    def test_cas_task_binding_and_hidden_result_privacy(self):
        self.ready()
        self.store.action(self.token, 'start_task', {})
        self.store.save(self.token,'code v1',0,ORDER[0])
        for revision, task in [(0,ORDER[0]),(1,ORDER[1])]:
            with self.assertRaises(StudyError):
                self.store.save(self.token,'stale',revision,task)
        run = self.store.finish_task(self.token,response={'difficulty':2,'confidence':6})
        self.store.update_run(run['id'],'complete',{'status':'passed','cases':[{'secret':True}]})
        self.assertNotIn('result',self.store.get_run(self.token,run['id']))
        other = self.store.create(count=1)[0]['token']
        with self.assertRaises(StudyError):
            self.store.get_run(other,run['id'])
        retry = self.store.retry_grade(run['id'])
        self.assertNotEqual(retry['id'],run['id'])
        self.assertEqual(retry['code'],'code v1')
        self.assertEqual(len(self.store.export()['runs']),3)

    def test_all_six_submissions_complete_one_condition(self):
        self.ready()
        for task in ORDER:
            self.store.action(self.token, 'start_task', {})
            self.assertEqual(self.store.state(self.token)['task']['id'],task)
            self.store.finish_task(self.token,response={'difficulty':3,'confidence':4})
        state = self.store.state(self.token)
        self.assertEqual(state['stage'],'complete')
        self.assertEqual(state['task_index'],6)
        data = self.store.export()
        self.assertEqual(len([r for r in data['runs'] if r['purpose']=='grade']),6)
        self.store = Store(self.temp.name,clock=lambda:self.now)
        self.assertEqual({r['status'] for r in self.store.export()['runs'] if r['purpose']=='grade'},{'interrupted'})
