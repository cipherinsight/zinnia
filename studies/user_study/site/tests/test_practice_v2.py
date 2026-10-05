"""Practice contract and preservation of previous training records."""
import unittest
import tempfile
from .. import content
from ..store import Store

class PracticeContentTests(unittest.TestCase):
    def test_examples_and_bilingual_requirements(self):
        for task in ('practice_zinnia_v2', 'practice_rust_v2'):
            payload = content.task_payload(task, 'rust' if 'rust' in task else 'zinnia')
            self.assertEqual([e['expected'] for e in payload['examples']], [[5,4],[3],[0,3,0],[0,0]])
            self.assertEqual(payload['ranks'], [2,1])
            self.assertTrue(content.is_practice(task))
            for locale in range(2):
                for phrase in payload['requirements'][locale]:
                    self.assertIn(phrase, payload['descriptions'][locale])
            with self.assertRaises(ValueError):
                content.validate_input(task, {'x': [[1,2]], 'baseline': [1]})

    def test_original_contract_is_preserved(self):
        for task in ('practice','practice_zinnia','practice_rust'):
            self.assertEqual(content.answer(task, {'x':[1,2], 'y':[3,4]}), [4,6])

class PracticePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)

    def test_started_original_practice_keeps_contract_after_restart(self):
        p = self.store.create_crossover(count=1, preview=True)[0]
        with self.store.db() as db:
            db.execute('UPDATE participants SET practice_version=1 WHERE token=?', (p['token'],))
        self.store.action(p['token'], 'consent', {'agree':True})
        self.store.action(p['token'], 'background', {'answers':dict.fromkeys(['python','rust','arrays','zk'],'basic')})
        self.store.action(p['token'], 'tutorial_done', {})
        reopened = Store(self.tmp.name)
        state = reopened.state(p['token'])
        self.assertEqual(state['task']['id'], 'practice_'+state['condition'])
        self.assertEqual(state['task']['ranks'], [1,1])
