"""Task-version preservation and revised contract boundary checks."""
import json
import tempfile
import unittest
from pathlib import Path
from ...tasks import TASKS as LEGACY_TASKS
from ...revision2 import content as v2
from .. import content
from ..store import Store

class RevisionTests(unittest.TestCase):
    def test_new_assignments_and_legacy_sessions_keep_distinct_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old_order=[t.name for t in LEGACY_TASKS]
            old_budgets=dict.fromkeys(old_order,300)
            (root/'config.json').write_text(json.dumps({'admin_key':'test-key','budgets':old_budgets}))
            store=Store(root)
            self.assertEqual(store.config['admin_key'],'test-key')
            new=store.create(1,order='progressive',preview=True,condition='rust')[0]
            self.assertEqual(new['order'],content.ORDER)
            # Simulate a saved pre-revision participant and verify normal resume.
            with store.db() as db:
                db.execute('UPDATE participants SET task_order=?,budgets=?,stage="task",task_index=0 WHERE token=?',(json.dumps(old_order),json.dumps(old_budgets),new['token']))
                p=store.participant(db,new['token'])
                store.start_session(db,p,'calibrate')
                db.execute('UPDATE sessions SET code=? WHERE token=?',('original pilot draft',new['token']))
            reopened=Store(root).state(new['token'])
            self.assertEqual(reopened['order'],old_order)
            self.assertEqual(reopened['budgets'],old_budgets)
            self.assertEqual(reopened['task']['id'],'calibrate')
            self.assertEqual(content.answer('calibrate',{'x':[-10,0,50,200]}),[0,5,105,255])
            with store.db() as db:
                self.assertEqual(db.execute('SELECT code FROM sessions WHERE token=?',(new['token'],)).fetchone()[0],'original pilot draft')

    def test_all_examples_valid_and_highlights_and_manual_topics_exist(self):
        manual=json.loads((Path(__file__).parents[1]/'static/manual.json').read_text())
        topics={t['id'] for t in manual}
        for task in v2.ORDER:
            for data in content.cases(task)+content.cases(task,True):content.validate_input(task,data)
            for language in ('zinnia','rust'):
                p=content.task_payload(task,language)
                self.assertTrue(set(p['manual_topics'])<=topics)
                for i,phrases in enumerate(p['requirements']):
                    for phrase in phrases:self.assertIn(phrase,p['descriptions'][i])

    def test_small_hand_computed_examples(self):
        self.assertEqual(content.answer('centered_gram_v2',{'x':[[[1,2],[3,6]]]}),[[[8,16],[16,32]]])
        self.assertEqual(content.answer('weighted_distances_v2',{'x':[[[1,2]]],'y':[[[3,5]]],'weight':[2,0]}),[[[8]]])
        # Saturation must precede the per-channel minimum and energy.
        self.assertEqual(content.answer('channel_energy_v2',{'x':[[[-64],[64]]],'gain':[4],'bias':[16]}),[[255**2]])
        # Cross-correlation keeps filter order; a true convolution flips it.
        self.assertEqual(content.answer('convolution_v2',{'x':[[[[1],[2],[3]]]],'filters':[[[[2],[3]]]]}),[[[[8],[13]]]])

    def test_mismatched_shapes_and_ranges_rejected(self):
        for task,data in [
            ('weighted_distances_v2',{'x':[[[1,2]]],'y':[[[1,2]],[[3,4]]],'weight':[1,1]}),
            ('weighted_distances_v2',{'x':[[[1]]],'y':[[[2]]],'weight':[-1]}),
            ('block_covariance_v2',{'x':[[[[1]]]]}),
            ('masked_diffusion_v2',{'x':[[[[1]]]],'mask':[[2]],'mix':[[1]]}),
        ]:
            with self.subTest(task=task),self.assertRaises(ValueError):content.validate_input(task,data)
