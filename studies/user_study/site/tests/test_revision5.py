import unittest
from .. import content
from ...revision5 import content as v5

class ModerateTaskTests(unittest.TestCase):
    def test_public_hidden_inputs_and_shapes(self):
        self.assertEqual(content.ORDER,v5.ORDER)
        self.assertEqual(set(content.BUDGETS.values()),{600})
        for task in v5.ORDER:
            for data in v5.cases(task)+v5.cases(task,True):
                v5.validate_input(task,data)
                self.assertEqual(__import__('numpy').asarray(v5.answer(task,data)).ndim,v5.OUT_RANK[task])
            self.assertNotIn('hints',v5.task_payload(task,'zinnia'))

    def test_hand_calculations(self):
        checks=[
            (v5.ORDER[0],{'x':[[1],[3],[5]],'gain':[2],'bias':[-3]},[58]),
            (v5.ORDER[1],{'x':[[1,2],[3,6]],'y':[[2],[5]]},[[12],[24]]),
            (v5.ORDER[2],{'x':[[1,2,0]],'y':[[3,5,4]],'weight':[2,0,1]},[[24]]),
            (v5.ORDER[3],{'x':[[1,2,3],[4,5,6]],'filter':[[1,0],[0,2]]},[[11,14]]),
            (v5.ORDER[4],{'x':[[[1],[2]],[[3],[4]]]},[[[14]]]),
            (v5.ORDER[5],{'x':[[0,0,0],[0,64,0],[0,0,0]],'mask':[[1,1,1],[1,1,1],[1,1,1]]},[[0,0,0],[0,8,0],[0,0,0]])]
        for task,data,expected in checks:self.assertEqual(v5.answer(task,data),expected)

    def test_invalid_array_relationships(self):
        for task,data in [
            (v5.ORDER[0],{'x':[[1,2]],'gain':[1],'bias':[0,0]}),
            (v5.ORDER[1],{'x':[[1]],'y':[[1],[2]]}),
            (v5.ORDER[2],{'x':[[1,2]],'y':[[1,2]],'weight':[1,1]}),
            (v5.ORDER[3],{'x':[[1]],'filter':[[1,2]]}),
            (v5.ORDER[4],{'x':[[[1],[2]]]}),
            (v5.ORDER[5],{'x':[[1]],'mask':[[1,0]]})]:
            with self.assertRaises(ValueError):v5.validate_input(task,data)

    def test_historical_contract_still_resolves(self):
        old=content.task_payload('strided_filters_v4','rust')
        self.assertEqual(old['ranks'],[4,4])
        self.assertEqual(old['output_rank'],4)
        self.assertEqual(content.task_payload('image_filter_v5','rust')['ranks'],[2,2])
