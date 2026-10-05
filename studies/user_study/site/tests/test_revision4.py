"""Distinct stride/dilation, historical contracts, and matched negative controls."""
import tempfile
import unittest
import numpy as np
from ...revision3.rust_relationship_audit import VALID as OLD_RUST
from ...revision4.tasks import FILTER, FILTER_TASK
from ...revision4.reference import RUST
from .. import content
from ..execution import Runner

class FilterContractTests(unittest.TestCase):
    def test_stride_and_dilation_differ_and_old_contract_is_preserved(self):
        data={'x':[[[[1],[2],[3],[4],[5],[6]]]],'filters':[[[[2],[3]]]]}
        self.assertEqual(content.answer(FILTER,data),[[[[14],[24]]]])
        self.assertEqual(content.answer('strided_filters_v3',data),[[[[11],[21]]]])
        self.assertEqual(content.ORDER[3],FILTER)
        self.assertNotIn('strided_filters_v3',content.ORDER)

    def test_effective_filter_extent_and_examples(self):
        data={'x':np.ones((2,4,7,1),dtype=int).tolist(),'filters':np.ones((2,2,3,1),dtype=int).tolist()}
        result=np.asarray(content.answer(FILTER,data))
        self.assertEqual(result.shape,(2,1,1,2));self.assertTrue(np.all(result==6))
        data['x']=np.ones((2,3,7,1),dtype=int).tolist()
        with self.assertRaises(ValueError):content.validate_input(FILTER,data)
        for language in ('rust','zinnia'):
            payload=content.task_payload(FILTER,language)
            for i,phrases in enumerate(payload['requirements']):
                for phrase in phrases:self.assertIn(phrase,payload['descriptions'][i])
        for case in content.cases(FILTER)+content.cases(FILTER,True):content.validate_input(FILTER,case)

class FilterNegativeControlTests(unittest.TestCase):
    def test_same_wrong_algorithms_are_rejected_in_both_engines(self):
        shortcut='''def kernel(x, filters):
    a=x[:,::2,::2,:]
    B,H,W,C=a.shape
    K,U,V,C=filters.shape
    out=np.zeros((B,H-U+1,W-V+1,K),dtype=int)
    for b in range(B):
        for i in range(H-U+1):
            for j in range(W-V+1):
                for k in range(K):
                    out[b,i,j,k]=np.sum(a[b,i:i+U,j:j+V,:]*filters[k])
    return out
'''
        body=FILTER_TASK.variants['nested_loops'].replace('2*i+3*u','2*i+2*u').replace('2*j+3*v','2*j+2*v')
        wrong_spacing='def kernel(x, filters):\n'+'\n'.join('    '+line for line in body.splitlines())+'\n'
        controls=[('zinnia','old_shortcut',shortcut),('rust','old_shortcut',OLD_RUST['strided_filters_v3']['subsample_then_filter']),
                  ('zinnia','wrong_spacing',wrong_spacing),('rust','wrong_spacing',RUST[FILTER].replace('2*i+3*iu','2*i+2*iu').replace('2*j+3*iv','2*j+2*iv'))]
        with tempfile.TemporaryDirectory() as directory:
            runner=Runner(directory)
            try:
                for language,name,code in controls:
                    with self.subTest(language=language,control=name):
                        result=runner.execute(language,FILTER,code,content.cases(FILTER)+content.cases(FILTER,True))
                        # These controls compile and execute. Failure must be a
                        # genuine contract mismatch, not an unavailable API.
                        self.assertEqual(result['status'],'failed',result)
                        self.assertTrue(any(not case['passed'] for case in result['cases']))
                        if name=='wrong_spacing':self.assertTrue(any(case['passed'] for case in result['cases']))
            finally:runner.pool.shutdown()
