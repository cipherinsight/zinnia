import unittest
import numpy as np
from ...revision3 import content as v3
from .. import content

class RelationshipTests(unittest.TestCase):
    def test_hand_calculated_relationships(self):
        self.assertEqual(content.answer('channel_energy_v3',{'x':[[[1],[2]],[[1],[2]]],'gain':[[1],[3]],'bias':[[0],[0]]}),[[1],[9]])
        self.assertEqual(content.answer('cross_products_v3',{'x':[[[1,2],[3,6]]],'y':[[[2],[5]]]}),[[[12],[24]]])
        self.assertEqual(content.answer('projected_distances_v3',{'x':[[[1,2]]],'y':[[[0,0]]],'transform':[[[1,2,0],[0,-1,1]]],'weight':[[2,3,4]]}),[[[18]]])
        self.assertEqual(content.answer('strided_filters_v3',{'x':[[[[1],[2],[3],[4],[5]]]],'filters':[[[[2],[3]]]]}),[[[[11],[21]]]])
        a=np.arange(6).reshape(1,2,3,1)*np.array([1,2])
        result=np.asarray(content.answer('rectangular_covariance_v3',{'x':a.tolist()}))
        self.assertEqual(result.shape,(1,1,1,2,2))
        self.assertEqual(result[0,0,0].tolist(),[[630,1260],[1260,2520]])
        x=np.zeros((2,3,3,1),dtype=int);x[:,1,1,0]=8
        mask=np.zeros((2,3,3),dtype=int);mask[1]=1
        result=np.asarray(content.answer('batch_diffusion_v3',{'x':x.tolist(),'mask':mask.tolist(),'mix':[[[1]],[[1]]]}))
        self.assertEqual(result[:,1,1,0].tolist(),[8,1])

    def test_payloads_ranges_and_shape_errors(self):
        for task in v3.ORDER:
            for condition in ('zinnia','rust'):
                payload=content.task_payload(task,condition)
                for i,phrases in enumerate(payload['requirements']):
                    for phrase in phrases:self.assertIn(phrase,payload['descriptions'][i])
            for case in content.cases(task)+content.cases(task,True):content.validate_input(task,case)
        invalid=[('cross_products_v3',{'x':[[[1],[2]]],'y':[[[3]]]}),
                 ('projected_distances_v3',{'x':[[[1,2]]],'y':[[[1,2]]],'transform':[[[1],[2]]],'weight':[[1,2]]}),
                 ('strided_filters_v3',{'x':[[[[1],[2]]]],'filters':[[[[1],[2]]]]}),
                 ('rectangular_covariance_v3',{'x':[[[[1],[2]],[[3],[4]]]]})]
        for task,data in invalid:
            with self.subTest(task=task),self.assertRaises(ValueError):content.validate_input(task,data)
