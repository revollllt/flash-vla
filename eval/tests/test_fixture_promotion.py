import tempfile
from pathlib import Path
import unittest
import torch
from lab.optimize import fixture, promotion


class FixtureTests(unittest.TestCase):
    def test_snapshot_values_survive_mutation_and_preserve_alias(self):
        base=torch.arange(16,dtype=torch.float32)
        x=base[2:10]; y=base[4:12:2]
        record=fixture.snapshot([x,y,7,None])
        base.zero_()
        restored=fixture.restore(record)
        self.assertTrue(torch.equal(restored[0],torch.arange(2,10,dtype=torch.float32)))
        self.assertEqual(restored[1].stride(),(2,))
        restored[0][2]=99
        self.assertEqual(restored[1][0],99)
        self.assertEqual(restored[2:],[7,None])

    def test_latest_source_and_cross_target_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'a').mkdir(); (root/'b').mkdir()
            (root/'a/kernel').write_text('a'); (root/'b/kernel').write_text('b')
            a=dict(root=str(root/'a'),inputs=['kernel']); b=dict(root=str(root/'b'),inputs=['kernel'])
            evidence=dict(incumbent_source=a,candidate_source=b,acceptance={'bar':1},qualified_targets=['pi05'],gate_verdict='pass')
            self.assertTrue(promotion.applicability(evidence,a,b,{'bar':1},['pi05'])['applicable'])
            self.assertFalse(promotion.applicability(evidence,b,b,{'bar':1},['pi05'])['applicable'])
            self.assertFalse(promotion.applicability(evidence,a,b,{'bar':1},['pi0','pi05'])['applicable'])
            self.assertFalse(promotion.applicability(evidence,a,b,{'bar':2},['pi05'])['applicable'])
