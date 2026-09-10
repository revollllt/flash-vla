import unittest
from tools.profiling.kernel_trace.query import query, validate_tokens


def fixture():
    raw=dict(clock_domain='gpu-local',timer_unit='ns',dropped_records=0,coverage='selected CTA',ranges=[])
    base=dict(device_id='gpu',launch_id=1,replay_id=0,cta=[0,0,0],operation='tma',token=0,generation=0)
    raw['markers']=[dict(base,kind='issue',timestamp_ns=100,role='producer'),
                    dict(base,kind='completion_observed',timestamp_ns=200,role='consumer')]
    return raw


class QueryTests(unittest.TestCase):
    def test_cross_role_pair(self):
        result=validate_tokens(fixture())
        self.assertTrue(result['complete'])
        self.assertEqual(result['pairs'][0]['observed_window_ns'],100)

    def test_generation_and_replay_not_interchangeable(self):
        for key in ('generation','replay_id'):
            raw=fixture(); raw['markers'][1][key]=1
            result=validate_tokens(raw)
            self.assertFalse(result['complete']); self.assertEqual(result['pairs'],[])

    def test_duplicate_and_reversed(self):
        raw=fixture(); raw['markers'].append(raw['markers'][0])
        with self.assertRaises(ValueError): validate_tokens(raw)
        raw=fixture(); raw['markers'][1]['timestamp_ns']=0
        with self.assertRaises(ValueError): validate_tokens(raw)

    def test_unknown_clock_and_partial(self):
        raw=fixture(); raw['clock_domain']='unknown'
        with self.assertRaises(ValueError): query(raw)
        raw=fixture(); raw['dropped_records']=1
        self.assertFalse(query(raw)['tokens']['complete'])

    def test_filters_do_not_invent_coverage(self):
        raw=fixture(); raw['ranges']=[dict(stage='wait',role='consumer',semantic='wait_scope',start_ns=100,end_ns=200,task_id=1)]
        self.assertEqual(query(raw,task_id=2)['range_count'],0)
        self.assertEqual(query(raw,task_id=1)['coverage'],'selected CTA')
