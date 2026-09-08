import copy
import unittest
from benchmarks.kernel_trace.export_perfetto import convert, union_ns, overlap_ns


def fixture():
    base = dict(device_id="synthetic:0", launch_id=1, replay_id=0, cta=[0,0,0],
                warp_id=0, role="cuda_role", stage="math", semantic="software_scope",
                token=0, sm_begin=7, sm_end=7, start_ns=10**18, end_ns=10**18+10000)
    other = dict(base, warp_id=1, role="tensor_role", stage="mma", start_ns=10**18+5000, end_ns=10**18+15000)
    return dict(schema_version=1, synthetic=True, clock_domain="gpu-local", timer_unit="ns",
                timer_resolution_ns=None, dropped_records=0, coverage="synthetic", ranges=[base,other])


class ExportTests(unittest.TestCase):
    def test_large_epoch_integer_subtraction_and_us(self):
        trace, summary = convert(fixture())
        ranges = [e for e in trace["traceEvents"] if e["ph"] == "X"]
        self.assertEqual([e["ts"] for e in ranges], [0.0,5.0])
        self.assertEqual([e["dur"] for e in ranges], [10.0,10.0])
        self.assertTrue(trace["metadata"]["synthetic"])
        self.assertFalse(summary["production_gate_eligible"])
    def test_overlap_is_not_sum(self):
        _, s = convert(fixture())
        o=s["observed_scope_overlaps"][0]
        self.assertEqual(o["observed_scope_overlap_ns"],5000)
        self.assertEqual(o["observed_scope_union_ns"],15000)
    def test_interval_math(self):
        self.assertEqual(union_ns([(0,10),(5,15),(20,22)]),17)
        self.assertEqual(overlap_ns([(0,10),(5,15)],[(8,12)]),4)
    def test_migration_not_attributed(self):
        f=fixture(); f["ranges"][0]["sm_end"]=9
        _, s=convert(f)
        self.assertEqual(len(s["scope_statistics"]),1)
        self.assertFalse(s["observed_scope_overlaps"])
        self.assertTrue(s["warnings"])
    def test_dropped_default_rejected(self):
        f=fixture(); f["dropped_records"]=1
        with self.assertRaises(ValueError): convert(f)
    def test_partial_view_no_full_stats(self):
        f=fixture(); f["dropped_records"]=1
        _, s=convert(f,allow_partial=True)
        self.assertEqual(s["scope_statistics"],[])
        self.assertEqual(s["observed_scope_overlaps"],[])
    def test_duplicate_rejected(self):
        f=fixture(); f["ranges"].append(copy.deepcopy(f["ranges"][0]))
        with self.assertRaises(ValueError): convert(f)
    def test_repeated_launch_separate(self):
        f=fixture(); r=copy.deepcopy(f["ranges"][0]); r["launch_id"]=2; f["ranges"].append(r)
        _, s=convert(f); self.assertEqual(s["range_count"],3)
    def test_reversed_rejected(self):
        f=fixture(); f["ranges"][0]["end_ns"]=0
        with self.assertRaises(ValueError): convert(f)
    def test_multidevice_not_assumed_aligned(self):
        f=fixture(); f["ranges"][1]["device_id"]="other"
        with self.assertRaises(ValueError): convert(f)
    def test_missing_fields_rejected(self):
        f=fixture(); del f["ranges"][0]["start_ns"]
        with self.assertRaises(ValueError): convert(f)
    def test_zero_duration_preserved(self):
        f=fixture(); f["ranges"][0]["end_ns"]=f["ranges"][0]["start_ns"]
        t,_=convert(f); self.assertTrue(any(e.get("dur")==0 for e in t["traceEvents"]))
    def test_crossing_ranges_separate_tracks(self):
        f=fixture(); f["ranges"][1].update(warp_id=0,role="cuda_role",stage="second")
        t,_=convert(f); events=[e for e in t["traceEvents"] if e["ph"]=="X"]
        self.assertNotEqual(events[0]["tid"],events[1]["tid"])
    def test_sparse_sm_id(self):
        f=fixture()
        for r in f["ranges"]: r.update(sm_begin=173,sm_end=173)
        _,s=convert(f); self.assertEqual(s["scope_statistics"][0]["sm_id"],173)
    def test_float_timestamp_rejected(self):
        f=fixture(); f["ranges"][0]["start_ns"]=1.5
        with self.assertRaises(ValueError): convert(f)
    def test_fake_hardware_semantic_rejected(self):
        f=fixture(); f["ranges"][0]["semantic"]="tensor_core_active"
        with self.assertRaises(ValueError): convert(f)

if __name__ == "__main__": unittest.main()


class MarkerExportTests(unittest.TestCase):
    def test_pair_instants_preserve_units_and_roles(self):
        raw=fixture()
        marker=dict(device_id='synthetic:0',launch_id=1,replay_id=0,cta=[0,0,0],
                    operation='tma',token=0,generation=2,role='producer',kind='issue',timestamp_ns=10**18)
        raw['markers']=[marker,dict(marker,role='consumer',kind='completion_observed',timestamp_ns=10**18+5000)]
        trace,summary=convert(raw)
        events=[e for e in trace['traceEvents'] if e['ph']=='i']
        self.assertEqual([e['ts'] for e in events],[0,5])
        self.assertNotEqual(events[0]['tid'],events[1]['tid'])
        raw['markers'].pop()
        self.assertTrue(convert(raw)[1]['warnings'])
