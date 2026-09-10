import unittest
from tools.profiling.timeline import intervals, occurrences, region_occurrences


def event(start, end, site='a', inv=0, stream=7):
    return dict(start_us=start, end_us=end, call_site=site, invocation_id=inv, stream=stream)


class TimelineTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(intervals([event(0, 10), event(5, 15)]),
                         dict(sum_kernel_duration_us=20, interval_union_us=15, region_makespan_us=15))

    def test_serial_and_gap(self):
        stats = intervals([event(0, 10), event(20, 30)])
        self.assertEqual(stats['interval_union_us'], 20)
        self.assertEqual(stats['region_makespan_us'], 30)

    def test_empty_and_reversed(self):
        self.assertEqual(intervals([])['region_makespan_us'], 0)
        with self.assertRaises(ValueError):
            intervals([event(10, 0)])

    def test_repeated_groups_are_not_whole_graph_span(self):
        events = [event(0, 10, 'a', 0), event(5, 15, 'b', 1),
                  event(100, 110, 'a', 2), event(105, 115, 'b', 3)]
        result = region_occurrences(events, {'a', 'b'})
        self.assertEqual(result['sum_occurrence_makespan_us'], 30)
        self.assertEqual(len(occurrences(events)), 4)

    def test_multistream_unknown_group_but_valid_union(self):
        events = [event(0, 10, 'a', 0, 1), event(5, 15, 'b', 1, 2)]
        self.assertEqual(intervals(events)['interval_union_us'], 15)
        self.assertEqual(region_occurrences(events, {'a', 'b'})['status'], 'unsupported')

    def test_unattributed(self):
        events = [event(0, 10, None, None)]
        self.assertEqual(occurrences(events), [])
        self.assertEqual(region_occurrences(events, {'a'})['status'], 'unsupported')

    def test_incomplete_or_ambiguous_groups(self):
        for events in ([event(0, 10)], [event(0, 10, 'a', 0), event(10, 20, 'a', 1), event(20, 30, 'b', 2)]):
            self.assertEqual(region_occurrences(events, {'a', 'b'})['status'], 'unsupported')

    def test_multiple_kernels_one_invocation(self):
        events = [event(0, 3), event(5, 7), event(8, 10, 'b', 1)]
        self.assertEqual(len(occurrences(events)), 2)
        self.assertEqual(region_occurrences(events, {'a', 'b'})['sum_occurrence_makespan_us'], 10)
