"""GPU checks for graph stream ownership and producer/consumer ordering."""
import pytest
import torch

from flash_vla.runtime.cuda.graph import StreamGraph
from flash_vla.runtime.cuda.program import Program, Segment
from flash_vla.runtime.cuda.timing import capture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("helper", [False, True])
def test_replay_keeps_capture_stream_and_orders_caller_work(monkeypatch, helper):
    x = torch.zeros(4096, device="cuda")
    y = torch.empty_like(x)

    def run():
        torch.cuda._sleep(2_000_000)
        torch.add(x, 2, out=y)

    run()
    torch.cuda.synchronize()
    if helper:
        graph = capture(run)
    else:
        graph = StreamGraph()
        with graph.capture():
            run()
    raw_replay = torch.cuda.CUDAGraph.replay
    observed = []

    def replay(raw):
        observed.append(torch.cuda.current_stream().cuda_stream)
        return raw_replay(raw)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    callers = [torch.cuda.current_stream(), torch.cuda.Stream(), graph.stream]
    for value, caller in enumerate(callers):
        with torch.cuda.stream(caller):
            x.fill_(value)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            consumed = y.clone()
        caller.synchronize()
        assert torch.equal(consumed, torch.full_like(consumed, value + 2))
        assert start.elapsed_time(end) > 0.1
    assert observed == [graph.stream.cuda_stream] * len(callers)


def test_segments_have_distinct_streams_and_preserve_dataflow():
    x = torch.zeros(4096, device="cuda")
    y, z = torch.empty_like(x), torch.empty_like(x)
    program = Program([Segment("double", lambda: torch.mul(x, 2, out=y)),
                       Segment("offset", lambda: torch.add(y, 3, out=z))])
    streams = [graph.stream.cuda_stream for graph in program.graphs.values()]
    assert len(set(streams)) == 2
    for caller in (torch.cuda.current_stream(), torch.cuda.Stream()):
        with torch.cuda.stream(caller):
            x.fill_(7)
            program.replay_all()
            consumed = z.clone()
        caller.synchronize()
        assert torch.equal(consumed, torch.full_like(consumed, 17))
