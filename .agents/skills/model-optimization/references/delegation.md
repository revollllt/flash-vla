# Delegating the kernel loop

What holds for the kernel agents step 5 starts. The model loop stays serial;
this is the contract each kernel agent runs under, and what makes a run's
fan-out a recorded setting rather than an accident -- under the one-line rule
alone, comparable runs have fanned out anywhere from zero to thirty worktrees.

- **One hotspot, one agent, one worktree**, started from the current deployed
  version and stating that commit. Fan-out is bounded by the hotspots worth the
  work at once and by this round's budget; the run README records how many ran
  at once.
- **One GPU, one lock.** Design and compilation overlap; device work does not.
  Every GPU job of every agent -- timing, profiling, on-device correctness, and
  compilation heavy enough to move a timing -- takes the run's one `flock` file,
  named in the run's ignored environment file, for one bounded experiment and
  releases it. A busy lock means do analysis, not wait as if the experiment
  failed, and never bypass it. A model timing run has nothing else on the
  device (step 6).
- **A moved base invalidates a local gain.** When the deployed version advances
  while an agent is in flight, the agent rebases onto it and re-runs its
  numerical check and local timing before returning; a candidate measured
  against a base that no longer exists is not a candidate. An accepted change
  can alter what a kernel assumed -- a packed projection changes the row stride
  every later attention kernel reads.
- **The return packet is enough to continue without the agent.** Diff, exact
  commands, raw numbers, the base commit, and the paths tried and refused with
  the toolbox they were refused against. A stalled or exhausted agent is
  stopped and a fresh one continues from its packet; nothing is nursed, and a
  refused path is not re-run by the next agent unless the toolbox changed.
- **Same budget, same materials.** Kernel agents' compilation, validation and
  GPU waiting count against the round's budget, with GPU waiting recorded
  separately; they read the materials the delegating session may read and
  nothing else. Their model and harness are what the brief names, the
  delegating session's own unless it says otherwise -- comparing two runs needs
  that line in both briefs.
