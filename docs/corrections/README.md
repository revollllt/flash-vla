# Corrections

Conclusions this project reached and later had to withdraw, kept so the loop
stops re-reaching them. The operative form of each lesson is the review list in
[the optimization workflow](../optimization.md); this directory holds the cases
that put each item on that list, and is where a new case is added.

Records are organized by **failure mode, not by date**. A dated snapshot cannot
answer the question a reader actually has, which is whether this kind of mistake
has been made before.

## Two files, because the evidence is not the same kind

- [human.md](human.md) — caught by a reader outside the loop. Self-
  authenticating: someone asked about a number and the answer changed. The
  transcript and the commit are the record.
- [agent.md](agent.md) — caught by the loop itself. **Not self-authenticating.**
  The reasoning that produced the wrong conclusion also produces the
  retrospective about it, so an entry here is a claim by an interested party and
  needs an independent check before it is believed.

Keeping them apart is the point. Mixing them would let the weaker evidence
inherit the credibility of the stronger, and the first file's whole value is
that it answers a narrower question: **which kinds of error this loop does not
catch by itself.**

## Admitting an entry

Both files want the same shape — what was concluded, what was actually true,
the mechanism, and what it cost or bought — and both reject a case that is only
a bug. A bug is fixed once; a failure mode recurs, and only the second kind
earns a place here.

**human.md** needs the exchange that caused the change: the message, and the
commit or measurement that followed it.

**agent.md** needs, in addition, all three of:

1. **A falsifiable artifact**, not a narrative. A measurement, a commit that
   reverses the earlier one, a validator that now fails — something a reader can
   re-run. "On reflection this was wrong" is not admissible.
2. **The artifact contradicting the earlier conclusion**, rather than merely
   differing from it. A number measured under other conditions is a different
   measurement, not a refutation.
3. **An independent confirmation**, by a reader that did not produce the
   conclusion or the retrospective — a fresh context, following
   [correction-review](../../.agents/skills/correction-review/SKILL.md). Record
   who confirmed it and what they checked.

An entry that cannot meet these belongs in the run's own README as an ordinary
result, not here.

## The failure modes

Each links to its cases. The workflow's review list is the short form of the
same set; this is where the evidence for it lives.

| Mode | The check it produced |
|---|---|
| A number was carried, never measured | a compared number is from this run, or its provenance is stated |
| A ratio was taken against an unchecked denominator | a ratio names its denominator; an impossible ratio is a bug in the denominator first |
| An estimate assumed the wrong memory | an estimate names the memory tier it assumes, against this machine's measured constants |
| A negative outlived its toolbox | a negative carries the toolbox it was measured with, and expires when a technique later worth a factor was not in it |
| A limit was declared from one attempt | "unavailable" needs the boundary probed, not one error string |
| A dependency was treated as a property of the world | a blocker named in a next-steps list is a target; say what would remove it, and cost that |
