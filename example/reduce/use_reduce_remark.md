# DCBF Reduce Modes

This directory contains generic reduce JSON templates. The same naming is used in `example/Si_reduce_example`, which provides a runnable Si test case.

## candidate_only reduce

`candidate_only` reduces one candidate dataset by itself. It only uses `reduce.input_xyz` as the candidate pool and selects representative structures from that same pool.

Use this mode when you have one MD trajectory or one structure database and want to compress it into a smaller representative subset.

Generic template paths:

```text
example/reduce/candidate_only
example/reduce/candidate_only_UIP
```

Minimal meaning:

- `input_xyz`: candidate structures to be reduced, for example `md.xyz`
- `output_xyz`: selected representative structures
- `remain_xyz`: candidate structures not selected

## reference_guided reduce

`reference_guided` reduces a new candidate dataset against an existing reference/training dataset. It uses the reference set to decide which candidate structures add new descriptor-space coverage.

Use this mode in active-learning style workflows: keep an existing training set, then select additional useful structures from a new MD trajectory.

Generic template path:

```text
example/reduce/reference_guided
```

Minimal meaning:

- `current_xyz`: existing training/reference structures, for example `train.xyz`
- `interval_ref_xyz`: reference grid/coverage baseline; normally the same as `current_xyz`
- `input_xyz`: new candidate structures, for example `md.xyz`
- `output_xyz`: `current_xyz` plus newly selected candidate structures when `append_current=true`
- `remain_xyz`: candidate structures not selected
