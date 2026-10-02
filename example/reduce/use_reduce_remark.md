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

When the bundled universal potential is selected, Reduce uses the 84-element
`l2k3` model shipped with DCBF. Custom models and explicit element mappings
continue to take precedence.

## XYZ I/O mode

`reduce.xyz_io_mode` controls how Reduce reads, splits, and converts structure files. The default `fast_extxyz` uses a byte-range frame index, copies selected EXTXYZ blocks without rewriting them, and generates CFG shards in parallel with `encoding_cores`. This avoids building one in-memory `ase.Atoms` list for the full dataset and substantially reduces time and memory for large databases while preserving the original headers, labels, and numeric text.

- `fast_extxyz`: strict and fastest; requires standard EXTXYZ with `Lattice`, element information, and positions in `Properties`.
- `auto`: prefers the fast path but warns and falls back to ASE for unsupported XYZ/EXTXYZ or `.traj` input.
- `ase`: always uses the original ASE read, split, rewrite, and CFG-conversion path.

## Dimension-level minimum cover

`reduce.dimension_min_cover_workers` controls the minimum-cover strategy. The default `-1` uses all CPUs allocated or visible on the current node. Use `0` for the original joint cover, `1` for serial per-dimension cover, or a positive integer for that many worker processes. Every nonzero mode automatically takes the per-dimension union and then applies deterministic global reverse pruning. A structure is removed only when every original coverage or population target remains satisfied. The result can still differ from the original joint solver.

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

`reduce.chunk_size` defaults to `1000000`. It controls how many candidate
structures are handled in each reference-guided batch; an explicit JSON value
still overrides this default.

Reference-guided Reduce automatically checkpoints after every completed chunk.
If the process is interrupted, the next `dcbf reduce reduce.json` run verifies
the input files, model, elements, descriptor settings, chunk size, and other
selection parameters, then resumes from the next unfinished chunk. The default
`keep_intermediate=false` removes the checkpoint and intermediate descriptor
data after a successful run, but keeps an interrupted checkpoint for resume.
`keep_intermediate=true` keeps the checkpoint and intermediate files for
inspection. Candidate-only Reduce can reuse a complete descriptor cache, but
its minimum-cover calculation is not resumed in the middle. If inputs, the
model, or selection parameters change, the old checkpoint is rejected and the
run starts from chunk 0.
