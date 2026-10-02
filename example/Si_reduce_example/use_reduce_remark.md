# DCBF Reduce Modes

This example shows two reduce modes for Si datasets: `candidate_only` and `reference_guided`.

## candidate_only reduce

`candidate_only` reduces one candidate dataset by itself. It only uses `reduce.input_xyz` as the candidate pool and selects representative structures from that same pool.

Use this mode when you have one MD trajectory or one structure database and want to compress it into a smaller representative subset.

Minimal meaning:

- `input_xyz`: candidate structures to be reduced, for example `md.xyz`
- `output_xyz`: selected representative structures
- `remain_xyz`: candidate structures not selected

The bundled universal Reduce potential is an 84-element `l2k3` model. This Si
example keeps using its explicit local Si model instead.

In this Si example:

```bash
cd candidate_only
dcbf reduce reduce.candidate_only.Si.json
```

## reference_guided reduce

`reference_guided` reduces a new candidate dataset against an existing reference/training dataset. It uses the reference set to decide which candidate structures add new descriptor-space coverage.

Use this mode in active-learning style workflows: keep an existing training set, then select additional useful structures from a new MD trajectory.

Minimal meaning:

- `current_xyz`: existing training/reference structures, for example `train.xyz`
- `interval_ref_xyz`: reference grid/coverage baseline; normally the same as `current_xyz`
- `input_xyz`: new candidate structures, for example `md.xyz`
- `output_xyz`: `current_xyz` plus newly selected candidate structures when `append_current=true`
- `remain_xyz`: candidate structures not selected

`reduce.chunk_size` defaults to `1000000`; set it explicitly only when a
smaller batch is needed for memory control.

In this Si example:

```bash
cd reference_guided
dcbf reduce reduce.reference_guided.Si.json
```

## XYZ I/O mode

`reduce.xyz_io_mode` defaults to `fast_extxyz`. It indexes standard EXTXYZ frames by byte range, copies selected frames without reserializing them, and writes CFG shards in parallel with `encoding_cores`. This is faster and uses much less memory than loading the full dataset as `ase.Atoms` objects, while retaining the original headers, labels, and numeric text.

- `fast_extxyz`: strict and fastest; incompatible input raises an explicit error.
- `auto`: prefers the fast path and falls back to ASE with a warning when needed.
- `ase`: forces the original ASE implementation for compatibility or comparison.

## Dimension-level minimum cover

`reduce.dimension_min_cover_workers` defaults to `-1`, using all CPUs allocated or visible on the current node. Set it to `0` for the original joint cover, `1` for serial per-dimension cover, or a positive integer for limited parallel workers. Every nonzero mode automatically unions the independently selected structures and applies global reverse pruning while preserving all coverage and population requirements. The result may still differ from the joint solver.

Reference-guided Reduce automatically saves a checkpoint after each completed
chunk. Re-running the same `dcbf reduce reduce.json` command after an
interruption resumes from the next unfinished chunk only after all input,
model, descriptor, chunk, and selection settings match. The default
`keep_intermediate=false` cleans checkpoint and intermediate files after a
successful run, while an interrupted run keeps its checkpoint for automatic
resume. Set `keep_intermediate=true` to retain intermediate files for
inspection. Candidate-only Reduce may reuse a complete descriptor cache, but
does not resume an interrupted minimum-cover solve. Changed inputs, models, or
selection parameters invalidate the old checkpoint and restart the run.
