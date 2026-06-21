# DCBF Update Notes

Updated: 2026-06-19

- The program name has been changed from `ocbf` to `dcbf`, and the related commands, example configs, and directory naming were updated accordingly.
- The deployment now bundles the SUS2 developer version, which provides faster execution and higher accuracy than the earlier packaged version.
- The packaged runtime now includes a PLUMED-enabled LAMMPS environment. Example templates are provided through `init/lmp_in_plumed.py` for PLUMED metadynamics and `init/lmp_in_mcmd.py` for MCMD-style custom MD workflows. See `init/use_mcmd_plumed_remark.md` for how to switch templates and prepare `input.plumed.tmpl` for the target system.
- `core_hours.txt` now includes sampling-stage SUS2MD/LAMMPS core-hour accounting and accumulates multi-structure sampling runs.
- Added `dcbf coverage-pca` for PCA-based coverage analysis between loop/input datasets and a query dataset, with figure, CSV, and PCA text outputs.
- In `coverage-pca`, the query dataset means the target dataset used to evaluate coverage against the sampled loop data. It can be provided explicitly with `--query` as an external xyz/traj dataset, or generated automatically as `query.xyz` from workspace structures through LAMMPS when `--run-dir` mode is used.
- `plot-errors` now uses `sus2_plot_errors_v3.py` from the current deployment package.
- Parameter names and input rules were further standardized, especially for coverage-related settings, the `body_list` name, and the `dq_width_*` replacements for older parameter names.

Note:

- This file intentionally keeps only the main update points. For the exact parameter list and current defaults, use `dcbf -h`, `dcbf coverage-pca -h`, and the example JSON files.
