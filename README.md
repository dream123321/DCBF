# DCBF

Dual-space Chemical-Bond-level Fine Sampling (DCBF) is an active-learning workflow for SUS2/MLIP potential training, molecular dynamics sampling, dataset reduction, and post-analysis.

This repository contains the source code and examples. For normal HPC use, the recommended installation route is the one-button deployment package from GitHub Releases.

## Recommended Installation

Download the latest deployment package from Releases:

[Download `dcbf_one-button_deployment.tar.gz`](https://github.com/dream123321/DCBF/releases/download/deploy-20260621/dcbf_one-button_deployment.tar.gz)

```bash
tar -zxvf dcbf_one-button_deployment.tar.gz
cd dcbf_one-button_deployment
bash install.sh
source activate.sh
bash verify.sh
dcbf -h
dcbf train -h
dcbf coverage-pca -h
```

DFT software such as VASP must be installed by the user. For ABACUS workflows, install the required ASE ABACUS interface separately.

## Quick Start

After installation, start from the sample configuration:

```bash
source /path/to/dcbf_one-button_deployment/activate.sh
cd /path/to/dcbf_one-button_deployment/source/DCBF/example/sample
```

Edit the JSON file to match your cluster queue, `dft_env`, `dft_command`, and structure paths. Then run:

```bash
dcbf run dcbf.init_dataset.vasp.test.json --prepare-only
dcbf run dcbf.init_dataset.vasp.test.json
```

Stop a managed run with:

```bash
dcbf kill .
```

## Main Commands

```bash
dcbf create-init
dcbf run dcbf.init_dataset.vasp.test.json
dcbf train data.extxyz --template l2k3 --submit
dcbf reduce reduce.json
dcbf coverage-pca --input all_sample_data.xyz --query query.xyz
dcbf plot-errors dft.xyz mlip.xyz
```

## Source Installation

For development-only use:

```bash
conda create -n dcbf_env python=3.10
conda activate dcbf_env
cd dcbf
python -m pip install -r requirement.txt
python setup.py install
```

The full deployment package already bundles the tested runtime, SUS2 developer version, PLUMED-enabled LAMMPS, and example templates.

## Current Update Highlights

- Program name changed from `ocbf` to `dcbf`; commands, examples, and directory naming were updated accordingly.
- The deployment bundles the SUS2 developer version for faster execution and higher accuracy.
- The runtime includes PLUMED-enabled LAMMPS. Example templates are provided through `init/lmp_in_plumed.py` for PLUMED metadynamics and `init/lmp_in_mcmd.py` for MCMD-style custom workflows.
- `core_hours.txt` now includes sampling-stage SUS2MD/LAMMPS core-hour accounting and accumulates multi-structure sampling runs.
- Added `dcbf coverage-pca` for PCA-based coverage analysis between loop/input datasets and a query dataset, with figure, CSV, and PCA text outputs.
- `plot-errors` now uses `sus2_plot_errors_v3.py` from the current deployment package.
- Parameter names and input rules were standardized, especially `coverage_mode`, `coverage_grid`, `body_list`, and `dq_width_*`.

See [`update.md`](update.md) for the concise release notes.

## Examples

- `example/sample`: one-button sampling example.
- `example/sample_json`: JSON templates for different structure-selection workflows.
- `example/reduce`: candidate-only and reference-guided reduce examples.
- `example/Si_reduce_example`: small Si reduce test case.
- `example/Si_plumed_example`: PLUMED/MCMD template example.

## Citation

If you use SUS2-MLIP, please cite the SUS2-MLIP reference listed by `mlp-sus2` and the relevant DCBF workflow documentation.
