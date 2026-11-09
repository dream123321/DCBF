# DCBF Installation

Optimal Chemical-Bond-level Fine Sampling

Current source release: SUS2-SH CPU `v3-sus2sh-20261109` (based on `v3-20261003`).

## 1. Recommended: SUS2-SH CPU Deployment Package

Support ldd (GNU libc) version ≥ 2.17

## Download

Download the DCBF SUS2-SH CPU deployment package for this build.

```bash
tar -zxvf DCBF_SUS2SH-CPU.tar.gz
cd DCBF_SUS2SH-CPU
bash install.sh
source activate.sh
bash verify.sh
dcbf -h
```

The DFT software needs to be installed by yourself (VASP, ABACUS, CP2K or QE).

```bash
cd source/DCBF/example/sample
```

Then modify the submission queue, `dft_env` and `dft_command`. Before starting the task, source the path/to/source.sh.

```bash
source path/to/source.sh
dcbf run dcbf.init_dataset.vasp.test.json
```

## 2. Kill A Managed Background Run

```bash
dcbf kill .
```

Or specify the run directory/config path:

```bash
dcbf kill dcbf.init_dataset.vasp.test.json
```

## 3. Manual Build From Source

### 3.1 Install Python dependencies

```bash
conda create --name dcbf_env python=3.10
cd source/DCBF/dcbf
python -m pip install -r requirement.txt
python setup.py install
```

### 3.2 Build superlinear (SUS2-MLIP native core)

Build the native core from the bundled `superlinear-native-src.tar.gz`
([superlinear](https://github.com/hu-yanxiao/superlinear)).

Expected outputs:

- `bin/mlp-sus2`
- `lib/lib_mlip_interface.a`

### 3.3 Build pysuperlinear

`pysuperlinear` is the Python/ASE interface of the same superlinear codebase.
The CPU bundle ships it prebuilt. To rebuild it, use
[superlinear](https://github.com/hu-yanxiao/superlinear) at commit
`e093be57fef9622a3072bf05c7c640f8bcdfcbb5` (`interfaces/python`). This replaces
the legacy `pymlip` binding.

### 3.4 Build LAMMPS Interface

```bash
tar -zxzf sus2-embedded-interface.tar.gz
cd /path/to/lammps/src
cp -r /path/to/sus2-embedded-interface/ML-SUS2 ./ML-SUS2
make no-user-mlip || true
make no-ML-SUS2 || true
make yes-ML-SUS2
make mpi -j 16
```

Expected output:

- `lmp_mpi`

The SUS2-SH CPU workflow uses `pair_style sus2/embedded`; the interface also
keeps `pair_style sus2mtp`.

Legacy `pymlip` and `sus2-interface-20260410` instructions remain available in
tag `v3-20261003`.
