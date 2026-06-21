# DCBF Installation

Dual-space Chemical-Bond-level Fine Sampling

## 1. Recommended: One-Button Deployment Package
Support ldd (GNU libc) version ≥ 2.17 

## Download

Download the DCBF deployment package provided for this build.


```bash
tar -zxvf dcbf_one-button_deployment.tar.gz
cd dcbf_one-button_deployment
bash install.sh
source activate.sh
bash verify.sh
dcbf -h "View more functions" (dcbf train -h)
```

The DFT software needs to be installed by yourself. if scf_cal_engine = abacus, please  install ase-abacus (pip install git https://gitlab.com/1041176461/ase-abacus.git).  

```bash
cd source/DCBF/example/sample
```
Then modify the submission queue, dft_env and dft_command. Before starting the task, you need to source the path/to/source.sh.
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

### 3.2 Build SUS2-MLIP

install [SUS2-MLIP](https://github.com/hu-yanxiao/SUS2-MLIP)


Expected outputs:
- `bin/mlp-sus2`
- `lib/lib_mlip_interface.a`

### 3.3 Build pymlip

```bash
tar -zxvf pysus2mlip.tar.gz
cd pysus2mlip
```

Edit `setup.py` so that:
- `mlip_include_dir` points to:
  - `<SUS2-MLIP_latest>/src/common`
  - `<SUS2-MLIP_latest>/src`
  - `<SUS2-MLIP_latest>/dev_src`
- `extra_objects` points to:
  - `<SUS2-MLIP_latest>/lib/lib_mlip_interface.a`

Then install:

```bash
CC=gcc CXX=g++ python -m pip install .
```

### 3.4 Build LAMMPS Interface
SUS2-MLIP models can be used in [LAMMPS](https://github.com/lammps/lammps) simulation via the sus2-interface.

```bash

tar -zxzf sus2-interface-20260410.tar.gz
cd /path/to/lammps/src
cp -r /path/to/sus2-interface-20260410/ML-SUS2 ./ML-SUS2
make no-user-mlip || true
make no-ML-SUS2 || true
make yes-ML-SUS2
make mpi -j 16
```

Expected output:
- `lmp_mpi`
