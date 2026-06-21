# VASP POTCAR and k-point customization

This example directory now supports custom VASP pseudopotentials and engine-specific k-point inputs.

## VASP

### Custom POTCAR

If you place an `init/POTCAR_dir/` directory here, DCBF  will build `POTCAR` from your files instead of calling `vaspkit`.

Supported layouts inside `POTCAR_dir/`:

- `Li/POTCAR`
- `F/POTCAR`
- `POTCAR_Li`
- `POTCAR_F`

The builder searches by element symbol and concatenates files in POSCAR element order.

### Custom KPOINTS

If you place `init/KPOINTS`, it will be copied directly into every VASP SCF task directory.

Use the native VASP `KPOINTS` file format from the VASP documentation.

When `init/KPOINTS` is provided, DCBF removes `KSPACING` and `KGAMMA`
from the copied `INCAR` in each task directory so the explicit `KPOINTS` file
is the only active k-point definition.

## ABACUS

If you place `init/KPT`, it will be copied directly into every ABACUS SCF task directory.

Use the native ABACUS `KPT` file format from the ABACUS documentation.

When `init/KPT` is provided, DCBF removes `kspacing` from the copied
`INPUT` in each task directory so the explicit `KPT` file is the only active
k-point definition.

## CP2K

DCBF keeps CP2K in the previous stable mode.

- it copies your `init/cp2k.inp` template
- it patches the lattice vectors for each sampled structure
- it does not add any extra CP2K k-point customization layer

## Quantum ESPRESSO

QE already supports custom k-points in two ways:

- put a native `K_POINTS` card directly in `init/qe.in`
- or provide `init/qe_kpoints.yaml` for the helper-based mesh generation already supported by this branch

If `init/qe.in` already contains a native `K_POINTS` card, DCBF keeps
that card and does not override it with `qe_kpoints.yaml`.
