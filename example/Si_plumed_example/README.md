# Si PLUMED/MCMD Example

This folder keeps only the reusable input templates for a Si PLUMED or MCMD-style sampling setup.

- `init/lmp_in.py`: template compatible with the DCBF sampling workflow.
- `init/input.plumed.tmpl`: minimal PLUMED metadynamics template. Edit atom indices, collective variables, force constants, temperature placeholders, and units for your own system before production runs.

Runtime outputs such as `COLVAR`, `HILLS`, `log.lammps`, dumps, and submitted job folders are intentionally not tracked in the source repository. They are generated in the working directory during sampling.
