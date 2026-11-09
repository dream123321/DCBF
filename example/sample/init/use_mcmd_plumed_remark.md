# MCMD and PLUMED Template Usage

How to use MCMD / PLUMED templates in sample/init

1. The program only reads one active LAMMPS template:
   init/lmp_in.py

2. Switch by copying or renaming the desired template to lmp_in.py:
   - Normal MD:
     use init/lmp_in.py
   - MCMD:
     copy init/lmp_in_mcmd.py -> init/lmp_in.py
   - PLUMED metadynamics:
     copy init/lmp_in_plumed.py -> init/lmp_in.py

3. PLUMED mode also needs:
   - init/input.plumed.tmpl
   The workflow will generate a local input.plumed in each run directory
   and replace __TEMP__ with the current MD temperature automatically.

4. If input.plumed.tmpl uses extra files such as:
   - NDX_FILE=water.ndx
   then put the required file in init/ too, for example:
   - init/water.ndx

5. The current sample input.plumed.tmpl is a simple Si smoke-test example:
   - CV: DISTANCE ATOMS=1,2
   - no water.ndx needed

   Important: input.plumed.tmpl is not a universal PLUMED input.
   For a real system, edit it according to your own structure and literature reference:
   choose the proper collective variable(s), atom ids or groups, SIGMA, HEIGHT,
   PACE, BIASFACTOR, wall settings, temperature, and any required index files.
   Do not directly reuse the Si smoke-test CV unless it matches your target system.

6. Current examples are separate:
   - lmp_in_mcmd.py = MCMD
   - lmp_in_plumed.py = PLUMED
   They are not merged into one template yet.
