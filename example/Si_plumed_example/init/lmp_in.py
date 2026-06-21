#
# Longer PLUMED-enabled LAMMPS templates for a 3x3x3 Si example.
# Designed for about 100 dumped frames with 1 ps interval.
#

npt = """variable dt equal 0.001
variable StepNPT equal 99000
variable PreStep equal 0
variable nevery equal 1000
variable out_dump_file_0 string force.0.dump
# variable out_dump_file string

units metal
boundary p p p
atom_style atomic

box tilt large
read_data data.in

pair_style sus2mtp current_0.mtp
pair_coeff * *

neighbor 2.0 bin
neigh_modify delay 1 check yes

timestep    ${dt}
variable Tdamp equal "v_dt * 100"
variable Pdamp equal "v_dt * 1000"
velocity        all create ${T} ${random} mom yes rot yes dist gaussian
thermo_style   custom step temp pe etotal press
thermo 1000

fix plumed_fix all plumed plumedfile input.plumed outfile plumed.out
fix 100 all npt temp ${T} ${T} ${Tdamp} aniso 0.0 0.0 ${Pdamp}
fix 101 all momentum 1 linear 0 0 0
velocity all scale ${T}
velocity all zero linear

run             ${PreStep}

reset_timestep 0

compute pe all pe/atom
dump 0 all custom ${nevery} ${out_dump_file_0} id type x y z fx fy fz c_pe
dump_modify 0 format line "%6d %3d %20.15e %20.15e %20.15e %20.15e %20.15e %20.15e %20.15e" sort id

dump 1 all custom ${nevery} ${out_dump_file} id type x y z fx fy fz c_pe
dump_modify 1 sort id

run             ${StepNPT}
unfix plumed_fix
"""

nvt = """variable dt equal 0.001
variable StepNVT equal 99000
variable PreStep equal 0
variable nevery equal 1000
variable out_dump_file_0 string force.0.dump
# variable out_dump_file string

units metal
boundary p p p
atom_style atomic

box tilt large
read_data data.in

pair_style sus2mtp current_0.mtp
pair_coeff * *

neighbor 2.0 bin
neigh_modify delay 1 check yes

timestep    ${dt}
variable Tdamp equal "v_dt * 100"
variable Pdamp equal "v_dt * 1000"
velocity        all create ${T} ${random} mom yes rot yes dist gaussian
thermo_style   custom step temp pe etotal press
thermo 1000

fix plumed_fix all plumed plumedfile input.plumed outfile plumed.out
fix 100 all nvt temp ${T} ${T} ${Tdamp}
fix 101 all momentum 1 linear 0 0 0
velocity all scale ${T}
velocity all zero linear

run             ${PreStep}

reset_timestep 0

compute pe all pe/atom
dump 0 all custom ${nevery} ${out_dump_file_0} id type x y z fx fy fz c_pe
dump_modify 0 format line "%6d %3d %20.15e %20.15e %20.15e %20.15e %20.15e %20.15e %20.15e" sort id

dump 1 all custom ${nevery} ${out_dump_file} id type x y z fx fy fz c_pe
dump_modify 1 sort id

run             ${StepNVT}
unfix plumed_fix
"""
