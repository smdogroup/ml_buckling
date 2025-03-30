# run both axial and shear datasets
mpirun -n 4 python 1_gen_data.py --metal --axial --clear
mpirun -n 4 python 1_gen_data.py --metal --no-axial --clear

mpirun -n 4 python 1_gen_data.py --no-metal --axial
mpirun -n 4 python 1_gen_data.py --no-metal --no-axial
