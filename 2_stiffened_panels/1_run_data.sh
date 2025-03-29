# run both axial and shear datasets
mpirun -n 4 python 1_gen_data.py --axial --clear
mpirun -n 4 python 1_gen_data.py --no-axial --clear