# re-archive both models (useful if testing changes to kernel functions and theta)
rm *.txt
python _combine_stiff_unstiff_data.py --load Nx
mpirun -n 4 python 2_train_model.py --load Nx >> axial_hp_opt.txt
python _combine_stiff_unstiff_data.py --load Nxy
mpirun -n 4 python 2_train_model.py --load Nxy >> shear_hp_opt.txt