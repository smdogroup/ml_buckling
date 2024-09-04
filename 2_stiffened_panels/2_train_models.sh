# re-archive both models (useful if testing changes to kernel functions and theta)
mpirun -n 4 python 2_train_model.py --load Nx
mpirun -n 4 python 2_train_model.py --load Nxy