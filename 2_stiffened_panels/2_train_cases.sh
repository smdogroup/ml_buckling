# different training cases to examine model fit
python 2_kfold_opt.py --load "Nx" --ntrain 1000 --kfold 10
python 2_kfold_opt.py --load "Nxy" --ntrain 1000 --kfold 10

python 2_kfold_opt.py --load "Nx" --ntrain 1000 --kfold 30
python 2_kfold_opt.py --load "Nxy" --ntrain 1000 --kfold 30

python 2_kfold_opt.py --load "Nx" --ntrain 2000 --kfold 30
python 2_kfold_opt.py --load "Nxy" --ntrain 2000 --kfold 30

python 2_kfold_opt.py --load "Nx" --ntrain 3000 --kfold 30
python 2_kfold_opt.py --load "Nxy" --ntrain 3000 --kfold 30