# train each case
python data/_combine_stiff_unstiff_data.py --load "Nx"
python 2_kfold_opt.py --load "Nx" --ntrain 1500 --kfold 20 --archive

python data/_combine_stiff_unstiff_data.py --load "Nxy"
python 2_kfold_opt.py --load "Nxy" --ntrain 1500 --kfold 20 --archive