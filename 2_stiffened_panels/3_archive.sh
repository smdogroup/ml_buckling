# re-archive both models (useful if testing changes to kernel functions and theta)
python _combine_stiff_unstiff_data.py --load Nx
python 3_eval_and_archive_model.py --load Nx --archive
python _combine_stiff_unstiff_data.py --load Nxy
python 3_eval_and_archive_model.py --load Nxy --archive