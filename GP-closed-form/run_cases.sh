# shell script to generate main table data
#   switched bools to store_true now

export NUM_KFOLDS=20
# standard literature axial
python 1_train_and_eval.py --kernel "SE" --axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "SE" --axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-3_2" --axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-3_2" --axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-5_2" --axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-5_2" --axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "RQ" --axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "RQ" --axial --affine --no-log --kfolds ${NUM_KFOLDS}

# # standard literature shear
python 1_train_and_eval.py --kernel "SE" --no-axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "SE" --no-axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-3_2" --no-axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-3_2" --no-axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-5_2" --no-axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "matern-5_2" --no-axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "RQ" --no-axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "RQ" --no-axial --affine --no-log --kfolds ${NUM_KFOLDS}

# custom buckling kernels axial
python 1_train_and_eval.py --kernel "buckling+SE" --axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+SE" --axial --no-affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+SE" --axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+SE" --axial --no-affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --no-affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --no-affine --no-log --kfolds ${NUM_KFOLDS}

# custom buckling kernels shear
python 1_train_and_eval.py --kernel "buckling+SE" --no-axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+SE" --no-axial --no-affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+SE" --no-axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+SE" --no-axial --no-affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --no-affine --log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --affine --no-log --kfolds ${NUM_KFOLDS}
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --no-affine --no-log --kfolds ${NUM_KFOLDS}

# run kfold study on buckling+RQ
# ------------------------------

# first for axial
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --affine --log --kfolds 5
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --affine --log --kfolds 10
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --affine --log --kfolds 20
python 1_train_and_eval.py --kernel "buckling+RQ" --axial --affine --log --kfolds 40
# python 1_train_and_eval.py --kernel "buckling+RQ" --axial --affine --log --kfolds 80

# then for shear
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --affine --log --kfolds 5
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --affine --log --kfolds 10
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --affine --log --kfolds 20
python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --affine --log --kfolds 40
# python 1_train_and_eval.py --kernel "buckling+RQ" --no-axial --affine --log --kfolds 80
