# TODO for selection of best GP the closed-form

- [x] make new run script that runs hyperparameter opt and evaluation for a given kernel + data transform
    - [x] archive the hyperparam_opt script
    - [x] k-fold cross validation a method with k an argument
    - [x] make the evaluation with R^2 a method
    - [x] make the plotting a method?
    - [x] writes out the opt theta + R^2 to a file (for each case in its own folder)
    - [x] use argparse settings in this script
    - [x] append R^2 results to overall csv (that I can copy data over to main table)
    - [x] plots the model fit in that case folder
    - [x] add all theta hyperparam settings
- [x] make a bash script that runs all cases of each kernel + data transform for the data table
- [x] make a bash script for the different kfold runs of buckling + RQ kernel (included in same one)git 

Get results:
- [x] run the run_cases.sh
- [x] compile data for main table into master.csv
- [x] make plot of axial + shear buckling+RQ extrap error with num_kfold

In the paper:
- [x] change hyperparam opt explanation to k-fold cross validation (eqn for that)
    - [x] explain that the MAP or MLE methods led to overfitting the model / bad extrapolation error
    - [ ] explain I only do this hyperparameter fitting on the interpolation zone, not extrapolation
- [x] make a table of the different primitive kernel functions in Appendix or in main paper part
- [ ] explain why I went with buckling + RQ
    - [ ] explain why the buckling + SE has really good R^2 for longer length scales but hyperparameter opt pushes to lower length scales due to some discontinuity / mode switching behavior
    - [ ] explain that the RQ kernel has a range of length scales in it (SE doesn't) => leads to better hyperparam optimized models than SE
- [x] make a plot of k-fold cross validation (k vs extrap R^2 of the buckling + SE)
- [x] update the plot of the best interpolating and best extrapolating mdoels (standard vs buckling + RQ kernels)
- [ ] cite this paper in showing how to select different kernel terms? https://arxiv.org/pdf/1302.4922 and https://www.cs.toronto.edu/~duvenaud/cookbook/
- [x] and cite / discuss this for k-fold cross validation in text, https://www.mdpi.com/2079-9292/10/16/1973 and https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf