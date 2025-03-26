# TODO for selection of best GP the closed-form

- [ ] make new run script that runs hyperparameter opt and evaluation for a given kernel + data transform
    - [ ] archive the hyperparam_opt script
    - [ ] k-fold cross validation a method with k an argument
    - [ ] make the evaluation with R^2 a method
    - [ ] writes out the opt theta + R^2 to a file (for each case in its own folder)
    - [ ] use argparse settings in this script
    - [ ] append R^2 results to overall csv (that I can copy data over to main table)
    - [ ] plots the model fit in that case folder
- [ ] make a bash script that runs all cases of each kernel + data transform for the data table
- [ ] make a plot that shows the buckling + RQ kernel extrap R^2 increases with increasing k in k-fold

In the paper:
- [ ] change hyperparam opt explanation to k-fold cross validation (eqn for that)
    - [ ] explain that the MAP or MLE methods led to overfitting the model / bad extrapolation error
    - [ ] explain I only do this hyperparameter fitting on the interpolation zone, not extrapolation
- [ ] make a table of the different primitive kernel functions in Appendix or in main paper part
- [ ] explain why I went with buckling + RQ
    - [ ] explain why the buckling + SE has really good R^2 for longer length scales but hyperparameter opt pushes to lower length scales due to some discontinuity / mode switching behavior
    - [ ] explain that the RQ kernel has a range of length scales in it (SE doesn't) => leads to better hyperparam optimized models than SE
- [ ] make a plot of k-fold cross validation (k vs extrap R^2 of the buckling + SE)
- [ ] update the plot of the best interpolating and best extrapolating mdoels (standard vs buckling + RQ kernels)