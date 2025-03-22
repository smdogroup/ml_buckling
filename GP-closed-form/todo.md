# TODO for selection of best GP the closed-form

- [x] script to generate the closed-form solution data => axial vs shear
    - [x] bools for affine transform, log transform on data
    - [x] flag to include the low rho_0, high gamma extrapolation zone from dataset
    - [ ] train vs test split input
- [ ] 2D and 3D plots of the dataset to check it makes sense
- [ ] hand-coded GP with SE kernel and hyperparam opt with cholesky and pyoptSparse
- [ ] surrogate model with commercial GPs
- [ ] work on custom kernel GPs
- [ ] make table comparing all the GPs
    - [ ] commercial GP: linear+log, affine + no affine
    - [ ] SE kernel: linear+log, affine + no affine
    - [ ] custom kernels: log, affine + no affine
- [ ] identify the best kernels!