# TODO in Stiffened Panels

- [ ] before regen datasets of metal + composite and axial+shear, validate models better
    - [ ] make tests/ directory and unittests of eig_FEA not None on some high gamma, low rho0
    - [ ] compare to old gamma = 3.0 metal datasets, why solve failing now?
    - [ ] try changing sigma = 5 to 10, scale up strain by CF, eig = 1 to 2.5 and then scale up output
    - [ ] investigate why crippling and high error 1e4 in solutions (how to get better eigenvalue solves)
    - [ ] complete verification or tests/
    - [ ] ant optimization not causing problems? (seemed good now..)
    - [ ] run metal axial dataset again and see if indeed getting more smooth dataset
- [ ] rerun metal + composite, axial + shear datasets (4 total runs)
- [ ] then move onto generating surrogate model again
    - [ ] hyperparam optimize axial
    - [ ] see if dataset more consistent with unstiffened and slopes in extrap region better