# TODO in Stiffened Panels

- [ ] before regen datasets of metal + composite and axial+shear, validate models better
    - [x] make tests/ directory and unittests of eig_FEA not None on some high gamma, low rho0
    - [x] compare to old gamma = 3.0 metal datasets, why solve failing now?
    - [x] try changing sigma = 5 to 10, scale up strain by CF, eig = 1 to 2.5 and then scale up output
    - [x] investigate why crippling and high error 1e4 in solutions (how to get better eigenvalue solves)
    - [x] complete verification or tests/
    - [x] ant optimization not causing problems? (seemed good now..)
    - [ ] come up with strategy to check the next stiffener count (odd or even) if None. If increasing stiffener count (start checking at certain rho0^*) increases buckling load significantly maybe more than 10% then stick with that one next time. Check either one fewer or one more stiffener until find one that works.. (since sometimes need more, sometimes need less stiffeners depending on slenderness). If both fail, go down for low slenderness for low gamma, go up for more slender 
    - [ ] run metal axial dataset again and see if indeed getting more smooth dataset
- [ ] rerun metal + composite, axial + shear datasets (4 total runs)
- [ ] then move onto generating surrogate model again
    - [ ] hyperparam optimize axial
    - [ ] see if dataset more consistent with unstiffened and slopes in extrap region better